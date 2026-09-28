"""Continuous-batching model of one serving engine (one GPU or one TP group).

Why this exists. The per-request timing strategies price every request as if
it had the GPU to itself: each request sleeps its own prefill time, then its
own per-token decode time. A real serving engine (vLLM's V1 scheduler) runs
one *step* at a time over everything in flight: every running sequence
decodes one token in the same forward pass, and waiting prefills are chunked
into whatever token budget is left. Two consequences matter for the storage
workload an emulator must reproduce:

* Decode time per token grows with the *total* context of the batch (each
  step reads the weights once plus every running sequence's KV), so turns
  slow down when sessions overlap and speed up when they do not.
* A long prefill in the batch (a new 50k-token session) stalls every other
  sequence's decode for the steps it occupies.

Measured on the real GB10 (Llama 3.2 1B, AgentX, 2026-09-27): per-token decode
= 11.9 ms + 0.156 ms per 1k tokens of batch context + ~78 ms x (fraction of
time another request is prefilling), 12 % median error over 131 requests. The
emulator without this model was flat at ~49 ms/token regardless of batch,
which halved the turn rate and changed which sessions overlapped in the KV
cache tiers.

Model. Step time is a roofline over the whole batch::

    flops = 2 * P_lin * (decode tokens + prefill tokens)
          + 4 * L_attn * H * d * (sum(decode ctx) + sum(n * c + n^2 / 2 per chunk))
    bytes = W + kv_bytes_per_token * (sum(decode ctx) + sum(c + n per chunk))
    step_ms = overhead_ms + 1000 * max(flops / (peak_flops * eff_flops),
                                       bytes / (peak_bw * eff_bw))

with W the weight bytes read per step and P_lin the non-embedding parameters.
Two efficiencies per GPU are the only fitted numbers; on GB10 one real capture
gives eff_bw = 0.90 and eff_flops = 0.45 (see DEFAULT_* below), which
reproduce single-stream decode, batched decode, cold 49k-117k-token
prefills, and warm-turn TTFT together.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field

from kvbench.core.gpu_profiles import GPUProfile, get_gpu_profile
from kvbench.core.models import ModelProfile, get_model_profile

logger = logging.getLogger(__name__)

# Fitted on the GB10 AgentX capture (8 GB host tier) by open-loop replay of
# its 139 real requests through this model, using the tokens vLLM actually
# generated (aiperf re-tokenizes output text and under-counts by ~17 %; its
# osl_mismatch_diff_pct recovers the true count). Median per-request latency
# error 3 %, run totals 0.99x latency / 0.97x TTFT; held-out 24 GB and
# 8 GB + 16 GB-disk captures: 4 % / 3 %, totals 1.02x / 0.99x. Used when a
# GPU has no fitted values of its own.
DEFAULT_EFF_BW = 0.90
DEFAULT_EFF_FLOPS = 0.45
DEFAULT_FRONTEND_US_PER_TOKEN = 1.0
DEFAULT_KV_LOAD_GB_S = 12.0
# vLLM V1 default max_num_batched_tokens for the OpenAI server on large-memory
# devices; decode tokens count against it.
DEFAULT_TOKEN_BUDGET = 8192
DEFAULT_MAX_NUM_SEQS = 256


@dataclass(frozen=True)
class StepCostModel:
    """Roofline cost of one engine step (per rank; ranks run in lockstep)."""

    weight_bytes: float
    kv_bytes_per_token: float
    linear_flops_per_token: float
    attn_flops_per_pair: float
    flops_per_s: float
    bytes_per_s: float
    overhead_ms: float = 0.0
    # KV connector load (LMCache start_load_kv) runs inside the forward step
    # in vLLM, so a turn's cached prefix is copied in while every other
    # sequence in the batch waits. None = not modelled.
    load_bytes_per_s: float | None = None

    @classmethod
    def from_profiles(
        cls,
        model: ModelProfile | str,
        gpu: GPUProfile | str,
        tp_size: int = 1,
        eff_bw: float = DEFAULT_EFF_BW,
        eff_flops: float = DEFAULT_EFF_FLOPS,
        overhead_ms: float = 0.0,
        weight_bytes_per_param: float | None = None,
        load_gb_s: float | None = None,
    ) -> StepCostModel:
        m = get_model_profile(model) if isinstance(model, str) else model
        g = get_gpu_profile(gpu) if isinstance(gpu, str) else gpu
        tp = max(1, int(tp_size))
        embed = m.vocab_size * m.hidden
        params = m.estimated_params_billions * 1e9
        # Weights read per step: everything except the embedding table (a
        # row gather); the LM head is a full read. MoE profiles already give
        # active params, which have no separate embedding term to remove.
        body = params - embed if m.params_billions_override is None else params
        wbpp = m.bytes_per_element if weight_bytes_per_param is None else weight_bytes_per_param
        heads = m.num_heads or 1
        attn_layers = m.effective_kv_layers
        return cls(
            weight_bytes=body * wbpp / tp,
            kv_bytes_per_token=m.total_kv_cache_bytes_per_token / tp,
            linear_flops_per_token=2.0 * body / tp,
            attn_flops_per_pair=4.0 * attn_layers * heads * m.head_dim / tp,
            flops_per_s=g.bf16_tflops * 1e12 * eff_flops,
            bytes_per_s=g.hbm_bandwidth_tb_s * 1e12 * eff_bw,
            overhead_ms=overhead_ms,
            load_bytes_per_s=(load_gb_s * 1e9) if load_gb_s else None,
        )

    def load_ms(self, tokens: int) -> float:
        """In-step time to load ``tokens`` of cached prefix into the engine."""
        if not self.load_bytes_per_s or tokens <= 0:
            return 0.0
        return 1000.0 * tokens * self.kv_bytes_per_token / self.load_bytes_per_s

    def step_ms(
        self,
        decode_ctx_sum: int,
        n_decode: int,
        chunks: list[tuple[int, int]] | tuple = (),
    ) -> float:
        """Time of one step: ``n_decode`` sequences decoding one token over
        ``decode_ctx_sum`` total context, plus prefill ``chunks`` of
        ``(new_tokens, context_before)``."""
        if n_decode == 0 and not chunks:
            return 0.0
        pre_tokens = sum(n for n, _ in chunks)
        pairs = decode_ctx_sum + sum(n * c + n * n / 2.0 for n, c in chunks)
        flops = self.linear_flops_per_token * (n_decode + pre_tokens) + self.attn_flops_per_pair * pairs
        nbytes = self.weight_bytes + self.kv_bytes_per_token * (
            decode_ctx_sum + sum(c + n for n, c in chunks)
        )
        return self.overhead_ms + 1000.0 * max(flops / self.flops_per_s, nbytes / self.bytes_per_s)

    def single_decode_ms(self, context: int) -> float:
        return self.step_ms(context, 1)

    def prefill_alone_ms(self, n_new: int, context: int, token_budget: int = DEFAULT_TOKEN_BUDGET) -> float:
        """Prefill time for one request alone on the engine (chunked)."""
        total, done = 0.0, 0
        while done < n_new:
            n = min(token_budget, n_new - done)
            total += self.step_ms(0, 0, [(n, context + done)])
            done += n
        return total


@dataclass
class Sequence:
    """One request inside the engine."""

    ctx: int
    prefill_left: int
    decode_left: int
    load_tokens: int = 0
    loaded: bool = False
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    submitted: float = field(default_factory=time.monotonic)
    cancelled: bool = False


class BatchingEngine:
    """vLLM-V1-style step loop: running decodes first, then chunked prefills."""

    def __init__(
        self,
        cost: StepCostModel,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
        max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
        sleep: bool = True,
    ) -> None:
        self.cost = cost
        self.token_budget = int(token_budget)
        self.max_num_seqs = int(max_num_seqs)
        self.sleep = sleep
        self._waiting: deque[Sequence] = deque()
        self._running: list[Sequence] = []
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        # stats
        self.steps = 0
        self.busy_ms = 0.0
        self.max_batch = 0
        self.lag_ms_max = 0.0

    # -- request side -----------------------------------------------------
    def submit(self, n_new: int, context: int, max_tokens: int, load_tokens: int = 0) -> Sequence:
        """Queue a request. Its first queue item arrives when the prefill
        completes (the first token); then one item per decode step."""
        # vLLM always runs at least the last prompt token through the model.
        seq = Sequence(
            ctx=int(context),
            prefill_left=max(1, int(n_new)),
            decode_left=max(1, int(max_tokens)),
            load_tokens=int(load_tokens),
        )
        self._waiting.append(seq)
        self._ensure_task()
        assert self._wake is not None
        self._wake.set()
        return seq

    def cancel(self, seq: Sequence) -> None:
        seq.cancelled = True

    @property
    def in_flight(self) -> int:
        return len(self._waiting) + len(self._running)

    def info(self) -> dict[str, float]:
        return {
            "steps": self.steps,
            "busy_ms": round(self.busy_ms, 1),
            "max_batch": self.max_batch,
            "in_flight": self.in_flight,
            "lag_ms_max": round(self.lag_ms_max, 1),
            "token_budget": self.token_budget,
        }

    # -- engine side ------------------------------------------------------
    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._wake = asyncio.Event()
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    def _schedule(self) -> tuple[list[Sequence], list[tuple[Sequence, int]]]:
        self._running = [s for s in self._running if not s.cancelled]
        while self._waiting and self._waiting[0].cancelled:
            self._waiting.popleft()
        decoders = [s for s in self._running if s.prefill_left == 0]
        budget = self.token_budget - len(decoders)
        chunks: list[tuple[Sequence, int]] = []
        # Partially prefilled running sequences continue first (FIFO order).
        for s in self._running:
            if budget <= 0:
                break
            if s.prefill_left > 0:
                n = min(budget, s.prefill_left)
                chunks.append((s, n))
                budget -= n
        # Then admit waiting requests while budget and slots remain.
        while budget > 0 and self._waiting and len(self._running) < self.max_num_seqs:
            s = self._waiting.popleft()
            if s.cancelled:
                continue
            self._running.append(s)
            n = min(budget, s.prefill_left)
            chunks.append((s, n))
            budget -= n
        return decoders, chunks

    async def _loop(self) -> None:
        assert self._wake is not None
        t_next = time.monotonic()
        while True:
            if not self._waiting and not self._running:
                self._wake.clear()
                await self._wake.wait()
                t_next = time.monotonic()
                continue
            decoders, chunks = self._schedule()
            if not decoders and not chunks:
                await asyncio.sleep(0)
                continue
            ms = self.cost.step_ms(
                sum(s.ctx for s in decoders), len(decoders), [(n, s.ctx) for s, n in chunks]
            )
            for s, _n in chunks:
                if not s.loaded:
                    ms += self.cost.load_ms(s.load_tokens)
                    s.loaded = True
            self.steps += 1
            self.busy_ms += ms
            self.max_batch = max(self.max_batch, len(decoders) + len(chunks))
            if self.sleep:
                # Deadline-based pacing so event-loop overhead does not
                # accumulate into the emulated time; if the loop falls far
                # behind, restart the clock instead of bursting to catch up.
                t_next += ms / 1000.0
                now = time.monotonic()
                if t_next > now:
                    await asyncio.sleep(t_next - now)
                else:
                    lag = (now - t_next) * 1000.0
                    self.lag_ms_max = max(self.lag_ms_max, lag)
                    if lag > 50.0:
                        t_next = now
                    await asyncio.sleep(0)
            else:
                await asyncio.sleep(0)
            for s in decoders:
                if s.cancelled:
                    continue
                s.ctx += 1
                s.decode_left -= 1
                s.queue.put_nowait(ms)
            for s, n in chunks:
                if s.cancelled:
                    continue
                s.prefill_left -= n
                s.ctx += n
                if s.prefill_left == 0:
                    # The step that finishes the prefill samples the first token.
                    s.decode_left -= 1
                    s.queue.put_nowait(ms)
            self._running = [s for s in self._running if not s.cancelled and (s.prefill_left > 0 or s.decode_left > 0)]


def build_engine(config) -> BatchingEngine | None:
    """Engine for a KVBenchConfig when ``timing.batching`` is on (and not
    full-speed); None otherwise."""
    t = config.timing
    if not getattr(t, "batching", False) or t.full_speed:
        return None
    cost = StepCostModel.from_profiles(
        config.server.model_profile,
        config.gpu.gpu_profile,
        tp_size=config.gpu.tp_size,
        eff_bw=t.batching_eff_bw,
        eff_flops=t.batching_eff_flops,
        overhead_ms=t.batching_overhead_ms,
        weight_bytes_per_param=t.batching_weight_bytes_per_param,
        load_gb_s=t.kv_load_gb_s,
    )
    eng = BatchingEngine(cost, token_budget=t.batching_token_budget, max_num_seqs=t.batching_max_num_seqs)
    logger.info(
        "Batching engine: eff_bw=%.2f eff_flops=%.2f budget=%d; single-stream decode %.1f ms @1k, %.1f ms @100k; "
        "cold prefill 50k %.2f s, 117k %.2f s",
        t.batching_eff_bw, t.batching_eff_flops, eng.token_budget,
        cost.single_decode_ms(1000), cost.single_decode_ms(100_000),
        cost.prefill_alone_ms(50_000, 0, eng.token_budget) / 1e3,
        cost.prefill_alone_ms(116_815, 0, eng.token_budget) / 1e3,
    )
    return eng
