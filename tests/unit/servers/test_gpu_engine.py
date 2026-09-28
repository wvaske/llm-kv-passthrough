"""Tests for the continuous-batching engine model."""

from __future__ import annotations

import asyncio
import time

import pytest

from kvbench.servers.gpu_engine import BatchingEngine, StepCostModel


def _cost(**kw) -> StepCostModel:
    return StepCostModel.from_profiles("llama-3.2-1b", "GB10", **kw)


class TestStepCostModel:
    def test_decode_is_memory_bound_and_grows_with_batch_context(self) -> None:
        c = _cost()
        one = c.step_ms(80_000, 1)
        four = c.step_ms(4 * 80_000, 4)
        # weights are read once per step, KV once per sequence
        assert four > one
        assert four < 4 * one
        per_ktok = (c.step_ms(90_000, 1) - c.step_ms(80_000, 1)) / 10
        assert per_ktok == pytest.approx(c.kv_bytes_per_token * 1000 / c.bytes_per_s * 1000, rel=1e-6)

    def test_gb10_llama_matches_measured_single_stream(self) -> None:
        # Real GB10 Llama 3.2 1B alone: ~28.5 ms/token at ~87k context.
        assert 20.0 < _cost().single_decode_ms(87_000) < 35.0

    def test_cold_prefill_is_quadratic_in_length(self) -> None:
        c = _cost()
        t50 = c.prefill_alone_ms(50_000, 0)
        t100 = c.prefill_alone_ms(100_000, 0)
        assert 2.0 * t50 < t100 < 4.0 * t50
        # measured 4.1 s (49k) and 17.5 s (117k) on the real GB10
        assert 3.0 < c.prefill_alone_ms(49_078, 0) / 1000 < 6.0
        assert 12.0 < c.prefill_alone_ms(116_815, 0) / 1000 < 22.0

    def test_load_cost(self) -> None:
        c = _cost(load_gb_s=15.0)
        assert c.load_ms(0) == 0.0
        assert c.load_ms(100_000) == pytest.approx(100_000 * c.kv_bytes_per_token / 15e9 * 1000)
        assert _cost().load_ms(100_000) == 0.0


async def _drain(seq, n: int) -> list[float]:
    return [await seq.queue.get() for _ in range(n)]


class TestBatchingEngine:
    @pytest.mark.asyncio
    async def test_single_request_token_stream(self) -> None:
        eng = BatchingEngine(_cost(), sleep=False)
        seq = eng.submit(n_new=300, context=10_000, max_tokens=5)
        steps = await asyncio.wait_for(_drain(seq, 5), 5)
        assert len(steps) == 5
        # first item is the prefill step, the rest are single-sequence decode steps
        assert steps[0] > steps[1]
        await eng.close()

    @pytest.mark.asyncio
    async def test_concurrent_decodes_share_steps(self) -> None:
        eng = BatchingEngine(_cost(), sleep=False)
        a = eng.submit(1, 50_000, 20)
        b = eng.submit(1, 50_000, 20)
        sa, sb = await asyncio.wait_for(asyncio.gather(_drain(a, 20), _drain(b, 20)), 5)
        # after both prefills, the same step serves both sequences
        assert sa[-1] == pytest.approx(sb[-1])
        assert sa[-1] > _cost().single_decode_ms(50_000)
        assert eng.max_batch >= 2
        await eng.close()

    @pytest.mark.asyncio
    async def test_long_prefill_is_chunked_and_stalls_decode(self) -> None:
        eng = BatchingEngine(_cost(), token_budget=2048, sleep=False)
        d = eng.submit(1, 20_000, 10)
        await d.queue.get()
        p = eng.submit(8_000, 0, 1)
        decode_steps = await asyncio.wait_for(_drain(d, 4), 5)
        await asyncio.wait_for(p.queue.get(), 5)
        # while the 8k prefill is being chunked, decode steps carry its cost
        assert max(decode_steps) > 5 * _cost().single_decode_ms(20_000)
        await eng.close()

    @pytest.mark.asyncio
    async def test_cancel_frees_the_slot(self) -> None:
        eng = BatchingEngine(_cost(), sleep=False)
        s = eng.submit(1, 1_000, 10_000)
        await s.queue.get()
        eng.cancel(s)
        for _ in range(20):
            await asyncio.sleep(0)
        assert eng.in_flight == 0
        await eng.close()

    @pytest.mark.asyncio
    async def test_paced_in_wall_time(self) -> None:
        cost = _cost()
        eng = BatchingEngine(cost, sleep=True)
        s = eng.submit(1, 80_000, 11)
        t0 = time.monotonic()
        steps = await asyncio.wait_for(_drain(s, 11), 10)
        elapsed = (time.monotonic() - t0) * 1000
        assert elapsed == pytest.approx(sum(steps), rel=0.25, abs=15)
        await eng.close()
