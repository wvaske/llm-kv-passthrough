# KV-Bench: Project Status & Production Release Plan

**Assessment date:** 2026-07-08
**Version under review:** 1.0.0 (as declared in `pyproject.toml`)
**Assessment method:** full test-suite run with coverage, lint/type-check runs, live server smoke tests with empirical bug verification, and three independent deep code reviews (core engine & servers; storage & connectors; docs, deployment & packaging).

---

## 1. Executive Summary

KV-Bench is structurally complete but functionally unfinished. All six planned phases have code, docs, and passing tests (446 tests, ~6s), and a single-node server genuinely works end-to-end: the CLI runs, `/health` responds, the OpenAI-shaped chat completions API works in both blocking and SSE streaming modes, and repeated prompts show real cache-driven latency reduction (measured 131ms cold → 21ms warm prefill).

However, the three layers of the system — the physics (roofline latency model), the caching (tokenization/chunk hashing), and the distribution (proxy, remote storage) — **do not connect to each other**. The carefully built `LatencyCalculator` is dead code; servers use hardcoded per-token sleeps. Cache keys are derived from prompt *length*, not content (verified live: two same-length, different-content prompts hit each other's cache). The CLI and every deployment artifact set configuration that the server silently ignores (verified live: `--model llama-3.1-70b --gpu L4` serves the default 8B/H100 profile). The "disaggregated" proxy never sends a byte to its prefill/decode backends. The Docker image does not build, the docs site does not build, and there is no CI, no LICENSE file, and no integration tests.

**Bottom line:** as a demo of the architecture, it is in good shape. As a *benchmarking tool* — whose entire value is producing numbers people can trust — its two headline outputs (cache hit rates and GPU-emulated latencies) are currently invalid. The declared version of 1.0.0 is not justified; this is a ~0.3–0.4 state. The release plan below (Section 5) sequences the work to reach a trustworthy 1.0 in six milestones, estimated at **8–11 engineering weeks**.

---

## 2. Implementation Status vs. Plan

Status against the deliverables in `KV_BENCH_PROJECT_PLAN.md`:

| Phase | Deliverable | Plan target | Actual status |
|---|---|---|---|
| 1 | Config system, GPU/model profiles | ≥95% cov | ✅ **Done & solid.** 7 GPU + 7 model profiles, validated Pydantic config, 96–100% coverage. Caveats: config is never consumed by the server (see C1); H100/H200/L4/L40S TFLOPS use with-sparsity numbers vs A100's dense number. |
| 2 | Latency calculator, token processor | ≥90% cov | ⚠️ **Built but disconnected.** 97–99% coverage, but `LatencyCalculator` and `TokenProcessor` are never used by any server or CLI path. The roofline model double-counts tensor parallelism (super-linear speedup). |
| 3 | 7 storage backends + factory | ≥90% cov | ⚠️ **2 of 7 production-grade.** Memory (87%) and local-disk (81%) are real (local-disk lacks eviction). Redis (13%), MinIO (16%), Ceph (18%) have never executed their I/O paths, even against mocks; Redis Sentinel calls an API that doesn't exist in redis-py. NFS/Weka are thin `local_disk` subclasses whose docstrings claim locking/retry/parallel-I/O features that are not implemented. |
| 4 | Servers, proxy, connectors | ≥85% cov | ⚠️ **Single-node works; distributed is fictitious.** Prefill/decode/combined servers function but use hardcoded latencies and a length-only fake tokenizer. Proxy simulates backends locally — no HTTP, endpoints never contacted. LMCache connector's key format and chunk hashing do not match real LMCache. Mooncake/Dynamo are honest stubs, but the factory silently accepts them, so "comparisons" between connectors measure nothing. |
| 5 | Integration & E2E tests, ≥90% cov | ≥90% cov | ❌ **Materially missing.** `tests/integration/` is empty. `tests/e2e/test_distributed.py` is in-process only (no processes, no network) despite its name. Overall coverage 72% vs 90% target. No pytest markers → `make test-integration` collects 0 tests, `make test-e2e` deselects everything. |
| 6 | Docs, Docker, Ansible | complete | ⚠️ **Present but broken at the seams.** ~3,600 lines of docs exist but `mkdocs build` fails (wrong `docs_dir`, missing API-reference pages). Dockerfile fails to build (missing README in build context; bash process-substitution under `/bin/sh`). Ansible playbooks/templates exist but render invalid configs (`backend_type: s3`/`mooncake`, `memory_allocation: eager` — all rejected by the schema) and `pip install kvbench` from PyPI, where the package isn't published. No `roles/` (planned). |

**Planned but empty packages:** `src/kvbench/kv/` (manager/chunk/metadata), `src/kvbench/distributed/` (registry/coordinator/health), `src/kvbench/metrics/` (Prometheus exporter/collectors) contain only docstring `__init__.py` files. The plan's `examples/` and `benchmarks/` directories do not exist.

**Plan success criteria scorecard:** of the nine success criteria in the project plan, only "Pluggable connector architecture" and (partially) "Configurable resources" hold today. GenAI-Perf compatibility is plausible but unproven; LMCache interop, multi-host deployment, all-backends-operational, ≥90% coverage, complete docs, and working deployment automation are all unmet.

---

## 3. Quality Assessment

### 3.1 What genuinely works (verified live)

- `pip install -e .` and the `kvbench` CLI (`serve`, `list-profiles`, `info`, `version`).
- Single-node server: `/health`, `/v1/chat/completions` (blocking + SSE streaming with correct chunk framing and `data: [DONE]`), `/metrics` (JSON).
- The cache path end-to-end: chunk store/lookup through connector → storage, with measurable warm-vs-cold latency difference.
- Memory storage backend: correct LRU with recency, TTL on access, capacity enforcement, single-lock concurrency.
- Test infrastructure: 446 tests pass deterministically and fast; pre-commit config is coherent.
- Docs *content* is substantial and mostly accurate about profiles/schemas (build tooling aside).

### 3.2 Critical defects (release blockers)

| # | Defect | Evidence |
|---|---|---|
| C1 | **Server ignores all configuration.** CLI builds a config, exports `KVBENCH_*` env vars, then `create_app()` constructs `KVBenchConfig()` from defaults; `from_env()` is never called in `src/`. Every CLI flag (except host/port), every YAML config, every env var in all 6 compose files, the Ansible templates, and `lmcache_test.sh` is a silent no-op. | Verified live: `kvbench serve --model llama-3.1-70b --gpu L4` → `/health` reports `llama-3.1-8b`, `combined`, memory storage. |
| C2 | **Cache keys depend on prompt length, not content.** `prefill.py`/`combined.py` `_simulate_tokenize` returns `list(range(num_tokens))`. Any two prompts of equal length share all cache entries; hit-rate metrics — the tool's core output — are meaningless. The real `TokenProcessor` is unused (and itself hashes whole-text+position, so common prefixes never match — structurally 0% prefix-hit rate). | Verified live: prompt of 2000 A's then 2000 B's → 2 cache hits on the second request. |
| C3 | **The disaggregated proxy is fictitious.** No HTTP client exists in `src/`; `_simulate_prefill_request`/`_simulate_decode_request` sleep locally with hardcoded 0.1ms/8ms per token. Configured backends receive zero traffic; results are identical whether they are up or down. The 5-container distributed compose topology is cosmetic. | `proxy.py:205-311`; no `httpx` import anywhere in `src/`. |
| C4 | **GPU/model emulation is disconnected.** Servers hardcode per-token latencies; `gpu_profile`/`model_profile` affect nothing but two scalars. The roofline `LatencyCalculator` is dead code, and it contains a tensor-parallelism double-count (bandwidth ×tp **and** bytes ÷tp → measured 60× "speedup" at TP=8 where 8× is the physical ceiling). | `latency.py:100-102, 232-262`; grep confirms no instantiation outside tests. |
| C5 | **Docker image does not build.** (a) `pyproject.toml` declares `readme = "README.md"` but the builder stage doesn't copy it → build backend errors. (b) `RUN pip wheel ... <(echo ...)` is a bashism under `/bin/sh`. Four compose files reference `image: kvbench:latest`, which cannot exist. | Reproduced: both failures during `docker`-equivalent build. |
| C6 | **Redis Sentinel path calls a nonexistent API.** `redis.asyncio.Sentinel.from_url` does not exist in redis-py (verified against installed 8.x). The branch has never run. | `redis_backend.py:82`; verified via `hasattr` check. |
| C7 | **MkDocs site does not build.** `mkdocs.yml` lives inside `docs/` with no `docs_dir` override → `docs/docs` not found; nav references a nonexistent `api/` section. `make docs` fails. | Reproduced by the audit. |
| C8 | **No LICENSE file, no CI, no CHANGELOG.** README/pyproject claim MIT but the file is absent — the package is legally undistributable as-is. `.github/` does not exist; none of the failing quality gates (below) are enforced anywhere. | Repo inspection. |

### 3.3 Major defects (must fix before or shortly after 1.0)

**Correctness of results (the product):**
- Silent failure masking in remote backends: MinIO and Ceph catch **all** exceptions and report infrastructure failures (bad endpoint, auth failure, cluster down) as cache misses; MinIO marks the bucket "ensured" even when both `head_bucket` and `create_bucket` fail. A benchmark against an unreachable store runs to completion and produces plausible-looking garbage. This is the most dangerous failure mode a benchmarking tool can have.
- Chunk hashes are not prefix-chained (real vLLM/LMCache chain each block's hash to its parent), so identical chunks after different prefixes falsely collide — false cache hits by design.
- LMCache "compatibility" isn't: key format (`lmcache@…@kv_bytes` vs real `vllm@model@ws@wid@hash`) and hash algorithm both diverge, so interop with a real LMCache deployment is impossible. The connector factory also hardcodes `model_name="llama-3.1-8b"`, `world_size=1` and ignores `lmcache_remote_url` — different models against shared storage would cross-contaminate cache keys.
- GPU spec inconsistency: Hopper/Ada profiles use NVIDIA's 2:4-sparsity TFLOPS, A100 uses dense — cross-GPU comparisons skewed ~2× (relevant once the calculator is wired in). `estimate_ttft` also underestimates cached-prefill attention (quadratic in miss-tokens only) and double-counts the first decode token.
- Connector-comparison theater: the factory accepts `mooncake`/`dynamo` (honest stubs internally) without warning, and `ConnectorConfig` exposes `mooncake_endpoint`/`dynamo_table` fields that are validated then ignored (`dynamo_table` is even documented as a *DynamoDB* table — a different product). "LMCache vs Mooncake vs Dynamo" runs compare three near-identical passthroughs.

**Robustness:**
- `local_disk`: no eviction (fills once, rejects all writes forever, `evictions` stat never increments); `get()` rewrites `.meta` files outside the lock (concurrent-access corruption); non-atomic writes (crash → truncated value later served as a valid hit — no checksum); persisted stats trusted blindly (drift can permanently brick the cache as "full"); synchronous `os.walk` on the event loop.
- Redis (non-cluster) `keys()` uses the blocking `KEYS` command; `put`/`delete` `GET` the full old value just for size accounting (doubling traffic on multi-MB chunks); TTL'd items never decrement stats → utilization >100% on long runs.
- NFS backend docstrings claim file locking and retry logic; neither exists (`put`/`get` are literal `super()` calls). Multi-host NFS use — its only reason to exist — races on stats and meta files. Weka similarly claims parallel I/O it doesn't do; NFS-vs-Weka-vs-local comparisons measure shard depth only.
- Ceph `connect()`/`open_ioctx()` are blocking C calls on the event loop (can freeze the server for the full mon timeout); TOCTOU between `stat` and `read`.
- Decode server requires a connector it never reads from — the *read* side of KV passthrough is never exercised by any server. Decode streaming never increments request counters (metrics denominators wrong).
- Unbounded `_metadata` dict growth in the LMCache connector (entries never reaped on backend eviction) — linear memory leak over long runs.

**API fidelity / ecosystem:**
- `finish_reason` is always `"stop"` (should be `"length"` when `max_tokens` is hit) — harnesses keying on it misclassify every response.
- No `usage` in final stream chunk, no `stream_options.include_usage` — GenAI-Perf and similar tools rely on this for token accounting; `n`, `stop`, `seed` accepted and silently ignored.
- SSE error paths emit no `data: [DONE]`; non-stream errors always map to 500 (even "no healthy servers", which should be 503).
- `/metrics` returns JSON, not Prometheus exposition format, while `prometheus-client` is declared-but-unused and the shipped Prometheus scrape configs poll a `:9090` port nothing listens on.

**Deployment/docs drift:**
- README documents `s3` and `mooncake` *storage backends* (neither exists; it's `minio`, and Mooncake is a connector) and omits `ceph`; Ansible templates emit those invalid names plus ~10 config fields that don't exist in the schema → rendered configs fail validation → systemd crash-loop.
- Compose files reference ~a third env vars that map to no config field; `docker-compose.distributed.yml` passes endpoint lists as JSON strings `from_env` can't parse.
- Docs command examples crash or don't exist: `kvbench serve --storage redis` (no way to pass `redis_url` via CLI → ValidationError), `kvbench benchmark` (command doesn't exist).
- `pyproject`: `pyyaml` is imported but undeclared (arrives only transitively); `pydantic-settings`, `orjson`, `structlog`, `prometheus-client` declared but never imported; `pip install kvbench[all]` fails (`rados` isn't pip-installable); all URLs are `your-org` placeholders.

### 3.4 Quality gates: configured but failing

| Gate | Configured target | Actual |
|---|---|---|
| pytest | all pass | ✅ 446/446 pass |
| Coverage | ≥90% (plan) | ❌ 72% branch coverage; CLI 0%, Redis 13%, MinIO 16%, Ceph 18% |
| ruff | clean | ❌ 13 errors (unused imports/variable) |
| black | clean | ❌ 27 of 69 files would be reformatted |
| mypy | `strict = true` | ❌ 55 errors in 15 files |
| CI | — | ❌ none exists to enforce any of the above |

### 3.5 Test-suite quality

The ~3,300 lines of tests are numerous, fast, and well-organized, but they assert *direction and shape*, not *magnitude or cross-request behavior* — which is precisely why every critical bug passes:

- Latency tests assert "positive" and "increases", never absolute values against known hardware numbers (misses the sparsity-TFLOPS error) and bound TP scaling only from below (`ratio > 2.0` — an upper bound of `≤ 4.0` would have caught the 16× super-linear bug).
- Cache tests repeat the *identical* prompt; no test sends two different same-length prompts (that test would fail today).
- Connector tests verify the implementation's key strings against itself, not against real LMCache fixtures.
- Storage "coverage" of Redis/MinIO/Ceph consists of constructor calls in factory tests — zero I/O methods ever executed, even against fakeredis/moto/mocked clients.
- No tests for `app.py` at all (no `TestClient`): the HTTP layer, SSE framing, error-status mapping, and the CLI→app config wiring (where C1 lives) are untested.

---

## 4. Scope Recommendation for 1.0

Trustworthy numbers from a smaller surface beat broken numbers from a large one. Recommended cuts:

- **Keep (core product):** combined/prefill/decode servers, memory + local-disk + Redis(standalone/cluster) + MinIO backends, LMCache connector, real proxy, GenAI-Perf compatibility, Docker + compose, docs, CI.
- **Descope to "experimental" (documented as such, factory warns loudly):** Ceph, NFS, Weka backends — keep the code, strip the false claims from docstrings, exclude from supported matrix until integration-tested against real services.
- **Remove or gate behind explicit `--i-know-this-is-a-stub`:** Mooncake and Dynamo connectors in comparisons; delete the ignored `mooncake_endpoint`/`dynamo_table` config fields.
- **Delete:** Redis Sentinel branch (broken, unreachable), empty `kv/` package (fold plans into `connectors/`), `rados` pip extra.
- **Defer to 1.1:** `distributed/` (registry/coordinator/health), Ansible roles, `kvbench benchmark` CLI, Mixtral MoE modeling.

Re-version the current state as **0.4.0** and reserve 1.0.0 for the exit criteria below.

---

## 5. Release Plan to Production Quality

Six milestones, ordered so each unblocks the next. Estimates assume one experienced engineer; parallelize M3/M4 with a second.

### M1 — Make the numbers real (core correctness) — ~2 weeks
The product is its numbers; nothing else matters until these are right.

1. **Wire configuration through** (fixes C1): pass the CLI/YAML/env-built `KVBenchConfig` into `create_app()` via a proper uvicorn factory (`--factory` or app-state injection); adopt `pydantic-settings BaseSettings` as originally planned (dependency already declared) instead of the hand-rolled `from_env`; validate `gpu_profile`/`model_profile` against the registries at startup; support list-valued env vars (endpoints).
2. **Real content-based tokenization & prefix-chained hashing** (fixes C2): replace `list(range(n))` with the (fixed) `TokenProcessor` everywhere; make `simulate_tokenize` a streaming/word-level hash so shared prefixes produce shared tokens in O(n); chain chunk hashes to their parent hash (vLLM/LMCache semantics); one canonical hash function (delete the divergent copies in `prefill.py`/`combined.py`).
3. **Wire `LatencyCalculator` into all servers** (fixes C4): prefill sleeps = roofline prefill latency for miss tokens given full context; decode sleeps = roofline decode latency at current context length; delete the hardcoded 0.1ms/8ms constants.
4. **Fix the roofline model itself:** remove the TP double-count (divide bytes by tp OR multiply bandwidth, not both — add a `≤ tp_size×` scaling regression test); use dense TFLOPS uniformly (H100 SXM ≈ 989.5 dense) with an optional sparsity flag; fix `estimate_ttft` (attention of miss-tokens against *full* context; first token comes from prefill, don't add a decode step); causal-attention factor; fix the 100GbE constant (12.5 GB/s).
5. **Close the loop on decode:** decode server *loads* KV chunks through the connector (the read path), so storage read performance actually appears in TTFT/ITL as designed.
6. **Tests that would have caught all of the above:** same-length-different-content cache-miss test; absolute latency sanity bounds against published hardware numbers; TP upper-bound test; `TestClient` suite for `app.py` including config wiring.

**Exit criteria:** changing `--gpu`/`--model`/`--storage` measurably changes results; two same-length prompts do not share cache; TP=8 speedup ≤ 8×; cold/warm delta scales with model size.

### M2 — Make "distributed" true — ~1.5 weeks
1. Implement real HTTP forwarding in the proxy (`httpx.AsyncClient`, connection pooling, one shared client, clean shutdown): prefill request → chosen prefill endpoint; decode/streaming → chosen decode endpoint; propagate the KV-handoff metadata between them; model KV-transfer latency using the (fixed) `kv_transfer_latency` with a configurable interconnect bandwidth + per-transfer fixed cost.
2. Real health checking (periodic `/health` probes flipping the `healthy` flag the failover logic already reads); 503 on no-healthy-backends.
3. Correct HTTP semantics while in here: `finish_reason: "length"`, `usage` in final stream chunk + `stream_options.include_usage`, `[DONE]` on error paths, 4xx/503 mapping.
4. **Integration tests (populate `tests/integration/`):** spin up prefill+decode+proxy in-process on ephemeral ports; assert backends receive traffic, kill a backend and assert failover; add the `integration` pytest markers so the Makefile targets work.

**Exit criteria:** `docker-compose.distributed.yml` topology serves traffic through real backends; stopping a backend changes proxy behavior; e2e test exists that fails if the proxy stops forwarding.

### M3 — Storage backends: trustworthy or clearly experimental — ~2 weeks
1. **Fail loudly, never fake a miss:** replace blanket `except Exception` in MinIO/Ceph/Redis with typed not-found handling (`NoSuchKey`, `ClientError` code inspection); infrastructure errors raise/propagate and increment `errors`; fix `_ensure_bucket` to not mark success on failure. A benchmark must crash rather than fabricate results.
2. **Redis:** delete the Sentinel branch (C6); `SCAN` instead of `KEYS`; `STRLEN` for size accounting instead of full `GET`; fix TTL/overwrite stats drift; lock `_get_client` (also MinIO — the current race leaks aiohttp sessions).
3. **local_disk:** add LRU (or clock) eviction using the existing metadata; atomic writes (tmp file + `rename`) for values, meta, and stats; move `get()` meta updates under the lock (or stop rewriting meta on read); reconcile persisted stats against the directory on startup; `os.walk` via executor.
4. **Honesty pass on NFS/Weka:** either implement the claimed locking/retry (fcntl advisory locks, bounded retries) or rewrite the docstrings/docs to say "local_disk with tuned sharding, single-writer only" and mark experimental.
5. **Ceph:** move `connect()`/`open_ioctx()` to the executor; fix TOCTOU (read then stat, or read with generous length); mark experimental until tested against a real cluster.
6. **Integration tests against real services** in CI via docker service containers: Redis (real), MinIO (real), moto/fakeredis for the unit tier. Target: every supported backend's I/O paths executed in CI.
7. Storage factory: stop using local `cpu_memory_gb` as remote-backend "capacity" (make capacity optional/unknown for remote backends); wire `use_ssl`/`key_prefix`/`region` from config; validate `nvme_path` writability at startup.
8. Connector fixes: pass real `model_name`/`world_size`/`worker_id` from config into `LMCacheConfig`; reap `_metadata` on eviction (or make it a bounded LRU); decide the LMCache-interop question — either match the real key format + chained hashes (and prove it with fixture keys from a real LMCache run) or rename the claim to "LMCache-style".

**Exit criteria:** benchmark against an unreachable backend fails fast with a clear error; Redis + MinIO integration tests green in CI; supported-vs-experimental matrix documented.

### M4 — Release engineering — ~1.5 weeks (parallelizable with M3)
1. **Fix the Docker build** (C5): copy `README.md` into the builder, drop the process-substitution bashism, install from the built wheel; add a compose smoke test (build → up → curl `/health` → completions request) to CI.
2. **CI pipeline** (GitHub Actions): ruff + black + mypy + pytest with coverage gate (start at 75%, ratchet to 90%) + docker build + mkdocs build + integration tests with Redis/MinIO service containers. Fix the current backlog to get it green: 13 ruff errors, 27 unformatted files, 55 mypy errors.
3. **Docs build** (C7): move `mkdocs.yml` to repo root with `docs_dir: docs`; add API reference via mkdocstrings or drop the nav entry; fix wrong examples (`--storage redis`, nonexistent `kvbench benchmark`); reconcile the backend-name table (`s3`/`mooncake` → `minio`, add `ceph`); publish to GitHub Pages.
4. **Legal/packaging** (C8): add `LICENSE` (MIT), `CHANGELOG.md` (Keep-a-Changelog), replace all `your-org` placeholder URLs; pyproject hygiene — add `pyyaml`, remove `orjson`/`structlog`/unused extras, drop the un-installable `rados` pin, move `httpx` per M2 usage (it becomes a runtime dep for the proxy).
5. **Metrics:** implement the Prometheus exposition endpoint with the already-declared `prometheus-client` (requests, tokens, cache hits/misses, latency histograms, storage stats) in `src/kvbench/metrics/`; keep `/stats` as JSON; make the shipped Prometheus scrape configs true. Fix decode-streaming request counters and the double-close of connectors.
6. **Deployment configs made truthful:** purge nonexistent env vars/fields from all compose files and Ansible templates; fix `memory_allocation: eager`; endpoints as parseable lists; Ansible installs from a built wheel or git tag (not unpublished PyPI); CLI gains `--redis-url`/backend-option flags or documented env-var equivalents.

**Exit criteria:** fresh clone → `docker compose up` works; CI green on all gates; `mkdocs build --strict` passes; `pip install .` on a clean machine runs `kvbench serve` with YAML config end-to-end.

### M5 — Validation & hardening — ~1.5 weeks
1. **GenAI-Perf compatibility proof:** run `genai-perf profile` against combined and proxy topologies in CI (or a nightly job); capture artifacts. This is a plan success criterion that has never been executed.
2. **Multi-host validation:** deploy the distributed compose stack (and Ansible path onto ≥2 VMs), run a benchmark battery, verify cache sharing across hosts through Redis/MinIO — i.e., the project's original purpose, exercised once for real.
3. **Result-validity tests:** golden-number regression tests for latency outputs per (GPU, model, context) tuple; hit-rate tests with known workload structure (shared system prompt → expected prefix-hit fraction); soak test (30+ min sustained load) watching for the metadata leak, stats drift, and utilization >100% class of bugs.
4. Coverage ratchet to ≥90% on `core/`, `servers/`, `connectors/`; ≥80% on storage (remote backends via integration tier). CLI tests via `typer.testing.CliRunner` (currently 0%).
5. Load/failure drills: backend kill mid-benchmark, storage-full behavior, malformed requests, concurrent client swarm.

**Exit criteria:** all nine plan success criteria demonstrably met or explicitly descoped in docs; a written "known limitations" page.

### M6 — Release — ~0.5 week
1. Version discipline: cut `v0.9.0-rc1` from the above, run the full validation battery, fix fallout, then tag **v1.0.0**.
2. Publish: PyPI (`kvbench`), GHCR container image (`ghcr.io/...:1.0.0` — makes `image: kvbench:latest` compose files real), GitHub Release with changelog, docs site live.
3. Post-release: enable branch protection requiring the CI gates; issue templates; label the 1.1 backlog (Ceph/NFS/Weka graduation, Mooncake/Dynamo real connectors, `distributed/` package, `kvbench benchmark` command, Ansible roles).

### Timeline summary

| Milestone | Duration | Cumulative |
|---|---|---|
| M1 Core correctness | 2 wk | 2 wk |
| M2 Real distribution | 1.5 wk | 3.5 wk |
| M3 Storage trustworthiness | 2 wk | 5.5 wk |
| M4 Release engineering | 1.5 wk (parallel w/ M3 possible) | 5.5–7 wk |
| M5 Validation | 1.5 wk | 7–8.5 wk |
| M6 Release | 0.5 wk | **7.5–9 wk** |

With buffer for unknowns discovered during M3/M5 (the never-executed backends), plan **8–11 weeks** single-engineer, ~6 weeks with two.

---

## 6. Top 10 Immediate Actions (this week)

1. Re-version `pyproject.toml` to `0.4.0` — stop advertising 1.0.
2. Add `LICENSE` (MIT) — the repo is currently undistributable.
3. Wire `KVBenchConfig` into `create_app()` (C1) — one focused PR; everything downstream depends on it.
4. Replace `list(range(n))` tokenization with content-based hashing (C2) and add the same-length-different-content regression test.
5. Fix the Dockerfile (C5) — two-line fix; unblocks all compose usage.
6. Move `mkdocs.yml` to repo root (C7) — one-line fix.
7. Stand up minimal CI (ruff/black/mypy/pytest) even before the gates are green, with allowed-failure markers, so drift stops compounding.
8. Delete the Redis Sentinel branch (C6) and add `pyyaml` to dependencies.
9. Add loud runtime warnings when Mooncake/Dynamo stubs or untested backends (Ceph/NFS/Weka) are selected.
10. Fix MinIO/Ceph exception swallowing so infrastructure failures stop masquerading as cache misses.
