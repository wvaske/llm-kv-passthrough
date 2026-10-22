#!/usr/bin/env python3
"""CPU-only check of kvbench.kv.lmcache_mp_stack (scratch dir on the boot drive, never a test drive).
1. store a 40-chunk prompt, lookup = full hit, retrieve = full hit (windowed: window 8 < 40 chunks, so it streams from L2)
2. GQA tp8 -> 8 objects per chunk; MLA tp8 -> 1 object per chunk
3. watermark: write ~3x L2 capacity -> L2 usage stays under the watermark, files deleted in batches
Usage: <kvbench venv python> scripts/drive-test/check_lmcache_mp_stack.py <path to lmcache_mp_stack.py>"""
import asyncio, importlib.util, json, os, shutil, sys, time, random
spec = importlib.util.spec_from_file_location("lmcache_mp_stack", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
ROOT = os.environ.get("KV_MP_SCRATCH", "/root/kv_v1/mp_unit"); ok = True
def check(name, cond, detail):
    global ok; ok &= bool(cond); print(("PASS " if cond else "FAIL ") + name + ": " + str(detail))

def cfg(path, cap_gb, l1_gb=0.5, window=4, wm=0.85, ratio=0.2):
    os.makedirs(path, exist_ok=True)
    l2 = {"type": "fs_native", "base_path": path, "relative_tmp_dir": ".tmp", "num_workers": 4, "use_odirect": False,
          "max_capacity_gb": cap_gb, "eviction": {"eviction_policy": "LRU", "trigger_watermark": wm, "eviction_ratio": ratio}}
    f = path + ".json"
    json.dump({"server_args": ["--l1-size-gb", str(l1_gb), "--eviction-policy", "LRU", "--eviction-trigger-watermark", "0.5", "--chunk-size", "256",
                               "--l2-adapter", json.dumps(l2)], "restore_window_chunks": window}, open(f, "w"))
    return f

def files(path):
    fs = [x for x in os.listdir(path) if x.endswith(".data")]
    return len(fs), sum(os.path.getsize(os.path.join(path, x)) for x in fs)

async def main():
    shutil.rmtree(ROOT, ignore_errors=True)
    # 1 + 2a: GQA llama tp8 (8 objects/chunk, 4 MiB each), 40 chunks, L1 0.5 GiB < 40 x 32 MiB, window 4
    p = ROOT + "/gqa"; s = mod.LMCacheMPStack("llama-3.1-8b", config_file=cfg(p, 4.0), tp_size=8)
    await s.start()
    toks = [random.randrange(50000) for _ in range(40 * 256)]
    await s.store(toks)
    time.sleep(3)
    n, b = files(p)
    check("GQA tp8: 8 objects per chunk", n == 40 * 8, f"{n} files, {b/2**20:.0f} MiB (expect 320 x 4 MiB)")
    hit = await s.lookup(toks); check("lookup full hit", hit == 40 * 256, hit)
    got = await s.retrieve(toks); check("windowed retrieve full hit (L1 0.5 GiB < 1.25 GiB prompt)", got == 40 * 256, f"{got} tokens, windows {s.usage_info()['counters']['restore_windows']}")
    miss = await s.lookup([random.randrange(50000) for _ in range(10 * 256)]); check("unknown prefix = miss", miss == 0, miss)
    await s.close()
    # 2b: MLA deepseek tp8 -> 1 object per chunk (17.2 MiB)
    p = ROOT + "/mla"; s = mod.LMCacheMPStack("deepseek-v3.2", config_file=cfg(p, 4.0), tp_size=8)
    await s.start(); await s.store([random.randrange(50000) for _ in range(10 * 256)]); time.sleep(2)
    n, b = files(p); check("MLA tp8: 1 object per chunk", n == 10, f"{n} files, {b/n/2**20 if n else 0:.2f} MiB each (expect 17.16)")
    await s.close()
    # 2c: active_ranks [0] on GQA tp8 = one GPU's shard (4 MiB objects, 1 per chunk); researched packed profile (kimi-k3, replicated)
    p = ROOT + "/ar"; s = mod.LMCacheMPStack("llama-3.1-8b", config_file=cfg(p, 4.0), tp_size=8, active_ranks=[0])
    await s.start(); t2 = [random.randrange(50000) for _ in range(10 * 256)]; await s.store(t2); time.sleep(2)
    n, b = files(p); hit = await s.lookup(t2)
    check("active_ranks [0]: 1 x 4 MiB object per chunk", n == 10 and b == 10 * 4 * 2**20 and hit == 10 * 256, f"{n} files, {b/2**20:.0f} MiB, hit {hit}")
    await s.close()
    p = ROOT + "/kimi"; s = mod.LMCacheMPStack("kimi-k3", config_file=cfg(p, 4.0), tp_size=8)
    await s.start(); await s.store([random.randrange(50000) for _ in range(10 * 256)]); time.sleep(2)
    n, b = files(p); check("kimi-k3 (packed, replicated) tp8: 1 object per chunk", n == 10, f"{n} files, {b/n/2**20 if n else 0:.2f} MiB each (expect {27648*256/2**20:.2f})")
    await s.close()
    # 3: watermark 0.85/0.2 on a 1 GiB L2, write ~3 GiB (llama tp1: 32 MiB objects)
    p = ROOT + "/wm"; s = mod.LMCacheMPStack("llama-3.1-8b", config_file=cfg(p, 1.0), tp_size=1)
    await s.start(); trace = []
    for i in range(12):
        await s.store([random.randrange(50000) for _ in range(8 * 256)])
        time.sleep(0.7); n, b = files(p); trace.append(round(b / 2**30, 2))
    time.sleep(3); n, b = files(p); trace.append(round(b / 2**30, 2))
    c = s.usage_info()["counters"]
    check("watermark: L2 stays <= ~0.85 GiB after settle", b / 2**30 <= 0.9, f"L2 GiB over time {trace}")
    check("watermark: deletes happened in batches", c["l2_files_deleted"] > 0, c)
    await s.close()
    shutil.rmtree(ROOT, ignore_errors=True)
    print("RESULT:", "ALL PASS" if ok else "FAILED")

asyncio.run(main()); sys.exit(0 if ok else 1)
