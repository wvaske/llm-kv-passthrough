"""In-memory payload CPU ceiling; not storage bandwidth or modeled GPU speed.

Run with the same limits as integration tests, for example:
prlimit --as=4294967296 -- env MALLOC_ARENA_MAX=2 OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
    .venv/bin/python scripts/benchmark-payloads.py
"""

import argparse
import json
import time

import numpy as np
import torch

from kvbench.core.payload import SCRATCH_BYTES, fill_sfc64

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--page-mib", type=int, default=4, choices=range(1, 65))
parser.add_argument("--iterations", type=int, default=128, choices=range(1, 1025))
args = parser.parse_args()
tensor = torch.empty(args.page_mib * 1024**2, dtype=torch.uint8, device="cpu")
array = tensor.numpy()
results = {}
for algorithm in ("torch", "sfc64"):
    start = time.perf_counter()
    for seed in range(args.iterations):
        if algorithm == "torch":
            tensor.random_(0, 256, generator=torch.Generator().manual_seed(seed))
        else:
            fill_sfc64(array, np.random.SFC64(seed))
    elapsed = time.perf_counter() - start
    results[algorithm] = {"seconds": elapsed, "MiB_per_s": args.page_mib * args.iterations / elapsed}
print(json.dumps({"page_mib": args.page_mib, "iterations": args.iterations,
                  "sfc64_scratch_bytes": SCRATCH_BYTES, "numpy_version": np.__version__, "results": results}, indent=2))
