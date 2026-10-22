#!/usr/bin/env bash
# Bounded local validation. The limit includes virtual memory, so allocator
# arenas and numerical-library thread counts are deliberately kept small.
set -euo pipefail
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
export LMCACHE_LOG_LEVEL=ERROR
# Repeated model/rank fixtures must not spawn background HTTP usage reporters.
export LMCACHE_TRACK_USAGE=false
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""
export ROCR_VISIBLE_DEVICES=""
python_bin="${PYTHON:-.venv/bin/python}"
exec prlimit --as="${KVBENCH_TEST_MEMORY_BYTES:-4294967296}" -- \
  "$python_bin" -m pytest -p no:cacheprovider "$@"
