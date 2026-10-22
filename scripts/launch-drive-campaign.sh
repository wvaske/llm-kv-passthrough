#!/usr/bin/env bash
# Lab-specific supervised launch. Does real, paced filler writes followed by
# the endurance ABBA protocol. Source/configs are frozen in a detached worktree.
set -euo pipefail
root="$(git rev-parse --show-toplevel)"
revision="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then
    printf '%s\n' 'Commit/review changes before freezing a drive campaign.' >&2
    exit 1
fi
artifacts=/mnt/development/wvaske/kvbench-runs
snapshot="$artifacts/source-$revision"
unit="kvbench-waf-${revision:0:8}"
mkdir -p "$artifacts"
git worktree add --detach "$snapshot" "$revision"
sudo -n systemd-run --unit="$unit" --uid="$(id -un)" \
    --working-directory="$snapshot" \
    --setenv="PYTHONPATH=$snapshot/src" --setenv=MALLOC_ARENA_MAX=2 \
    --setenv=PYTHONDONTWRITEBYTECODE=1 --setenv=OMP_NUM_THREADS=1 \
    --setenv=OPENBLAS_NUM_THREADS=1 \
    --property=MemoryHigh=3G --property=MemoryMax=4G --property=MemorySwapMax=0 \
    --property=CPUQuota=800% --property=Nice=10 --property=IOWeight=10 \
    --property='IOWriteBandwidthMax=/dev/nvme6n1 3G' \
    --property='IOReadBandwidthMax=/dev/nvme6n1 3G' \
    --property=TimeoutStopSec=90 \
    "$root/.venv/bin/python" -m kvbench.benchmark.drive campaign \
    --config "$snapshot/examples/drive-compare.yaml" --rate-mib 2048 \
    --filler-generator sfc64 --fsync-mib 256
printf 'Unit: %s\nSource: %s\nArtifacts: %s\n' "$unit" "$snapshot" "$artifacts"
