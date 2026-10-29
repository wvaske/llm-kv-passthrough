#!/usr/bin/env bash
# run_agentx_matrix.sh - AgentX (Claude Code trace) KV-storage test points for one or more model
# profiles on one block device, emulating one LMCache instance (one TP rank) of an N-GPU node.
#
# Each model is one test point, run sequentially on a fresh LMCache disk tier:
#   kvbench serve --model M --gpu GPU --tp-size N --kv-ranks R --full-speed   (one real LMCache engine)
#   kvbench warmup --fill-factor F                                           (steady state, excluded)
#   aiperf profile ... AgentX traces (or a synthetic multi-turn stand-in)     (measured load)
# and the device's read/write bytes, LMCache's own counters and AIPerf's aggregates are written
# to --out/<run_id>/ plus a flat dashboard feed (--out/dashboard/{runs,events}.json).
#
# Works on NVMe namespaces and ZFS zvols alike: the device is resolved with readlink -f, counters
# come from /proc/diskstats, and nothing lab-specific (nvme-cli, tmux, bpftrace) is required.
#
# Example (dev host, zvol):
#   sudo scripts/agentx/run_agentx_matrix.sh --dev /dev/zvol/databases/testdev --mnt /mnt/testdev --format 1 \
#     --models qwen3.8-flash-next,glm-5.3,muse-spark --gpu B300_SXM --tp 8 --ranks 0 \
#     --workload agentx --traces /root/agentx/weka_all --users 16 --duration 900 --out /mnt/development/$USER/agentx-runs
set -uo pipefail

MODELS=qwen3.8-flash-next,glm-5.3,muse-spark; GPU=B300_SXM; TP=8; RANKS=0
DEV=""; MNT=""; FORMAT=0; FSTYPE=xfs; MOUNT_OPTS="noatime"
WORKLOAD=agentx; TRACES=""; PUBLIC_DATASET=""; MAXISL=131072; TRACE_DELAYS=0; MAX_CTX=2097152; MAX_OUT=65536
USERS=16; DURATION=900; REQ_TIMEOUT=3600; FILL=1.05; WARM_CONC=4
CPU_GB=2; DISK_GB=0; TIER_PCT=90; CHUNK=256; IO_THREADS=16; ODIRECT=true; POLICY=LRU
ISL=32768; OSL=512; TURNS=8; SESSIONS=200; SEED=42     # synthetic stand-in shape (long multi-turn sessions)
OUT=./agentx-runs; TAG=""; PORT=8201; KVB=""; AIPERF="${AIPERF:-aiperf}"; HOST_LABEL=$(hostname -s)
HOTKV=""

while [ $# -gt 0 ]; do
  k=${1#--}; v=${2:-}; shift 2 || { echo "missing value for --$k" >&2; exit 2; }
  case $k in
    models) MODELS=$v;; gpu) GPU=$v;; tp) TP=$v;; ranks) RANKS=$v;; dev) DEV=$v;; mnt) MNT=$v;; format) FORMAT=$v;;
    fstype) FSTYPE=$v;; mount-opts) MOUNT_OPTS=$v;; workload) WORKLOAD=$v;; traces) TRACES=$v;; public-dataset) PUBLIC_DATASET=$v;;
    max-isl) MAXISL=$v;; trace-delays) TRACE_DELAYS=$v;; max-ctx) MAX_CTX=$v;; max-out) MAX_OUT=$v;;
    users) USERS=$v;; duration) DURATION=$v;; req-timeout) REQ_TIMEOUT=$v;; fill) FILL=$v;; warm-conc) WARM_CONC=$v;;
    cpu-gb) CPU_GB=$v;; disk-gb) DISK_GB=$v;; tier-pct) TIER_PCT=$v;; chunk) CHUNK=$v;; io-threads) IO_THREADS=$v;; odirect) ODIRECT=$v;;
    cache-policy) POLICY=$v;; isl) ISL=$v;; osl) OSL=$v;; turns) TURNS=$v;; sessions) SESSIONS=$v;; seed) SEED=$v;;
    out) OUT=$v;; tag) TAG=$v;; port) PORT=$v;; kvbench) KVB=$v;; aiperf) AIPERF=$v;; host-label) HOST_LABEL=$v;; hot-kv-gb) HOTKV=$v;;
    *) echo "unknown option --$k" >&2; exit 2;;
  esac
done
case $WORKLOAD in agentx|synthetic) ;; *) echo "--workload must be agentx or synthetic" >&2; exit 2;; esac
[ -n "$MNT" ] || { echo "--mnt is required (filesystem on the device under test)" >&2; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd); [ -n "$KVB" ] || KVB=$(cd "$HERE/../.." && pwd)
PY="$KVB/.venv/bin/python"; [ -x "$PY" ] || PY=$(command -v python3)
KVBENCH="$KVB/.venv/bin/kvbench"; [ -x "$KVBENCH" ] || KVBENCH=$(command -v kvbench)
FEED="$PY $HERE/dashboard_feed.py"; DASH="$OUT/dashboard"; mkdir -p "$DASH"
log(){ echo "$(date '+%F %T') $*" | tee -a "$OUT/matrix.log" >&2; }

# ---------------- device / filesystem ----------------
if [ -n "$DEV" ]; then
  RDEV=$(readlink -f "$DEV") || { echo "cannot resolve $DEV" >&2; exit 2; }
  [ -b "$RDEV" ] || { echo "$DEV ($RDEV) is not a block device" >&2; exit 2; }
  if [ "$FORMAT" = 1 ]; then
    # Refuse to format anything that is currently mounted or that another mount depends on.
    findmnt -rno TARGET -S "$RDEV" | while read -r t; do umount "$t" || exit 3; done || { echo "could not unmount $RDEV" >&2; exit 3; }
    log "formatting $DEV ($RDEV) as $FSTYPE and mounting at $MNT"
    "mkfs.$FSTYPE" -f "$RDEV" >> "$OUT/matrix.log" 2>&1 || { echo "mkfs.$FSTYPE $RDEV failed" >&2; exit 3; }
    mkdir -p "$MNT" && mount -o "$MOUNT_OPTS" "$RDEV" "$MNT" || { echo "mount failed" >&2; exit 3; }
  fi
fi
SRC=$(findmnt -no SOURCE --target "$MNT") || { echo "$MNT is not on a mounted filesystem" >&2; exit 2; }
SRC_R=$(readlink -f "${SRC%%\[*}" 2>/dev/null || echo "$SRC")
if [ -n "$DEV" ] && [ "$SRC_R" != "$RDEV" ]; then echo "$MNT is mounted from $SRC, not $DEV ($RDEV)" >&2; exit 2; fi
BLK=$(basename "$SRC_R")      # zd0 / nvme6n1 / vda ... (diskstats key); empty if the source is not a block device
[ -b "$SRC_R" ] || BLK=""
diskstat(){ { [ -n "$BLK" ] && grep -w "$BLK" /proc/diskstats; } > "$1" 2>/dev/null || : > "$1"; }
FS_GIB=$(( $(df -B1 --output=size "$MNT" | tail -1 | tr -dc 0-9) / 1073741824 ))
[ "$DISK_GB" = 0 ] && DISK_GB=$(( FS_GIB * TIER_PCT / 100 ))
log "device=${DEV:-<fs only>} ($SRC_R, diskstats key '${BLK:-none}') mnt=$MNT fs=${FS_GIB} GiB disk tier=${DISK_GB} GiB cpu tier=${CPU_GB} GiB"
if [ "$WORKLOAD" = agentx ] && [ -z "$TRACES" ] && [ -z "$PUBLIC_DATASET" ]; then
  echo "--workload agentx needs --traces <weka trace dir> or --public-dataset semianalysis-cc-traces-weka[...]" >&2; exit 2; fi

# ---------------- per-model test points ----------------
IFS=, read -ra MLIST <<< "$MODELS"
SERVER_PID=""
cleanup(){ [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null; [ -n "${IOS:-}" ] && kill "$IOS" 2>/dev/null; true; }
trap cleanup EXIT INT TERM
RC=0
for MODEL in "${MLIST[@]}"; do
  RUN="$(date -u +%Y%m%dT%H%M%S)-${MODEL}${TAG:+-$TAG}"; W="$OUT/$RUN"; mkdir -p "$W"
  $FEED start --out "$DASH" --run-id "$RUN" --model "$MODEL" --gpu "$GPU" --tp "$TP" --ranks "$RANKS" --workload "$WORKLOAD" \
    --host "$HOST_LABEL" --device "${DEV:-$SRC_R}" --mount "$MNT" --users "$USERS" --duration "$DURATION" --chunk "$CHUNK" \
    --cpu-gib "$CPU_GB" --disk-gib "$DISK_GB" --note "${PUBLIC_DATASET:+public dataset $PUBLIC_DATASET}${TRACES:+traces $TRACES}"
  CACHE="$MNT/kvbench_agentx"; rm -rf "$CACHE"; mkdir -p "$CACHE"; sync
  command -v fstrim >/dev/null && fstrim -v "$MNT" >> "$W/run.log" 2>&1
  printf 'cache_policy: %s\nchunk_size: %s\nlocal_cpu: true\nmax_local_cpu_size: %s\nlocal_disk: "file://%s/"\nmax_local_disk_size: %s\nextra_config:\n  use_odirect: %s\n  disk_io_threads: %s\n' \
    "$POLICY" "$CHUNK" "$CPU_GB" "$CACHE" "$DISK_GB" "$ODIRECT" "$IO_THREADS" > "$W/lmcache.yaml"
  log "[$RUN] serve --model $MODEL --gpu $GPU --tp-size $TP --kv-ranks $RANKS --full-speed (port $PORT)"
  ( cd "$KVB" && KVBENCH_KV__MAX_CPU_MEMORY_GB=$CPU_GB KVBENCH_SCHEDULER__REQUEST_TIMEOUT_S=$REQ_TIMEOUT KVBENCH_SCHEDULER__MAX_CONTEXT_TOKENS=$MAX_CTX \
      KVBENCH_SCHEDULER__MAX_REQUEST_BYTES=67108864 KVBENCH_SCHEDULER__MAX_OUTPUT_TOKENS=$MAX_OUT \
      exec "$KVBENCH" serve --model "$MODEL" --gpu "$GPU" --tp-size "$TP" --kv-ranks "$RANKS" --full-speed ${HOTKV:+--hot-kv-gb $HOTKV} \
      --max-active $((USERS*2)) --max-pending $((USERS*64)) --lmcache-config "$W/lmcache.yaml" --port "$PORT" --host 127.0.0.1 ) > "$W/serve.log" 2>&1 &
  SERVER_PID=$!
  UP=0; for t in $(seq 90); do curl -sf "localhost:$PORT/health" >/dev/null && { UP=1; break; }
    grep -q "Application startup failed\|Error" "$W/serve.log" 2>/dev/null && break; sleep 2; done
  if [ "$UP" != 1 ]; then
    log "[$RUN] FAIL: server did not come up: $(grep -m1 -E 'Error' "$W/serve.log" | cut -c1-200)"
    $FEED finish --out "$DASH" --run-id "$RUN" --status failed --note "server did not start: $(grep -m1 -E 'Error' "$W/serve.log" | cut -c1-160)"
    kill $SERVER_PID 2>/dev/null; SERVER_PID=""; RC=1; continue
  fi
  curl -s "localhost:$PORT/kvbench/manifest" > "$W/manifest.json"; curl -s "localhost:$PORT/kvbench/state" > "$W/state-start.json"
  BPT=$($PY -c "import json;d=json.load(open('$W/state-start.json'))['capacity'];print(d.get('active_stored_bytes_per_token'),d.get('modeled_stored_bytes_per_token'))")
  $FEED phase --out "$DASH" --run-id "$RUN" --phase warmup --message "server up; rank0/model B/token: $BPT; warmup fill-factor $FILL" \
    --set "kv_bytes_per_token_rank0=${BPT%% *}" --set "kv_bytes_per_token_model=${BPT##* }"
  "$KVBENCH" warmup --url "http://localhost:$PORT" --fill-factor "$FILL" --concurrency "$WARM_CONC" > "$W/warmup.log" 2>&1
  WARM_GB=$(grep -h "Warmup complete" "$W/warmup.log" | awk '{print $3}' | head -1)
  log "[$RUN] warmup done (${WARM_GB:-?} GiB)"
  sync; diskstat "$W/diskstats-before.txt"
  command -v iostat >/dev/null && [ -n "$BLK" ] && { iostat -dxm "$BLK" 10 > "$W/iostat.log" 2>&1 & IOS=$!; }
  if [ "$WORKLOAD" = agentx ]; then
    if [ -n "$PUBLIC_DATASET" ]; then WL="--public-dataset $PUBLIC_DATASET"; else WL="--custom-dataset-type weka_trace --input-file $TRACES"; fi
    WL="$WL --synthesis-max-isl $MAXISL"; [ "$TRACE_DELAYS" = 1 ] || WL="$WL --ignore-trace-delays"
  else
    WL="--synthetic-input-tokens-mean $ISL --output-tokens-mean $OSL --num-sessions $SESSIONS --num-dataset-entries $((SESSIONS*TURNS)) --conversation-turn-mean $TURNS"
  fi
  $FEED phase --out "$DASH" --run-id "$RUN" --phase load --message "AIPerf $WORKLOAD load: $USERS users x ${DURATION}s" --set "warmup_gib=${WARM_GB:-null}"
  log "[$RUN] load: $USERS users, ${DURATION}s, $WORKLOAD"
  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0} timeout $((DURATION+REQ_TIMEOUT+900)) "$AIPERF" profile --url "http://localhost:$PORT" --model "$MODEL" \
    --endpoint-type chat --streaming --tokenizer gpt2 --use-server-token-count --no-gpu-telemetry --no-server-metrics \
    --concurrency "$USERS" --benchmark-duration "$DURATION" --request-timeout-seconds "$REQ_TIMEOUT" $WL \
    --random-seed "$SEED" --ui none --artifact-dir "$W/aiperf" > "$W/aiperf.log" 2>&1; ARC=$?
  [ -n "${IOS:-}" ] && { kill "$IOS" 2>/dev/null; IOS=""; }
  sync; diskstat "$W/diskstats-after.txt"
  curl -s "localhost:$PORT/kvbench/state" > "$W/state.json"; curl -s "localhost:$PORT/stats" > "$W/stats.json"
  curl -s "localhost:$PORT/metrics" > "$W/metrics.prom"; ls "$CACHE" 2>/dev/null | wc -l > "$W/cache-files.txt"
  STATUS=ok; [ "$ARC" = 0 ] || STATUS=failed
  $FEED finish --out "$DASH" --run-id "$RUN" --status "$STATUS" --manifest "$W/manifest.json" --state "$W/state.json" --stats "$W/stats.json" \
    --aiperf-dir "$W/aiperf" --disk-before "$W/diskstats-before.txt" --disk-after "$W/diskstats-after.txt" ${WARM_GB:+--warmup-gib "$WARM_GB"} \
    --note "$( [ "$ARC" = 0 ] && echo "aiperf exit 0" || echo "aiperf exit $ARC: $(grep -m1 -iE 'error' "$W/aiperf.log" | cut -c1-160)")"
  log "[$RUN] $STATUS; artifacts in $W"
  kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; SERVER_PID=""
  [ "$STATUS" = ok ] || RC=1
done
log "matrix finished rc=$RC; dashboard feed: $DASH/runs.json $DASH/events.json"
exit $RC
