#!/bin/bash
# kvbench_run.sh - one KV-Bench (v1 rate-based) test point, called by mubench's generic `cmd` load.
# Starts N `kvbench serve` replicas on one drive, warms the cache to steady state, runs one AIPerf per replica,
# and writes ONE flat JSON summary to --output (mubench pulls it back as <run_uuid>.cmd-execution.json).
# Every option is passed straight through, so new sweep parameters need only a YAML change.
set -uo pipefail

# ---------------- defaults (override with --key value) ----------------
MODEL=llama-3.1-8b; GPU=B200_SXM; SERVERS=8; USERS=16; DURATION=600
PREFILL_TPS=0; DECODE_TPS=0          # 0 -> --full-speed (no compute sleeps; storage-bound)
CPU_GB=2; DISK_GB=0; CHUNK=256; IO_THREADS=16; ODIRECT=true
MNT=/mnt/d25a; DEV=""                # DEV auto-derived from MNT when empty
SESSIONS=500; ENTRIES=2000; TURNS=4; ISL=4096; OSL=128; FILL=1.05; SEED=42
OUTPUT=/tmp/kvbench_result.json; TAG=""; IOTRACE=0
WORKLOAD=synthetic; TRACES=/root/agentx/weka_all; MAXISL=131072; TRACE_DELAYS=0
DISCARD=0; FSTRIM_EVERY=0; POLICY=LRU; TIER_PCT=90; HEAVY_TURNS=100; HEAVY_ISL=0; REQ_TIMEOUT=3600
MAX_OUT=0                            # emulator max_output_tokens; 0 = auto (agentx 65536: traces ask up to 40k; else max(2xOSL,4096))
MAX_PENDING=0                        # emulator queue per server; 0 = auto 64 x users (was 8 x users -> 503 "queue is full" in AgentX)
DTRACE=0                             # 1 = bpftrace every discard (TRIM) during the load: /root/kv_discard.bt -> discard.csv.zst + kvbench.discard_*
MAX_CTX=2097152                      # emulator scheduler.max_context_tokens (model max_seq_len also caps; llama-3.1-8b-10m allows 10,485,760)
ADMIT=1                              # LMCache disk admission: write a chunk to disk on its Nth store (1 = every store = stock; 2 = write on 2nd use). Needs /root/patch_lmcache_admit.py
DMAX=0                               # max bytes per TRIM command (/sys/block/<dev>/queue/discard_max_bytes); 0 = kernel default (hw max). Restored after the run.
KVB=/root/llm-kv-passthrough-v1; AIPERF=/root/aiperf-venv/bin/aiperf; BASEPORT=8100

while [ $# -gt 0 ]; do
  k=${1#--}; v=${2:-}; shift 2 || { echo "missing value for $k" >&2; exit 2; }
  case $k in
    model) MODEL=$v;; gpu) GPU=$v;; servers) SERVERS=$v;; users) USERS=$v;; duration) DURATION=$v;;
    prefill-tps) PREFILL_TPS=$v;; decode-tps) DECODE_TPS=$v;; cpu-gb) CPU_GB=$v;; disk-gb) DISK_GB=$v;;
    chunk) CHUNK=$v;; io-threads) IO_THREADS=$v;; odirect) ODIRECT=$v;; mnt) MNT=$v;; dev) DEV=$v;;
    sessions) SESSIONS=$v;; entries) ENTRIES=$v;; turns) TURNS=$v;; isl) ISL=$v;; osl) OSL=$v;; fill) FILL=$v;; seed) SEED=$v;;
    output) OUTPUT=$v;; tag) TAG=$v;; kvbench) KVB=$v;; iotrace) IOTRACE=$v;; workload) WORKLOAD=$v;; traces) TRACES=$v;; max-isl) MAXISL=$v;; trace-delays) TRACE_DELAYS=$v;; discard) DISCARD=$v;; fstrim-every) FSTRIM_EVERY=$v;; cache-policy) POLICY=$v;; tier-pct) TIER_PCT=$v;; heavy-turns) HEAVY_TURNS=$v;; heavy-isl) HEAVY_ISL=$v;; req-timeout) REQ_TIMEOUT=$v;; max-out) MAX_OUT=$v;; max-pending) MAX_PENDING=$v;; discard-trace) DTRACE=$v;; discard-max) DMAX=$v;; max-ctx) MAX_CTX=$v;; disk-admit) ADMIT=$v;;
    *) echo "unknown option --$k" >&2; exit 2;;
  esac
done

W=$(mktemp -d /root/kv_v1/run.XXXXXX); LOG=$W/run.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG" >&2; }
fail(){ log "FAIL: $*"; python3 - "$OUTPUT" "$*" "$W" <<'EOF'
import json,sys; json.dump({"kvbench.valid": False, "kvbench.error": sys.argv[2], "kvbench.workdir": sys.argv[3]}, open(sys.argv[1],"w"))
EOF
  cleanup; exit 1; }
cleanup(){ [ -n "${DMAX_ORIG:-}" ] && { echo "$DMAX_ORIG" > "/sys/block/$DEV/queue/discard_max_bytes" || log "WARNING: could not restore discard_max_bytes"; }; [ -n "${BT:-}" ] && pkill -INT -f "bpftrace /root/kv_iotrace.bt"; [ -n "${DT:-}" ] && pkill -INT -f "bpftrace /root/kv_discard.bt"; for i in $(seq 1 "$SERVERS"); do tmux kill-session -t "kvr$i" 2>/dev/null; done; pkill -f "kvbench serve.*--port $((BASEPORT+1))" 2>/dev/null; kill $IOS 2>/dev/null; true; }
IOS=""
# Drive counters at every phase boundary -> run-to-run WAF ledger. host = NVMe data_units_written x 512000 B,
# nand = OCP "Physical media units written" (bytes). Appended, never rewritten, so gaps between runs are counted too.
LEDGER=${KV_LEDGER:-/root/kv_v1/WAF_LEDGER.csv}
snap(){ local ctrl="/dev/${DEV%n[0-9]*}" h n fb emax emin
  h=$(nvme smart-log "$ctrl" -o json 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin)['data_units_written']*512000)" 2>/dev/null)
  read -r n fb emax emin < <(nvme ocp smart-add-log "$ctrl" -o json 2>/dev/null | python3 -c "
import json,sys;d=json.load(sys.stdin);p=d['Physical media units written']
print(p['hi']*2**64+p['lo'], d['Percent free blocks'], d['Max User data erase counts'], d['Min User data erase counts'])" 2>/dev/null)
  [ -s "$LEDGER" ] || echo "utc,epoch,phase,serial,device,workdir,tag,model,chunk,host_bytes,nand_bytes,free_blocks_pct,erase_max,erase_min" > "$LEDGER"
  echo "$(date -u '+%F %T'),$(date +%s),$1,$SERIAL,$DEV,$W,$TAG,\"$MODEL\",$CHUNK,$h,$n,$fb,$emax,$emin" >> "$LEDGER"
  echo "$1 $h $n $fb" >> "$W/smart.txt"
  mkdir -p "$W/smart"; nvme smart-log "$ctrl" -o json > "$W/smart/$1.smart.json" 2>/dev/null
  nvme ocp smart-add-log "$ctrl" -o json > "$W/smart/$1.ocp.json" 2>/dev/null; }
# one-time record of everything that defines the test environment (versions, drive, kernel/queue settings, mount)
env_record(){ local E="$W/env"; mkdir -p "$E"; local ctrl="/dev/${DEV%n[0-9]*}"
  nvme id-ctrl "$ctrl" -o json > "$E/id-ctrl.json" 2>/dev/null; nvme id-ns "/dev/$DEV" -o json > "$E/id-ns.json" 2>/dev/null
  nvme list -o json > "$E/nvme-list.json" 2>/dev/null
  for q in /sys/block/$DEV/queue/*; do [ -f "$q" ] && printf "%s\t%s\n" "$(basename $q)" "$(cat $q 2>/dev/null)"; done > "$E/queue.tsv"
  findmnt -J "$MNT" > "$E/mount.json" 2>/dev/null; xfs_info "$MNT" > "$E/xfs_info.txt" 2>&1; uname -a > "$E/uname.txt"; cat /proc/cmdline > "$E/cmdline.txt"
  "$KVB/.venv/bin/python" -c "import importlib.metadata as m;print('\n'.join(sorted(f'{d.metadata[\"Name\"]}=={d.version}' for d in m.distributions())))" > "$E/kvbench_venv_packages.txt" 2>/dev/null
  "$AIPERF" --version > "$E/aiperf_version.txt" 2>&1; git -C "$KVB" log -1 --format="%H %s" > "$E/kvbench_git.txt" 2>/dev/null; git -C "$KVB" status --short >> "$E/kvbench_git.txt" 2>/dev/null
  "$KVB/.venv/bin/python" /root/patch_lmcache_disk.py --check > "$E/lmcache_patch.txt" 2>&1; "$KVB/.venv/bin/python" /root/patch_lmcache_admit.py --check >> "$E/lmcache_patch.txt" 2>&1; cp "$0" "$E/kvbench_run.sh"
  cp /root/kv_discard.bt /root/kv_clean.sh "$E/" 2>/dev/null; sysctl -a 2>/dev/null | grep -E "^vm\.(dirty|swappiness)" > "$E/sysctl_vm.txt"; }

# --model may be a comma list (mixed run): server i gets entry ((i-1) mod count).
IFS=, read -ra MODELS <<< "$MODEL"
model_of(){ echo "${MODELS[$(( ($1 - 1) % ${#MODELS[@]} ))]}"; }
# --workload may be a comma list too (mixed run on one drive): server i gets entry ((i-1) mod count), e.g. 16 entries for an explicit split.
IFS=, read -ra WLS <<< "$WORKLOAD"
wl_of(){ echo "${WLS[$(( ($1 - 1) % ${#WLS[@]} ))]}"; }
for x in "${WLS[@]}"; do case $x in synthetic|agentx|heavy) ;; *) echo "unknown workload '$x'" >&2; exit 2;; esac; done
KV_KIB=$("$KVB/.venv/bin/python" -c "
import sys; from kvbench.core.models import get_model_profile as g
m=sys.argv[1].split(','); n=int(sys.argv[2])
print(round(sum(g(m[i % len(m)]).total_kv_cache_bytes_per_token for i in range(n))/n/1024, 1))" "$MODEL" "$SERVERS") || KV_KIB=""

# ---------------- guards ----------------
SRC=$(findmnt -no SOURCE "$MNT") || fail "$MNT is not a mountpoint"
[ -z "$DEV" ] && DEV=$(basename "$SRC")
[ "$(basename "$SRC")" = "$DEV" ] || fail "$MNT is mounted from $SRC, not $DEV"
SERIAL=$(cat /sys/block/"$DEV"/device/serial 2>/dev/null | xargs)
pgrep -f "\.venv/bin/kvbench serve" >/dev/null && fail "another kvbench serve is already running"
# LMCache 0.5.4 disk-tier accounting bug (re-stored chunks counted twice -> tier "full" while nearly empty, FileNotFoundError on
# evict). Fixed by /root/patch_lmcache_disk.py (backport of LMCache 0.5.6). Refuse to measure with an unpatched LMCache.
LMC_PATCH=$("$KVB/.venv/bin/python" /root/patch_lmcache_disk.py --check 2>/dev/null | head -1)
case "$LMC_PATCH" in "patched already"*) ;; *) fail "LMCache disk backend not patched ($LMC_PATCH) - run: $KVB/.venv/bin/python /root/patch_lmcache_disk.py";; esac
if [ "$ADMIT" != 1 ]; then
  "$KVB/.venv/bin/python" /root/patch_lmcache_admit.py --check >/dev/null 2>&1 || fail "--disk-admit $ADMIT needs the admission patch - run: $KVB/.venv/bin/python /root/patch_lmcache_admit.py"
fi
MOUNT_OPTS=$(findmnt -no OPTIONS "$MNT")
# discard must come from the mount (kv_clean.sh fresh ... discard); verify instead of trusting the flag
if [ "$DISCARD" = 1 ]; then echo "$MOUNT_OPTS" | grep -qw discard || fail "--discard 1 but $MNT is not mounted with discard (pass it to kv_clean.sh fresh)"
else echo "$MOUNT_OPTS" | grep -qw discard && DISCARD=1; fi
log "mount options: $MOUNT_OPTS"
env_record
DQ=/sys/block/$DEV/queue; DMAX_ORIG=""
# sysfs only accepts <= UINT_MAX bytes (4 GiB - 1), so the boot default (hw max, 2 TB) cannot be written back. A block request
# is capped at ~4 GiB anyway (max seen 3.996 GiB at the 2 TB default), so the largest accepted 4 KiB-aligned value = default behaviour.
DMAX_DEFAULT=$(( (4294967295 / $(cat $DQ/discard_granularity)) * $(cat $DQ/discard_granularity) ))
[ "$(cat $DQ/discard_max_hw_bytes)" -lt "$DMAX_DEFAULT" ] && DMAX_DEFAULT=$(cat $DQ/discard_max_hw_bytes)
if [ "$DMAX" != 0 ]; then
  DMAX_ORIG=$DMAX_DEFAULT
  echo "$DMAX" > $DQ/discard_max_bytes || fail "cannot set discard_max_bytes=$DMAX"
  [ "$(cat $DQ/discard_max_bytes)" -le "$DMAX" ] || fail "discard_max_bytes not applied (now $(cat $DQ/discard_max_bytes))"
elif [ "$(cat $DQ/discard_max_bytes)" -lt "$DMAX_DEFAULT" ]; then
  # an earlier run (or a crash) left a cap behind - put the default behaviour back before measuring
  echo "$DMAX_DEFAULT" > $DQ/discard_max_bytes || fail "cannot restore discard_max_bytes default"
  log "TRIM: discard_max_bytes was capped at a previous value; reset to $DMAX_DEFAULT (default behaviour)"
fi
DMAX_EFF=$(cat $DQ/discard_max_bytes)
log "TRIM: discard_max_bytes=$DMAX_EFF (hw max $(cat $DQ/discard_max_hw_bytes), max_discard_segments $(cat $DQ/max_discard_segments), granularity $(cat $DQ/discard_granularity))"
snap run_start
rm -rf "$MNT/lmcache_v1"; mkdir -p "$MNT/lmcache_v1"; sync; fstrim -v "$MNT" >> "$LOG" 2>&1
snap after_trim
# Disk tier: 0 = auto. The total tier is a FIXED fraction (TIER_PCT, default 90%) of the empty filesystem, split across
# servers, so every run sees the same total cache footprint regardless of server count (2-server smokes included).
TIER_PCT=${TIER_PCT:-90}
[ "$DISK_GB" = 0 ] && DISK_GB=$(( $(df -BG --output=size "$MNT" | tail -1 | tr -dc 0-9) * TIER_PCT / 100 / SERVERS ))
RATE_ARGS="--full-speed"; MODE=full-speed
if [ "$PREFILL_TPS" != 0 ] || [ "$DECODE_TPS" != 0 ]; then RATE_ARGS="--prefill-tps $PREFILL_TPS --decode-tps $DECODE_TPS"; MODE=rate; fi
FS_GB=$(df -B1 --output=size "$MNT" | tail -1 | tr -dc 0-9); FS_GB=$(( FS_GB / 1000000000 ))
log "model=$MODEL (avg ${KV_KIB} KiB/token) gpu=$GPU servers=$SERVERS users=$USERS dur=${DURATION}s mode=$MODE cpu=${CPU_GB}GiB chunk=$CHUNK dev=$DEV ($SERIAL) mnt=$MNT"
log "disk tier: ${DISK_GB} GiB ($(( DISK_GB * 1073741824 / 1000000000 )) GB) per server x $SERVERS = $(( DISK_GB * SERVERS * 1073741824 / 1000000000 )) GB of the ${FS_GB} GB filesystem (tier_pct $TIER_PCT)"

# ---------------- servers ----------------
[ "$MAX_OUT" = 0 ] && { if [[ ",$WORKLOAD," == *",agentx,"* ]]; then MAX_OUT=65536; else MAX_OUT=$(( OSL*2 > 4096 ? OSL*2 : 4096 )); fi; }
[ "$MAX_PENDING" = 0 ] && MAX_PENDING=$(( USERS*64 ))
[ "$ADMIT" != 1 ] && log "disk admission: a chunk is written to the disk tier on its ${ADMIT}th store (DCWE439_ADMIT)"
[ ${#WLS[@]} -gt 1 ] && log "mixed workload: $(for i in $(seq 1 "$SERVERS"); do printf 's%s=%s ' $i $(wl_of $i); done)"
log "emulator limits: request_timeout=${REQ_TIMEOUT}s max_context_tokens=$MAX_CTX max_output_tokens=$MAX_OUT max_active=$((USERS*2)) max_pending=$MAX_PENDING"
echo 3 > /proc/sys/vm/drop_caches
for i in $(seq 1 "$SERVERS"); do
  mkdir -p "$MNT/lmcache_v1/s$i"
  printf 'cache_policy: %s\nchunk_size: %s\nlocal_cpu: true\nmax_local_cpu_size: %s\nlocal_disk: "file://%s/lmcache_v1/s%s/"\nmax_local_disk_size: %s\nextra_config:\n  use_odirect: %s\n  disk_io_threads: %s\n' \
    "$POLICY" "$CHUNK" "$CPU_GB" "$MNT" "$i" "$DISK_GB" "$ODIRECT" "$IO_THREADS" > "$W/lmcache-s$i.yaml"
  [ "$ADMIT" != 1 ] && printf '  disk_admit_min_puts: %s\n  disk_admit_seen_max: 4000000\n' "$ADMIT" >> "$W/lmcache-s$i.yaml"
  tmux new-session -d -s "kvr$i" "cd $KVB && KVBENCH_KV__MAX_CPU_MEMORY_GB=$CPU_GB KVBENCH_SCHEDULER__REQUEST_TIMEOUT_S=$REQ_TIMEOUT KVBENCH_SCHEDULER__MAX_CONTEXT_TOKENS=$MAX_CTX KVBENCH_SCHEDULER__MAX_REQUEST_BYTES=67108864 KVBENCH_SCHEDULER__MAX_OUTPUT_TOKENS=$MAX_OUT .venv/bin/kvbench serve --model $(model_of $i) --gpu $GPU $RATE_ARGS --max-active $((USERS*2)) --max-pending $MAX_PENDING --lmcache-config $W/lmcache-s$i.yaml --port $((BASEPORT+i)) > $W/serve-s$i.log 2>&1"
done
for i in $(seq 1 "$SERVERS"); do for t in $(seq 90); do
  curl -sf "localhost:$((BASEPORT+i))/health" >/dev/null && break
  grep -q "Application startup failed" "$W/serve-s$i.log" 2>/dev/null && fail "server s$i failed to start: $(grep -m1 -E 'Error' "$W/serve-s$i.log" | cut -c1-200)"
  sleep 2; done; done
UP=0; for i in $(seq 1 "$SERVERS"); do curl -sf "localhost:$((BASEPORT+i))/health" >/dev/null && UP=$((UP+1)); done
[ "$UP" = "$SERVERS" ] || fail "only $UP/$SERVERS servers came up"
curl -s "localhost:$((BASEPORT+1))/kvbench/manifest" > "$W/manifest-s1.json"

# ---------------- optional file-level syscall trace (open/read/write/close/unlink on the cache dir) ----------------
BT=""
if [ "$IOTRACE" = 1 ]; then
  { echo "# mono_to_epoch=$(python3 -c 'import time;print(time.time()-time.clock_gettime(time.CLOCK_MONOTONIC))')"; } > "$W/iotrace.hdr"
  ( BPFTRACE_MAX_STRLEN=64 bpftrace /root/kv_iotrace.bt 2> "$W/iotrace.err" | cat "$W/iotrace.hdr" - | zstd -q -3 -T4 -o "$W/iotrace.csv.zst" ) &
  BT=$!; sleep 8; log "iotrace started (bpftrace /root/kv_iotrace.bt -> $W/iotrace.csv.zst)"
fi

# ---------------- warmup (excluded from measurement) ----------------
log "warmup fill-factor $FILL"; PIDS=""
for i in $(seq 1 "$SERVERS"); do "$KVB/.venv/bin/kvbench" warmup --url "http://localhost:$((BASEPORT+i))" --fill-factor "$FILL" --concurrency 4 > "$W/warmup-s$i.log" 2>&1 & PIDS="$PIDS $!"; done
wait $PIDS
WARM_GB=$(grep -h "Warmup complete" "$W"/warmup-s*.log | awk '{g+=$3} END{printf "%.0f", g}')

# ---------------- measured load ----------------
wl_args(){ local wl=$1
if [ "$wl" = agentx ]; then
  # Real Claude Code sessions (10..3000 turns, ISL 80k-650k). --synthesis-max-isl drops turns longer than the emulated
  # context; --ignore-trace-delays + --concurrency = full-speed closed loop (TRACE_DELAYS=1 keeps recorded think-times).
  WL_ARGS="--custom-dataset-type weka_trace --input-file $TRACES --synthesis-max-isl $MAXISL"
  [ "$TRACE_DELAYS" = 1 ] || WL_ARGS="$WL_ARGS --ignore-trace-delays"
elif [ "$wl" = heavy ]; then
  # Per server: USERS-1 short sessions (TURNS turns) + 1 heavy session (HEAVY_TURNS turns, context ~HEAVY_TURNS x (ISL+OSL)).
  # Heavy session is line 1 of the file; sequential sampling hands it to the first slot at load start.
  [ -s "$W/heavy.jsonl" ] || python3 /root/make_heavy_dataset.py "$W/heavy.jsonl" --isl "$ISL" --osl "$OSL" --turns-short "$TURNS" --sessions-short "$SESSIONS" \
    --turns-long "$HEAVY_TURNS" --heavy-isl "$HEAVY_ISL" --heavy 1 >> "$LOG" 2>&1 || fail "heavy dataset generation failed"
  WL_ARGS="--custom-dataset-type multi_turn --input-file $W/heavy.jsonl --dataset-sampling-strategy sequential"
else
  WL_ARGS="--synthetic-input-tokens-mean $ISL --output-tokens-mean $OSL --num-sessions $SESSIONS --num-dataset-entries $ENTRIES --conversation-turn-mean $TURNS"
fi; }
declare -A WLA; for x in "${WLS[@]}"; do [ -n "${WLA[$x]:-}" ] || { wl_args "$x"; WLA[$x]=$WL_ARGS; }; done
sync; echo 3 > /proc/sys/vm/drop_caches
DT=""
if [ "$DTRACE" = 1 ]; then
  DEVT=$(( (16#$(stat -c %t /dev/$DEV) << 20) | 16#$(stat -c %T /dev/$DEV) ))
  { echo "# mono_to_epoch=$(python3 -c 'import time;print(time.time()-time.clock_gettime(time.CLOCK_MONOTONIC))')"; } > "$W/discard.hdr"
  ( bpftrace /root/kv_discard.bt "$DEVT" "$DEV" 2> "$W/discard.err" | cat "$W/discard.hdr" - | zstd -q -3 -T2 -o "$W/discard.csv.zst" ) &
  DT=$!; sleep 5; log "discard trace started (bpftrace /root/kv_discard.bt $DEVT $DEV -> $W/discard.csv.zst)"
fi
snap load_start
iostat -dxm "$DEV" 5 > "$W/iostat.log" 2>&1 & IOS=$!
log "load: $SERVERS x $USERS users, ${DURATION}s"; PIDS=""
for i in $(seq 1 "$SERVERS"); do
  HF_HUB_OFFLINE=1 AIPERF_DATASET_CONFIGURATION_TIMEOUT=${AIPERF_DATASET_TIMEOUT:-3600} AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=${AIPERF_DATASET_TIMEOUT:-3600} timeout $((DURATION+REQ_TIMEOUT+900)) "$AIPERF" profile --url "http://localhost:$((BASEPORT+i))" --model "$(model_of $i)" \
    --endpoint-type chat --streaming --tokenizer gpt2 --use-server-token-count \
    --concurrency "$USERS" --benchmark-duration "$DURATION" --request-timeout-seconds "$REQ_TIMEOUT" ${WLA[$(wl_of $i)]} \
    --random-seed $((SEED+i)) --ui none --artifact-dir "$W/aiperf_s$i" > "$W/aiperf_s$i.log" 2>&1 & PIDS="$PIDS $!"
done
FT=""
if [ "$FSTRIM_EVERY" -gt 0 ] 2>/dev/null; then
  ( while sleep "$FSTRIM_EVERY"; do echo "$(date '+%F %T') $(fstrim -v "$MNT" 2>&1)" >> "$W/fstrim.log"; done ) & FT=$!
  log "periodic fstrim every ${FSTRIM_EVERY}s"
fi
( while sleep 60; do log "heartbeat: $(grep "^$DEV " "$W/iostat.log" | tail -1 | awk '{printf "r=%s w=%s MB/s util=%s%%", $3, $9, $NF}')"; done ) & HB=$!
wait $PIDS
kill $HB 2>/dev/null; [ -n "$FT" ] && kill $FT 2>/dev/null
kill $IOS 2>/dev/null; IOS=""
sync; snap load_end
if [ -n "$DT" ]; then pkill -INT -f "bpftrace /root/kv_discard.bt"; wait $DT 2>/dev/null; DT=""
  python3 /root/kv_discard_summary.py "$W/discard.csv.zst" > "$W/discard_summary.json" 2>> "$W/discard.err"
  log "discard trace stopped: $(du -h "$W/discard.csv.zst" | cut -f1) compressed; $(python3 -c "import json;d=json.load(open('$W/discard_summary.json'));print(d['rq_GiB'].get('count'),'cmds, p50',d['rq_GiB'].get('p50'),'GiB,',d['nvme_ranges_per_cmd'].get('p50'),'ranges')" 2>/dev/null)"
fi
if [ -n "$BT" ]; then pkill -INT -f "bpftrace /root/kv_iotrace.bt"; wait $BT 2>/dev/null; BT=""
  LS=$(awk -F, '$3=="load_start"{print $2}' "$LEDGER" | tail -1); LE=$(awk -F, '$3=="load_end"{print $2}' "$LEDGER" | tail -1)
  python3 /root/kv_iotrace_summary.py "$W/iotrace.csv.zst" > "$W/iotrace_all.json" 2>> "$W/iotrace.err"
  python3 /root/kv_iotrace_summary.py "$W/iotrace.csv.zst" "$LS" "$LE" > "$W/iotrace_load.json" 2>> "$W/iotrace.err"
  log "iotrace stopped: $(du -h "$W/iotrace.csv.zst" | cut -f1) compressed"
fi
for i in $(seq 1 "$SERVERS"); do curl -s "localhost:$((BASEPORT+i))/kvbench/state" > "$W/state-s$i.json"; curl -s "localhost:$((BASEPORT+i))/stats" > "$W/stats-s$i.json"; done
# Disk-tier health per server: a stuck tier logs "Disk space under pressure" on every put; store failures log "KV serving request failed".
for i in $(seq 1 "$SERVERS"); do
  echo "s$i $(grep -c 'Disk space under pressure' "$W/serve-s$i.log") $(grep -c 'KV serving request failed' "$W/serve-s$i.log") $(grep -c 'DCWE439_PATCH missing file' "$W/serve-s$i.log") $(ls "$MNT/lmcache_v1/s$i" 2>/dev/null | wc -l)"
done > "$W/tier_health.txt"
log "tier health (server, disk-full warnings, store failures, missing-file evicts, files): $(tr '\n' ';' < "$W/tier_health.txt")"
cleanup

# ---------------- one flat JSON summary ----------------
python3 - "$W" "$OUTPUT" "$DEV" <<EOF
import csv, glob, json, os, sys
W, OUT, DEV = sys.argv[1:4]
r = {"kvbench.valid": True, "kvbench.version": "v1-rate-based@" + os.popen("git -C $KVB rev-parse --short HEAD").read().strip(),
     "kvbench.wrapper_rev": "$(md5sum "$0" | cut -c1-8)", "kvbench.workload": "$WORKLOAD", "kvbench.heavy_turns": $HEAVY_TURNS, "kvbench.request_timeout_s": $REQ_TIMEOUT, "kvbench.max_output_tokens": $MAX_OUT, "kvbench.max_pending": $MAX_PENDING, "kvbench.max_context_tokens": $MAX_CTX, "kvbench.lmcache_patch": "$LMC_PATCH", "kvbench.discard_max_bytes": ${DMAX_EFF:-0}, "kvbench.discard_max_set": $DMAX, "kvbench.heavy_isl": $HEAVY_ISL, "kvbench.disk_admit_min_puts": $ADMIT, "kvbench.discard": $DISCARD, "kvbench.fstrim_every_s": $FSTRIM_EVERY,
     "kvbench.cache_policy": "$POLICY", "kvbench.tier_pct": $TIER_PCT, "kvbench.mount_opts": "$MOUNT_OPTS",
     "kvbench.fstrim_calls": int(os.popen("grep -c trimmed $W/fstrim.log 2>/dev/null").read().strip() or 0), "kvbench.max_isl": $MAXISL, "kvbench.trace_delays": $TRACE_DELAYS,
     "kvbench.model": "$MODEL", "kvbench.model_count": ${#MODELS[@]},
     "kvbench.kv_kib_per_token": float("${KV_KIB:-0}") or None, "kvbench.gpu": "$GPU", "kvbench.mode": "$MODE", "kvbench.prefill_tps": float("$PREFILL_TPS"),
     "kvbench.decode_tps": float("$DECODE_TPS"), "kvbench.servers": $SERVERS, "kvbench.users_per_server": $USERS,
     "kvbench.total_users": $SERVERS*$USERS, "kvbench.duration_s": $DURATION, "kvbench.cpu_gb": float("$CPU_GB"),
     "kvbench.disk_gb_per_server": $DISK_GB, "kvbench.tier_total_gb": $((DISK_GB*SERVERS)), "kvbench.chunk_tokens": $CHUNK, "kvbench.io_threads": $IO_THREADS,
     "kvbench.odirect": "$ODIRECT" == "true", "kvbench.sessions": $SESSIONS, "kvbench.dataset_entries": $ENTRIES, "kvbench.turns": $TURNS, "kvbench.isl": $ISL,
     "kvbench.osl": $OSL, "kvbench.device": DEV, "kvbench.serial": "$SERIAL", "kvbench.mount": "$MNT", "kvbench.tag": "$TAG",
     "kvbench.warmup_gb": float("${WARM_GB:-0}"), "kvbench.workdir": W}
rps = tps = req = 0.0; ttft = []; canc = 0
for f in glob.glob(f"{W}/aiperf_s*/profile_export_aiperf.csv"):
    m = {x[0]: x for x in csv.reader(open(f)) if len(x) > 1}
    g = lambda k: float(m[k][1]) if k in m and m[k][1] else 0.0
    rps += g("Request Throughput (requests/sec)"); tps += g("Output Token Throughput (tokens/sec)"); req += g("Request Count")
for f in glob.glob(f"{W}/aiperf_s*/profile_export.jsonl"):
    for line in open(f):
        try: x = json.loads(line).get("metrics", {}).get("time_to_first_token")
        except Exception: continue
        if not x: continue
        v = x.get("value") if isinstance(x, dict) else x
        if v is not None: ttft.append((v / 1e6 if isinstance(x, dict) and x.get("unit") == "ns" else v) / 1000.0)
WLMAP = {i + 1: w for i, w in enumerate("$(for i in $(seq 1 "$SERVERS"); do printf '%s ' $(wl_of $i); done)".split())}
in_tok = out_tok = 0; per_wl = {}
for f in glob.glob(f"{W}/aiperf_s*/profile_export.jsonl"):
    wl = WLMAP.get(int(f.split("aiperf_s")[1].split("/")[0]), "?"); pw = per_wl.setdefault(wl, {"req": 0, "in": 0, "err": 0, "ttft": []})
    for line in open(f):
        try: d = json.loads(line); mt = d.get("metrics", {})
        except Exception: continue
        pw["req"] += 1; pw["err"] += 1 if d.get("error") else 0
        if d.get("error"): continue
        v = mt.get("input_sequence_length"); v = v.get("value") if isinstance(v, dict) else v; in_tok += v or 0; pw["in"] += v or 0
        o = mt.get("output_sequence_length"); o = o.get("value") if isinstance(o, dict) else o; out_tok += o or 0
        t = mt.get("time_to_first_token")
        if t: tv = t.get("value") if isinstance(t, dict) else t; pw["ttft"].append((tv / 1e6 if isinstance(t, dict) and t.get("unit") == "ns" else tv) / 1000.0)
r.update({"kvbench.input_tokens": in_tok, "kvbench.output_tokens": out_tok})
if len(set(WLMAP.values())) > 1:
    for wl, pw in per_wl.items():
        tt = sorted(pw["ttft"]); r.update({f"kvbench.mix_{wl}_requests": pw["req"], f"kvbench.mix_{wl}_failed": pw["err"], f"kvbench.mix_{wl}_input_tokens": pw["in"],
                                           f"kvbench.mix_{wl}_ttft_p50_s": round(tt[len(tt) // 2], 3) if tt else None})
hv_turn = hv_ctx = 0; sh_ctx = 0
for f in glob.glob(f"{W}/aiperf_s*/profile_export.jsonl"):
    for line in open(f):
        try: d = json.loads(line); m = d.get("metadata", {}); isl = d.get("metrics", {}).get("input_sequence_length")
        except Exception: continue
        v = isl.get("value") if isinstance(isl, dict) else isl
        if str(m.get("conversation_id", "")).startswith("heavy"):
            hv_turn = max(hv_turn, (m.get("turn_index") or 0) + 1); hv_ctx = max(hv_ctx, v or 0)
        elif v: sh_ctx = max(sh_ctx, v)
if hv_turn: r.update({"kvbench.heavy_max_turn": hv_turn, "kvbench.heavy_max_context_tokens": hv_ctx, "kvbench.short_max_context_tokens": sh_ctx})
nreq = nerr = 0; err_kind = {}
for f in glob.glob(f"{W}/aiperf_s*/profile_export.jsonl"):
    for line in open(f):
        try: d = json.loads(line)
        except Exception: continue
        nreq += 1; e = d.get("error")
        if e:
            nerr += 1; es = str(e)
            k = "timeout_or_no_content" if "No responses" in es or "imeout" in es else "queue_full" if "queue is full" in es else \
                "max_output" if "max_output" in es else "context" if "context" in es else "other"
            err_kind[k] = err_kind.get(k, 0) + 1
r.update({"kvbench.requests_sent": nreq, "kvbench.requests_failed": nerr,
          "kvbench.error_pct": round(100 * nerr / nreq, 2) if nreq else None, **{f"kvbench.errors_{k}": v for k, v in err_kind.items()}})
ttft.sort(); n = len(ttft); p = lambda q: ttft[min(n - 1, int(q * n))] if n else None
r.update({"kvbench.requests": req, "kvbench.req_per_s": round(rps, 3), "kvbench.out_tok_per_s": round(tps, 1),
          "kvbench.ttft_n": n, "kvbench.ttft_avg_s": round(sum(ttft) / n, 3) if n else None,
          **{f"kvbench.ttft_p{int(q*100)}_s": (round(p(q), 3) if n else None) for q in (0.5, 0.9, 0.95, 0.99)}})
rows = [l.split() for l in open(f"{W}/iostat.log") if l.startswith(DEV + " ")]
if rows:
    rd = [float(x[2]) for x in rows]; wr = [float(x[8]) for x in rows]; ra = [float(x[5]) for x in rows]; ut = [float(x[-1]) for x in rows]
    busy = [i for i, x in enumerate(rd) if x > 100]
    r.update({"kvbench.dev_read_gb": round(sum(rd) * 5 / 1000, 1), "kvbench.dev_write_gb": round(sum(wr) * 5 / 1000, 1),
              "kvbench.dev_read_mbps_avg": round(sum(rd) / len(rd), 1), "kvbench.dev_write_mbps_avg": round(sum(wr) / len(wr), 1),
              "kvbench.dev_read_mbps_peak": max(rd), "kvbench.dev_util_pct_avg": round(sum(ut) / len(ut), 1),
              "kvbench.dev_r_await_ms_busy": round(sum(ra[i] for i in busy) / len(busy), 2) if busy else None})
hit = lk = st = 0
for f in glob.glob(f"{W}/state-s*.json"):
    try: k = json.load(open(f))["kv_stats"]
    except Exception: continue
    hit += k.get("hit_tokens", 0); lk += k.get("lookup_tokens", 0); st += k.get("submitted_store_bytes", 0)
r.update({"kvbench.token_hit_rate": round(hit / lk, 4) if lk else None, "kvbench.kv_store_gb": round(st / 1e9, 1)})
if len(set(WLMAP.values())) > 1:   # per-workload split of hits and KV bytes handed to LMCache (incl. warmup), from each server's state
    for wl in set(WLMAP.values()):
        h = l = s_ = 0
        for i, w in WLMAP.items():
            if w != wl: continue
            try: k = json.load(open(f"{W}/state-s{i}.json"))["kv_stats"]
            except Exception: continue
            h += k.get("hit_tokens", 0); l += k.get("lookup_tokens", 0); s_ += k.get("submitted_store_bytes", 0)
        r.update({f"kvbench.mix_{wl}_token_hit_rate": round(h / l, 4) if l else None, f"kvbench.mix_{wl}_kv_store_gb": round(s_ / 1e9, 1)})
adm = {"puts": 0, "skipped": 0}
for f in glob.glob(f"{W}/serve-s*.log"):
    last = None
    for line in open(f, errors="replace"):
        if "DCWE439_ADMIT stats" in line: last = line
    if last:
        import ast
        try:
            d = ast.literal_eval(last.split("DCWE439_ADMIT stats ", 1)[1].rsplit(" seen_set", 1)[0])
            adm["puts"] += d.get("puts", 0); adm["skipped"] += d.get("skipped", 0)
        except Exception: pass
if adm["puts"]: r.update({"kvbench.admit_puts": adm["puts"], "kvbench.admit_skipped": adm["skipped"], "kvbench.admit_skipped_pct": round(100 * adm["skipped"] / adm["puts"], 1)})
S = {}
for line in open(f"{W}/smart.txt") if os.path.exists(f"{W}/smart.txt") else []:
    x = line.split()
    if len(x) >= 3 and x[1].isdigit() and x[2].isdigit(): S[x[0]] = (int(x[1]), int(x[2]), x[3] if len(x) > 3 else None)
def waf(a, b):
    if a in S and b in S and S[b][0] > S[a][0]: return round((S[b][1] - S[a][1]) / (S[b][0] - S[a][0]), 4)
for ph in S: r[f"kvbench.smart_{ph}_host_bytes"], r[f"kvbench.smart_{ph}_nand_bytes"] = S[ph][0], S[ph][1]
if "load_start" in S and "load_end" in S:   # endurance cost per unit of work: drive bytes per input token served during the load
    hb = S["load_end"][0] - S["load_start"][0]; nb = S["load_end"][1] - S["load_start"][1]
    r.update({"kvbench.host_gb_load": round(hb / 1e9, 1), "kvbench.nand_gb_load": round(nb / 1e9, 1)})
    if in_tok: r.update({"kvbench.host_kib_per_input_token": round(hb / in_tok / 1024, 2), "kvbench.nand_kib_per_input_token": round(nb / in_tok / 1024, 2)})
    if req: r.update({"kvbench.nand_mb_per_request": round(nb / req / 1e6, 1)})
r.update({"kvbench.waf_load": waf("load_start", "load_end"), "kvbench.waf_warmup": waf("after_trim", "load_start"),
          "kvbench.waf_run": waf("run_start", "load_end"),
          "kvbench.free_blocks_pct_load_start": S.get("load_start", (0, 0, None))[2], "kvbench.free_blocks_pct_load_end": S.get("load_end", (0, 0, None))[2]})
for scope in ("load", "all"):
    fn = f"{W}/iotrace_{scope}.json"
    if not os.path.exists(fn): continue
    try: t = json.load(open(fn))
    except Exception: continue
    pre = "kvbench.iotrace_" + scope + "_"
    r[pre + "span_s"] = t.get("trace_span_s")
    for op in ("open", "read", "write", "close", "unlink"):
        for k in ("count", "per_s", "errors", "MBps", "lat_ms_p50", "lat_ms_p90", "lat_ms_p99", "lat_ms_max"):
            if op in t and t[op].get(k) is not None: r[f"{pre}{op}_{k}"] = t[op][k]
        if op in t and t[op].get("sizes"): r[f"{pre}{op}_size_top"] = int(max(t[op]["sizes"], key=t[op]["sizes"].get))
    for k, v in (t.get("files") or {}).items():
        if v is not None: r[f"{pre}files_{k}"] = v
    for k, v in (t.get("max_inflight") or {}).items(): r[f"{pre}max_inflight_{k}"] = v
    for k, v in (t.get("opens") or {}).items(): r[f"{pre}opens_{k}"] = v
if os.path.exists(f"{W}/iotrace.csv.zst"): r["kvbench.iotrace_file"] = f"{W}/iotrace.csv.zst"
if os.path.exists(f"{W}/discard_summary.json"):
    try:
        t = json.load(open(f"{W}/discard_summary.json"))
        r["kvbench.discard_trace_file"] = f"{W}/discard.csv.zst"
        for k in ("span_s", "errors", "bios_per_s", "cmds_per_s", "discard_GBps", "bios_per_rq", "cmds_at_255_ranges_pct"):
            r[f"kvbench.discard_{k}"] = t.get(k)
        for grp in ("bio_MiB", "rq_GiB", "latency_ms", "nvme_ranges_per_cmd"):
            for k, v in (t.get(grp) or {}).items(): r[f"kvbench.discard_{grp}_{k}"] = v
    except Exception:
        pass
th = [l.split() for l in open(f"{W}/tier_health.txt")] if os.path.exists(f"{W}/tier_health.txt") else []
stuck = [x[0] for x in th if len(x) >= 5 and int(x[1]) > 0]
r.update({"kvbench.tier_stuck_servers": len(stuck), "kvbench.tier_stuck_list": " ".join(stuck),
          "kvbench.tier_full_warnings": sum(int(x[1]) for x in th if len(x) >= 5),
          "kvbench.store_failures": sum(int(x[2]) for x in th if len(x) >= 5),
          "kvbench.evict_missing_files": sum(int(x[3]) for x in th if len(x) >= 5)})
if stuck:
    r["kvbench.valid"] = False; r["kvbench.error"] = f"disk tier stuck on {len(stuck)} server(s): {' '.join(stuck)} (LMCache 'Disk space under pressure')"
if r.get("kvbench.error_pct") is not None and r["kvbench.error_pct"] > 2.0:
    r["kvbench.valid"] = False; r["kvbench.error"] = f"{r['kvbench.error_pct']} % of requests failed (limit 2 %): {err_kind}"
if n == 0: r["kvbench.valid"] = False; r["kvbench.error"] = "no AIPerf requests recorded"
json.dump(r, open(OUT, "w"), indent=1)
print(json.dumps(r))
EOF
log "done -> $OUTPUT"
