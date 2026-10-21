#!/bin/bash
# kv_clean.sh - KV-Bench drive-test cleanup on the SUT. Patterns live in this file (not on the ssh command line) so pkill
# can never match the calling shell.
#   stop                 : stop KV-Bench / AIPerf / tmux from any earlier test        (mubench post_cmd)
#   fresh <mnt> <dev> [mountopts] : stop + nvme format --ses=1 + mkfs.xfs + mount [-o mountopts] + idle (mubench pre_cmd)
#   all                  : stop + also kill leftover datacapture readers (pmd_reader, telegraf) - BETWEEN runs only,
#                          never as pre_cmd (mubench starts the captures before pre_cmd runs)
#   check                : print number of leftover processes (0 = clean)
#   dev                  : print the kernel name of the DUT (the one drive whose serial is in /root/kv_dut_serials)
# <dev> may be a kernel name (nvme1n1), 'dut' = the single serial in /root/kv_dut_serials, or 'sn:<SERIAL>' (must also be listed there).
# KV_INST=<0..9> (env): parallel drives on one host. stop/check/all then touch ONLY that instance (wrapper --instance N:
#   ports 8100+100N+1.., tmux kvr<N>_*, iostat/bpftrace of its device). Without KV_INST the old host-wide behaviour applies.
set -u
MODE=${1:-stop}; MNT=${2:-/mnt/d16}; DEV=${3:-nvme2n1}; IDLE=${IDLE:-60}
LOG=/root/kv_v1/clean.log; mkdir -p /root/kv_v1/results
WORK='llm-kv-passthrough-v1/.venv/bin/python|aiperf|kvbench_run\.sh'
CAPT='pmd_reader|telegraf --config /tmp/dcwe'
if [ -n "${KV_INST:-}" ]; then
  # instance-scoped: its wrapper (--instance N), its servers (ports 81N01-81N99 / 8N01.. for N=0 -> 81xx), its AIPerf clients
  PR="$(( 81 + KV_INST ))[0-9][0-9]"
  # anchored to the real binaries: the tmux server process carries the first session's command line and must not match
  WORK="^bash /root/kvbench_run\.sh .*--instance $KV_INST( |\$)|^/root/llm-kv-passthrough-v1/\.venv/bin/python3? \.venv/bin/kvbench serve .*--port $PR |^(timeout [0-9]+ )?/root/aiperf-venv/bin/(python3? /root/aiperf-venv/bin/)?aiperf profile --url http://localhost:$PR |^/root/llm-kv-passthrough-v1/\.venv/bin/python3? /root/llm-kv-passthrough-v1/\.venv/bin/kvbench warmup --url http://localhost:$PR"
  CAPT='^x_never_matches_x$'      # datacapture readers are shared host-wide -> mubench stops them itself
fi
log(){ echo "$(date -u '+%F %T') [$MODE] $*" | tee -a "$LOG"; }
count(){ pgrep -f "$1" | grep -vx "$$" | wc -l; }

# DUT by serial: the only namespace whose controller serial is listed in /root/kv_dut_serials (refuse if 0 or > 1 match)
dut_dev(){ local n m=""; for n in /sys/block/nvme*n1; do grep -qx "$(cat $n/device/serial 2>/dev/null | xargs)" /root/kv_dut_serials 2>/dev/null && m="$m $(basename $n)"; done
  set -- $m; [ $# = 1 ] && echo "$1"; }
if [ "$MODE" = check ]; then echo $(( $(count "$WORK") + $(count "$CAPT") )); exit 0; fi
if [ "$MODE" = dev ]; then d=$(dut_dev); [ -n "$d" ] && echo "$d" || { echo "no unique DUT (serials in /root/kv_dut_serials)" >&2; exit 1; }; exit 0; fi
if [ "$DEV" = dut ]; then DEV=$(dut_dev) || true; [ -n "$DEV" ] || { log "REFUSING: no unique DUT found by serial (/root/kv_dut_serials)"; exit 1; }; fi
if [ "${DEV#sn:}" != "$DEV" ]; then   # sn:<SERIAL> -> kernel name of that controller's namespace 1
  want=${DEV#sn:}; DEV=""; for n in /sys/block/nvme*n1; do [ "$(cat $n/device/serial 2>/dev/null | xargs)" = "$want" ] && DEV=$(basename $n); done
  [ -n "$DEV" ] || { log "REFUSING: no namespace with serial $want"; exit 1; }
fi
if [ "$MODE" = dev ]; then :; fi

pkill -f "$WORK"
if [ -n "${KV_INST:-}" ]; then for t in $(tmux ls -F '#S' 2>/dev/null | grep "^kvr${KV_INST}_"); do tmux kill-session -t "=$t"; done; else tmux kill-server 2>/dev/null; fi
for i in $(seq 30); do [ "$(count "$WORK")" = 0 ] && break; sleep 1; done
pkill -9 -f "$WORK" 2>/dev/null
if [ "$MODE" = all ]; then pkill -f "$CAPT"; sleep 2; pkill -9 -f "$CAPT" 2>/dev/null; fi
log "processes stopped (workload left=$(count "$WORK"), captures left=$(count "$CAPT"))"
[ "$MODE" = fresh ] || exit 0

# Only format a drive listed as a DUT (one serial per line in /root/kv_dut_serials). Never the boot drive.
SER=$(cat /sys/block/$DEV/device/serial 2>/dev/null | xargs)
grep -qx "$SER" /root/kv_dut_serials 2>/dev/null || { log "REFUSING: $DEV serial '$SER' is not in /root/kv_dut_serials"; exit 1; }
findmnt -no SOURCE / | grep -q "/dev/${DEV}" && { log "REFUSING: $DEV holds the root filesystem"; exit 1; }
umount "$MNT" 2>/dev/null; umount -l "$MNT" 2>/dev/null
for m in $(findmnt -rno TARGET -S /dev/$DEV 2>/dev/null); do umount "$m" 2>/dev/null || umount -l "$m"; log "unmounted $DEV from $m"; done   # mounted elsewhere (other mountpoint)
t0=$(date +%s)
# Always 4 KiB LBA (the drive may come back from DRT at 512 B): pick the LBA format whose data size is 2^12.
LBAF=$(nvme id-ns /dev/$DEV -o json | python3 -c "import json,sys;d=json.load(sys.stdin);print(next(i for i,f in enumerate(d['lbafs']) if f['ds']==12 and f['ms']==0))")
nvme format /dev/$DEV --ses=1 --lbaf=$LBAF --force >> "$LOG" 2>&1 || { log "nvme format failed"; exit 1; }
log "nvme format --ses=1 done in $(( $(date +%s) - t0 )) s"
# MOUNTOPTS (env or 4th arg) e.g. "discard" - XFS ignores discard on remount, so it must be set here at first mount.
MOPTS="noatime${4:+,$4}${MOUNTOPTS:+,$MOUNTOPTS}"
mkfs.xfs -f -K /dev/$DEV > /dev/null 2>> "$LOG" && mkdir -p "$MNT" && mount -o "$MOPTS" /dev/$DEV "$MNT" || { log "mkfs/mount failed"; exit 1; }
mkdir -p "$MNT/lmcache_v1"
log "mounted $(findmnt -no SOURCE,SIZE,AVAIL "$MNT" | xargs); idling ${IDLE}s"
sync; echo 3 > /proc/sys/vm/drop_caches; sleep "$IDLE"
log "ready"
