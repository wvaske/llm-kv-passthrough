#!/bin/bash
# Parallel smoke: 3 drives at once, one wrapper instance per drive (8 servers each), 8 min load.
# Exercises: kv_clean.sh fresh sn:<SERIAL> + KV_INST scoping, wrapper --instance (ports/tmux), DCWE439_COUNTERS, kvsize models.
# PASS = every instance rc 0, valid, error_pct <= 2, tier counters present.
D0=${D0:?serial of drive 0}; D1=${D1:?serial of drive 1}; D2=${D2:?serial of drive 2}
run(){ local i=$1 sn=$2 m=$3 mnt=/mnt/kv$1
  KV_INST=$i IDLE=3 bash /root/kv_clean.sh fresh $mnt sn:$sn discard | tail -1
  bash /root/kvbench_run.sh --instance $i --model $m --servers 8 --users 8 --duration 480 --cpu-gb 4 --chunk 256 --sessions 600 \
    --mnt $mnt --tag smoke_par$i --workload synthetic --discard 1 --tier-pct 30 --req-timeout 1800 \
    --output /root/kv_v1/smoke_par$i.json > /root/kv_v1/smoke_par$i.log 2>&1
  echo "instance $i rc=$?"; }
run 0 $D0 kvsize-70k & sleep 20; run 1 $D1 kvsize-13k & sleep 20; run 2 $D2 kvsize-4k & wait
python3 - <<'PY'
import json
ok = True
for i in range(3):
    try: r = json.load(open(f"/root/kv_v1/smoke_par{i}.json"))
    except Exception as e: print(i, "no result", e); ok = False; continue
    g = lambda k: r.get("kvbench." + k)
    good = g("valid") and (g("error_pct") or 0) <= 2 and g("tier_files_written")
    ok &= bool(good)
    print(f"inst {i} {g('model')} {g('kv_kib_per_token')} KiB/tok dev {g('device')} sn {g('serial')} valid {g('valid')} err {g('error_pct')} req/s {g('req_per_s')} "
          f"waf_load {g('waf_load')} hit {g('token_hit_rate')} files_w {g('tier_files_written')} GB_w {round((g('tier_bytes_written') or 0)/1e9,1)} files_rm {g('tier_files_removed')} {g('error') or ''}")
print("RESULT:", "PASS" if ok else "FAIL")
PY
for i in 0 1 2; do KV_INST=$i bash /root/kv_clean.sh stop >/dev/null; done; echo "left=$(bash /root/kv_clean.sh check)"; echo SMOKE_DONE
