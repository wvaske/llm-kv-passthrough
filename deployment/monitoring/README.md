# NVMe WAF monitoring

## Self-updating campaign dashboard

[KV-Bench — Live Campaign](http://172.16.2.20:3000/d/kvbench-campaign) refreshes
every **10 seconds**, automatically following preparation, idle baseline,
preconditioning, measurement, drain, and the next arm/run ID. No run selection
or page reload is needed. It includes:

- Service state, current/next phase, goal and comparison sequence.
- Phase progress, remaining bytes, estimated time and estimated finish.
- Filesystem occupancy versus target and completed comparison arms.
- Host/NAND writes, phase WAF, memory/limit, CPU use and temperature.
- Source freshness, observer heartbeat and error/qualification indicators.

`kvbench-campaign-watch.service` is a separate **read-only** observer on dev,
running as `wvaske` with a 128 MiB memory cap and 10% CPU ceiling. It reads the
atomic experiment snapshot, small artifact manifests/phase summaries, and
systemd status. It does not restart or modify the frozen benchmark process,
scan KV payloads, or issue privileged NVMe commands.

Latest status: `/var/lib/kvbench-nvme/campaign-status.json`. Every observation is
appended to `campaign-status.stream` for the existing Telegraf tail input and
stored as measurement `kvbench_campaign`. The existing stream logrotate rule
covers this file. Grafana uses the same `influxdb-telegraf` datasource.

Preparation progress measures the increase from that run's starting occupancy
to its target, rather than dividing total occupancy by the target. ETA uses
up to five minutes of observed progress/counter deltas, with an instantaneous
rate fallback while the window warms. It is approximate and suppressed when
stale, invalid, stopped or draining. A drain's fixed tail starts only after
outstanding requests and I/O complete, so the observer does not invent a drain
completion time. Missing counters/progress are shown as unavailable, not zero.

Live stat queries are anchored to the latest observer timestamp. A missing ETA
or rate therefore cannot silently show an older valid number as current. The
state/feed-age cards retain a 30-day last-seen fallback and explicitly label a
stalled feed, rather than turning every panel blank. A clean systemd stop
is distinguished from a completed, qualified experiment. Preparation WAF is
explicitly labeled as pending qualification, not a cache-comparison result.

Dashboard source: `build_campaign_dashboard.py`; regenerate the provisioned
`kvbench-campaign.json` with:

```bash
python deployment/monitoring/build_campaign_dashboard.py
```

Live deployment uses the existing Grafana file provider on lp-01 at
`/mnt/configs/configs-grafana-provisioning/dashboards/kvbench-campaign.json`.
All 32 panel queries were executed through Grafana against live data and
returned without errors. Tests cover stale/reset telemetry, restarted/stopped
units, phase/arm transitions, ETA bounds and qualified versus smoke results.

## Device counters and WAF

String fields are flattened for line-protocol display and only quote/backslash
characters are escaped. In particular, JSON-style `\n` is not a valid Telegraf
string escape and can discard the entire status point. Exact multiline errors
remain in the JSON status and benchmark artifacts. A regression test exercises
the real Telegraf parser with failed-campaign errors.

The read-only sampler collects standard NVMe SMART host writes and OCP SMART
physical-media writes every **5 seconds**. It appends exact raw integer counters
and calculated deltas to JSONL, then publishes an append-only Influx stream and
an atomic latest-sample snapshot. The native Telegraf agent tails the stream so
phase changes and fast samples are not lost to polling. Grafana uses the `influxdb-telegraf`
Flux datasource and `telegraf` bucket.

## Deployed on dev (2026-10-09)

- `kvbench-nvme.service`: user `wvaske`, 256 MiB memory limit, read-only `sudo -n
  nvme` commands targeting `/dev/nvme6`, occupancy of `/mnt/work`.
- `/var/lib/kvbench-nvme/samples.jsonl`: raw counters, units implicit in schema,
  timestamps, device identity, interval and run WAF, validity and errors.
- `/var/lib/kvbench-nvme/latest.influx`: atomically replaced metric sample.
- `/var/lib/kvbench-nvme/latest.stream`: every monitoring sample for Telegraf.
- `/var/lib/kvbench-nvme/experiment.stream`: every experiment sample when run.
- `/etc/telegraf/telegraf.d/kvbench-nvme.conf`: tails all `.stream` files.
- `/etc/logrotate.d/kvbench-nvme`: seven daily logs. The sampler opens the file
  for each sample, so rename rotation does not lose an open stream.
- Dashboard provisioned on lp-01 at
  `/mnt/configs/configs-grafana-provisioning/dashboards/kvbench-nvme-waf.json`.
- [Live dashboard](http://172.16.2.20:3000/d/kvbench-nvme-waf).

All seven Flux queries were executed against live data. Counter, rate,
occupancy, drive-fill and validity panels returned data. WAF correctly remained
absent while host-write deltas were zero, even when NAND increased due to idle
background activity. This check is telemetry validation, not an endurance result.

## Install elsewhere

Adapt the paths, user, controller and mount in the service. The service user
needs permission for the specific read-only `nvme id-ctrl`, `nvme smart-log`
and `nvme ocp smart-add-log` commands (with JSON output); do not grant the
collector write/format/sanitize commands. Install the service, Telegraf drop-in
and logrotate file to the corresponding system directories, create the data
directory owned by the service user, enable/start the service, and reload
Telegraf with SIGHUP. Put the dashboard JSON in Grafana's provisioned directory.

No controller is discovered or selected implicitly. Unsupported OCP schemas,
missing NAND counters, command failures, identity changes, resets and sampling
gaps generate invalid samples rather than fabricated NAND bytes or zero WAF.
Original JSONL counters retain 128-bit precision; dashboard numeric fields are
floating point, with deltas and ratios calculated first using exact integers.

For a standalone bounded check:

```bash
python -m kvbench.benchmark.nvme --device /dev/nvme6 --mount /mnt/work \
  --sudo --interval 5 --samples 12 --output /tmp/nvme-check.jsonl
```

The dashboard's window WAF divides **sums of byte deltas**, not averages of
individual ratios. Cumulative WAF includes idle GC after the baseline. Any
counter reset/gap invalidates cumulative WAF until a new collector run starts.
Zero host deltas omit interval WAF but preserve NAND deltas and rates.
Host/NAND counters are controller-wide: other writers affect them. Endurance
experiments additionally qualify single-device backing, namespaces and occupancy.

To stop just this monitoring addition: stop/disable `kvbench-nvme.service`,
remove its Telegraf drop-in and reload Telegraf. Existing monitoring is independent.
