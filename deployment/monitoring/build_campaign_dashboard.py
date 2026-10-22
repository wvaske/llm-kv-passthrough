"""Generate the auto-follow campaign dashboard JSON for Grafana provisioning.

Usage: python deployment/monitoring/build_campaign_dashboard.py
The generated dashboard is stored beside this source and is deployed separately.
"""

import json
from pathlib import Path

DS = {"type": "influxdb", "uid": "influxdb-telegraf"}
FILTER = 'r._measurement == "kvbench_campaign" and r.host == "dev" and r.serial == "2337448A0113"'
NOW = f'from(bucket: "telegraf") |> range(start: -2m) |> filter(fn: (r) => {FILTER})'
STAMP = NOW + ' |> filter(fn: (r) => r._field == "observer_epoch_seconds") |> group() |> last() |> keep(columns: ["_time"])'


def current(field, lookback="-2m"):
    # Anchor every stat to the latest observer point. If ETA/rates are absent
    # because telemetry went stale, do not display an older valid value as live.
    stamp = STAMP.replace("range(start: -2m)", f"range(start: {lookback})")
    source = NOW.replace("range(start: -2m)", f"range(start: {lookback})")
    return (f'stamp = {stamp}\nvalue = {source} |> filter(fn: (r) => r._field == "{field}") |> group()\n'
            'join(tables: {value: value, stamp: stamp}, on: ["_time"]) |> keep(columns: ["_time", "_value"]) '
            f'|> rename(columns: {{_value: "{field}"}})')


panels = [{"id": 1, "type": "text", "title": "Live campaign — follows phase/run changes automatically",
           "gridPos": {"x": 0, "y": 0, "w": 24, "h": 3},
           "options": {"mode": "markdown", "content":
               "Updates every **10 seconds** and follows the active phase/run automatically. ETA uses a rolling observed rate and is hidden when stale, stopped or draining. **Preparation WAF is not a cache-comparison result.** Live cards use the latest observer point; charts follow the time picker. [Detailed WAF dashboard](/d/kvbench-nvme-waf)"}}]


def stat(title, field, x, y, w=4, h=4, unit="none", kind="stat", lookback="-2m", **defaults):
    panels.append({"id": len(panels) + 1, "type": kind, "title": title,
                   "gridPos": {"x": x, "y": y, "w": w, "h": h}, "datasource": DS,
                   "fieldConfig": {"defaults": {"unit": unit, "noValue": "—", **defaults}, "overrides": []},
                   "options": {"reduceOptions": {"calcs": ["lastNotNull"], "fields": f"/^{field}$/", "values": False},
                               "textMode": "value", "colorMode": "value", "graphMode": "none"},
                   "targets": [{"refId": "A", "query": current(field, lookback)}]})


state_map = [{"type": "value", "options": {
    "RUNNING": {"color": "green"}, "COMPLETE": {"color": "green"},
    "STOPPED": {"color": "orange"}, "FAILED": {"color": "red"},
    "STARTING": {"color": "orange"}, "STOPPING": {"color": "orange"},
    "ACTIVATING": {"color": "orange"},
    "RUNNING — WAITING FOR TELEMETRY": {"color": "orange"},
    "STATUS FEED STALE — CHECK TELEMETRY": {"color": "red"},
    "RUNNING — TELEMETRY STALE/INVALID": {"color": "orange"}, "OBSERVER ERROR": {"color": "red"},
}}]
stat("Campaign state", "state_label", 0, 3, 4, mappings=state_map, lookback="-30d")
panels[-1]["targets"][0]["query"] += ' |> map(fn: (r) => ({r with state_label: if uint(v: now()) - uint(v: r._time) > uint(v: 30000000000) then "STATUS FEED STALE — CHECK TELEMETRY" else r.state_label}))'
stat("Current phase", "phase_label", 4, 3, 8)
stat("Current phase goal", "phase_goal_label", 12, 3, 6)
stat("Next step", "next_phase_label", 18, 3, 6)
stat("Current phase progress", "phase_progress", 0, 7, unit="percentunit", kind="gauge", min=0, max=1)
stat("Estimated phase time remaining", "phase_eta_seconds", 4, 7, unit="dtdurations")
stat("Estimated phase finish (local time)", "phase_eta_unix_ms", 8, 7, 8, unit="dateTimeAsLocal")
stat("Remaining bytes in phase", "phase_remaining_bytes", 16, 7, unit="decbytes")
stat("Filesystem occupancy (target below)", "occupancy", 20, 7, unit="percentunit", kind="gauge", min=0, max=1)

stat("Host writes since phase baseline", "run_host_bytes", 0, 11, unit="decbytes")
stat("NAND writes since phase baseline", "run_nand_bytes", 4, 11, unit="decbytes")
stat("Current phase WAF", "waf_run", 8, 11, unit="suffix:x", decimals=3)
stat("Completed comparison arms", "completed_arms", 12, 11, unit="short", decimals=0)
stat("Comparison byte-goal progress", "comparison_progress", 16, 11, unit="percentunit", min=0, max=1)
stat("Result qualification", "qualification_label", 20, 11, mappings=[{"type": "value", "options": {
    "QUALIFIED": {"color": "green"}, "FAILED": {"color": "red"},
    "PENDING — NO ENDURANCE RESULT": {"color": "orange"}, "PENDING": {"color": "orange"},
    "SMOKE — NOT QUALIFIED": {"color": "orange"}, "Unknown": {"color": "gray"},
}}])

plan_query = (f'stamp = {STAMP}\nvalue = {NOW} |> filter(fn: (r) => r._field =~ /^arm_[0-9]+_status$/) |> group()\n'
              'join(tables: {value: value, stamp: stamp}, on: ["_time"]) |> keep(columns: ["_field", "_value"]) '
              '|> rename(columns: {_field: "Arm", _value: "Status"}) |> sort(columns: ["Arm"])')
panels.append({"id": len(panels) + 1, "type": "table", "title": "Comparison sequence",
               "gridPos": {"x": 0, "y": 15, "w": 12, "h": 7}, "datasource": DS,
               "targets": [{"refId": "A", "query": plan_query, "format": "table"}]})


def series(title, fields, x, y, w=12, unit="none"):
    selection = " or ".join(f'r._field == "{field}"' for field in fields)
    query = (f'from(bucket: "telegraf") |> range(start: v.timeRangeStart, stop: v.timeRangeStop) '
             f'|> filter(fn: (r) => {FILTER} and ({selection})) '
             '|> aggregateWindow(every: v.windowPeriod, fn: mean, createEmpty: false)')
    panels.append({"id": len(panels) + 1, "type": "timeseries", "title": title,
                   "gridPos": {"x": x, "y": y, "w": w, "h": 7}, "datasource": DS,
                   "fieldConfig": {"defaults": {"unit": unit, "min": 0, "custom": {"spanNulls": False}}, "overrides": []},
                   "targets": [{"refId": "A", "query": query}]})


series("Campaign memory and hard limit (includes file cache)", ["memory_bytes", "memory_limit_bytes"], 12, 15, unit="bytes")
series("Host and NAND write rates", ["host_bytes_per_second", "nand_bytes_per_second"], 0, 22, unit="Bps")
series("Campaign CPU usage (1.0 = one core)", ["cpu_cores"], 12, 22, unit="short")
series("Occupancy and target", ["occupancy", "occupancy_target"], 0, 29, unit="percentunit")
# Never average WAF ratios over a display window; last retains the actual
# published phase-cumulative ratio, whose numerator includes the drain tail.
series("Phase-cumulative WAF — baseline resets at each phase", ["waf_run"], 12, 29, unit="suffix:x")
panels[-1]["targets"][0]["query"] = panels[-1]["targets"][0]["query"].replace("fn: mean", "fn: last")

bool_map = [{"type": "value", "options": {"true": {"text": "OK", "color": "green"},
                                         "false": {"text": "CHECK", "color": "red"}}}]
stat("Observer health", "observer_ok", 0, 36, mappings=bool_map)
stat("Fresh, valid source telemetry", "telemetry_ok", 4, 36, mappings=bool_map)
stat("Source sample age", "telemetry_age_seconds", 8, 36, unit="s",
     thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None},
                                                 {"color": "red", "value": 30}]})
stat("Status feed age", "observer_epoch_seconds", 12, 36, unit="s", lookback="-30d",
     thresholds={"mode": "absolute", "steps": [{"color": "green", "value": None},
                                                 {"color": "red", "value": 30}]})
panels[-1]["targets"][0]["query"] += ' |> map(fn: (r) => ({r with observer_epoch_seconds: float(v: uint(v: now())) / 1000000000.0 - r.observer_epoch_seconds}))'
stat("Drive temperature", "temperature_celsius", 16, 36, unit="celsius")
stat("Campaign elapsed", "campaign_elapsed_seconds", 20, 36, unit="dtdurations")
stat("Unit", "unit", 0, 40, 6)
stat("Telemetry run ID (auto-followed)", "telemetry_run_id", 6, 40, 8)
stat("Error / stale-data detail", "last_error", 14, 40, 6)
stat("Workload streams", "workload_streams", 20, 40, unit="short", decimals=0)
stat("Current artifact directory", "artifact_path", 0, 44, 24, 3)

dashboard = {"uid": "kvbench-campaign", "title": "KV-Bench — Live Campaign",
             "tags": ["kvbench", "nvme", "campaign"], "schemaVersion": 39, "version": 1,
             "refresh": "10s", "timezone": "browser", "time": {"from": "now-6h", "to": "now"},
             "panels": panels, "links": [{"title": "Detailed NVMe WAF", "url": "/d/kvbench-nvme-waf", "type": "link"}]}

if __name__ == "__main__":
    path = Path(__file__).with_name("kvbench-campaign.json")
    path.write_text(json.dumps(dashboard, indent=2) + "\n")
    print(f"Generated {path} ({len(panels)} panels)")
