#!/usr/bin/env python3
"""acceptance_ratio.py — read spec-decode acceptance from /metrics.

Usage: python3 acceptance_ratio.py [base_url]
Keeps per-position labels (position="0"..). Cumulative-only dump used to
collapse all positions onto one scalar — that hid the pos0..pos6 curve.

When DFLASH2_ACC_PROBE=1, also prints dflash2_acc_* from HTTP /metrics (if any)
and from $LOG_DIR/dflash2-acc.prom (worker textfile; preferred across processes).
"""
import os
import sys
import urllib.request
from pathlib import Path

base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
text = urllib.request.urlopen(base_url + "/metrics", timeout=10).read().decode()

vals = {}
per_pos = {}
debug = {}
for line in text.splitlines():
    if line.startswith("#") or not line.strip():
        continue
    name, _, value = line.rpartition(" ")
    metric = name.split("{")[0]
    try:
        num = float(value)
    except ValueError:
        continue
    if metric.startswith("dflash2_acc_"):
        debug[metric] = num
        continue
    if "spec_decode" not in metric:
        continue
    if metric.endswith("accepted_tokens_per_pos_total") and 'position="' in name:
        pos = name.split('position="', 1)[1].split('"', 1)[0]
        per_pos[int(pos)] = num
        continue
    if metric not in vals:
        vals[metric] = num

# Worker process cannot publish into API /metrics; read textfile from LOG_DIR.
_log = os.environ.get("LOG_DIR") or os.path.expanduser("~/logs/glm53")
prom_path = Path(_log) / "dflash2-acc.prom"
file_debug = {}
if prom_path.is_file():
    for line in prom_path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("dflash2_acc_"):
            try:
                file_debug[parts[0]] = float(parts[1])
            except ValueError:
                pass

drafted = vals.get("vllm:spec_decode_num_draft_tokens_total")
accepted = vals.get("vllm:spec_decode_num_accepted_tokens_total")
drafts = vals.get("vllm:spec_decode_num_drafts_total")

print("spec_decode metrics found:")
for k in sorted(vals):
    print(f"  {k} = {vals[k]}")
if per_pos:
    print("per_pos accepted (labeled):")
    for p in sorted(per_pos):
        hits = per_pos[p]
        rate = hits / drafts if drafts else float("nan")
        prev = per_pos.get(p - 1)
        cond = hits / prev if prev else float("nan")
        print(f"  pos{p}: accepted={hits:.0f} rate_vs_drafts={rate:.3f} cond={cond:.3f}")

if debug:
    print("dflash2_acc debug gauges (HTTP /metrics):")
    for k in sorted(debug):
        print(f"  {k} = {debug[k]}")
else:
    print("dflash2_acc HTTP /metrics: (none — worker gauges not visible on API process)")
if file_debug:
    print(f"dflash2_acc textfile ({prom_path}):")
    for k in sorted(file_debug):
        print(f"  {k} = {file_debug[k]}")
else:
    print(f"dflash2_acc textfile: (none at {prom_path} — expected when DFLASH2_ACC_PROBE=0)")

if drafted is None or accepted is None:
    print("ERROR: spec_decode draft/accepted counters not found")
    sys.exit(1)
ratio = accepted / drafted if drafted else float("nan")
print(f"\ndrafted={drafted:.0f} accepted={accepted:.0f} ratio={ratio:.3f}")
if 0.6 <= ratio <= 0.8:
    print("HEALTHY (in the 0.6-0.8 band)")
elif ratio < 0.3:
    print("UNHEALTHY: ratio near 0.15-class -> aux capture / mHC contraction suspect")
else:
    print("check band manually (W4A16 Golden soak typically ~0.43–0.44)")
