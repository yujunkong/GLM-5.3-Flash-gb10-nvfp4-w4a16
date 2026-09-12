#!/usr/bin/env python3
"""Runtime TPS drift / stability telemetry (ops patch, not a decode-path change).

Every --interval seconds appends one CSV row: GPU clock/temp/util/mem, docker
stats (optional), and /metrics accept counters. Does not alter serving knobs.

Example:
  python3 scripts/tps_drift_logger.py \\
    --out "${LOG_DIR:-$HOME/logs/glm53}/tps-drift.csv" \\
    --interval 30
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import time
import urllib.request
from pathlib import Path


def gpu_row() -> dict:
    # Comment: nvidia-smi CSV — keep columns stable for long-run joins.
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,temperature.gpu,clocks.sm,utilization.gpu,"
            "utilization.memory,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=15,
    ).strip().splitlines()[0]
    p = [x.strip() for x in out.split(",")]
    return {
        "smi_ts": p[0],
        "temp_c": p[1],
        "sm_clock_mhz": p[2],
        "gpu_util_pct": p[3],
        "mem_util_pct": p[4],
        "mem_used_mib": p[5],
        "mem_total_mib": p[6],
        "power_w": p[7] if len(p) > 7 else "",
    }


def accept_row(api: str) -> dict:
    drafted = accepted = ""
    try:
        text = urllib.request.urlopen(api.rstrip("/") + "/metrics", timeout=10).read().decode()
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                drafted = line.rsplit(" ", 1)[-1]
            elif (
                line.startswith("vllm:spec_decode_num_accepted_tokens_total")
                and "per_pos" not in line
            ):
                accepted = line.rsplit(" ", 1)[-1]
    except Exception as e:  # noqa: BLE001 — telemetry must not crash the loop
        return {"drafted": "", "accepted": "", "accept_ratio": "", "metrics_err": str(e)}
    ratio = ""
    try:
        if drafted and accepted:
            ratio = f"{float(accepted) / float(drafted):.6f}"
    except Exception:
        pass
    return {
        "drafted": drafted,
        "accepted": accepted,
        "accept_ratio": ratio,
        "metrics_err": "",
    }


def docker_mem(name: str) -> dict:
    try:
        out = subprocess.check_output(
            [
                "docker",
                "stats",
                "--no-stream",
                "--format",
                "{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}",
                name,
            ],
            text=True,
            timeout=20,
        ).strip()
        cpu, mem, mem_pct = out.split("\t")
        return {
            "docker_cpu_pct": cpu.rstrip("%"),
            "docker_mem": mem,
            "docker_mem_pct": mem_pct.rstrip("%"),
        }
    except Exception as e:  # noqa: BLE001
        return {
            "docker_cpu_pct": "",
            "docker_mem": "",
            "docker_mem_pct": "",
            "docker_err": str(e),
        }


FIELDS = [
    "wall_iso",
    "smi_ts",
    "temp_c",
    "sm_clock_mhz",
    "gpu_util_pct",
    "mem_util_pct",
    "mem_used_mib",
    "mem_total_mib",
    "power_w",
    "drafted",
    "accepted",
    "accept_ratio",
    "metrics_err",
    "docker_cpu_pct",
    "docker_mem",
    "docker_mem_pct",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(Path.home() / "logs" / "glm53" / "tps-drift.csv"),
        help="CSV path (append)",
    )
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--container", default="glm53")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists() or out.stat().st_size == 0

    with out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            w.writeheader()
            f.flush()
        while True:
            row = {"wall_iso": dt.datetime.now().isoformat(timespec="seconds")}
            try:
                row.update(gpu_row())
            except Exception as e:  # noqa: BLE001
                row.update({k: "" for k in FIELDS if k not in row})
                row["metrics_err"] = f"smi:{e}"
            row.update(accept_row(args.api))
            row.update(docker_mem(args.container))
            w.writerow(row)
            f.flush()
            if args.once:
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
