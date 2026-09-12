#!/usr/bin/env python3
"""Timed decode soak with clock/temp snapshots. Checklist evidence helper."""
from __future__ import annotations
import argparse, json, statistics, subprocess, time, urllib.request

def gpu_snap():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.current.graphics,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10,
        ).strip().split(",")
        return {
            "gpu_clock_mhz": float(out[0].strip()),
            "gpu_temp_c": float(out[1].strip()),
            "power_w": float(out[2].strip()) if len(out) > 2 else None,
        }
    except Exception as e:
        return {"error": str(e)}

def one_decode(base, model, max_tokens, prompt):
    url = base.rstrip("/") + "/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    completion = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for b in r:
            line = b.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            if first is None:
                first = time.perf_counter()
            evt = json.loads(line[6:])
            u = evt.get("usage")
            if u and u.get("completion_tokens") is not None:
                completion = u["completion_tokens"]
    t1 = time.perf_counter()
    ttft = (first - t0) if first else None
    toks = completion or max_tokens
    decode_s = (t1 - (first or t0))
    return {
        "ttft_s": round(ttft, 4) if ttft else None,
        "total_s": round(t1 - t0, 4),
        "completion_tokens": toks,
        "decode_tok_s": round((toks - 1) / decode_s, 3) if decode_s > 0 else None,
        "snap": gpu_snap(),
    }

def accept_ratio(api):
    text = urllib.request.urlopen(api.rstrip("/") + "/metrics", timeout=10).read().decode()
    drafted = accepted = None
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
            drafted = float(line.rsplit(" ", 1)[-1])
        elif line.startswith("vllm:spec_decode_num_accepted_tokens_total") and "per_pos" not in line:
            accepted = float(line.rsplit(" ", 1)[-1])
    if drafted and accepted is not None:
        return {"drafted": drafted, "accepted": accepted, "ratio": accepted / drafted}
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="glm-5.3-flash")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    prompt = (
        "Write a compact Python function that parses a log line, validates fields, "
        "and returns a typed dictionary. Explain edge cases after the code."
    )
    # discard warmup
    one_decode(a.base, a.model, min(64, a.max_tokens), prompt + "\n# warmup")
    t_end = time.time() + a.minutes * 60
    rows = []
    n = 0
    while time.time() < t_end:
        n += 1
        row = one_decode(a.base, a.model, a.max_tokens, prompt + f"\n# salt={n}")
        row["n"] = n
        row["wall_mono"] = time.time()
        rows.append(row)
        print(json.dumps(row), flush=True)
    toks = [r["decode_tok_s"] for r in rows if r.get("decode_tok_s")]
    temps = [r["snap"].get("gpu_temp_c") for r in rows if isinstance(r.get("snap"), dict) and "gpu_temp_c" in r["snap"]]
    clocks = [r["snap"].get("gpu_clock_mhz") for r in rows if isinstance(r.get("snap"), dict) and "gpu_clock_mhz" in r["snap"]]
    summary = {
        "label": a.label,
        "minutes": a.minutes,
        "runs": len(rows),
        "decode_tok_s_median": statistics.median(toks) if toks else None,
        "decode_tok_s_mean": statistics.mean(toks) if toks else None,
        "decode_tok_s_min": min(toks) if toks else None,
        "decode_tok_s_max": max(toks) if toks else None,
        "decode_drift_pct": (
            round(100.0 * (statistics.mean(toks[-3:]) - statistics.mean(toks[:3])) / statistics.mean(toks[:3]), 2)
            if len(toks) >= 6 else None
        ),
        "gpu_temp_c_min": min(temps) if temps else None,
        "gpu_temp_c_max": max(temps) if temps else None,
        "gpu_temp_c_delta": (max(temps) - min(temps)) if temps else None,
        "gpu_clock_mhz_median": statistics.median(clocks) if clocks else None,
        "accept": accept_ratio(a.api),
        "snap_end": gpu_snap(),
    }
    print(json.dumps({"summary": True, **summary}), flush=True)
    out = {"rows": rows, "summary": summary}
    open(a.out, "w").write(json.dumps(out, indent=2) + "\n")

if __name__ == "__main__":
    main()
