"""Main verification entry point.

Runs all available claim verifiers and writes a combined EVAL.md
plus per-claim JSON artifacts.

Usage:  uv run python -m src.verify
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

RESULTS = {}
ALL_VERDICTS = {}


def run_all():
    t0 = time.time()

    from src.claims.c1_definition import run as run_c1
    claims_to_run = [("C1", run_c1)]

    for label, fn in claims_to_run:
        try:
            res = fn()
            RESULTS[label] = res
            ALL_VERDICTS[label] = res.get("verdict", "ERROR")
        except Exception:
            tb = traceback.format_exc()
            RESULTS[label] = {"claim": label, "verdict": "ERROR", "traceback": tb}
            ALL_VERDICTS[label] = "ERROR"
            print(f"\n[ERROR] {label} failed:\n{tb}")

    elapsed = time.time() - t0
    write_eval(elapsed)
    return ALL_VERDICTS


def write_eval(elapsed: float):
    artifacts_dir = os.path.join(os.path.dirname(__file__), "..", ".openresearch", "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)

    lines = ["# EVAL.md — APUB Reproduction (arXiv 2403.08966)", ""]
    lines.append(f"**Runtime:** {elapsed:.1f}s")
    lines.append("")
    lines.append("| Claim | Verdict | Key Evidence |")
    lines.append("|-------|---------|--------------|")
    for label, res in RESULTS.items():
        ev = ""
        if label == "C1":
            ev = (f"integral==CVaR (diff={res.get('integral_vs_cvar_max_diff', '?')}), "
                  f"APUB>=mean={res.get('apub_geq_sample_mean')}, "
                  f"APUB>=Efron={res.get('apub_geq_efron')}")
        lines.append(f"| {label} | {res.get('verdict', '?')} | {ev} |")
    lines.append("")

    for label, res in RESULTS.items():
        lines.append(f"## {label}: {res.get('verdict', '?')}")
        lines.append("")
        for k, v in res.items():
            if k.startswith("detail_"):
                continue
            if isinstance(v, (list, dict)):
                continue
            lines.append(f"- **{k}:** {v}")
        lines.append("")

    with open(os.path.join(artifacts_dir, "EVAL.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(artifacts_dir, "all_results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)


if __name__ == "__main__":
    verdicts = run_all()
    failed = [k for k, v in verdicts.items() if v not in ("VERIFIED", "FALSIFIED")]
    if failed:
        print(f"\nNon-passing claims: {failed}")
        sys.exit(1)
    print("\nAll claims passed.")
