"""Merge per-pod validation_result.json files into a BENCHMARKS.md table.

Usage:
    python scripts/validation/aggregate_results.py \\
        --results-dir docs/validation_results \\
        --output docs/validation_results/summary.md
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="docs/validation_results")
    ap.add_argument("--output", default="docs/validation_results/summary.md")
    args = ap.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    rows: list[dict] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            if "config" in data and "measured" in data:
                rows.append(data)
        except (json.JSONDecodeError, OSError):
            continue

    if not rows:
        print(f"No valid result JSONs in {results_dir}")
        return 1

    md = [
        "# kv-planner — RunPod validation campaign results",
        "",
        f"Generated: {dt.datetime.utcnow().isoformat(timespec='seconds')}Z",
        f"Configs measured: **{len(rows)}**",
        "",
        "## Summary table",
        "",
        "| GPU | Model | Precision | Predicted tok/s | Measured tok/s | Predicted TPOT ms | Measured TPOT ms | MAPE TPOT | MBU derived |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        cfg = r["config"]
        pred = r["predicted"]
        meas = r["measured"]
        acc = r.get("accuracy", {}) or {}
        cal = r.get("calibration", {}) or {}
        md.append(
            f"| {cfg['gpu_key']} | {cfg['model_slug']} | {cfg['precision']} | "
            f"{pred['throughput_tok_s']:.0f} | {meas.get('aggregate_tok_s', 0):.0f} | "
            f"{pred['tpot_ms']:.1f} | {meas.get('tpot_ms_p50', 0) or 0:.1f} | "
            f"{acc.get('mape_tpot_pct', '—')}% | "
            f"{cal.get('derived_mbu', '—')} |"
        )

    md.extend([
        "",
        "## Per-config detail",
        "",
    ])
    for r in rows:
        cfg = r["config"]
        md.extend([
            f"### {cfg['gpu_key']} — {cfg['model_slug']} ({cfg['precision']})",
            "",
            "```json",
            json.dumps(r, indent=2),
            "```",
            "",
        ])

    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"Wrote {out} — {len(rows)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
