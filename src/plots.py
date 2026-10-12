"""Regenerate the paper's Figure 2 from a real summary.csv.

    python -m src.plots --summary results/summary.csv --out figures/

Panel (a) is the fallback population by model. Panel (b) is structural parallel
accuracy against Outcome Equivalence Rate, with the parity diagonal. Distance
above the diagonal is the mis-scoring the paper is about, so keep the axes
square and keep the diagonal visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODE_ORDER = ["M2_serialised_complete", "M3_serialised_truncated",
              "M4_serialised_drift", "M5_collapsed"]
MODE_LABELS = {"M2_serialised_complete": "M2 serialised-complete",
               "M3_serialised_truncated": "M3 truncated",
               "M4_serialised_drift": "M4 drift",
               "M5_collapsed": "M5 collapsed"}
MODE_COLORS = {"M2_serialised_complete": "#4c9a6b",
               "M3_serialised_truncated": "#e3922f",
               "M4_serialised_drift": "#d96b6b",
               "M5_collapsed": "#8c2f39"}


def read_rows(path: str, harness: str, regime: str) -> list[dict]:
    with open(path) as fh:
        rows = [r for r in csv.DictReader(fh)
                if r["harness"] == harness and r["regime"] == regime]
    for r in rows:
        r["mode_shares"] = json.loads(r["mode_shares"])
    return rows


def plot(rows: list[dict], out_path: str) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.2))
    models = [r["model"] for r in rows]

    bottoms = [0.0] * len(rows)
    for mode in MODE_ORDER:
        vals = []
        for r in rows:
            shares = {m: float(r["mode_shares"].get(m, 0.0)) for m in MODE_ORDER}
            total = sum(shares.values()) or 1.0
            vals.append(100 * shares[mode] / total)
        ax_a.bar(models, vals, bottom=bottoms, color=MODE_COLORS[mode],
                 label=MODE_LABELS[mode], edgecolor="white", linewidth=0.6)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax_a.set_ylabel("share of fallback traces (%)")
    ax_a.set_title("(a) Fallback population by model", loc="left", fontsize=10)
    ax_a.set_ylim(0, 100)
    ax_a.tick_params(axis="x", rotation=45, labelsize=8)
    ax_a.legend(fontsize=7, frameon=False, loc="lower right")

    ax_b.plot([0, 100], [0, 100], "--", color="#999999", linewidth=0.9,
              label="parity (no mis-scoring)")
    for r in rows:
        x, y = float(r["par_ast"]), float(r["oer"])
        ax_b.scatter(x, y, s=55, color="#2f6f9f", zorder=3)
        ax_b.annotate(r["model"], (x, y), textcoords="offset points",
                      xytext=(6, 4), fontsize=7)
    ax_b.set_xlabel("Parallel AST accuracy (%)")
    ax_b.set_ylabel("Outcome Equivalence Rate (%)")
    ax_b.set_title("(b) Structural score vs outcome correctness", loc="left", fontsize=10)
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(0, 100)
    ax_b.set_aspect("equal")
    ax_b.legend(fontsize=7, frameon=False, loc="lower right")

    for ax in (ax_a, ax_b):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    parser.add_argument("--out", default="figures")
    parser.add_argument("--harness", default="h1_native")
    parser.add_argument("--regime", default="ALL")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = read_rows(args.summary, args.harness, args.regime)
    if not rows:
        raise SystemExit(f"no rows for harness={args.harness} regime={args.regime}")
    plot(rows, os.path.join(args.out, "fig2_results.png"))


if __name__ == "__main__":
    main()
