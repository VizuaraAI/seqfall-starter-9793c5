#!/usr/bin/env bash
# Two-second end-to-end check: no GPU, no downloads. Run this first.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
"$PY" -m src.tasks
"$PY" -m src.evaluate --config configs/default.yaml --limit 24 --out-dir results/smoke
"$PY" -m src.plots --summary results/smoke/summary.csv --out figures
echo "smoke run OK: see results/smoke/summary.csv and figures/fig2_results.png"
