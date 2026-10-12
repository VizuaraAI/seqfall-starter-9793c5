#!/usr/bin/env bash
# Full evaluation for whatever models are uncommented in the config.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
CONFIG="${1:-configs/default.yaml}"
OUT="${2:-results/$(date +%Y%m%d_%H%M%S)}"
"$PY" -m src.evaluate --config "$CONFIG" --out-dir "$OUT"
"$PY" -m src.plots --summary "$OUT/summary.csv" --out "$OUT/figures"
echo "results in $OUT"
