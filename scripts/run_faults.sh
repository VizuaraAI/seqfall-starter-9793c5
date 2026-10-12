#!/usr/bin/env bash
# A3: recovery under injected tool failures.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
OUT="${1:-results/faults_$(date +%Y%m%d_%H%M%S)}"
"$PY" -m src.evaluate --config configs/faults.yaml --out-dir "$OUT"
echo "recovery_rate column in $OUT/summary.csv is the number you want"
