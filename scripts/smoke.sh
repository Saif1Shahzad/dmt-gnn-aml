#!/usr/bin/env bash
set -euo pipefail
RUN_ROOT="${1:-runs/smoke}"
dmtgnn synthetic --out "$RUN_ROOT/raw" --events 120
dmtgnn prepare --format csv --time-unit s --input "$RUN_ROOT/raw/transactions.csv" --out "$RUN_ROOT/data"
dmtgnn train --data "$RUN_ROOT/data" --config configs/smoke.yaml --out "$RUN_ROOT/model"
dmtgnn evaluate --data "$RUN_ROOT/data" --checkpoint "$RUN_ROOT/model/checkpoint.pt" --out "$RUN_ROOT/test"
dmtgnn replay --data "$RUN_ROOT/data" --checkpoint "$RUN_ROOT/model/checkpoint.pt" --mode batch --batch-size 10 --out "$RUN_ROOT/replay"
dmtgnn report --runs "$RUN_ROOT" --out "$RUN_ROOT/report"
