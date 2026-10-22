#!/usr/bin/env bash
# Run only offline analysis. The recordings, controller and official XML stay fixed.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TASK_ROOT"
TASK_PYTHON="${TASK_PYTHON:-/home/zn/文档/Wheel-Legged-Gym/mjlab_training/.venv/bin/python}"
STUDY_OUT="${1:-artifacts/fixed-h7-calibration-20261021/replicate}"
if [[ -e "$STUDY_OUT" || -e "${STUDY_OUT}-rounding" ]]; then
    echo 'Use a new output directory; existing results will not be overwritten.' >&2
    exit 2
fi
export OPENBLAS_NUM_THREADS=1
"$TASK_PYTHON" sim2sim/calibrate_fixed_h7_huananhu.py smoke --out "$STUDY_OUT"
"$TASK_PYTHON" sim2sim/calibrate_fixed_h7_huananhu.py fit --budget 40 --out "${STUDY_OUT}-small"
"$TASK_PYTHON" sim2sim/calibrate_fixed_h7_huananhu.py rl --small --out "${STUDY_OUT}-small"
"$TASK_PYTHON" sim2sim/calibrate_fixed_h7_huananhu.py fit --budget 240 --out "$STUDY_OUT"
"$TASK_PYTHON" sim2sim/calibrate_fixed_h7_huananhu.py rl --stride 20 --out "$STUDY_OUT"
"$TASK_PYTHON" scripts/agent/check_fixed_h7_calibration.py --out "$STUDY_OUT"
"$TASK_PYTHON" scripts/agent/check_fixed_h7_rounding.py --out "${STUDY_OUT}-rounding"
"$TASK_PYTHON" scripts/agent/report_fixed_h7_calibration.py --out "$STUDY_OUT" --rounding "${STUDY_OUT}-rounding"
