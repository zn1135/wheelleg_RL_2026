#!/usr/bin/env bash
set -euo pipefail
CT_ARENA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$CT_ARENA_ROOT/mjlab_training/src:$CT_ARENA_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "$CT_ARENA_ROOT/mjlab_training/.venv/bin/python" -m sim2sim.rmuc2027_arena "$@"
