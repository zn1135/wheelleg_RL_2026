#!/usr/bin/env bash
# 本仓独立环境可通过 setup_mjlab.sh 复用已安装的 mjlab，无需修改其仓库。
set -euo pipefail
CT_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CT_MJLAB_PYTHON="$CT_REPO_ROOT/mjlab_training/.venv/bin/python"
if [[ ! -x "$CT_MJLAB_PYTHON" ]]; then
    echo '请先运行 scripts/setup_mjlab.sh <已安装mjlab的python路径>，或 uv sync --project mjlab_training --locked --python 3.12' >&2
    exit 1
fi
export PYTHONPATH="$CT_REPO_ROOT/mjlab_training/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$CT_MJLAB_PYTHON" -m ct_mjlab.cli "$@"
