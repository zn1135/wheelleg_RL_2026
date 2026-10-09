#!/usr/bin/env bash
# 只在本仓创建轻量运行环境；现有 mjlab 环境只读。
set -euo pipefail
CT_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CT_EXISTING_PYTHON=${1:?用法：scripts/setup_mjlab.sh <现有mjlab环境的python绝对路径>}
"$CT_EXISTING_PYTHON" - "$CT_REPO_ROOT" <<'PY'
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import sysconfig
import subprocess
import venv
root=Path(sys.argv[1])
assert sys.version_info[:2]==(3,12), '需要现有 Python 3.12 mjlab 环境'
versions={name:metadata.version(name) for name in ('mjlab','torch','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','tensordict','onnx')}
assert versions['mjlab']=='1.6.0' and versions['rsl-rl-lib']=='5.4.2', versions
site=Path(sysconfig.get_paths()['purelib'])
target=root/'mjlab_training/.venv'
if not (target/'pyvenv.cfg').exists():venv.EnvBuilder(with_pip=False).create(target)
subprocess.run([str(target/'bin/python'),'-c',
    'import sys; assert sys.version_info[:2] == (3,12), "本仓已有环境不是 Python 3.12，请另行处理"'],check=True)
local=target/'lib/python3.12/site-packages';local.mkdir(parents=True,exist_ok=True)
(local/'reused_mjlab_runtime.pth').write_text(str(site)+'\n'+str(root/'mjlab_training/src')+'\n')
(target/'reused-runtime.json').write_text(json.dumps(dict(python=sys.executable,versions=versions),indent=2)+'\n')
print('复用运行时：',sys.executable,versions)
PY
uv pip install --python "$CT_REPO_ROOT/mjlab_training/.venv/bin/python" --no-deps 'onnxruntime==1.26.0'
