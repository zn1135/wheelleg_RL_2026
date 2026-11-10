"""核对受保护仿真脚本，只允许已整理的轮速采集路径变化。"""
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RELOCATED = {
    'sim2sim/calibrate_fixed_h7_huananhu.py': (
        'wheel-bench-test-20261021',
        'wheel-bench-1p70-test-20261021',
        'wheel-bench-3p00-300ms-speed30-test-20261021',
        'fixed-h7-calibration-20261021',
        'real2sim-latest-gap-20261021',
    ),
    'sim2sim/probe_h7_drive_response.py': ('fixed-h7-fixed300-data-20261021',),
    'sim2sim/real2sim_backlash_huananhu.py': (
        'wheel-sine-test-20261021', 'real2sim-dynamics-20261021',
    ),
}


def matches_baseline(path, expected):
    path = Path(path)
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return True
    try:
        name = str(path.relative_to(ROOT))
    except ValueError:
        return False
    if name not in RELOCATED:
        return False
    for directory in RELOCATED[name]:
        new = re.escape(f'artifacts/real2sim/{directory}'.encode())
        old = f'artifacts/{directory}'.encode()
        data = re.sub(new + rb'(?![A-Za-z0-9_-])', lambda _: old, data)
    return hashlib.sha256(data).hexdigest() == expected
