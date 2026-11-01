"""检查已核对有 argparse 的代表入口；只调用 --help，不训练、不连接设备。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
MJLAB = (
    'motion.assess', 'supermimic.check_chuanliantui_supermimic',
    'supermimic.check_supermimic_motion_retrain', 'supermimic.score_supermimic_motion',
    'supermimic.score_supermimic_motion_retrain', 'imitation.check_lqr_vmc',
    'imitation.select_preserved_bc', 'imitation.render_preserved_bc',
    'real2sim.check_real2sim_huananhu', 'real2sim.report_fixed_h7_calibration',
    'maintenance.wait_training_completion', 'maintenance.restore_script_archive',
)



def check(entry):
    python, name = entry
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'mjlab_training/src') if python == sys.executable else str(ROOT),
               CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    command = [python, '-m', 'scripts.agent.' + name, '--help']
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=45)
        passed = result.returncode == 0 and 'usage:' in result.stdout.lower()
        return dict(module=name, python=python, passed=passed, exit_code=result.returncode,
                    diagnostic='' if passed else (result.stdout + result.stderr)[-2200:])
    except subprocess.TimeoutExpired:
        return dict(module=name, python=python, passed=False, diagnostic='--help exceeded 45s')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--isaac-python', type=Path)
    args = parser.parse_args()
    entries = [(sys.executable, name) for name in MJLAB]
    if args.isaac_python:
        entries.append((str(args.isaac_python), 'isaac.export_chuanliantui_onnx'))
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(check, entries))
    print(json.dumps(dict(results=results, passed=sum(r['passed'] for r in results), total=len(results),
                          scope='代表入口导入与参数检查；其他脚本由静态检查覆盖，未运行训练/物理/实采。'), ensure_ascii=False))
    raise SystemExit(0 if all(r['passed'] for r in results) else 1)


if __name__ == '__main__':
    main()
