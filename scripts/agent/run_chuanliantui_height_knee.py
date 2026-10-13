"""Only the two explicitly authorized A2500 +1000 training cases."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
from ct_mjlab.experiments import height_knee as h

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=h.CASES, required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    h.train(p.parse_args())
