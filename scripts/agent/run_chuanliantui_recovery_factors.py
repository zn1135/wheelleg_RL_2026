"""Run explicit 2000-update recovery continuations through the native mjlab runner."""
import argparse
from pathlib import Path
from ct_mjlab.experiments.recovery_factors import CASES, train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=CASES, required=True)
    p.add_argument('--resume', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--iterations', type=int, required=True)
    p.add_argument('--num-envs', type=int, default=4096)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    train(p.parse_args())


if __name__ == '__main__':
    main()
