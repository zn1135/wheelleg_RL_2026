"""旧命令兼容入口；训练统一进入scripts/train.py和任务注册器。"""
import argparse
import sys
from pathlib import Path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == 'train':
        from ct_mjlab.scripts.train import main as train_main
        return train_main(['--task', 'fudan_ppo_walk', *argv[1:]])
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('train', help='兼容旧命令，实际进入统一注册训练链路')
    export = sub.add_parser('export')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    from .export import export as export_policy
    return export_policy(args.checkpoint, args.output)

if __name__ == '__main__':
    main()
