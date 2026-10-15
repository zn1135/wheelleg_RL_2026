"""Shared argument parsing and registered-task dispatch."""
import sys

import torch

from ..envs import task_registry
from ..utils.helpers import build_parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    return task_registry.run(args.task, args, parser)


def command_main(command, argv=None):
    return main([command, *(sys.argv[1:] if argv is None else argv)])
