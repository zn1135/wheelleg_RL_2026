"""ct-mjlab / scripts/mjlab.sh entry: dispatch `<command> [args]` to ct_mjlab.scripts.<command>."""
from importlib import import_module
import sys

COMMANDS = ('train', 'play', 'eval', 'export', 'check')


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        sys.exit(f'usage: ct-mjlab {{{",".join(COMMANDS)}}} [args]  (use <command> --help)')
    command, sys.argv = argv[0], [f'ct-mjlab {argv[0]}', *argv[1:]]
    return import_module(f'ct_mjlab.scripts.{command}').main()


if __name__ == '__main__':
    main()
