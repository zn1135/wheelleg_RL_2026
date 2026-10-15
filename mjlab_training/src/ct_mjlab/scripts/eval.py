"""Eval a registered mjlab task."""
from ct_mjlab.scripts._entry import command_main


def main(argv=None):
    return command_main('eval', argv)


if __name__ == '__main__':
    main()
