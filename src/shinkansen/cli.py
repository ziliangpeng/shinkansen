"""shinkansen CLI entrypoint."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .accelerator import summary as accelerator_summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shinkansen", description=__doc__)
    parser.add_argument("--version", action="version", version=f"shinkansen {__version__}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("info", help="Print accelerator and environment info.")
    sub.add_parser("train", help="Run a training job (not implemented yet).")

    args = parser.parse_args(argv)

    if args.command == "info" or args.command is None:
        print(accelerator_summary())
        return 0

    if args.command == "train":
        print("train: not implemented yet — laying track.", file=sys.stderr)
        return 2

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
