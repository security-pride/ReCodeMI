#!/usr/bin/env python3
from _common import build_parser, run_baseline


def main():
    parser = build_parser("margin")
    args = parser.parse_args()
    run_baseline("margin", args)


if __name__ == "__main__":
    main()
