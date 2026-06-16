#!/usr/bin/env python3
from _common import build_parser, run_baseline


def main():
    parser = build_parser("entropy")
    args = parser.parse_args()
    run_baseline("entropy", args)


if __name__ == "__main__":
    main()
