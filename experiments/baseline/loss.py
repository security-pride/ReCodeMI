#!/usr/bin/env python3
from _common import build_parser, run_baseline


def main():
    parser = build_parser("loss")
    args = parser.parse_args()
    run_baseline("loss", args)


if __name__ == "__main__":
    main()
