#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the RQ2 single-run LAT evaluation.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/stack2/RQ2"))
    parser.add_argument("--single-run-dir", type=Path)
    parser.add_argument("--parent-prefix", default="stack2/RQ2")
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.single_run_dir or args.dataset_root / "single_run_merged"
    split_paths = {split: data_dir / f"{split}.jsonl" for split in ("fit", "validate", "test")}
    for path in split_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing RQ2 dataset split: {path}")

    output_dir = args.output_root / args.model / args.parent_prefix / "single_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "main.py",
        "--fit",
        str(split_paths["fit"]),
        "--validate",
        str(split_paths["validate"]),
        "--test",
        str(split_paths["test"]),
        "--output-dir",
        str(output_dir),
        "--model",
        args.model,
        "--concept",
        "familiarity",
        "--stimulus",
        "starcoder2-mia-test",
    ]
    if args.model_dir:
        command.extend(["--model-dir", args.model_dir])
    print("[RQ2]", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
