#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def split_code_in_half(code: str) -> tuple[str, str]:
    if not isinstance(code, str):
        code = "" if code is None else str(code)
    midpoint = len(code) // 2
    return code[:midpoint], code[midpoint:]


def write_dataset(input_jsonl: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with (output_dir / "train.jsonl").open("w", encoding="utf-8") as out_train, (
        output_dir / "test.jsonl"
    ).open("w", encoding="utf-8") as out_test:
        for count, obj in enumerate(iter_jsonl(input_jsonl), 1):
            if "code_1" not in obj or "code_2" not in obj:
                raise KeyError(f"Row {count} must contain code_1 and code_2")
            train_input, train_gt = split_code_in_half(obj["code_1"])
            test_input, test_gt = split_code_in_half(obj["code_2"])
            out_train.write(
                json.dumps(
                    {"id": str(count), "input": train_input, "gt": train_gt},
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_test.write(
                json.dumps(
                    {"id": str(count), "input": test_input, "gt": test_gt},
                    ensure_ascii=False,
                )
                + "\n"
            )
    if count == 0:
        raise ValueError(f"No records found in {input_jsonl}")
    print(f"AdvPrompt-MIA dataset written to: {output_dir} ({count} pairs)")


def parse_args() -> argparse.Namespace:
    stack2_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the AdvPrompt-MIA paired dataset.")
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=stack2_dir / "original_stack2_apps.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=stack2_dir / "AdvPrompt-MIA")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_dataset(args.input_jsonl, args.output_dir)


if __name__ == "__main__":
    main()
