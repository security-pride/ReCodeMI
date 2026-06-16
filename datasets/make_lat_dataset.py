#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_jsonl(
    path: Path,
    skip_starter_code: bool = False,
    require_solution: bool = False,
) -> list[dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if skip_starter_code and sample.get("starter_code"):
                continue
            solutions = sample.get("solutions")
            if require_solution and not (
                isinstance(solutions, list)
                and solutions
                and isinstance(solutions[0], str)
                and solutions[0]
            ):
                continue
            data.append(sample)
    return data


def split_code_middle(code: Any) -> dict[str, str]:
    if not isinstance(code, str):
        code = "" if code is None else str(code)
    lines = code.splitlines(keepends=True)
    if not lines:
        return {"input": "", "gt": ""}

    mid_line_idx = len(lines) // 2
    mid_line = lines[mid_line_idx]
    mid_col = len(mid_line) // 2
    return {
        "input": "".join(lines[:mid_line_idx]) + mid_line[:mid_col],
        "gt": mid_line[mid_col:] + "".join(lines[mid_line_idx + 1 :]),
    }


def build_code_generation(mem_code: list[str]) -> list[dict[str, str]]:
    return [split_code_middle(code) for code in mem_code[:2]]


def generate_sample(index: int, stack2_sample: dict, apps_sample: dict) -> dict:
    stack2_code = stack2_sample["code"]
    apps_code = apps_sample["solutions"][0]
    mem_code = [stack2_code, apps_code]
    return {
        "index": index,
        "syn_code": [stack2_code, apps_code],
        "mem_code": mem_code,
        "syn_labels": [1, 0],
        "mem_labels": [1, 0],
        "code_generation": build_code_generation(mem_code),
    }


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_split(
    path: Path,
    apps_data: list[dict],
    stack2_data: list[dict],
) -> None:
    rows = (
        generate_sample(index, stack2_sample, apps_sample)
        for index, (stack2_sample, apps_sample) in enumerate(zip(stack2_data, apps_data))
    )
    write_jsonl(path, rows)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the base Stack2/APPS LAT dataset.")
    parser.add_argument("--apps-file", type=Path, default=script_dir / "APPS/APPS.jsonl")
    parser.add_argument(
        "--stack2-file",
        type=Path,
        default=script_dir / "stack2/the-stack-v2-python-90k.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=script_dir / "stack2")
    parser.add_argument("--sample-count", type=int, default=800)
    parser.add_argument("--fit-ratio", type=float, default=0.1)
    parser.add_argument("--validate-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_count <= 0:
        raise ValueError("--sample-count must be positive")
    if args.fit_ratio < 0 or args.validate_ratio < 0:
        raise ValueError("Split ratios must be non-negative")
    if args.fit_ratio + args.validate_ratio >= 1:
        raise ValueError("--fit-ratio + --validate-ratio must be less than 1")

    apps_data = load_jsonl(
        args.apps_file,
        skip_starter_code=True,
        require_solution=True,
    )
    stack2_data = load_jsonl(args.stack2_file)
    if len(apps_data) < args.sample_count:
        raise ValueError(
            f"APPS has {len(apps_data)} eligible rows, fewer than {args.sample_count}"
        )
    if len(stack2_data) < args.sample_count:
        raise ValueError(f"Stack2 has {len(stack2_data)} rows, fewer than {args.sample_count}")

    rng = random.Random(args.seed)
    apps_data = rng.sample(apps_data, args.sample_count)
    stack2_data = rng.sample(stack2_data, args.sample_count)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original_rows = []
    for index, (stack2_sample, apps_sample) in enumerate(zip(stack2_data, apps_data)):
        original_rows.append(
            {
                "index": index,
                "code_1": stack2_sample["code"],
                "code_2": apps_sample["solutions"][0],
                "stack2_id": stack2_sample.get("sample_id") or stack2_sample.get("index"),
                "apps_id": apps_sample["id"],
            }
        )
    write_jsonl(args.output_dir / "original_stack2_apps.jsonl", original_rows)

    fit_end = int(args.sample_count * args.fit_ratio)
    validate_end = fit_end + int(args.sample_count * args.validate_ratio)
    split_bounds = {
        "fit": (0, fit_end),
        "validate": (fit_end, validate_end),
        "test": (validate_end, args.sample_count),
    }
    for split, (start, end) in split_bounds.items():
        write_split(
            args.output_dir / f"{split}.jsonl",
            apps_data[start:end],
            stack2_data[start:end],
        )

    print(f"Base dataset written to: {args.output_dir}")
    print(
        f"samples={args.sample_count}, fit={fit_end}, "
        f"validate={validate_end - fit_end}, test={args.sample_count - validate_end}"
    )


if __name__ == "__main__":
    main()
