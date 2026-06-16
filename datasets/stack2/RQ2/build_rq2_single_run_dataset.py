#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


LEVELS = [
    "v1",
    "v2",
    "v3",
    "v4",
]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def ensure_list_str(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            else:
                out.append("" if item is None else str(item))
        return out
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    return [str(value)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build RQ2 single-run dataset: one test file containing "
            "seen, unseen, and 4 cumulative independent variants."
        )
    )
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("datasets/stack2"),
        help="Root containing base splits and the RQ2/v1..v4 subfolders.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/stack2/RQ2/single_run_merged"),
        help="Output dir for fit/validate/test jsonl.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input_root.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_test = root / "test.jsonl"
    if not base_test.is_file():
        raise FileNotFoundError(f"Missing base test file: {base_test}")

    level_tests = [root / "RQ2" / lvl / "test.jsonl" for lvl in LEVELS]
    for p in level_tests:
        if not p.is_file():
            raise FileNotFoundError(f"Missing level test file: {p}")

    base_rows = list(iter_jsonl(base_test))
    level_rows = [list(iter_jsonl(p)) for p in level_tests]
    n = len(base_rows)
    for lvl, rows in zip(LEVELS, level_rows):
        if len(rows) != n:
            raise ValueError(f"Row count mismatch: {lvl} has {len(rows)} rows, base has {n}")

    out_test = out_dir / "test.jsonl"
    with out_test.open("w", encoding="utf-8") as f:
        for i in range(n):
            base = dict(base_rows[i])
            syn_base = ensure_list_str(base.get("syn_code"))
            mem_base = ensure_list_str(base.get("mem_code"))
            seen_syn = syn_base[0] if len(syn_base) > 0 else ""
            unseen_syn = syn_base[1] if len(syn_base) > 1 else ""
            seen_mem = mem_base[0] if len(mem_base) > 0 else ""
            unseen_mem = mem_base[1] if len(mem_base) > 1 else ""

            syn_variants: List[str] = []
            mem_variants: List[str] = []
            for lvl_rows in level_rows:
                row = lvl_rows[i]
                syn_list = ensure_list_str(row.get("syn_code"))
                mem_list = ensure_list_str(row.get("mem_code"))
                syn_variants.append(syn_list[2] if len(syn_list) > 2 else (syn_list[-1] if syn_list else ""))
                mem_variants.append(mem_list[2] if len(mem_list) > 2 else (mem_list[-1] if mem_list else ""))

            out_row = dict(base)
            out_row["syn_code"] = [seen_syn, unseen_syn] + syn_variants
            out_row["mem_code"] = [seen_mem, unseen_mem] + mem_variants
            out_row["syn_labels"] = [1, 0, 2, 2, 2, 2]
            out_row["mem_labels"] = [1, 0, 2, 2, 2, 2]
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    # fit/validate can reuse any level copy (they are identical in builder logic).
    src_fit = root / "RQ2" / LEVELS[-1] / "fit.jsonl"
    src_validate = root / "RQ2" / LEVELS[-1] / "validate.jsonl"
    if not src_fit.is_file():
        src_fit = root / "fit.jsonl"
    if not src_validate.is_file():
        src_validate = root / "validate.jsonl"
    if not src_fit.is_file() or not src_validate.is_file():
        raise FileNotFoundError("Missing fit/validate source files.")

    shutil.copy2(src_fit, out_dir / "fit.jsonl")
    shutil.copy2(src_validate, out_dir / "validate.jsonl")

    print(f"[single-run] output_dir={out_dir}")
    print(f"[single-run] fit={out_dir / 'fit.jsonl'}")
    print(f"[single-run] validate={out_dir / 'validate.jsonl'}")
    print(f"[single-run] test={out_test}")
    print("[single-run] test choices per row: seen, unseen, v1, v2, v3, v4")


if __name__ == "__main__":
    main()
