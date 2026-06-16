#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_METHODS = (
    "loss",
    "ppl",
    "zlib",
    "min_k",
    "confidence",
    "entropy",
    "msp",
    "margin",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write baseline test metrics to CSV.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-csv", type=Path)
    return parser.parse_args()


def normalize_methods(raw: str) -> list[str]:
    methods = raw.replace(",", " ").split()
    expanded = DEFAULT_METHODS if methods == ["all"] else methods
    unknown = sorted(set(expanded) - set(DEFAULT_METHODS))
    if unknown:
        raise ValueError(f"Unsupported baseline methods: {unknown}")
    return list(dict.fromkeys(expanded))


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_csv = args.out_csv.resolve() if args.out_csv else out_dir / "baseline_metrics.csv"
    rows = []

    for method in normalize_methods(args.methods):
        result_path = out_dir / f"{method}_results.json"
        if not result_path.is_file():
            raise FileNotFoundError(f"Missing baseline result: {result_path}")
        data = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = data.get("test_candidate_metrics") or {}
        rows.append(
            {
                "method": method,
                "model": args.model,
                "auroc": metrics.get("auroc"),
                "acc": metrics.get("accuracy"),
                "f1": metrics.get("f1_score"),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "model", "auroc", "acc", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[baseline] metrics csv saved: {out_csv} (rows={len(rows)})")


if __name__ == "__main__":
    main()
