#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from single_axis_mia_eval import load_json, run_axis


DEFAULT_MODELS = [
    "bigcode/starcoder2-15b",
    "bigcode/starcoder2-7b",
    "sapienzanlp/Minerva-7B-base-v1.0",
    "JetBrains/Mellum-4b-base",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the RQ4 AUROC-only summary CSV.")
    parser.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--parent-file", default="stack2")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-label2-as-positive", action="store_true")
    return parser.parse_args()


def extract_auroc(result: dict, model: str, setting: str) -> float:
    if "error" in result:
        raise ValueError(f"{model} {setting} failed: {result['error']}")
    value = (result.get("test_metrics") or {}).get("auroc")
    if value is None:
        raise ValueError(f"{model} {setting} has no test AUROC")
    return float(value)


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    models = [model.strip() for model in args.models.split(",") if model.strip()]
    rows = []

    for model in models:
        full_dir = root / "results" / model / args.parent_file / "RQ4" / "full"
        test_path = full_dir / "LAT.json"
        val_path = full_dir / "LAT_val.json"
        if not test_path.is_file() or not val_path.is_file():
            raise FileNotFoundError(f"Missing existing RQ4 LAT files under: {full_dir}")

        test_data = load_json(test_path)
        val_data = load_json(val_path)
        mem_result = run_axis(
            "mem",
            val_data,
            test_data,
            include_label2_as_positive=args.include_label2_as_positive,
        )
        syn_result = run_axis(
            "syn",
            val_data,
            test_data,
            include_label2_as_positive=args.include_label2_as_positive,
        )
        full_result = test_data.get("linear_classifier") or {}

        for setting, value in (
            ("mem_only", extract_auroc(mem_result, model, "mem_only")),
            ("syn_only", extract_auroc(syn_result, model, "syn_only")),
            ("full_recodemi", extract_auroc(full_result, model, "full_recodemi")),
        ):
            rows.append(
                {
                    "source": "RQ4",
                    "model_name": model,
                    "setting": setting,
                    "metric": "test_auroc",
                    "value": value,
                }
            )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "model_name", "setting", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"output={output}")


if __name__ == "__main__":
    main()
