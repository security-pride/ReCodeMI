#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write final 2D classifier test metrics to CSV.")
    parser.add_argument("--lat-json", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lat_json = args.lat_json.resolve()
    data = json.loads(lat_json.read_text(encoding="utf-8"))
    metrics = (data.get("linear_classifier") or {}).get("test_metrics") or {}
    required = {"accuracy", "f1_score", "auroc"}
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"Missing final classifier test metrics in {lat_json}: {missing}")

    out_csv = args.out_csv.resolve() if args.out_csv else lat_json.parent / "re_mia_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "auroc", "acc", "f1"])
        writer.writeheader()
        writer.writerow(
            {
                "model": args.model,
                "auroc": metrics["auroc"],
                "acc": metrics["accuracy"],
                "f1": metrics["f1_score"],
            }
        )
    print(f"[re-mia] metrics csv saved: {out_csv}")


if __name__ == "__main__":
    main()
