#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate recorded RQ2 metrics and analysis files.")
    parser.add_argument("--results-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.results_root.resolve()
    lat_path = result_dir / "LAT.json"
    if not lat_path.is_file():
        candidate = result_dir / "single_run"
        if (candidate / "LAT.json").is_file():
            result_dir = candidate
            lat_path = result_dir / "LAT.json"
        else:
            raise FileNotFoundError(f"Missing RQ2 LAT.json under: {result_dir}")

    lat = _load_json(lat_path)
    geometry_path = result_dir / "rq2_geometry_analysis.json"
    geometry = _load_json(geometry_path) if geometry_path.is_file() else {}
    classifier = lat.get("linear_classifier", {}) or {}
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": lat.get("model_name") or lat.get("model"),
        "best_layer": classifier.get("best_layer", lat.get("best_layer")),
        "test_metrics": classifier.get("test_metrics", {}),
        "timing": lat.get("timing", {}),
        "geometry": geometry,
        "files": {
            "lat_json": str(lat_path),
            "lat_val_json": str(result_dir / "LAT_val.json"),
            "geometry_json": str(geometry_path) if geometry_path.is_file() else None,
            "geometry_samples_csv": str(result_dir / "rq2_geometry_samples.csv"),
            "geometry_group_summary_csv": str(result_dir / "rq2_geometry_group_summary.csv"),
            "geometry_spearman_csv": str(result_dir / "rq2_geometry_spearman.csv"),
        },
    }
    output = result_dir / "RQ2_RESULTS_SUMMARY.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RQ2] summary: {output}")


if __name__ == "__main__":
    main()
