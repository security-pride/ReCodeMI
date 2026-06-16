#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODELS = [
    "bigcode/starcoder2-15b",
    "bigcode/starcoder2-7b",
    "bigcode/starcoder2-3b",
    "sapienzanlp/Minerva-7B-base-v1.0",
    "JetBrains/Mellum-4b-base",
]
SETTINGS = ["mem_only", "syn_only", "full_recodemi"]
GROUP_TO_LEVEL = {"seen": 0, "v1": 1, "v2": 2, "v3": 3, "v4": 4, "unseen": 5}


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_models(value: Optional[str]) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()] if value else list(DEFAULT_MODELS)


def _normalize_setting(value: str) -> str:
    aliases = {
        "full-recodemi": "full_recodemi",
        "full_recodemi": "full_recodemi",
    }
    normalized = str(value).strip()
    return aliases.get(normalized, normalized)


def _rq4_rows(root: Path, model: str, parent_file: str) -> List[Dict[str, Any]]:
    rq4_dir = root / "results" / model / parent_file / "RQ4"
    candidates = [
        rq4_dir / "ablation" / "single_axis_eval_summary.csv",
        rq4_dir / "single_axis_eval_summary.csv",
    ]
    summary_path = next((path for path in candidates if path.is_file()), None)
    if summary_path is None:
        return []
    rows: List[Dict[str, Any]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            setting = _normalize_setting(raw.get("setting", ""))
            if setting not in SETTINGS:
                continue
            for metric, field in (
                ("test_auroc", "test_auroc"),
                ("test_accuracy", "test_accuracy"),
                ("test_f1", "test_f1"),
            ):
                rows.append(
                    {
                        "source": "RQ4",
                        "model_name": model,
                        "setting": setting,
                        "metric": metric,
                        "value": raw.get(field),
                        "n": "",
                    }
                )
    return rows


def _find_rq2_lat(root: Path, model: str, parent_prefix: str) -> Optional[Path]:
    candidates = [
        root / "results" / model / parent_prefix / "single_run" / "LAT.json",
        root / "results" / model / parent_prefix / "LAT.json",
    ]
    return next((path for path in candidates if path.is_file()), None)


def _label_type(label: Any) -> int:
    if isinstance(label, list):
        values = [int(value) for value in label]
        if 2 in values:
            return 2
        if len(values) >= 2:
            if values[0] > 0 and values[1] > 0:
                return 1
            if values[0] == 0 and values[1] > 0:
                return 2
            return 0
        return 1 if values and values[0] > 0 else 0
    value = int(label)
    return 2 if value == 2 else int(value > 0)


def _index_map(question_id: int, count: int) -> Dict[int, int]:
    old_indices = list(range(count))
    random.Random(question_id).shuffle(old_indices)
    return {old_index: new_index for new_index, old_index in enumerate(old_indices)}


def _collect_rq2_rows(data: Dict[str, Any], layer: int, old_indices: List[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in data.get("results", []):
        if int(item.get("layer", 99999)) != layer:
            continue
        labels = item.get("correct_label_list", [])
        mem_scores = item.get("score_mem_X", [])
        syn_scores = item.get("score_syn_Y", [])
        count = min(len(labels), len(mem_scores), len(syn_scores))
        try:
            question_id = int(item.get("question_id"))
        except (TypeError, ValueError):
            continue
        for index in range(count):
            label = _label_type(labels[index])
            group = "seen" if label == 1 else "unseen" if label == 0 else None
            if group:
                rows.append(
                    {
                        "group": group,
                        "mutation_level": GROUP_TO_LEVEL[group],
                        "mem_score": float(mem_scores[index]),
                        "syn_score": float(syn_scores[index]),
                    }
                )
        mapping = _index_map(question_id, count)
        for level, old_index in enumerate(old_indices, 1):
            new_index = mapping.get(old_index)
            if new_index is None or new_index >= count:
                continue
            group = f"v{level}"
            rows.append(
                {
                    "group": group,
                    "mutation_level": GROUP_TO_LEVEL[group],
                    "mem_score": float(mem_scores[new_index]),
                    "syn_score": float(syn_scores[new_index]),
                }
            )
    return rows


def _feature(row: Dict[str, Any], setting: str) -> np.ndarray:
    if setting == "mem_only":
        return np.asarray([row["mem_score"]], dtype=float)
    if setting == "syn_only":
        return np.asarray([row["syn_score"]], dtype=float)
    return np.asarray([row["mem_score"], row["syn_score"]], dtype=float)


def _relative_positions(rows: List[Dict[str, Any]], setting: str) -> List[float]:
    seen = np.asarray([_feature(row, setting) for row in rows if row["group"] == "seen"])
    unseen = np.asarray([_feature(row, setting) for row in rows if row["group"] == "unseen"])
    if seen.size == 0 or unseen.size == 0:
        return [math.nan] * len(rows)
    seen_center = np.mean(seen, axis=0)
    unseen_center = np.mean(unseen, axis=0)
    values: List[float] = []
    for row in rows:
        point = _feature(row, setting)
        d_seen = float(np.linalg.norm(point - seen_center))
        d_unseen = float(np.linalg.norm(point - unseen_center))
        denominator = d_seen + d_unseen
        values.append(d_seen / denominator if denominator > 1e-12 else 0.5)
    return values


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    index = 0
    while index < values.size:
        end = index
        while end + 1 < values.size and values[order[end + 1]] == values[order[index]]:
            end += 1
        ranks[order[index : end + 1]] = 0.5 * (index + end) + 1.0
        index = end + 1
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[Optional[float], int]:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left, right = left[mask], right[mask]
    if left.size < 2:
        return None, int(left.size)
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    left_rank = left_rank - np.mean(left_rank)
    right_rank = right_rank - np.mean(right_rank)
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator < 1e-12:
        return None, int(left.size)
    return float(np.dot(left_rank, right_rank) / denominator), int(left.size)


def _rq2_rows(root: Path, model: str, parent_prefix: str, old_indices: List[int]) -> List[Dict[str, Any]]:
    lat_path = _find_rq2_lat(root, model, parent_prefix)
    if lat_path is None:
        return []
    data = _load_json(lat_path)
    layer = int(data.get("linear_classifier", {}).get("best_layer", data.get("best_layer", -1)))
    samples = _collect_rq2_rows(data, layer, old_indices)
    levels = [sample["mutation_level"] for sample in samples]
    rows: List[Dict[str, Any]] = []
    for setting in SETTINGS:
        rho, count = _spearman(levels, _relative_positions(samples, setting))
        rows.append(
            {
                "source": "RQ2",
                "model_name": model,
                "setting": setting,
                "metric": "relative_exposure_spearman",
                "value": rho,
                "n": count,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge selected RQ4 and RQ2 metrics into one CSV.")
    parser.add_argument("--root-dir", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--parent-file", default="stack2")
    parser.add_argument("--rq2-parent-prefix", default="stack2/RQ2")
    parser.add_argument("--models")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--independent-old-indices", default="2,3,4,5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root_dir.resolve()
    old_indices = [int(value.strip()) for value in args.independent_old_indices.split(",") if value.strip()]
    if len(old_indices) != 4:
        raise ValueError("--independent-old-indices must contain four values.")
    rows: List[Dict[str, Any]] = []
    for model in _parse_models(args.models):
        rows.extend(_rq4_rows(root, model, args.parent_file))
        rows.extend(_rq2_rows(root, model, args.rq2_parent_prefix, old_indices))

    output = args.output.resolve() if args.output else root / "results" / "RQ4" / "rq4_rq2_metrics.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["source", "model_name", "setting", "metric", "value", "n"]
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[aggregate] rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()
