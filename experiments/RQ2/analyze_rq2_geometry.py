#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

GROUP_TO_LEVEL = {
    "seen": 0,
    "v1": 1,
    "v2": 2,
    "v3": 3,
    "v4": 4,
    "unseen": 5,
}
GROUP_ORDER = ["seen", "v1", "v2", "v3", "v4", "unseen"]
GROUP_DISPLAY = {
    "seen": "Seen",
    "v1": "V1",
    "v2": "V2",
    "v3": "V3",
    "v4": "V4",
    "unseen": "Unseen",
}


def iter_layer_items(lat_json: Dict, layer: int):
    for item in lat_json.get("results", []):
        if int(item.get("layer", 99999)) == int(layer):
            yield item


def label_type(label) -> int:
    if isinstance(label, list):
        vals = [int(v) for v in label]
        if 2 in vals:
            return 2
        if len(vals) >= 2:
            mem = vals[0]
            syn = vals[1]
            if mem > 0 and syn > 0:
                return 1
            if mem == 0 and syn > 0:
                return 2
            return 0
        if len(vals) == 1:
            v = vals[0]
            if v == 2:
                return 2
            return 1 if v > 0 else 0
    v = int(label)
    if v == 2:
        return 2
    return 1 if v > 0 else 0


def _old_to_new_index_map(question_id: int, num_choices: int) -> Dict[int, int]:
    import random

    old_indices = list(range(num_choices))
    rng = random.Random(question_id)
    rng.shuffle(old_indices)
    return {old_idx: new_pos for new_pos, old_idx in enumerate(old_indices)}


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = a.size
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(x.size)
    if n < 2:
        return None, n

    rx = _rankdata_average(x)
    ry = _rankdata_average(y)

    rx = rx - np.mean(rx)
    ry = ry - np.mean(ry)
    denominator = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denominator < 1e-12:
        return None, n

    return float(np.dot(rx, ry) / denominator), n


def _collect_single_run_rows(lat_json: Dict, layer: int, independent_old_indices: List[int]) -> List[Dict]:
    rows: List[Dict] = []

    for item in iter_layer_items(lat_json, int(layer)):
        labels = item.get("correct_label_list", [])
        xs = item.get("score_mem_X", [])
        ys = item.get("score_syn_Y", [])
        qid = item.get("question_id")

        n = min(len(labels), len(xs), len(ys))
        if n == 0:
            continue

        try:
            qid_int = int(qid)
        except Exception:
            continue

        seen_idx = [i for i in range(n) if label_type(labels[i]) == 1]
        unseen_idx = [i for i in range(n) if label_type(labels[i]) == 0]

        if seen_idx:
            for i in seen_idx:
                rows.append(
                    {
                        "question_id": qid_int,
                        "group": "seen",
                        "mutation_level": GROUP_TO_LEVEL["seen"],
                        "mem_score": float(xs[i]),
                        "syn_score": float(ys[i]),
                    }
                )

        if unseen_idx:
            for i in unseen_idx:
                rows.append(
                    {
                        "question_id": qid_int,
                        "group": "unseen",
                        "mutation_level": GROUP_TO_LEVEL["unseen"],
                        "mem_score": float(xs[i]),
                        "syn_score": float(ys[i]),
                    }
                )

        old_to_new = _old_to_new_index_map(qid_int, n)
        for idx, old_idx in enumerate(independent_old_indices, 1):
            group_name = f"v{idx}"
            if group_name not in GROUP_TO_LEVEL:
                continue
            new_idx = old_to_new.get(int(old_idx))
            if new_idx is None or new_idx >= n:
                continue
            rows.append(
                {
                    "question_id": qid_int,
                    "group": group_name,
                    "mutation_level": GROUP_TO_LEVEL[group_name],
                    "mem_score": float(xs[new_idx]),
                    "syn_score": float(ys[new_idx]),
                }
            )

    return rows


def _compute_centers(rows: List[Dict]) -> Dict[str, np.ndarray]:
    seen_pts = np.array([[r["mem_score"], r["syn_score"]] for r in rows if r["group"] == "seen"], dtype=float)
    unseen_pts = np.array([[r["mem_score"], r["syn_score"]] for r in rows if r["group"] == "unseen"], dtype=float)

    if seen_pts.size == 0 or unseen_pts.size == 0:
        raise ValueError("Not enough points to compute seen/unseen centers.")

    return {
        "seen": np.mean(seen_pts, axis=0),
        "unseen": np.mean(unseen_pts, axis=0),
    }


def _attach_relative_exposure_positions(rows: List[Dict], centers: Dict[str, np.ndarray]) -> None:
    c_seen = centers["seen"]
    c_unseen = centers["unseen"]

    for r in rows:
        p = np.array([r["mem_score"], r["syn_score"]], dtype=float)
        d_seen = float(np.linalg.norm(p - c_seen))
        d_unseen = float(np.linalg.norm(p - c_unseen))
        denom = d_seen + d_unseen
        r["d_seen"] = d_seen
        r["d_unseen"] = d_unseen
        r["relative_exposure_position_r"] = float(d_seen / denom) if denom > 1e-12 else 0.5


def _group_stats(rows: List[Dict], key: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for g in GROUP_ORDER:
        vals = [float(r[key]) for r in rows if r["group"] == g]
        if not vals:
            out[g] = {"n": 0, "mean": None, "std": None, "median": None}
            continue
        arr = np.asarray(vals, dtype=float)
        out[g] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "median": float(np.median(arr)),
        }
    return out


def _write_group_summary_csv(rows: List[Dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Group",
                "d_seen mean ± std",
                "d_unseen mean ± std",
                "r(x) mean ± std",
            ]
        )
        for g in GROUP_ORDER:
            sub = [r for r in rows if r["group"] == g]
            if not sub:
                ds = "NA"
                du = "NA"
                rx = "NA"
            else:
                d_seen = np.asarray([float(r["d_seen"]) for r in sub], dtype=float)
                d_unseen = np.asarray([float(r["d_unseen"]) for r in sub], dtype=float)
                rel = np.asarray([float(r["relative_exposure_position_r"]) for r in sub], dtype=float)
                ds = f"{float(np.mean(d_seen)):.6f} ± {float(np.std(d_seen, ddof=0)):.6f}"
                du = f"{float(np.mean(d_unseen)):.6f} ± {float(np.std(d_unseen, ddof=0)):.6f}"
                rx = f"{float(np.mean(rel)):.6f} ± {float(np.std(rel, ddof=0)):.6f}"

            writer.writerow([g, ds, du, rx])


def _spearman_summary(rows: List[Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    lvl = np.array([int(r["mutation_level"]) for r in rows], dtype=float)
    mem = np.array([float(r["mem_score"]) for r in rows], dtype=float)
    syn = np.array([float(r["syn_score"]) for r in rows], dtype=float)
    d_seen = np.array([float(r["d_seen"]) for r in rows], dtype=float)
    d_unseen = np.array([float(r["d_unseen"]) for r in rows], dtype=float)
    rel = np.array([float(r["relative_exposure_position_r"]) for r in rows], dtype=float)

    out = {}
    for name, arr in [
        ("mem_score", mem),
        ("syn_score", syn),
        ("d_seen", d_seen),
        ("d_unseen", d_unseen),
        ("relative_exposure_position_r", rel),
    ]:
        rho, n = _spearmanr(lvl, arr)
        out[name] = {
            "spearman_rho": rho,
            "n": int(n),
        }
    return out


def _write_spearman_csv(spearman: Dict[str, Dict[str, Optional[float]]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "spearman_rho", "n"])
        for metric in ["mem_score", "syn_score", "d_seen", "d_unseen", "relative_exposure_position_r"]:
            item = spearman.get(metric, {}) or {}
            writer.writerow([metric, item.get("spearman_rho"), item.get("n")])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze RQ2 single-run geometry and correlation with mutation levels.")
    p.add_argument("--lat-json", type=Path, required=True, help="Path to single-run LAT.json")
    p.add_argument(
        "--independent-old-indices",
        type=str,
        default="2,3,4,5",
        help="Comma-separated old choice indices for v1..v4.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Output JSON path (default: <LAT_dir>/rq2_geometry_analysis.json)",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <LAT_dir>/rq2_geometry_samples.csv)",
    )
    p.add_argument(
        "--group-summary-csv",
        type=Path,
        default=None,
        help="Output group summary CSV path (default: <LAT_dir>/rq2_geometry_group_summary.csv)",
    )
    p.add_argument(
        "--spearman-csv",
        type=Path,
        default=None,
        help="Output Spearman summary CSV path (default: <LAT_dir>/rq2_geometry_spearman.csv)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lat_json_path = args.lat_json.resolve()
    if not lat_json_path.is_file():
        raise FileNotFoundError(f"Missing LAT json: {lat_json_path}")

    old_indices = [int(x.strip()) for x in args.independent_old_indices.split(",") if x.strip()]
    if len(old_indices) != 4:
        raise ValueError("--independent-old-indices must have exactly 4 entries for v1..v4.")

    with lat_json_path.open("r", encoding="utf-8") as f:
        lat_json = json.load(f)

    best_layer = lat_json.get("linear_classifier", {}).get("best_layer")
    if best_layer is None:
        best_layer = lat_json.get("best_layer", -3)
    best_layer = int(best_layer)

    rows = _collect_single_run_rows(lat_json, best_layer, old_indices)
    if not rows:
        raise ValueError("No valid samples extracted from LAT.json on best layer.")

    centers = _compute_centers(rows)
    _attach_relative_exposure_positions(rows, centers)

    spearman = _spearman_summary(rows)

    summary = {
        "lat_json": str(lat_json_path),
        "best_layer": int(best_layer),
        "group_to_level": GROUP_TO_LEVEL,
        "centers": {
            "seen": {"mem_score": float(centers["seen"][0]), "syn_score": float(centers["seen"][1])},
            "unseen": {"mem_score": float(centers["unseen"][0]), "syn_score": float(centers["unseen"][1])},
        },
        "num_samples": int(len(rows)),
        "by_group": {
            "mem_score": _group_stats(rows, "mem_score"),
            "d_seen": _group_stats(rows, "d_seen"),
            "d_unseen": _group_stats(rows, "d_unseen"),
            "relative_exposure_position_r": _group_stats(rows, "relative_exposure_position_r"),
        },
        "spearman_vs_mutation_level": spearman,
    }

    out_json = args.out_json.resolve() if args.out_json else (lat_json_path.parent / "rq2_geometry_analysis.json")
    out_csv = args.out_csv.resolve() if args.out_csv else (lat_json_path.parent / "rq2_geometry_samples.csv")
    out_group_csv = (
        args.group_summary_csv.resolve()
        if args.group_summary_csv
        else (lat_json_path.parent / "rq2_geometry_group_summary.csv")
    )
    out_spearman_csv = (
        args.spearman_csv.resolve() if args.spearman_csv else (lat_json_path.parent / "rq2_geometry_spearman.csv")
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "question_id",
        "group",
        "mutation_level",
        "mem_score",
        "syn_score",
        "d_seen",
        "d_unseen",
        "relative_exposure_position_r",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})

    _write_group_summary_csv(rows, out_group_csv)
    _write_spearman_csv(spearman, out_spearman_csv)

    print(f"[rq2-geometry] saved json: {out_json}")
    print(f"[rq2-geometry] saved csv: {out_csv}")
    print(f"[rq2-geometry] saved group summary csv: {out_group_csv}")
    print(f"[rq2-geometry] saved spearman csv: {out_spearman_csv}")
    print("[rq2-geometry] spearman vs mutation level:")
    for k, v in summary["spearman_vs_mutation_level"].items():
        print(f"  - {k}: rho={v.get('spearman_rho')} (n={v.get('n')})")


if __name__ == "__main__":
    main()
