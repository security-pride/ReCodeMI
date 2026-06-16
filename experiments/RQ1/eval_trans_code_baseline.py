#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS_ROOT = os.path.dirname(HERE)
PROJECT_ROOT = os.path.dirname(EXPERIMENTS_ROOT)
BASELINE_DIR = os.path.join(EXPERIMENTS_ROOT, "baseline")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# Avoid local script name collision with stdlib zlib (baseline/zlib.py).
if BASELINE_DIR in sys.path:
    sys.path.remove(BASELINE_DIR)
    _stdlib_zlib = importlib.import_module("zlib")
    sys.path.insert(0, BASELINE_DIR)
else:
    _stdlib_zlib = importlib.import_module("zlib")
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

from loaders import load_triplets_jsonl  # noqa: E402
from _common import (  # noqa: E402
    FlatSample,
    ScoreEngine,
    fit_binary_classifier_1d,
    flatten_triplets,
    orient_scores_by_validation,
    predict_classifier_1d,
)


LEVELS = [
    "v1",
    "v2",
    "v3",
    "v4",
]

FINAL_BASELINE_METHODS = [
    "loss",
    "ppl",
    "zlib",
    "min_k",
    "confidence",
    "entropy",
    "msp",
    "margin",
]


@dataclass
class GroupScoreSample:
    row_index: int
    group: str
    score_raw: float
    score: float
    member_like_score: float
    score_before_orientation: float
    pred_prob_seen: float
    pred_seen: int


def _split_code_middle(code: str):
    if not isinstance(code, str):
        code = "" if code is None else str(code)
    lines = code.splitlines(keepends=True)
    if not lines:
        return "", ""
    mid_line_idx = len(lines) // 2
    mid_line = lines[mid_line_idx]
    mid_col = len(mid_line) // 2
    input_part = "".join(lines[:mid_line_idx]) + mid_line[:mid_col]
    gt_part = mid_line[mid_col:] + "".join(lines[mid_line_idx + 1 :])
    return input_part, gt_part


def _as_str(x) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _get_pair_from_code(code: str):
    prompt, gt = _split_code_middle(code)
    return {"input": prompt, "gt": gt}


def _extract_seen_unseen_pairs(row: dict):
    mem_list = row.get("mem_code")

    seen_code = ""
    unseen_code = ""
    if isinstance(mem_list, list) and len(mem_list) > 0:
        seen_code = _as_str(mem_list[0])
    if isinstance(mem_list, list) and len(mem_list) > 1:
        unseen_code = _as_str(mem_list[1])

    return {
        "seen": _get_pair_from_code(seen_code),
        "unseen": _get_pair_from_code(unseen_code),
    }


def load_group_pairs_by_level(base_test_path: str, level_dirs: List[str]):
    rows_by_level = {}
    for lvl in level_dirs:
        p = os.path.join(lvl, "test.jsonl")
        rows = []
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows.append((line_no, row))
        rows_by_level[lvl] = rows

    base_rows = []
    with open(base_test_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            base_rows.append((line_no, row))

    # align by row order
    n = len(base_rows)
    for lvl in level_dirs:
        if len(rows_by_level[lvl]) != n:
            raise ValueError(f"Row count mismatch: {lvl} has {len(rows_by_level[lvl])}, base has {n}")

    groups = ["seen", *LEVELS, "unseen"]
    grouped_pairs = {g: [] for g in groups}

    for idx in range(n):
        base_line_no, base_row = base_rows[idx]
        row_idx = int(base_row.get("index", base_line_no - 1))
        su = _extract_seen_unseen_pairs(base_row)

        grouped_pairs["seen"].append((row_idx, su["seen"]["input"], su["seen"]["gt"]))
        grouped_pairs["unseen"].append((row_idx, su["unseen"]["input"], su["unseen"]["gt"]))

        for lvl_name, lvl_dir in zip(LEVELS, level_dirs):
            _, lvl_row = rows_by_level[lvl_dir][idx]
            syn_list = lvl_row.get("syn_code")
            trans_code = ""
            if isinstance(syn_list, list) and len(syn_list) >= 3:
                trans_code = _as_str(syn_list[2])
            elif isinstance(syn_list, list) and len(syn_list) > 0:
                trans_code = _as_str(syn_list[-1])
            pair = _get_pair_from_code(trans_code)
            grouped_pairs[lvl_name].append((row_idx, pair["input"], pair["gt"]))

    return grouped_pairs


def compute_codegen_scores_by_methods(
    engine: ScoreEngine,
    methods: List[str],
    flat_samples: List[FlatSample],
    min_k_ratio: float,
    progress_prefix: str = "",
) -> Dict[str, np.ndarray]:
    score_map = {m: np.empty(len(flat_samples), dtype=np.float32) for m in methods}
    for i, sample in enumerate(flat_samples):
        if i % 50 == 0:
            print(f"{progress_prefix}scoring {i}/{len(flat_samples)}")
        for method in methods:
            s = engine.score_completion(
                sample.prompt or "",
                sample.gt or "",
                method=method,
                min_k_ratio=min_k_ratio,
            )
            if np.isnan(s):
                s = 0.0
            score_map[method][i] = float(s)
    return score_map


def compute_group_scores_by_methods(
    engine: ScoreEngine,
    methods: List[str],
    grouped_pairs: Dict[str, List[tuple]],
    min_k_ratio: float,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for group, pairs in grouped_pairs.items():
        out[group] = {m: np.empty(len(pairs), dtype=np.float32) for m in methods}
        for i, (_, prompt, gt) in enumerate(pairs):
            if i % 50 == 0:
                print(f"[{group}] scoring {i}/{len(pairs)}")
            for method in methods:
                s = engine.score_completion(
                    prompt,
                    gt,
                    method=method,
                    min_k_ratio=min_k_ratio,
                )
                if np.isnan(s):
                    s = 0.0
                out[group][method][i] = float(s)
    return out


def compute_monotonic_consistency(
    grouped_pairs: Dict[str, List[tuple]],
    group_scores_member_like: Dict[str, np.ndarray],
) -> Dict[str, float]:
    chain_groups = ["seen", "v1", "v2", "v3", "v4"]
    for g in chain_groups:
        if g not in grouped_pairs or g not in group_scores_member_like:
            return {
                "num_samples": 0,
                "full_chain_count": 0,
                "full_chain_rate": 0.0,
                "pairwise_counts": {},
                "pairwise_rates": {},
            }

    row_to_scores: Dict[int, Dict[str, float]] = {}
    for g in chain_groups:
        pairs = grouped_pairs[g]
        scores = group_scores_member_like[g]
        for (row_idx, _, _), s in zip(pairs, scores):
            rid = int(row_idx)
            row_to_scores.setdefault(rid, {})
            row_to_scores[rid][g] = float(s)

    complete_rows = []
    for rid, gs in row_to_scores.items():
        if all(k in gs for k in chain_groups):
            complete_rows.append((rid, gs))

    n = len(complete_rows)
    if n == 0:
        return {
            "num_samples": 0,
            "full_chain_count": 0,
            "full_chain_rate": 0.0,
            "pairwise_counts": {
                "seen_ge_v1": 0,
                "v1_ge_v2": 0,
                "v2_ge_v3": 0,
                "v3_ge_v4": 0,
            },
            "pairwise_rates": {
                "seen_ge_v1": 0.0,
                "v1_ge_v2": 0.0,
                "v2_ge_v3": 0.0,
                "v3_ge_v4": 0.0,
            },
        }

    pairwise_counts = {
        "seen_ge_v1": 0,
        "v1_ge_v2": 0,
        "v2_ge_v3": 0,
        "v3_ge_v4": 0,
    }
    full_chain_count = 0

    for _, gs in complete_rows:
        c1 = gs["seen"] >= gs["v1"]
        c2 = gs["v1"] >= gs["v2"]
        c3 = gs["v2"] >= gs["v3"]
        c4 = gs["v3"] >= gs["v4"]
        pairwise_counts["seen_ge_v1"] += int(c1)
        pairwise_counts["v1_ge_v2"] += int(c2)
        pairwise_counts["v2_ge_v3"] += int(c3)
        pairwise_counts["v3_ge_v4"] += int(c4)
        full_chain_count += int(c1 and c2 and c3 and c4)

    pairwise_rates = {k: (float(v) / float(n)) for k, v in pairwise_counts.items()}
    return {
        "num_samples": int(n),
        "full_chain_count": int(full_chain_count),
        "full_chain_rate": float(full_chain_count / n),
        "pairwise_counts": pairwise_counts,
        "pairwise_rates": pairwise_rates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RQ1 baseline eval with seen + 4 cumulative transform intensities + unseen. "
            "Produces score JSON results for downstream plotting."
        )
    )
    parser.add_argument("--fit", type=str, required=True)
    parser.add_argument("--validate", type=str, required=True)
    parser.add_argument("--test", type=str, required=True, help="Base RQ1 test.jsonl path.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument(
        "--method",
        choices=[
            "loss",
            "ppl",
            "zlib",
            "min_k",
            "confidence",
            "entropy",
            "msp",
            "margin",
        ],
        default="loss",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="",
        help=(
            "Comma-separated methods to run in one process. Supports: "
            "loss,ppl,zlib,min_k,confidence,entropy,msp,margin,all."
        ),
    )
    parser.add_argument("--level-dirs", type=str, required=True, help="Comma-separated RQ1 level dirs in order: v1,v2,v3,v4")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-k-ratio", type=float, default=0.2)
    parser.add_argument(
        "--disable-score-orientation",
        action="store_true",
        help="Disable score direction alignment by validation AUROC (keep z-scored raw direction).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    start = time.time()
    if args.methods and args.methods.strip():
        raw = [x.strip() for x in args.methods.split(",") if x.strip()]
    else:
        raw = [args.method]
    methods: List[str] = []
    seen_methods = set()
    supported_methods = [
        "loss",
        "ppl",
        "zlib",
        "min_k",
        "confidence",
        "entropy",
        "msp",
        "margin",
    ]
    supported_methods_set = set(supported_methods)
    for m in raw:
        expanded = FINAL_BASELINE_METHODS if m == "all" else [m]
        for mm in expanded:
            if mm not in supported_methods_set:
                raise ValueError(f"Unsupported method: {mm}")
            if mm not in seen_methods:
                seen_methods.add(mm)
                methods.append(mm)
    print(f"[rq1] methods={methods}")

    level_dirs = [x.strip() for x in args.level_dirs.split(",") if x.strip()]
    if len(level_dirs) != 4:
        raise ValueError("--level-dirs must contain exactly 4 directories (v1,v2,v3,v4).")
    for d in level_dirs:
        if not os.path.exists(os.path.join(d, "test.jsonl")):
            raise FileNotFoundError(f"Missing test.jsonl in level dir: {d}")

    fit_triplet = load_triplets_jsonl(args.fit)
    val_triplet = load_triplets_jsonl(args.validate)

    fit_flat, _ = flatten_triplets(
        fit_triplet,
        text_source="code_generation",
        include_label2_as_positive=False,
    )
    val_flat, _ = flatten_triplets(
        val_triplet,
        text_source="code_generation",
        include_label2_as_positive=False,
    )
    if not val_flat:
        raise ValueError("No evaluable validation samples found.")

    grouped_pairs = load_group_pairs_by_level(args.test, level_dirs)
    group_order = ["seen", *LEVELS, "unseen"]
    for g in group_order:
        if g not in grouped_pairs or not grouped_pairs[g]:
            raise ValueError(f"No samples for group={g}")

    print(f"[rq1] loading model: {args.model}")
    engine = ScoreEngine(
        model_name=args.model,
        model_dir=args.model_dir,
        device=args.device,
        max_length=args.max_length,
    )

    fit_scores_raw_map = compute_codegen_scores_by_methods(
        engine,
        methods,
        fit_flat,
        args.min_k_ratio,
        progress_prefix="[fit] ",
    )
    val_scores_raw_map = compute_codegen_scores_by_methods(
        engine,
        methods,
        val_flat,
        args.min_k_ratio,
        progress_prefix="[val] ",
    )

    group_scores_raw_map = compute_group_scores_by_methods(
        engine,
        methods,
        grouped_pairs,
        args.min_k_ratio,
    )
    val_labels = np.array([s.label for s in val_flat], dtype=np.int64)
    method_to_json = {}
    for method_name in methods:
        fit_scores_raw = fit_scores_raw_map[method_name]
        val_scores_raw = val_scores_raw_map[method_name]
        group_scores_raw = {k: v[method_name] for k, v in group_scores_raw_map.items()}

        fit_mean = float(np.mean(fit_scores_raw)) if fit_scores_raw.size > 0 else 0.0
        fit_std = float(np.std(fit_scores_raw)) if fit_scores_raw.size > 0 else 1.0
        if not np.isfinite(fit_std) or fit_std < 1e-8:
            fit_std = 1.0

        val_scores = (val_scores_raw - fit_mean) / fit_std
        group_scores = {k: (v - fit_mean) / fit_std for k, v in group_scores_raw.items()}
        group_scores_before_orientation = {k: v.copy() for k, v in group_scores.items()}

        if args.disable_score_orientation:
            sign = 1.0
            orientation = {
                "sign": 1.0,
                "raw_validation_auroc": None,
                "oriented_validation_auroc": None,
                "orientation_fallback_used": False,
                "num_valid_scores": int(np.sum(~np.isnan(val_scores))),
            }
        else:
            orientation = orient_scores_by_validation(
                val_scores=val_scores,
                val_labels=val_labels,
            )
            val_scores = orientation["val_scores"]
            sign = float(orientation["sign"])
            group_scores = {k: sign * v for k, v in group_scores.items()}

        clf_info = fit_binary_classifier_1d(
            val_scores,
            val_labels,
            n_splits=args.cv_folds,
            random_state=args.random_state,
        )

        group_pred = {}
        group_prob = {}
        for g in group_order:
            p, prob = predict_classifier_1d(clf_info, group_scores[g])
            group_pred[g] = p
            group_prob[g] = prob

        stats: Dict[str, Dict[str, float]] = {}
        details: List[GroupScoreSample] = []
        for g in group_order:
            seen_cnt = int(np.sum(group_pred[g] == 1))
            unseen_cnt = int(np.sum(group_pred[g] == 0))
            total = int(group_pred[g].shape[0])
            mean_score = float(np.mean(group_scores[g])) if total > 0 else 0.0
            stats[g] = {
                "seen_pred_1": seen_cnt,
                "unseen_pred_0": unseen_cnt,
                "total": total,
                "mean_score": mean_score,
                "mean_member_like_score": mean_score,
            }

            for (row_idx, _, _), s_raw_score, s_oriented, s_raw, pred, prob in zip(
                grouped_pairs[g],
                group_scores_raw[g],
                group_scores[g],
                group_scores_before_orientation[g],
                group_pred[g],
                group_prob[g],
            ):
                details.append(
                    GroupScoreSample(
                        row_index=int(row_idx),
                        group=g,
                        score_raw=float(s_raw_score),
                        score=float(s_oriented),
                        member_like_score=float(s_oriented),
                        score_before_orientation=float(s_raw),
                        pred_prob_seen=float(prob),
                        pred_seen=int(pred),
                    )
                )

        monotonic = compute_monotonic_consistency(grouped_pairs, group_scores)

        result = {
            "method": method_name,
            "model": args.model,
            "groups": group_order,
            "data_files": {
                "fit": args.fit,
                "validate": args.validate,
                "test_base": args.test,
                "level_dirs": level_dirs,
            },
            "min_k_ratio": float(args.min_k_ratio),
            "classifier": {
                "type": clf_info.get("type"),
                "coef": clf_info.get("coef"),
                "intercept": clf_info.get("intercept"),
                "majority_class": clf_info.get("majority_class"),
                "cv_folds": clf_info.get("cv_folds"),
                "selected_C": clf_info.get("selected_C"),
            },
            "score_preprocess": {
                "type": "zscore_from_fit",
                "fit_mean": fit_mean,
                "fit_std": fit_std,
                "fit_num_samples": int(len(fit_flat)),
                "orientation": {
                    "rule": "validation_auc_flip_if_below_0.5" if not args.disable_score_orientation else "disabled_raw_score",
                    "enabled": bool(not args.disable_score_orientation),
                    "sign": float(orientation["sign"]),
                    "raw_validation_auroc": (
                        float(orientation["raw_validation_auroc"])
                        if orientation["raw_validation_auroc"] is not None
                        else None
                    ),
                    "oriented_validation_auroc": (
                        float(orientation["oriented_validation_auroc"])
                        if orientation["oriented_validation_auroc"] is not None
                        else None
                    ),
                    "orientation_fallback_used": bool(orientation.get("orientation_fallback_used", False)),
                },
            },
            "group_stats": stats,
            "monotonic_consistency": monotonic,
            "timing": {
                "runtime_seconds": float(time.time() - start),
                "num_fit_candidates": int(len(fit_flat)),
                "num_validate_candidates": int(len(val_flat)),
                "num_group_candidates_per_group": {g: int(len(grouped_pairs[g])) for g in group_order},
            },
            "group_candidate_details": [
                {
                    "row_index": int(x.row_index),
                    "group": x.group,
                    "score_raw": float(x.score_raw),
                    "score": float(x.score),
                    "member_like_score": float(x.member_like_score),
                    "score_before_orientation": float(x.score_before_orientation),
                    "pred_prob_seen": float(x.pred_prob_seen),
                    "pred_seen": int(x.pred_seen),
                }
                for x in details
            ],
        }

        out_json = os.path.join(args.output_dir, f"{method_name}_trans_code_levels_results.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        method_to_json[method_name] = out_json

        print(f"[{method_name}] results saved to {out_json}")
        for g in group_order:
            c = stats[g]
            print(
                f"[{method_name}] {g}: mean_score={c['mean_score']:.6f}, "
                f"seen={int(c['seen_pred_1'])}, unseen={int(c['unseen_pred_0'])}, total={int(c['total'])}"
            )
        print(
            f"[{method_name}] monotonic seen>=v1>=v2>=v3>=v4: "
            f"{monotonic['full_chain_count']}/{monotonic['num_samples']} "
            f"({monotonic['full_chain_rate']:.6f})"
        )

    if len(methods) > 1:
        summary_path = os.path.join(args.output_dir, "all_methods_trans_code_levels_results.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "methods": methods,
                    "model": args.model,
                    "result_files": method_to_json,
                    "data_files": {
                        "fit": args.fit,
                        "validate": args.validate,
                        "test_base": args.test,
                        "level_dirs": level_dirs,
                    },
                    "min_k_ratio": float(args.min_k_ratio),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"[rq1] multi-method summary saved to {summary_path}")


if __name__ == "__main__":
    main()
