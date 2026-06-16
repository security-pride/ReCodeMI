#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_to_binary(label):
    if isinstance(label, (list, tuple)):
        if len(label) == 0:
            return None
        try:
            v = int(label[0])  # mem label
        except (TypeError, ValueError):
            return None
        return v if v in (0, 1) else None
    try:
        v = int(label)
    except (TypeError, ValueError):
        return None
    return v if v in (0, 1) else None


def label_to_binary_nonzero_positive(label):
    if isinstance(label, (list, tuple)):
        if len(label) == 0:
            return None
        try:
            v = int(label[0])  # mem label
        except (TypeError, ValueError):
            return None
        return 1 if v > 0 else 0
    try:
        v = int(label)
    except (TypeError, ValueError):
        return None
    return 1 if v > 0 else 0


def collect_layer_data(lat_data, axis):
    feat_key = "score_mem_X" if axis == "mem" else "score_syn_Y"
    by_layer = defaultdict(lambda: {"X": [], "y": []})

    for item in lat_data.get("results", []):
        layer = int(item["layer"])
        scores = item.get(feat_key, [])
        labels = item.get("correct_label_list", [])
        n = min(len(scores), len(labels))
        for i in range(n):
            y = label_to_binary(labels[i])
            if y is None:
                continue
            by_layer[layer]["X"].append([float(scores[i])])
            by_layer[layer]["y"].append(int(y))
    return by_layer


def fit_best_layer(val_data_by_layer, n_splits=5, random_state=42):
    per_layer_results = {}
    layer_models = {}
    best_layer = None
    best_score = -1.0

    for layer, data in val_data_by_layer.items():
        sample_count = len(data["y"])
        class_counts = {
            "0": int(sum(1 for v in data["y"] if v == 0)),
            "1": int(sum(1 for v in data["y"] if v == 1)),
        }
        report = {
            "status": "skipped",
            "num_samples": int(sample_count),
            "class_counts": class_counts,
        }

        if sample_count == 0:
            report["reason"] = "no_valid_binary_labels"
            per_layer_results[layer] = report
            continue

        X = np.array(data["X"])
        y = np.array(data["y"])

        if len(set(y)) < 2:
            report["reason"] = "single_class_only"
            per_layer_results[layer] = report
            continue

        min_class_count = min(class_counts["0"], class_counts["1"])
        effective_splits = min(n_splits, min_class_count)
        if effective_splits < 2:
            report["reason"] = "insufficient_samples_for_cv"
            per_layer_results[layer] = report
            continue

        skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
        fold_aurocs, fold_accs, fold_f1s = [], [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_va)
            y_prob = clf.predict_proba(X_va)[:, 1]

            fold_accs.append(accuracy_score(y_va, y_pred))
            fold_f1s.append(f1_score(y_va, y_pred, zero_division=0))
            if len(np.unique(y_va)) == 2:
                fold_aurocs.append(roc_auc_score(y_va, y_prob))
            else:
                fold_aurocs.append(float("nan"))

        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
        mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else 0.0
        valid_aurocs = [a for a in fold_aurocs if not np.isnan(a)]
        mean_auc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.0
        combined_score = 0.5 * mean_auc + 0.25 * mean_acc + 0.25 * mean_f1
        # combined_score = mean_acc

        final_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        final_clf.fit(X, y)
        layer_models[layer] = final_clf

        report.update(
            {
                "status": "fitted",
                "cv_folds": int(effective_splits),
                "cv_mean_auroc": mean_auc,
                "cv_mean_accuracy": mean_acc,
                "cv_mean_f1": mean_f1,
                "combined_score": float(combined_score),
                "coef": float(final_clf.coef_[0][0]),
                "intercept": float(final_clf.intercept_[0]),
            }
        )
        per_layer_results[layer] = report

        if combined_score > best_score:
            best_score = combined_score
            best_layer = layer

    if best_layer is None:
        return {"error": "no_valid_layer_for_classifier", "per_layer_results": per_layer_results}, None

    best_report = per_layer_results[best_layer]
    return {
        "best_layer": int(best_layer),
        # "selection_metric": "0.5*AUROC + 0.25*Accuracy + 0.25*F1",
        "selection_metric": "AUROC",
        "validation_combined_score": float(best_score),
        "validation_auroc": float(best_report["cv_mean_auroc"]),
        "validation_accuracy": float(best_report["cv_mean_accuracy"]),
        "validation_f1": float(best_report["cv_mean_f1"]),
        "coef": float(best_report["coef"]),
        "intercept": float(best_report["intercept"]),
        "per_layer_results": per_layer_results,
    }, layer_models[best_layer]


def apply_on_test(test_data, axis, best_layer, clf, include_label2_as_positive=False):
    feat_key = "score_mem_X" if axis == "mem" else "score_syn_Y"
    label_fn = label_to_binary_nonzero_positive if include_label2_as_positive else label_to_binary

    y_true_all, y_pred_all, y_prob_all = [], [], []
    total_questions, correct_rank1 = 0, 0

    for item in test_data.get("results", []):
        if int(item["layer"]) != int(best_layer):
            continue

        scores = item.get(feat_key, [])
        labels = item.get("correct_label_list", [])
        n = min(len(scores), len(labels))
        if n == 0:
            continue

        X_q, y_q, positive_indices = [], [], []
        for i in range(n):
            y = label_fn(labels[i])
            if y is None:
                continue
            X_q.append([float(scores[i])])
            y_q.append(int(y))
            if y == 1:
                positive_indices.append(i)

        if not X_q:
            continue

        X_q = np.array(X_q)
        y_pred_q = clf.predict(X_q)
        y_prob_q = clf.predict_proba(X_q)[:, 1]

        y_true_all.extend(y_q)
        y_pred_all.extend(y_pred_q.tolist())
        y_prob_all.extend(y_prob_q.tolist())

        conf = clf.decision_function(X_q)
        ranked_local = list(np.argsort(conf)[::-1])
        pred_top1 = ranked_local[0] if ranked_local else None
        if pred_top1 is not None and pred_top1 in positive_indices:
            correct_rank1 += 1
        total_questions += 1

    if len(y_true_all) == 0:
        return {"error": "no_test_samples_after_filtering"}

    metrics = {
        "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
        "precision": float(precision_score(y_true_all, y_pred_all, zero_division=0)),
        "recall": float(recall_score(y_true_all, y_pred_all, zero_division=0)),
        "f1_score": float(f1_score(y_true_all, y_pred_all, zero_division=0)),
        "auroc": float(roc_auc_score(y_true_all, y_prob_all)) if len(np.unique(y_true_all)) == 2 else None,
        "ranking_accuracy_top1": float(correct_rank1 / total_questions) if total_questions > 0 else 0.0,
        "num_questions_used_for_ranking": int(total_questions),
        "num_candidate_samples_for_cls_metrics": int(len(y_true_all)),
    }
    return metrics


def run_axis(axis, val_data, test_data, include_label2_as_positive=False):
    val_by_layer = collect_layer_data(val_data, axis)
    fit_info, best_clf = fit_best_layer(val_by_layer)
    if "error" in fit_info:
        return fit_info

    test_metrics = apply_on_test(
        test_data=test_data,
        axis=axis,
        best_layer=fit_info["best_layer"],
        clf=best_clf,
        include_label2_as_positive=include_label2_as_positive,
    )
    fit_info["test_metrics"] = test_metrics
    fit_info["test_label_policy"] = (
        "label>0_as_positive" if include_label2_as_positive else "label_in_{0,1}_only"
    )
    return fit_info


def _rankdata_average(values):
    arr = np.asarray(values, dtype=float)
    n = arr.shape[0]
    if n == 0:
        return arr
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def _spearman(x, y):
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    mask = np.isfinite(left) & np.isfinite(right)
    left = left[mask]
    right = right[mask]
    if left.size < 2:
        return None

    left_rank = _rankdata_average(left)
    right_rank = _rankdata_average(right)
    left_rank = left_rank - np.mean(left_rank)
    right_rank = right_rank - np.mean(right_rank)
    denominator = float(np.linalg.norm(left_rank) * np.linalg.norm(right_rank))
    if denominator < 1e-12:
        return None
    return float(np.dot(left_rank, right_rank) / denominator)


def compute_mem_syn_spearman(lat_data):
    by_layer = defaultdict(lambda: {"x": [], "y": []})

    for item in lat_data.get("results", []):
        layer = int(item.get("layer"))
        score_mem = item.get("score_mem_X", [])
        score_syn = item.get("score_syn_Y", [])
        n = min(len(score_mem), len(score_syn))
        for i in range(n):
            try:
                x = float(score_mem[i])
                y = float(score_syn[i])
            except (TypeError, ValueError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                by_layer[layer]["x"].append(x)
                by_layer[layer]["y"].append(y)

    if not by_layer:
        return {"error": "no_valid_points_for_correlation", "per_layer": {}}

    per_layer = {}
    for layer in sorted(by_layer.keys()):
        xs = by_layer[layer]["x"]
        ys = by_layer[layer]["y"]
        num_points = len(xs)
        if num_points < 2:
            per_layer[str(layer)] = {
                "num_points": int(num_points),
                "spearman": None,
                "error": "insufficient_points_for_correlation",
            }
            continue

        x = np.array(xs, dtype=float)
        y = np.array(ys, dtype=float)
        per_layer[str(layer)] = {
            "num_points": int(num_points),
            "spearman": _spearman(x, y),
            "error": "",
        }

    return {
        "num_layers": int(len(per_layer)),
        "per_layer": per_layer,
    }


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "best_layer",
                "val_auroc",
                "val_accuracy",
                "val_f1",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_auroc",
                "test_ranking_top1",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_spearman_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "layer",
                "num_points",
                "spearman",
                "error",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="RQ4: evaluate member inference with mem-only / syn-only axes from LAT results."
    )
    parser.add_argument(
        "--test-lat",
        default="results/bigcode/starcoder2-7b/stack2/RQ4/full/LAT.json",
        help="Path to LAT.json (test results).",
    )
    parser.add_argument(
        "--val-lat",
        default=None,
        help="Path to LAT_val.json. If omitted, use sibling LAT_val.json of --test-lat.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <LAT dir>/RQ4",
    )
    parser.add_argument(
        "--include-label2-as-positive",
        action="store_true",
        help="Treat any label > 0 as positive during test metric calculation.",
    )
    args = parser.parse_args()

    test_lat_path = args.test_lat
    val_lat_path = args.val_lat
    if val_lat_path is None:
        val_lat_path = os.path.join(os.path.dirname(test_lat_path), "LAT_val.json")

    test_data = load_json(test_lat_path)
    if os.path.exists(val_lat_path):
        val_data = load_json(val_lat_path)
        val_source = val_lat_path
    else:
        raise FileNotFoundError(f"Validation LAT file not found: {val_lat_path}")

    out_dir = args.out_dir or os.path.join(os.path.dirname(test_lat_path), "RQ4")
    os.makedirs(out_dir, exist_ok=True)

    mem_res = run_axis("mem", val_data, test_data, include_label2_as_positive=args.include_label2_as_positive)
    syn_res = run_axis("syn", val_data, test_data, include_label2_as_positive=args.include_label2_as_positive)

    full_two_d = test_data.get("linear_classifier", {})
    full_two_d_metrics = full_two_d.get("test_metrics", {}) if isinstance(full_two_d, dict) else {}

    full_spearman = compute_mem_syn_spearman(test_data)

    summary = {
        "inputs": {
            "test_lat": test_lat_path,
            "val_lat_used_for_layer_selection": val_source if val_source else test_lat_path,
            "used_test_as_val_fallback": False,
        },
        "label_policy": "label>0_as_positive" if args.include_label2_as_positive else "label_in_{0,1}_only",
        "data_reading_note": (
            "choices order is shuffled, but alignment is preserved by index. "
            "For each question-item, candidate i uses: "
            "score_mem_X[i], score_syn_Y[i], correct_label_list[i]. "
            "Binary class y is derived from correct_label_list[i][0] (mem label), "
            "following the same rule as evaluators_1.py."
        ),
        "results": {
            "mem_only": mem_res,
            "syn_only": syn_res,
            "full_recodemi": full_two_d,
        },
        "additional_metrics": {
            "mem_syn_spearman": {"full_recodemi": full_spearman},
        },
    }

    out_json = os.path.join(out_dir, "single_axis_eval_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    rows = []
    for name, res in [("mem_only", mem_res), ("syn_only", syn_res)]:
        if "error" in res:
            rows.append(
                {
                    "setting": name,
                    "best_layer": "ERROR",
                    "val_auroc": "",
                    "val_accuracy": "",
                    "val_f1": "",
                    "test_accuracy": "",
                    "test_precision": "",
                    "test_recall": "",
                    "test_f1": "",
                    "test_auroc": "",
                    "test_ranking_top1": "",
                }
            )
            continue
        tm = res.get("test_metrics", {})
        rows.append(
            {
                "setting": name,
                "best_layer": res.get("best_layer"),
                "val_auroc": res.get("validation_auroc"),
                "val_accuracy": res.get("validation_accuracy"),
                "val_f1": res.get("validation_f1"),
                "test_accuracy": tm.get("accuracy"),
                "test_precision": tm.get("precision"),
                "test_recall": tm.get("recall"),
                "test_f1": tm.get("f1_score"),
                "test_auroc": tm.get("auroc"),
                "test_ranking_top1": tm.get("ranking_accuracy_top1"),
            }
        )

    rows.append(
        {
            "setting": "full_recodemi",
            "best_layer": full_two_d.get("best_layer", "") if isinstance(full_two_d, dict) else "",
            "val_auroc": full_two_d.get("validation_auroc", "") if isinstance(full_two_d, dict) else "",
            "val_accuracy": full_two_d.get("validation_accuracy", "") if isinstance(full_two_d, dict) else "",
            "val_f1": full_two_d.get("validation_f1", "") if isinstance(full_two_d, dict) else "",
            "test_accuracy": full_two_d_metrics.get("accuracy", ""),
            "test_precision": full_two_d_metrics.get("precision", ""),
            "test_recall": full_two_d_metrics.get("recall", ""),
            "test_f1": full_two_d_metrics.get("f1_score", ""),
            "test_auroc": full_two_d_metrics.get("auroc", ""),
            "test_ranking_top1": full_two_d_metrics.get("ranking_accuracy_2d", ""),
        }
    )

    out_csv = os.path.join(out_dir, "single_axis_eval_summary.csv")
    write_csv(out_csv, rows)

    spearman_rows = []
    for setting, result in [("full_recodemi", full_spearman)]:
        if not isinstance(result, dict):
            spearman_rows.append(
                {
                    "setting": setting,
                    "layer": "",
                    "num_points": "",
                    "spearman": "",
                    "error": "missing_data",
                }
            )
            continue
        per_layer = result.get("per_layer", {})
        if not per_layer:
            spearman_rows.append(
                {
                    "setting": setting,
                    "layer": "",
                    "num_points": "",
                    "spearman": "",
                    "error": result.get("error", "no_per_layer_data"),
                }
            )
            continue
        for layer in sorted(per_layer.keys(), key=lambda x: int(x)):
            item = per_layer[layer]
            spearman_rows.append(
                {
                    "setting": setting,
                    "layer": layer,
                    "num_points": item.get("num_points", ""),
                    "spearman": item.get("spearman", ""),
                    "error": item.get("error", ""),
                }
            )

    out_spearman_csv = os.path.join(out_dir, "mem_syn_spearman_summary.csv")
    write_spearman_csv(out_spearman_csv, spearman_rows)

    print(f"[Done] JSON: {out_json}")
    print(f"[Done] CSV : {out_csv}")
    print(f"[Done] Spearman CSV: {out_spearman_csv}")
    print("Reading alignment rule:")
    print("  candidate i <-> score_mem_X[i], score_syn_Y[i], correct_label_list[i]")
    print("  y for MIA binary metrics comes from correct_label_list[i][0] (mem label).")


if __name__ == "__main__":
    main()
