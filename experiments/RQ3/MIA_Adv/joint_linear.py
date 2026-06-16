#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_feature(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return np.asarray(data["X"]), np.asarray(data["y"])


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["_line_index"] = line_index
            records.append(record)
    return records


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def compute_signal_records(
    split: str,
    victim_path: Path,
    inference_path: Path,
    group_size: int,
) -> list[dict]:
    import feature_builder

    victims = [record for record in read_jsonl(victim_path) if record["_line_index"] % group_size == 0]
    predictions = [line for index, line in enumerate(read_lines(inference_path)) if index % group_size == 0]
    if len(victims) != len(predictions):
        raise ValueError(
            f"Original record/prediction count mismatch for {split}: "
            f"{len(victims)} records vs {len(predictions)} predictions"
        )

    records = []
    for group_index, (victim, prediction) in enumerate(zip(victims, predictions)):
        ground_truth = victim.get("gt", "")
        similarity = feature_builder.compute_similarity(ground_truth, prediction)
        perplexity = feature_builder.calculate_perplexity(
            prediction,
            feature_builder.codegpt_model,
            feature_builder.codegpt_tokenizer,
        )
        records.append(
            {
                "split": split,
                "id": victim.get("id"),
                "group_idx": group_index,
                "line_index": int(victim["_line_index"]),
                "is_original": True,
                "similarity_signal": float(similarity),
                "perplexity_signal": float(perplexity),
            }
        )
    return records


def build_signals(args: argparse.Namespace) -> None:
    import feature_builder

    feature_builder.ensure_feature_models_loaded()
    records = compute_signal_records(
        "train",
        args.train_victim,
        args.train_infer,
        args.group_size,
    )
    records.extend(
        compute_signal_records(
            "test",
            args.test_victim,
            args.test_infer,
            args.group_size,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {args.output} ({len(records)} records)")


def load_signals(path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    records = read_jsonl(path)
    member = [record for record in records if record.get("split") == "train"]
    non_member = [record for record in records if record.get("split") == "test"]
    key = lambda item: int(item["group_idx"])
    ordered = sorted(member, key=key) + sorted(non_member, key=key)
    signals = np.asarray(
        [[record["similarity_signal"], record["perplexity_signal"]] for record in ordered],
        dtype=np.float64,
    )
    labels = np.asarray([1] * len(member) + [0] * len(non_member), dtype=np.int64)
    return signals, labels, ordered


def load_shadow_lat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    member = []
    non_member = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record.get("is_original", False):
                continue
            if record.get("mem_score") is None or record.get("syn_score") is None:
                continue
            target = member if record.get("split") == "train" else non_member
            target.append(record)

    key = lambda item: int(item.get("group_idx", item.get("line_index", 0)))
    ordered = sorted(member, key=key) + sorted(non_member, key=key)
    scores = np.asarray(
        [[record["mem_score"], record["syn_score"]] for record in ordered],
        dtype=np.float64,
    )
    labels = np.asarray([1] * len(member) + [0] * len(non_member), dtype=np.int64)
    return scores, labels


def require_aligned(name: str, expected: np.ndarray, actual: np.ndarray) -> None:
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise ValueError(f"Label mismatch for {name}: {expected.shape} vs {actual.shape}")


def impute_from_shadow(shadow_x: np.ndarray, target_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = np.nanmean(shadow_x, axis=0)
    means = np.where(np.isnan(means), 0.0, means)
    return np.where(np.isnan(shadow_x), means, shadow_x), np.where(np.isnan(target_x), means, target_x)


def balanced_split_indices(
    labels: np.ndarray,
    n_samples_per_class: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)
    positive = np.where(labels == 1)[0]
    negative = np.where(labels == 0)[0]
    sample_count = min(n_samples_per_class, len(positive), len(negative))
    if sample_count < 2:
        raise ValueError("Joint linear mapping requires at least two samples per class.")

    positive = rng.choice(positive, sample_count, replace=False)
    negative = rng.choice(negative, sample_count, replace=False)
    train_count = sample_count // 2
    positive = positive[rng.permutation(sample_count)]
    negative = negative[rng.permutation(sample_count)]
    train_idx = np.concatenate([positive[:train_count], negative[:train_count]])
    test_idx = np.concatenate([positive[train_count:], negative[train_count:]])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def fit_append(args: argparse.Namespace) -> None:
    shadow_signals, shadow_y, _ = load_signals(args.shadow_signals)
    target_signals, target_signal_y, target_records = load_signals(args.target_signals)
    target_x, target_y = load_feature(args.target_feature)
    shadow_lat, shadow_lat_y = load_shadow_lat(args.shadow_lat)
    require_aligned("shadow signals/LAT", shadow_y, shadow_lat_y.astype(shadow_y.dtype))
    require_aligned("target signals/feature", target_signal_y, target_y.astype(target_signal_y.dtype))

    shadow_signals, target_signals = impute_from_shadow(shadow_signals, target_signals)
    train_idx, test_idx = balanced_split_indices(
        shadow_y,
        args.n_samples_per_class,
        args.random_state,
    )

    mapper = make_pipeline(StandardScaler(), LinearRegression())
    mapper.fit(shadow_signals[train_idx], shadow_lat[train_idx])
    pseudo_lat = mapper.predict(target_signals).astype(np.float32)
    if args.mapper_oof and len(train_idx) >= 2:
        n_splits = min(args.mapper_folds, len(train_idx))
        if n_splits >= 2:
            folds = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=args.random_state,
            )
            for fit_positions, predict_positions in folds.split(
                shadow_signals[train_idx],
                shadow_y[train_idx],
            ):
                fit_indices = train_idx[fit_positions]
                predict_indices = train_idx[predict_positions]
                fold_mapper = make_pipeline(StandardScaler(), LinearRegression())
                fold_mapper.fit(shadow_signals[fit_indices], shadow_lat[fit_indices])
                pseudo_lat[predict_indices] = fold_mapper.predict(
                    target_signals[predict_indices]
                ).astype(np.float32)
    joint_x = np.concatenate([target_x.astype(np.float32), pseudo_lat], axis=1)

    args.mapper_out.parent.mkdir(parents=True, exist_ok=True)
    args.feature_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(mapper, args.mapper_out)
    np.savez(args.feature_out, X=joint_x, y=target_y)

    args.pseudo_out.parent.mkdir(parents=True, exist_ok=True)
    with args.pseudo_out.open("w", encoding="utf-8") as handle:
        for index, (record, label, scores) in enumerate(zip(target_records, target_y, pseudo_lat)):
            handle.write(
                json.dumps(
                    {
                        "index": index,
                        "split": record["split"],
                        "id": record.get("id"),
                        "group_idx": int(record["group_idx"]),
                        "line_index": int(record["line_index"]),
                        "label": int(label),
                        "pseudo_method": "joint_linear",
                        "pseudo_mem_score": float(scores[0]),
                        "pseudo_syn_score": float(scores[1]),
                        "similarity_signal": float(record["similarity_signal"]),
                        "perplexity_signal": float(record["perplexity_signal"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    diagnostics = {
        "method": "joint_linear",
        "mapper_input": "signals",
        "shadow_signal_shape": list(shadow_signals.shape),
        "target_signal_shape": list(target_signals.shape),
        "signal_columns": ["similarity_signal", "perplexity_signal"],
        "target_columns": ["mem_score", "syn_score"],
        "target_feature_shape": list(target_x.shape),
        "joint_feature_shape": list(joint_x.shape),
        "mapper_train_rows": int(len(train_idx)),
        "mapper_holdout_rows": int(len(test_idx)),
        "mapper_oof": bool(args.mapper_oof),
        "mapper_folds": int(args.mapper_folds),
        "random_state": int(args.random_state),
    }
    args.diagnostics_out.parent.mkdir(parents=True, exist_ok=True)
    args.diagnostics_out.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


def read_metrics(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None:
        raise ValueError(f"No metric row found in {path}")
    return row


def summarize(args: argparse.Namespace) -> None:
    rows = []
    for method, path in (("baseline", args.baseline_metrics), ("joint_linear", args.joint_metrics)):
        metrics = read_metrics(path)
        rows.append(
            {
                "method": method,
                "model": args.model,
                "auroc": metrics.get("auroc", metrics.get("auc")),
                "acc": metrics.get("accuracy"),
                "f1": metrics.get("f1_score"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "model", "auroc", "acc", "f1"])
        writer.writeheader()
        writer.writerows(rows)

    if args.remove_inputs:
        args.baseline_metrics.unlink(missing_ok=True)
        args.joint_metrics.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint linear ReCodeMIA utilities for AdvPrompt-MIA.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    signals = subparsers.add_parser("build-signals")
    signals.add_argument("--train-victim", type=Path, required=True)
    signals.add_argument("--test-victim", type=Path, required=True)
    signals.add_argument("--train-infer", type=Path, required=True)
    signals.add_argument("--test-infer", type=Path, required=True)
    signals.add_argument("--group-size", type=int, default=12)
    signals.add_argument("--output", type=Path, required=True)

    fit = subparsers.add_parser("fit-append")
    fit.add_argument("--shadow-signals", type=Path, required=True)
    fit.add_argument("--target-signals", type=Path, required=True)
    fit.add_argument("--target-feature", type=Path, required=True)
    fit.add_argument("--shadow-lat", type=Path, required=True)
    fit.add_argument("--mapper-out", type=Path, required=True)
    fit.add_argument("--feature-out", type=Path, required=True)
    fit.add_argument("--pseudo-out", type=Path, required=True)
    fit.add_argument("--diagnostics-out", type=Path, required=True)
    fit.add_argument("--mapper-oof", action="store_true")
    fit.add_argument("--mapper-folds", type=int, default=5)
    fit.add_argument("--random-state", type=int, default=676269283)
    fit.add_argument("--n-samples-per-class", type=int, default=2000)

    summary = subparsers.add_parser("summarize")
    summary.add_argument("--baseline-metrics", type=Path, required=True)
    summary.add_argument("--joint-metrics", type=Path, required=True)
    summary.add_argument("--model", required=True)
    summary.add_argument("--output", type=Path, required=True)
    summary.add_argument("--remove-inputs", action="store_true")

    args = parser.parse_args()
    if args.command == "build-signals":
        build_signals(args)
    elif args.command == "fit-append":
        fit_append(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
