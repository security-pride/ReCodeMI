#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler


LOGGER = logging.getLogger(__name__)


class AuditDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: list[int]):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]


class AuditModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, labels=None):
        logits = self.layers(inputs)
        if labels is None:
            return logits
        return logits, self.loss(logits, labels)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scalar_label(value) -> int:
    return int(value.item()) if isinstance(value, torch.Tensor) else int(value)


def rank_to_histogram(ranks: torch.Tensor, bins: int, output_size: int) -> np.ndarray:
    ranks = ranks.detach().cpu().float()
    if ranks.numel() == 0:
        return np.zeros(bins, dtype=np.float32)
    values = torch.histc(ranks, bins - 1, min=0, max=output_size).cpu().numpy()
    overflow = float(ranks.numel() - values.sum())
    return np.concatenate([values, [overflow]]).astype(np.float32) / float(ranks.numel())


def load_mem_syn(path: Path) -> list[np.ndarray]:
    records = torch.load(path)
    return [
        np.asarray([float(record["mem_score"]), float(record["syn_score"])], dtype=np.float32)
        for record in records
    ]


def load_shadow_data(args: argparse.Namespace) -> dict:
    train_ranks: list[torch.Tensor] = []
    train_labels: list[int] = []
    train_mem_syn: list[np.ndarray] = []
    holdout_ranks: list[torch.Tensor] = []
    holdout_labels: list[int] = []

    for index in range(args.number_of_shadow_model):
        rank_dir = (
            args.data_dir
            / f"shadow_{index}"
            / args.shadow_model_type
            / args.shadow_ranks_subdir
        )
        for label_name in ("member", "non_member"):
            rank_records = torch.load(rank_dir / f"{label_name}_ranks.pth")
            mem_syn_records = load_mem_syn(rank_dir / f"{label_name}_mem_syn.pth")
            if len(rank_records) != len(mem_syn_records):
                raise ValueError(f"Rank/mem-syn length mismatch in {rank_dir} for {label_name}")

            split = int(len(rank_records) * 0.8)
            train_ranks.extend([record["rank"] + 1 for record in rank_records[:split]])
            train_labels.extend([scalar_label(record["label"]) for record in rank_records[:split]])
            train_mem_syn.extend(mem_syn_records[:split])
            holdout_ranks.extend([record["rank"] + 1 for record in rank_records[split:]])
            holdout_labels.extend([scalar_label(record["label"]) for record in rank_records[split:]])

    return {
        "train_ranks": train_ranks,
        "train_labels": train_labels,
        "train_mem_syn": np.asarray(train_mem_syn, dtype=np.float32),
        "holdout_ranks": holdout_ranks,
        "holdout_labels": holdout_labels,
    }


def build_frequency(records: list[dict]) -> dict[int, int]:
    counts = Counter()
    for record in records:
        ground_truth = record.get("gt")
        if isinstance(ground_truth, torch.Tensor):
            counts.update(ground_truth.cpu().tolist())
        elif ground_truth is not None:
            counts.update(list(ground_truth))
    return {token: rank for rank, (token, _) in enumerate(counts.most_common())}


def load_frequency(data_dir: Path, model_type: str, label: str, records: list[dict]) -> dict[int, int]:
    path = data_dir / "target" / model_type / f"{label}_word_frequency.txt"
    if path.is_file():
        values = json.loads(path.read_text(encoding="utf-8"))
        return {int(item[0]): rank for rank, item in enumerate(values)}
    return build_frequency(records)


def sample_ranks(
    ranks: torch.Tensor,
    ground_truth: torch.Tensor,
    number: int,
    strategy: str,
    frequency: dict[int, int] | None,
) -> torch.Tensor:
    if number <= 0 or number >= len(ranks):
        return ranks
    if strategy == "random":
        indices = torch.randperm(len(ranks))[:number]
        return ranks[indices]

    scored = []
    for index, token in enumerate(ground_truth):
        token_id = int(token.item())
        if frequency is not None and token_id in frequency:
            scored.append((frequency[token_id], index))
    selected = [index for _, index in sorted(scored, reverse=True)[:number]]
    if not selected:
        return ranks[:number]
    return ranks[torch.tensor(selected, dtype=torch.long)]


def load_target_data(args: argparse.Namespace) -> tuple[list[torch.Tensor], list[int]]:
    rank_dir = args.data_dir / "target" / args.target_model_type / args.target_ranks_subdir
    ranks: list[torch.Tensor] = []
    labels: list[int] = []
    for label_name in ("member", "non_member"):
        records = torch.load(rank_dir / f"{label_name}_ranks.pth")
        frequency = None
        if args.number_of_queries > 0 and args.sampling_strategy == "frequency":
            frequency = load_frequency(args.data_dir, args.target_model_type, label_name, records)
        for record in records:
            ranks.append(
                sample_ranks(
                    record["rank"] + 1,
                    record["gt"],
                    args.number_of_queries,
                    args.sampling_strategy,
                    frequency,
                )
            )
            labels.append(scalar_label(record["label"]))
    return ranks, labels


def ranks_to_features(ranks: list[torch.Tensor], bins: int, output_size: int) -> np.ndarray:
    return np.asarray([rank_to_histogram(item, bins, output_size) for item in ranks], dtype=np.float32)


def fit_joint_linear(
    features: np.ndarray,
    mem_syn: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, object]:
    predictions = np.zeros_like(mem_syn, dtype=np.float32)
    split_count = min(folds, len(features))
    if split_count >= 2:
        splitter = KFold(n_splits=split_count, shuffle=True, random_state=seed)
        for fit_indices, predict_indices in splitter.split(features):
            mapper = make_pipeline(StandardScaler(), LinearRegression())
            mapper.fit(features[fit_indices], mem_syn[fit_indices])
            predictions[predict_indices] = mapper.predict(features[predict_indices]).astype(np.float32)
    else:
        mapper = make_pipeline(StandardScaler(), LinearRegression())
        mapper.fit(features, mem_syn)
        predictions = mapper.predict(features).astype(np.float32)

    final_mapper = make_pipeline(StandardScaler(), LinearRegression())
    final_mapper.fit(features, mem_syn)
    return predictions, final_mapper


def train_audit_model(args: argparse.Namespace, features: np.ndarray, labels: list[int], name: str):
    dataset = AuditDataset(features, labels)
    loader = DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=args.batch_size)
    model = AuditModel(features.shape[1], args.hidden_states).to(args.device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epoch):
        model.train()
        for step, (inputs, batch_labels) in enumerate(loader):
            inputs = inputs.to(args.device)
            batch_labels = batch_labels.to(args.device)
            _, loss = model(inputs, labels=batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.logging_steps == 0:
                LOGGER.info("%s epoch=%d step=%d loss=%.6f", name, epoch, step, loss.item())

    path = args.output_dir / f"{name}.pth"
    torch.save(model.state_dict(), path)
    return model


def predict_probabilities(args: argparse.Namespace, model, features: np.ndarray, labels: list[int]):
    dataset = AuditDataset(features, labels)
    loader = DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=args.batch_size)
    true_labels = []
    probabilities = []
    model.eval()
    with torch.no_grad():
        for inputs, batch_labels in loader:
            logits = model(inputs.to(args.device))
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            true_labels.extend(batch_labels.numpy().tolist())
            probabilities.extend(probs.tolist())
    return np.asarray(true_labels, dtype=np.int64), np.asarray(probabilities, dtype=np.float64)


def choose_threshold(labels: np.ndarray, probabilities: np.ndarray, metric: str) -> float:
    best_threshold = 0.5
    best_score = -float("inf")
    for threshold in np.unique(probabilities):
        predictions = (probabilities >= threshold).astype(np.int64)
        if metric == "f1":
            score = f1_score(labels, predictions, zero_division=0)
        elif metric == "accuracy":
            score = accuracy_score(labels, predictions)
        else:
            positive = labels == 1
            negative = labels == 0
            tpr = float(np.mean(predictions[positive] == 1)) if np.any(positive) else 0.0
            fpr = float(np.mean(predictions[negative] == 1)) if np.any(negative) else 0.0
            score = tpr - fpr
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def evaluate(args, model, features, labels, calibration_features, calibration_labels) -> dict:
    if args.fixed_threshold is None:
        holdout_labels, holdout_probs = predict_probabilities(
            args, model, calibration_features, calibration_labels
        )
        threshold = choose_threshold(holdout_labels, holdout_probs, args.threshold_metric)
    else:
        threshold = float(args.fixed_threshold)

    true_labels, probabilities = predict_probabilities(args, model, features, labels)
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "f1_score": float(f1_score(true_labels, predictions, zero_division=0)),
        "auroc": float(roc_auc_score(true_labels, probabilities)),
        "precision": float(precision_score(true_labels, predictions, zero_division=0)),
        "recall": float(recall_score(true_labels, predictions, zero_division=0)),
        "threshold": threshold,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NaturalCC baseline versus joint linear ReCodeMIA.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--hidden-states", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--logging-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--number-of-shadow-model", type=int, default=1)
    parser.add_argument("--shadow-model-type", default="code-gpt")
    parser.add_argument("--target-model-type", default="code-gpt")
    parser.add_argument("--output-size", type=int, default=98304)
    parser.add_argument("--number-of-queries", type=int, default=0)
    parser.add_argument("--sampling-strategy", choices=["random", "frequency"], default="frequency")
    parser.add_argument("--shadow-ranks-subdir", default="ranks")
    parser.add_argument("--target-ranks-subdir", default="ranks_starcoder2")
    parser.add_argument("--mapper-folds", type=int, default=5)
    parser.add_argument("--threshold-metric", choices=["youden", "f1", "accuracy"], default="youden")
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    shadow = load_shadow_data(args)
    shadow_train = ranks_to_features(shadow["train_ranks"], args.bins, args.output_size)
    shadow_holdout = ranks_to_features(shadow["holdout_ranks"], args.bins, args.output_size)
    shadow_joint, mapper = fit_joint_linear(
        shadow_train, shadow["train_mem_syn"], args.mapper_folds, args.seed
    )
    joblib.dump(mapper, args.output_dir / "rank_to_mem_syn_joint_linear.joblib")

    baseline_model = train_audit_model(args, shadow_train, shadow["train_labels"], "baseline")
    joint_model = train_audit_model(args, shadow_joint, shadow["train_labels"], "joint_linear")

    target_ranks, target_labels = load_target_data(args)
    target_features = ranks_to_features(target_ranks, args.bins, args.output_size)
    target_joint = mapper.predict(target_features).astype(np.float32)
    holdout_joint = mapper.predict(shadow_holdout).astype(np.float32)

    baseline_metrics = evaluate(
        args,
        baseline_model,
        target_features,
        target_labels,
        shadow_holdout,
        shadow["holdout_labels"],
    )
    joint_metrics = evaluate(
        args,
        joint_model,
        target_joint,
        target_labels,
        holdout_joint,
        shadow["holdout_labels"],
    )

    rows = []
    for method, metrics in (("baseline", baseline_metrics), ("joint_linear", joint_metrics)):
        rows.append(
            {
                "method": method,
                "model": args.model,
                "auroc": metrics["auroc"],
                "acc": metrics["accuracy"],
                "f1": metrics["f1_score"],
            }
        )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "model", "auroc", "acc", "f1"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
