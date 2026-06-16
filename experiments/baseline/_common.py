#!/usr/bin/env python3
import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

BASELINE_DIR = os.path.dirname(os.path.abspath(__file__))
LAT_DIR = os.path.dirname(BASELINE_DIR)
if LAT_DIR not in sys.path:
    sys.path.insert(0, LAT_DIR)
PROJECT_DIR = os.path.dirname(LAT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Avoid local script name collision with stdlib zlib (baseline/zlib.py).
if BASELINE_DIR in sys.path:
    sys.path.remove(BASELINE_DIR)
    _stdlib_zlib = importlib.import_module("zlib")
    sys.path.insert(0, BASELINE_DIR)
else:
    _stdlib_zlib = importlib.import_module("zlib")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from loaders import load_triplets_jsonl
from model import Model as LATModel

SUPPORTED_METHODS = (
    "loss",
    "ppl",
    "zlib",
    "min_k",
    "confidence",
    "entropy",
    "msp",
    "margin",
)

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


@dataclass
class FlatSample:
    question_id: int
    choice_index: int
    text: str
    label: int
    prompt: Optional[str] = None
    gt: Optional[str] = None


def _label_to_binary(label, include_label2_as_positive: bool = False) -> Optional[int]:
    if isinstance(label, (list, tuple, np.ndarray)):
        if len(label) == 0:
            return None
        label = label[0]

    try:
        v = int(label)
    except (TypeError, ValueError):
        return None

    if include_label2_as_positive:
        return 1 if v > 0 else 0

    return v if v in (0, 1) else None


def _pick_targets(triplet: dict, text_source: str) -> List[List[str]]:
    if text_source == "mem":
        return triplet["targets_mem"]
    if text_source == "syn":
        return triplet["targets"]
    raise ValueError(f"Unsupported text_source={text_source}")


def _split_code_middle_line(code: str) -> Dict[str, str]:
    if not isinstance(code, str):
        code = "" if code is None else str(code)
    lines = code.splitlines(keepends=True)
    if not lines:
        return {"input": "", "gt": ""}
    mid_line_idx = len(lines) // 2
    mid_line = lines[mid_line_idx]
    mid_col = len(mid_line) // 2
    input_part = "".join(lines[:mid_line_idx]) + mid_line[:mid_col]
    gt_part = mid_line[mid_col:] + "".join(lines[mid_line_idx + 1 :])
    return {"input": input_part, "gt": gt_part}


def _get_codegen_pairs_for_question(triplet: dict, qid: int) -> List[Dict[str, str]]:
    cg_all = triplet.get("code_generation")
    if isinstance(cg_all, list) and qid < len(cg_all) and isinstance(cg_all[qid], list):
        pairs = []
        for item in cg_all[qid]:
            if isinstance(item, dict):
                pairs.append(
                    {
                        "input": item.get("input", "") if isinstance(item.get("input", ""), str) else str(item.get("input", "")),
                        "gt": item.get("gt", "") if isinstance(item.get("gt", ""), str) else str(item.get("gt", "")),
                    }
                )
        if pairs:
            return pairs

    # Fallback: derive from mem_code on the fly if code_generation is not present.
    mem_targets = triplet.get("targets_mem")
    if isinstance(mem_targets, list) and qid < len(mem_targets) and isinstance(mem_targets[qid], list):
        return [_split_code_middle_line(code) for code in mem_targets[qid]]
    return []


def flatten_triplets(triplet: dict, text_source: str, include_label2_as_positive: bool):
    labels_all = triplet["labels"]
    tasks = triplet["tasks"]
    if not (len(labels_all) == len(tasks)):
        raise ValueError("Length mismatch among labels/tasks.")

    targets = None
    if text_source in {"mem", "syn"}:
        targets = _pick_targets(triplet, text_source)
        if len(targets) != len(labels_all):
            raise ValueError("Length mismatch among targets/labels/tasks.")

    flat_samples: List[FlatSample] = []
    question_meta = []

    for qid, (labels, task) in enumerate(zip(labels_all, tasks)):
        if text_source in {"mem", "syn"}:
            choices = targets[qid]
            if len(choices) != len(labels):
                raise ValueError(
                    f"Length mismatch at question {qid}: choices={len(choices)} labels={len(labels)}"
                )
            iter_indices = range(len(choices))
            codegen_pairs = None
        elif text_source == "code_generation":
            codegen_pairs = _get_codegen_pairs_for_question(triplet, qid)
            if not codegen_pairs:
                raise ValueError(
                    f"No code_generation/mem fallback data at question {qid}. "
                    "Please ensure dataset has code_generation or mem_code."
                )
            iter_indices = range(min(len(codegen_pairs), len(labels)))
        else:
            raise ValueError(f"Unsupported text_source={text_source}")

        sample_indices = []
        pos_choice_indices = []
        all_binary_labels = []

        for cidx in iter_indices:
            label_raw = labels[cidx]
            y = _label_to_binary(label_raw, include_label2_as_positive=include_label2_as_positive)
            if y is None:
                continue

            if text_source in {"mem", "syn"}:
                text = choices[cidx]
                prompt = None
                gt = None
            else:
                pair = codegen_pairs[cidx]
                prompt = pair.get("input", "")
                gt = pair.get("gt", "")
                text = prompt

            flat_idx = len(flat_samples)
            flat_samples.append(
                FlatSample(
                    question_id=qid,
                    choice_index=cidx,
                    text=text,
                    label=y,
                    prompt=prompt,
                    gt=gt,
                )
            )
            sample_indices.append(flat_idx)
            all_binary_labels.append(y)
            if y == 1:
                pos_choice_indices.append(cidx)

        question_meta.append(
            {
                "question_id": qid,
                "task": task,
                "num_choices": len(labels),
                "evaluable_flat_indices": sample_indices,
                "positive_choice_indices": pos_choice_indices,
                "binary_labels": all_binary_labels,
            }
        )

    return flat_samples, question_meta


class ScoreEngine:
    def __init__(self, model_name: str, model_dir: Optional[str], device: str, max_length: int):
        self.model_name = model_name
        self.model_path = model_dir if model_dir else model_name
        self.max_length = max_length

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        lat_model = LATModel(model_name, model_dir=model_dir)
        self.tokenizer = lat_model.tokenizer
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = lat_model.model
        self.model.eval()
        self.input_device = self._infer_input_device(self.model, self.device)

        self._logprob_cache: Dict[str, np.ndarray] = {}
        self._continuation_logprob_cache: Dict[str, np.ndarray] = {}
        self._token_signal_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._continuation_signal_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._score_cache: Dict[str, float] = {}

    @staticmethod
    def _infer_input_device(model, fallback: torch.device) -> torch.device:
        try:
            return next(model.parameters()).device
        except (StopIteration, AttributeError):
            return fallback

    @torch.no_grad()
    def _token_log_probs_from_ids(self, token_ids: List[int]) -> np.ndarray:
        if len(token_ids) < 2:
            return np.array([], dtype=np.float32)
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.input_device).unsqueeze(0)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        return token_log_probs.reshape(-1).detach().float().cpu().numpy()

    @torch.no_grad()
    def _token_signals_from_ids(self, token_ids: List[int]) -> Dict[str, np.ndarray]:
        if len(token_ids) < 2:
            empty = np.array([], dtype=np.float32)
            return {
                "logprob": empty,
                "confidence": empty,
                "entropy": empty,
                "msp": empty,
                "margin": empty,
            }

        input_ids = torch.tensor(token_ids, dtype=torch.long, device=self.input_device).unsqueeze(0)
        outputs = self.model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        confidences = torch.exp(token_log_probs)
        token_entropy = -(probs * log_probs).sum(dim=-1)
        token_msp = probs.max(dim=-1).values
        if probs.size(-1) >= 2:
            token_margin = torch.topk(probs, k=2, dim=-1).values
            token_margin = token_margin[..., 0] - token_margin[..., 1]
        else:
            token_margin = torch.zeros_like(token_msp)
        return {
            "logprob": token_log_probs.reshape(-1).detach().float().cpu().numpy(),
            "confidence": confidences.reshape(-1).detach().float().cpu().numpy(),
            "entropy": token_entropy.reshape(-1).detach().float().cpu().numpy(),
            "msp": token_msp.reshape(-1).detach().float().cpu().numpy(),
            "margin": token_margin.reshape(-1).detach().float().cpu().numpy(),
        }

    @torch.no_grad()
    def get_token_log_probs(self, text: str) -> np.ndarray:
        text = text if isinstance(text, str) else str(text)
        if text in self._logprob_cache:
            return self._logprob_cache[text]

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.input_device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.input_device)

        if input_ids.size(1) < 2:
            probs = np.array([], dtype=np.float32)
            self._logprob_cache[text] = probs
            return probs

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]

        token_log_probs = F.log_softmax(logits, dim=-1).gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)

        if attention_mask is not None:
            valid_mask = attention_mask[:, 1:].bool()
            token_log_probs = token_log_probs[valid_mask]
        else:
            token_log_probs = token_log_probs.reshape(-1)

        probs_np = token_log_probs.detach().float().cpu().numpy()
        self._logprob_cache[text] = probs_np
        return probs_np

    @torch.no_grad()
    def get_token_signals(self, text: str) -> Dict[str, np.ndarray]:
        text = text if isinstance(text, str) else str(text)
        if text in self._token_signal_cache:
            return self._token_signal_cache[text]

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(self.input_device)
        attention_mask = encoded.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.input_device)

        if input_ids.size(1) < 2:
            empty = np.array([], dtype=np.float32)
            signals = {
                "logprob": empty,
                "confidence": empty,
                "entropy": empty,
                "msp": empty,
                "margin": empty,
            }
            self._token_signal_cache[text] = signals
            self._logprob_cache[text] = empty
            return signals

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].float()
        labels = input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        token_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        confidences = torch.exp(token_log_probs)
        token_entropy = -(probs * log_probs).sum(dim=-1)
        token_msp = probs.max(dim=-1).values
        if probs.size(-1) >= 2:
            token_margin = torch.topk(probs, k=2, dim=-1).values
            token_margin = token_margin[..., 0] - token_margin[..., 1]
        else:
            token_margin = torch.zeros_like(token_msp)

        if attention_mask is not None:
            valid_mask = attention_mask[:, 1:].bool()
            token_log_probs = token_log_probs[valid_mask]
            confidences = confidences[valid_mask]
            token_entropy = token_entropy[valid_mask]
            token_msp = token_msp[valid_mask]
            token_margin = token_margin[valid_mask]
        else:
            token_log_probs = token_log_probs.reshape(-1)
            confidences = confidences.reshape(-1)
            token_entropy = token_entropy.reshape(-1)
            token_msp = token_msp.reshape(-1)
            token_margin = token_margin.reshape(-1)

        signals = {
            "logprob": token_log_probs.detach().float().cpu().numpy(),
            "confidence": confidences.detach().float().cpu().numpy(),
            "entropy": token_entropy.detach().float().cpu().numpy(),
            "msp": token_msp.detach().float().cpu().numpy(),
            "margin": token_margin.detach().float().cpu().numpy(),
        }
        self._token_signal_cache[text] = signals
        self._logprob_cache[text] = signals["logprob"]
        return signals

    @torch.no_grad()
    def get_continuation_log_probs(self, prompt: str, gt: str) -> np.ndarray:
        prompt = prompt if isinstance(prompt, str) else ("" if prompt is None else str(prompt))
        gt = gt if isinstance(gt, str) else ("" if gt is None else str(gt))
        cache_key = f"{prompt}\n<CONT>\n{gt}"
        if cache_key in self._continuation_logprob_cache:
            return self._continuation_logprob_cache[cache_key]

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        gt_ids = self.tokenizer.encode(gt, add_special_tokens=False)
        combined_ids = prompt_ids + gt_ids

        if len(combined_ids) < 2 or len(gt_ids) == 0:
            probs = np.array([], dtype=np.float32)
            self._continuation_logprob_cache[cache_key] = probs
            return probs

        if len(combined_ids) > self.max_length:
            overflow = len(combined_ids) - self.max_length
            combined_ids = combined_ids[overflow:]
            prompt_kept = max(0, len(prompt_ids) - overflow)
        else:
            prompt_kept = len(prompt_ids)

        gt_kept = len(combined_ids) - prompt_kept
        if gt_kept <= 0:
            probs = np.array([], dtype=np.float32)
            self._continuation_logprob_cache[cache_key] = probs
            return probs

        all_log_probs = self._token_log_probs_from_ids(combined_ids)
        if all_log_probs.size == 0:
            probs = np.array([], dtype=np.float32)
            self._continuation_logprob_cache[cache_key] = probs
            return probs

        start_idx = max(prompt_kept - 1, 0)
        continuation_probs = all_log_probs[start_idx:]
        self._continuation_logprob_cache[cache_key] = continuation_probs
        return continuation_probs

    @torch.no_grad()
    def get_continuation_signals(self, prompt: str, gt: str) -> Dict[str, np.ndarray]:
        prompt = prompt if isinstance(prompt, str) else ("" if prompt is None else str(prompt))
        gt = gt if isinstance(gt, str) else ("" if gt is None else str(gt))
        cache_key = f"{prompt}\n<CONT>\n{gt}"
        if cache_key in self._continuation_signal_cache:
            return self._continuation_signal_cache[cache_key]

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        gt_ids = self.tokenizer.encode(gt, add_special_tokens=False)
        combined_ids = prompt_ids + gt_ids

        empty = np.array([], dtype=np.float32)
        empty_signals = {
            "logprob": empty,
            "confidence": empty,
            "entropy": empty,
            "msp": empty,
            "margin": empty,
        }
        if len(combined_ids) < 2 or len(gt_ids) == 0:
            self._continuation_signal_cache[cache_key] = empty_signals
            self._continuation_logprob_cache[cache_key] = empty
            return empty_signals

        if len(combined_ids) > self.max_length:
            overflow = len(combined_ids) - self.max_length
            combined_ids = combined_ids[overflow:]
            prompt_kept = max(0, len(prompt_ids) - overflow)
        else:
            prompt_kept = len(prompt_ids)

        gt_kept = len(combined_ids) - prompt_kept
        if gt_kept <= 0:
            self._continuation_signal_cache[cache_key] = empty_signals
            self._continuation_logprob_cache[cache_key] = empty
            return empty_signals

        all_signals = self._token_signals_from_ids(combined_ids)
        start_idx = max(prompt_kept - 1, 0)
        signals = {name: values[start_idx:] for name, values in all_signals.items()}
        self._continuation_signal_cache[cache_key] = signals
        self._continuation_logprob_cache[cache_key] = signals["logprob"]
        return signals

    @staticmethod
    def _score_min_k(token_log_probs: np.ndarray, ratio: float) -> float:
        k_count = max(1, int(len(token_log_probs) * ratio))
        min_k_vals = np.sort(token_log_probs)[:k_count]
        return -float(np.mean(min_k_vals))

    def score(
        self,
        text: str,
        method: str,
        min_k_ratio: float = 0.2,
    ) -> float:
        key = f"{method}::{min_k_ratio}::{text}"
        if key in self._score_cache:
            return self._score_cache[key]

        if method in {"confidence", "entropy", "msp", "margin"}:
            token_values = self.get_token_signals(text)[method]
        else:
            token_values = self.get_token_log_probs(text)

        if token_values.size == 0:
            value = float("nan")
            self._score_cache[key] = value
            return value

        if method in {"confidence", "entropy", "msp", "margin"}:
            value = float(np.mean(token_values))
            self._score_cache[key] = value
            return value

        token_log_probs = token_values
        loss = -float(np.mean(token_log_probs))

        if method == "loss":
            value = loss
        elif method == "ppl":
            value = float(np.exp(min(loss, 50.0)))
        elif method == "zlib":
            comp_len = len(_stdlib_zlib.compress(text.encode("utf-8")))
            value = loss / max(comp_len, 1)
        elif method == "min_k":
            value = self._score_min_k(token_log_probs, min_k_ratio)
        else:
            raise ValueError(f"Unsupported method: {method}")

        self._score_cache[key] = value
        return value

    def score_completion(
        self,
        prompt: str,
        gt: str,
        method: str,
        min_k_ratio: float = 0.2,
    ) -> float:
        cache_key = f"completion::{method}::{min_k_ratio}::{prompt}\n<GT>\n{gt}"
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        if method in {"confidence", "entropy", "msp", "margin"}:
            token_values = self.get_continuation_signals(prompt, gt)[method]
        else:
            token_values = self.get_continuation_log_probs(prompt, gt)

        if token_values.size == 0:
            value = float("nan")
            self._score_cache[cache_key] = value
            return value

        if method in {"confidence", "entropy", "msp", "margin"}:
            value = float(np.mean(token_values))
            self._score_cache[cache_key] = value
            return value

        token_log_probs = token_values
        loss = -float(np.mean(token_log_probs))
        if method == "loss":
            value = loss
        elif method == "ppl":
            value = float(np.exp(min(loss, 50.0)))
        elif method == "zlib":
            comp_target = gt if isinstance(gt, str) and gt else (prompt + gt)
            comp_len = len(_stdlib_zlib.compress(comp_target.encode("utf-8")))
            value = loss / max(comp_len, 1)
        elif method == "min_k":
            value = self._score_min_k(token_log_probs, min_k_ratio)
        else:
            raise ValueError(f"Unsupported method: {method}")

        self._score_cache[cache_key] = value
        return value


def _nan_safe_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true.size > 0 else None,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if y_true.size > 0 else None,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if y_true.size > 0 else None,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)) if y_true.size > 0 else None,
        "auroc": None,
        "num_samples": int(y_true.size),
    }
    if y_true.size > 0 and len(np.unique(y_true)) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
    return metrics


def fit_binary_classifier_1d(
    scores: np.ndarray,
    labels: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
):
    unique = np.unique(labels)
    if unique.size < 2:
        majority = int(unique[0]) if unique.size == 1 else 0
        return {
            "type": "constant",
            "majority_class": majority,
            "model": None,
            "selection_metric": None,
            "cv_folds": 0,
            "selected_C": None,
            "cv_candidates": [],
        }

    class_counts = np.bincount(labels, minlength=2)
    min_class_count = int(min(class_counts[0], class_counts[1]))
    effective_splits = int(min(n_splits, min_class_count))

    c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    cv_candidates = []
    best_c = 1.0
    best_score = -1.0

    if effective_splits >= 2:
        skf = StratifiedKFold(
            n_splits=effective_splits,
            shuffle=True,
            random_state=random_state,
        )

        for c_value in c_grid:
            fold_accs = []
            fold_f1s = []
            fold_aurocs = []
            for train_idx, val_idx in skf.split(scores.reshape(-1, 1), labels):
                x_tr = scores[train_idx].reshape(-1, 1)
                y_tr = labels[train_idx]
                x_va = scores[val_idx].reshape(-1, 1)
                y_va = labels[val_idx]

                clf = LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=random_state,
                    C=float(c_value),
                )
                clf.fit(x_tr, y_tr)

                y_pred = clf.predict(x_va)
                y_prob = clf.predict_proba(x_va)[:, 1]

                fold_accs.append(float(accuracy_score(y_va, y_pred)))
                fold_f1s.append(float(f1_score(y_va, y_pred, zero_division=0)))
                if len(np.unique(y_va)) == 2:
                    fold_aurocs.append(float(roc_auc_score(y_va, y_prob)))

            mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
            mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else 0.0
            mean_auc = float(np.mean(fold_aurocs)) if fold_aurocs else 0.0
            combined = float(0.5 * mean_auc + 0.25 * mean_acc + 0.25 * mean_f1)

            cv_candidates.append(
                {
                    "C": float(c_value),
                    "cv_folds": int(effective_splits),
                    "cv_mean_auroc": mean_auc,
                    "cv_mean_accuracy": mean_acc,
                    "cv_mean_f1": mean_f1,
                    "combined_score": combined,
                }
            )

            if combined > best_score:
                best_score = combined
                best_c = float(c_value)

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
        C=float(best_c),
    )
    clf.fit(scores.reshape(-1, 1), labels)
    return {
        "type": "logistic_regression_cv" if effective_splits >= 2 else "logistic_regression_no_cv",
        "majority_class": None,
        "coef": float(clf.coef_[0][0]),
        "intercept": float(clf.intercept_[0]),
        "selection_metric": "0.5*AUROC + 0.25*Accuracy + 0.25*F1" if effective_splits >= 2 else None,
        "cv_folds": int(effective_splits) if effective_splits >= 2 else 0,
        "selected_C": float(best_c),
        "cv_candidates": cv_candidates,
        "model": clf,
    }


def predict_classifier_1d(classifier_info: dict, scores: np.ndarray):
    if classifier_info["type"] == "constant":
        pred = np.full(scores.shape[0], classifier_info["majority_class"], dtype=np.int64)
        prob = pred.astype(np.float32)
        return pred, prob

    clf = classifier_info["model"]
    prob = clf.predict_proba(scores.reshape(-1, 1))[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return pred, prob


def evaluate_question_ranking(question_meta: List[dict], probs: np.ndarray) -> dict:
    details = []
    total = 0
    correct = 0

    for q in question_meta:
        flat_indices = q["evaluable_flat_indices"]
        if not flat_indices:
            continue

        choice_indices = []
        choice_probs = []
        labels = []
        for fi, y in zip(flat_indices, q["binary_labels"]):
            # flat index -> original choice index is encoded in sample order assumption:
            # we reconstruct from detail data outside; here use order by indices
            # because flattened order for a question follows choice order.
            choice_indices.append(len(choice_indices))
            choice_probs.append(float(probs[fi]))
            labels.append(int(y))

        if not choice_probs:
            continue

        pred_local = int(np.argmax(choice_probs))
        positive_local = [idx for idx, y in enumerate(labels) if y == 1]
        is_evaluable = len(positive_local) > 0
        is_correct = pred_local in positive_local if is_evaluable else None

        if is_evaluable:
            total += 1
            if is_correct:
                correct += 1

        details.append(
            {
                "question_id": int(q["question_id"]),
                "predicted_choice_local_index": pred_local,
                "positive_choice_local_indices": positive_local,
                "is_evaluable": bool(is_evaluable),
                "is_correct": bool(is_correct) if is_correct is not None else None,
                "choice_probabilities": choice_probs,
            }
        )

    ranking_acc = float(correct / total) if total > 0 else 0.0
    return {
        "ranking_accuracy_top1": ranking_acc,
        "num_evaluable_questions": int(total),
        "num_correct_questions": int(correct),
        "details": details,
    }


def _normalize_methods(default_method: str, methods_arg: str) -> List[str]:
    if methods_arg and methods_arg.strip():
        raw = [x.strip() for x in methods_arg.split(",") if x.strip()]
    else:
        raw = [default_method]

    expanded: List[str] = []
    for m in raw:
        if m == "all":
            expanded.extend(DEFAULT_METHODS)
        else:
            expanded.append(m)

    methods: List[str] = []
    seen = set()
    for m in expanded:
        if m not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported method '{m}'. Supported: {list(SUPPORTED_METHODS)} or 'all'."
            )
        if m not in seen:
            seen.add(m)
            methods.append(m)
    return methods

def orient_scores_by_validation(val_scores, val_labels, test_scores=None, fit_scores=None):
    """
    用 validate 判断 raw score 方向。
    返回的 oriented score 满足：越大 = 越像 member(label=1)。
    """
    val_scores = np.asarray(val_scores, dtype=float)
    val_labels = np.asarray(val_labels, dtype=int)

    valid = ~np.isnan(val_scores)
    val_scores_valid = val_scores[valid]
    val_labels_valid = val_labels[valid]
    unique = np.unique(val_labels_valid)

    # Guard against degenerate validate slice where AUROC is undefined.
    if val_scores_valid.size == 0 or unique.size < 2:
        raw_auc = float("nan")
        sign = 1.0
        oriented_auc = float("nan")
        fallback = True
    else:
        raw_auc = float(roc_auc_score(val_labels_valid, val_scores_valid))
        sign = 1.0 if raw_auc >= 0.5 else -1.0
        oriented_auc = float(raw_auc if raw_auc >= 0.5 else 1.0 - raw_auc)
        fallback = False

    outputs = {
        "sign": sign,
        "raw_validation_auroc": raw_auc,
        "oriented_validation_auroc": oriented_auc,
        "orientation_fallback_used": bool(fallback),
        "num_valid_scores": int(val_scores_valid.size),
        "val_scores": sign * val_scores,
    }

    if test_scores is not None:
        outputs["test_scores"] = sign * np.asarray(test_scores, dtype=float)

    if fit_scores is not None:
        outputs["fit_scores"] = sign * np.asarray(fit_scores, dtype=float)

    return outputs

def run_baseline(method: str, args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    methods = _normalize_methods(method, getattr(args, "methods", ""))
    print(f"[baseline] methods={methods}")

    start = time.time()
    fit_triplet = None
    fit_flat = []
    fit_questions = []
    if args.fit and os.path.exists(args.fit):
        fit_triplet = load_triplets_jsonl(args.fit)
        fit_flat, fit_questions = flatten_triplets(
            fit_triplet,
            text_source=args.text_source,
            include_label2_as_positive=args.include_label2_as_positive,
        )
    elif args.fit:
        print(f"[{method}] WARNING: fit file not found, skipping fit split: {args.fit}")

    validate_triplet = load_triplets_jsonl(args.validate)
    test_triplet = load_triplets_jsonl(args.test)

    val_flat, val_questions = flatten_triplets(
        validate_triplet,
        text_source=args.text_source,
        include_label2_as_positive=args.include_label2_as_positive,
    )
    test_flat, test_questions = flatten_triplets(
        test_triplet,
        text_source=args.text_source,
        include_label2_as_positive=args.include_label2_as_positive,
    )

    if not val_flat:
        raise ValueError("No evaluable validation samples found.")
    if not test_flat:
        raise ValueError("No evaluable test samples found.")

    print(f"[baseline] loading model: {args.model}")
    engine = ScoreEngine(
        model_name=args.model,
        model_dir=args.model_dir,
        device=args.device,
        max_length=args.max_length,
    )

    def compute_scores_by_methods(flat_samples: List[FlatSample]) -> Dict[str, np.ndarray]:
        score_map = {m: np.empty(len(flat_samples), dtype=np.float32) for m in methods}
        for i, sample in enumerate(flat_samples):
            if i % 50 == 0:
                print(f"[baseline] scoring {i}/{len(flat_samples)}")
            for method_name in methods:
                if args.text_source == "code_generation":
                    s = engine.score_completion(
                        sample.prompt or "",
                        sample.gt or "",
                        method=method_name,
                        min_k_ratio=args.min_k_ratio,
                    )
                else:
                    s = engine.score(
                        sample.text,
                        method=method_name,
                        min_k_ratio=args.min_k_ratio,
                    )
                if np.isnan(s):
                    s = 0.0
                score_map[method_name][i] = float(s)
        return score_map

    fit_scores_raw_map = compute_scores_by_methods(fit_flat) if fit_flat else None
    val_scores_raw_map = compute_scores_by_methods(val_flat)
    test_scores_raw_map = compute_scores_by_methods(test_flat)

    val_labels = np.array([s.label for s in val_flat], dtype=np.int64)
    test_labels = np.array([s.label for s in test_flat], dtype=np.int64)
    fit_labels = np.array([s.label for s in fit_flat], dtype=np.int64) if fit_flat else None

    result_paths: Dict[str, str] = {}

    def _fmt(v):
        if v is None:
            return "NA"
        return f"{float(v):.6f}"

    for method_name in methods:
        fit_scores_raw = (
            fit_scores_raw_map[method_name]
            if fit_scores_raw_map is not None and method_name in fit_scores_raw_map
            else None
        )
        val_scores_raw = val_scores_raw_map[method_name]
        test_scores_raw = test_scores_raw_map[method_name]

        score_preprocess = {
            "type": "none",
            "fit_mean": None,
            "fit_std": None,
            "fit_num_samples": int(len(fit_flat)),
        }
        if fit_scores_raw is not None and fit_scores_raw.size > 0:
            fit_mean = float(np.mean(fit_scores_raw))
            fit_std = float(np.std(fit_scores_raw))
            if not np.isfinite(fit_std) or fit_std < 1e-8:
                fit_std = 1.0

            val_scores = (val_scores_raw - fit_mean) / fit_std
            test_scores = (test_scores_raw - fit_mean) / fit_std
            fit_scores = (fit_scores_raw - fit_mean) / fit_std
            score_preprocess = {
                "type": "zscore_from_fit",
                "fit_mean": float(fit_mean),
                "fit_std": float(fit_std),
                "fit_num_samples": int(len(fit_flat)),
            }
        else:
            val_scores = val_scores_raw
            test_scores = test_scores_raw
            fit_scores = fit_scores_raw

        # Preserve pre-orientation scores for debugging/auditing.
        val_scores_pre_orient = np.asarray(val_scores, dtype=np.float64).copy()
        test_scores_pre_orient = np.asarray(test_scores, dtype=np.float64).copy()
        fit_scores_pre_orient = (
            np.asarray(fit_scores, dtype=np.float64).copy() if fit_scores is not None else None
        )
        
        if not args.disable_score_orientation:
            orientation = orient_scores_by_validation(
                    val_scores=val_scores,
                    val_labels=val_labels,
                    test_scores=test_scores,
                    fit_scores=fit_scores_pre_orient,
                )

            val_scores = orientation["val_scores"]
            test_scores = orientation["test_scores"]
            fit_scores = orientation.get("fit_scores", fit_scores)

            score_preprocess["orientation"] = {
                "rule": "validation_auc_flip_if_below_0.5",
                "enabled": True,
                "sign": float(orientation["sign"]),
                "raw_validation_auroc": float(orientation["raw_validation_auroc"]),
                "oriented_validation_auroc": float(orientation["oriented_validation_auroc"]),
                "orientation_fallback_used": bool(orientation.get("orientation_fallback_used", False)),
                "num_valid_scores": int(orientation.get("num_valid_scores", 0)),
            }
        else:
            val_scores = np.asarray(val_scores, dtype=float)
            test_scores = np.asarray(test_scores, dtype=float)
            fit_scores = (
                np.asarray(fit_scores, dtype=float) if fit_scores is not None else fit_scores
            )
            score_preprocess["orientation"] = {
                "rule": "disabled_raw_score",
                "enabled": False,
                "sign": 1.0,
                "raw_validation_auroc": None,
                "oriented_validation_auroc": None,
                "orientation_fallback_used": False,
                "num_valid_scores": int(np.sum(~np.isnan(val_scores))),
            }

        clf_info = fit_binary_classifier_1d(
            val_scores,
            val_labels,
            n_splits=args.cv_folds,
            random_state=args.random_state,
        )
        fit_pred, fit_prob = (
            predict_classifier_1d(clf_info, fit_scores)
            if fit_scores is not None
            else (None, None)
        )
        val_pred, val_prob = predict_classifier_1d(clf_info, val_scores)
        test_pred, test_prob = predict_classifier_1d(clf_info, test_scores)

        fit_metrics = (
            _nan_safe_metrics(fit_labels, fit_pred, fit_prob)
            if fit_scores is not None
            else None
        )
        val_metrics = _nan_safe_metrics(val_labels, val_pred, val_prob)
        test_metrics = _nan_safe_metrics(test_labels, test_pred, test_prob)

        question_metrics = evaluate_question_ranking(test_questions, test_prob)

        test_candidate_details = []
        test_fp_map_by_question: Dict[str, List[int]] = {}
        test_fp_count = 0
        for idx, (s, y_t, y_p, p) in enumerate(zip(test_flat, test_labels, test_pred, test_prob)):
            is_fp = bool(int(y_t) == 0 and int(y_p) == 1)
            if is_fp:
                qkey = str(int(s.question_id))
                test_fp_map_by_question.setdefault(qkey, []).append(int(s.choice_index))
                test_fp_count += 1
            test_candidate_details.append(
                {
                    "question_id": int(s.question_id),
                    "choice_index": int(s.choice_index),
                    "y_true": int(y_t),
                    "y_pred": int(y_p),
                    "prob": float(p),
                    "is_fp": is_fp,
                    "score_before_orientation": float(test_scores_pre_orient[idx]),
                    "member_like_score": float(test_scores[idx]),
                }
            )

        for qkey in list(test_fp_map_by_question.keys()):
            test_fp_map_by_question[qkey] = sorted(set(test_fp_map_by_question[qkey]))

        runtime = time.time() - start
        result = {
            "method": method_name,
            "model": args.model,
            "text_source": args.text_source,
            "include_label2_as_positive": bool(args.include_label2_as_positive),
            "min_k_ratio": float(args.min_k_ratio),
            "data_files": {
                "fit": args.fit,
                "validate": args.validate,
                "test": args.test,
            },
            "classifier": {
                "type": clf_info["type"],
                "coef": clf_info.get("coef"),
                "intercept": clf_info.get("intercept"),
                "majority_class": clf_info.get("majority_class"),
                "selection_metric": clf_info.get("selection_metric"),
                "cv_folds": clf_info.get("cv_folds"),
                "selected_C": clf_info.get("selected_C"),
                "cv_candidates": clf_info.get("cv_candidates"),
            },
            "score_preprocess": score_preprocess,
            "fit_candidate_metrics": fit_metrics,
            "validation_candidate_metrics": val_metrics,
            "test_candidate_metrics": test_metrics,
            "test_question_metrics": question_metrics,
            "test_fp_count": int(test_fp_count),
            "test_fp_map_by_question": test_fp_map_by_question,
            "test_candidate_details": test_candidate_details,
            "timing": {
                "runtime_seconds": float(runtime),
                "num_validation_candidates": int(len(val_flat)),
                "num_test_candidates": int(len(test_flat)),
            },
        }

        out_path = os.path.join(args.output_dir, f"{method_name}_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        result_paths[method_name] = out_path
        print(f"[{method_name}] results saved to {out_path}")

        print(
            f"[{method_name}] fit(candidate): "
            f"acc={_fmt(fit_metrics.get('accuracy')) if fit_metrics else 'NA'}, "
            f"precision={_fmt(fit_metrics.get('precision')) if fit_metrics else 'NA'}, "
            f"recall={_fmt(fit_metrics.get('recall')) if fit_metrics else 'NA'}, "
            f"f1={_fmt(fit_metrics.get('f1_score')) if fit_metrics else 'NA'}, "
            f"auroc={_fmt(fit_metrics.get('auroc')) if fit_metrics else 'NA'}, "
            f"n={fit_metrics.get('num_samples') if fit_metrics else 0}"
        )
        print(
            f"[{method_name}] validation(candidate): "
            f"acc={_fmt(val_metrics.get('accuracy'))}, "
            f"precision={_fmt(val_metrics.get('precision'))}, "
            f"recall={_fmt(val_metrics.get('recall'))}, "
            f"f1={_fmt(val_metrics.get('f1_score'))}, "
            f"auroc={_fmt(val_metrics.get('auroc'))}, "
            f"n={val_metrics.get('num_samples')}"
        )
        print(
            f"[{method_name}] test(candidate): "
            f"acc={_fmt(test_metrics.get('accuracy'))}, "
            f"precision={_fmt(test_metrics.get('precision'))}, "
            f"recall={_fmt(test_metrics.get('recall'))}, "
            f"f1={_fmt(test_metrics.get('f1_score'))}, "
            f"auroc={_fmt(test_metrics.get('auroc'))}, "
            f"n={test_metrics.get('num_samples')}"
        )
        print(
            f"[{method_name}] test(question-top1): "
            f"ranking_acc={_fmt(question_metrics.get('ranking_accuracy_top1'))}, "
            f"num_evaluable={question_metrics.get('num_evaluable_questions')}, "
            f"num_correct={question_metrics.get('num_correct_questions')}"
        )
        print(
            f"[{method_name}] classifier selection: "
            f"type={clf_info.get('type')}, "
            f"cv_folds={clf_info.get('cv_folds')}, "
            f"selected_C={clf_info.get('selected_C')}"
        )
        print(f"[{method_name}] test false positives: {int(test_fp_count)}")

    if len(methods) > 1:
        summary_path = os.path.join(args.output_dir, "all_methods_results.json")
        summary = {
            "methods": methods,
            "model": args.model,
            "output_files": result_paths,
            "data_files": {
                "fit": args.fit,
                "validate": args.validate,
                "test": args.test,
            },
            "text_source": args.text_source,
            "include_label2_as_positive": bool(args.include_label2_as_positive),
            "min_k_ratio": float(args.min_k_ratio),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[baseline] all-method summary saved to {summary_path}")


def build_parser(method: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Run standalone {method} baseline for membership classification."
    )
    parser.add_argument("--fit", type=str, default="", help="Optional fit split path (kept for interface compatibility).")
    parser.add_argument("--validate", type=str, required=True, help="Path to validate jsonl.")
    parser.add_argument("--test", type=str, required=True, help="Path to test jsonl.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--model", type=str, required=True, help="HF model name/path.")
    parser.add_argument("--model-dir", type=str, default=None, help="Local model directory override.")
    parser.add_argument(
        "--text-source",
        type=str,
        choices=["mem", "syn", "code_generation"],
        default="code_generation",
        help="Scoring source: mem_code / syn_code / code_generation(input->gt continuation).",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto/cpu/cuda/cuda:0...")
    parser.add_argument("--max-length", type=int, default=2048, help="Max token length for scoring.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of folds on validate split for classifier selection.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for CV splitting and classifier fitting.")
    parser.add_argument(
        "--include-label2-as-positive",
        action="store_true",
        help="Treat label>0 as positive when converting labels.",
    )
    parser.add_argument(
        "--min-k-ratio",
        type=float,
        default=0.2,
        help="Only used by min_k baseline.",
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
    parser.add_argument(
        "--disable-score-orientation",
        action="store_true",
        help="Disable score direction alignment by validation AUROC (keep raw score direction).",
    )
    return parser
