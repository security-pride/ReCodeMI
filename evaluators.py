# Portions of this file are adapted from code-repe:
# https://github.com/sanadlab/code-repe

from abc import ABC, abstractmethod
from collections import defaultdict
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from .loaders import (
        aligned_shuffle_three_with_index_map,
        load_triplets_jsonl,
        load_memorization_jsonl,
    )
except ImportError:
    # Fallback for direct script execution from this directory.
    from loaders import (
        aligned_shuffle_three_with_index_map,
        load_triplets_jsonl,
        load_memorization_jsonl,
    )


class Evaluator(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def evaluate(
        self,
        tasks,
        targets_list,
        syn_labels_list,
        targets_mem_list=None,
        mem_labels_list=None,
        sample_ids=None,
    ):
        pass

    @classmethod
    def save_to_json(cls, results, output_file="results.json"):
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Results have been saved to {output_file}")


class LATEvaluator(Evaluator):
    STIMULUS = {
        "starcoder2-mia-test": lambda fit_file, validate_file: (
            load_triplets_jsonl(fit_file),
            load_triplets_jsonl(validate_file),
        ),
        "memeorization": lambda fit_file, validate_file: (
            load_memorization_jsonl(fit_file),
            load_memorization_jsonl(validate_file),
        ),
    }

    def __init__(
        self,
        model,
        concept,
        template="{target}",
        target_field_name="code",
        token_tags=None,
        no_wrapper=False,
    ):
        super().__init__(model)
        self.template = template
        self.concept = concept
        self.target_field_name = target_field_name
        self.token_tags = token_tags
        self.best_clf = None
        self.classifier_results = None

        self.raw_template = "{scenario}"
        if no_wrapper:
            self.prompt_template = "{scenario}"
        else:
            self.prompt_template = (
                f"{token_tags['user_start']}Consider the amount of {self.concept} in the following {self.target_field_name}.\n\n"
                + "{scenario}\n"
                + f"{token_tags['user_end']}"
                + f"{token_tags['assistant_start']}The amount of {self.concept} in the {self.target_field_name} is "
            )

    @staticmethod
    def _is_binary_scalar(value):
        try:
            v = int(value)
        except (TypeError, ValueError):
            return False
        return v in (0, 1)

    @staticmethod
    def _is_supported_scalar_label(value):
        try:
            v = int(value)
        except (TypeError, ValueError):
            return False
        return v in (0, 1, 2)

    def _label_to_binary(self, label):
        if self._is_binary_scalar(label):
            return int(label)
        return None

    def _normalize_scalar_label(self, label, label_name, sample_idx, choice_idx):
        if isinstance(label, (list, tuple, np.ndarray)):
            raise ValueError(
                f"{label_name} must be 1D scalar label in {{0,1,2}}, got sequence at "
                f"sample={sample_idx}, choice={choice_idx}: {label}"
            )
        if not self._is_supported_scalar_label(label):
            raise ValueError(
                f"{label_name} must be scalar in {{0,1,2}}, got {label!r} at "
                f"sample={sample_idx}, choice={choice_idx}."
            )
        return int(label)

    def evaluate(
        self,
        tasks,
        targets_list,
        syn_labels_list,
        targets_mem_list=None,
        mem_labels_list=None,
        sample_ids=None,
    ):
        eval_data_mem = []
        eval_data_syn = []

        if targets_mem_list is None:
            targets_mem_list = targets_list
        if mem_labels_list is None:
            mem_labels_list = syn_labels_list
        if sample_ids is None:
            sample_ids = list(range(len(tasks)))

        if not (
            len(tasks)
            == len(targets_list)
            == len(targets_mem_list)
            == len(syn_labels_list)
            == len(mem_labels_list)
            == len(sample_ids)
        ):
            raise ValueError(
                "Length mismatch among tasks, syn targets, mem targets, syn labels, mem labels, and sample ids."
            )

        syn_targets_shuffled_all = []
        mem_targets_shuffled_all = []
        syn_labels_shuffled_all = []
        mem_labels_shuffled_all = []
        shuffled_to_orig_all = []
        orig_to_shuffled_all = []
        choice_uids_shuffled_all = []

        for idx in range(len(tasks)):
            syn_targets = list(targets_list[idx])
            mem_targets = list(targets_mem_list[idx])
            syn_labels = list(syn_labels_list[idx])
            mem_labels = list(mem_labels_list[idx])

            if not (
                len(syn_targets)
                == len(mem_targets)
                == len(syn_labels)
                == len(mem_labels)
            ):
                raise ValueError(
                    f"Choice length mismatch at sample {idx}: "
                    f"len(syn_targets)={len(syn_targets)}, "
                    f"len(mem_targets)={len(mem_targets)}, "
                    f"len(syn_labels)={len(syn_labels)}, "
                    f"len(mem_labels)={len(mem_labels)}."
                )

            syn_labels_norm = [
                self._normalize_scalar_label(v, "syn_labels", idx, cidx)
                for cidx, v in enumerate(syn_labels)
            ]
            mem_labels_norm = [
                self._normalize_scalar_label(v, "mem_labels", idx, cidx)
                for cidx, v in enumerate(mem_labels)
            ]

            label_pairs = list(zip(syn_labels_norm, mem_labels_norm))
            (
                syn_targets_shuffled,
                mem_targets_shuffled,
                label_pairs_shuffled,
                shuffled_to_orig,
                orig_to_shuffled,
            ) = aligned_shuffle_three_with_index_map(
                syn_targets,
                mem_targets,
                label_pairs,
                seed=idx,
            )

            syn_labels_shuffled = [int(pair[0]) for pair in label_pairs_shuffled]
            mem_labels_shuffled = [int(pair[1]) for pair in label_pairs_shuffled]

            syn_targets_shuffled_all.append(syn_targets_shuffled)
            mem_targets_shuffled_all.append(mem_targets_shuffled)
            syn_labels_shuffled_all.append(syn_labels_shuffled)
            mem_labels_shuffled_all.append(mem_labels_shuffled)
            shuffled_to_orig_all.append([int(x) for x in shuffled_to_orig])
            orig_to_shuffled_all.append([int(x) for x in orig_to_shuffled])
            choice_uids_shuffled_all.append(
                [f"{sample_ids[idx]}:{orig_idx}" for orig_idx in shuffled_to_orig]
            )

            for syn_target, mem_target in zip(syn_targets_shuffled, mem_targets_shuffled):
                scenario = self.template.format(
                    target="```python\n" + syn_target.strip() + "\n```"
                )
                scenario_raw = "```python\n" + mem_target.strip() + "\n```"
                eval_data_mem.append(self.raw_template.format(scenario=scenario_raw))
                eval_data_syn.append(self.prompt_template.format(scenario=scenario))

        hidden_layers = list(range(-1, -self.model.num_hidden_layers(), -1))

        H_tests_mem = self.model.representation_pipeline(
            eval_data_mem, hidden_layers, custom_reader=self.model.rep_reader_mem
        )
        H_tests_syn = self.model.representation_pipeline(
            eval_data_syn, hidden_layers, custom_reader=self.model.rep_reader_syn
        )

        num_targets = [len(targets) for targets in syn_targets_shuffled_all]
        indices = np.cumsum(num_targets)
        start_indices = [0] + indices[:-1].tolist()
        end_indices = indices.tolist()
        H_grouped_mem = [
            H_tests_mem[start:end] for start, end in zip(start_indices, end_indices)
        ]
        H_grouped_syn = [
            H_tests_syn[start:end] for start, end in zip(start_indices, end_indices)
        ]

        layer_rank_stats = []
        detailed_results = []

        for layer in hidden_layers:
            layer_correct = 0
            layer_total = 0
            sign_syn = self.model.rep_reader_syn.direction_signs[layer]
            sign_mem = self.model.rep_reader_mem.direction_signs[layer]

            for i, (H_choices_mem, H_choices_syn) in enumerate(
                zip(H_grouped_mem, H_grouped_syn)
            ):
                H_layer_mem = [H[layer] for H in H_choices_mem]
                H_layer_syn = [H[layer] for H in H_choices_syn]

                syn_labels = syn_labels_shuffled_all[i]
                mem_labels = mem_labels_shuffled_all[i]

                evaluable_indices = [idx for idx, y in enumerate(syn_labels) if y in (0, 1)]

                if sign_syn == -1:
                    ranked_indices = sorted(
                        evaluable_indices, key=lambda idx: H_layer_syn[idx]
                    )
                else:
                    ranked_indices = sorted(
                        evaluable_indices,
                        key=lambda idx: H_layer_syn[idx],
                        reverse=True,
                    )
                predicted_label = ranked_indices[0] if ranked_indices else None

                correct_indices = [idx for idx, y in enumerate(syn_labels) if y == 1]
                is_evaluable = predicted_label is not None and len(correct_indices) > 0
                is_correct = (predicted_label in correct_indices) if is_evaluable else None

                if is_evaluable:
                    if is_correct:
                        layer_correct += 1
                    layer_total += 1

                predicted_choice = (
                    syn_targets_shuffled_all[i][int(predicted_label)]
                    if predicted_label is not None
                    else None
                )

                detailed_results.append(
                    {
                        "layer": int(layer),
                        "question_id": int(i),
                        "sample_id": sample_ids[i],
                        "question": tasks[i],
                        "choices": syn_targets_shuffled_all[i],
                        "predicted_label": int(predicted_label)
                        if predicted_label is not None
                        else None,
                        "predicted_choice": predicted_choice,
                        "predicted_ranking": [int(idx) for idx in ranked_indices],
                        "predicted_choices_ranked": [
                            syn_targets_shuffled_all[i][int(idx)] for idx in ranked_indices
                        ],
                        "correct_label": [int(idx) for idx in correct_indices],
                        "correct_choice": [
                            syn_targets_shuffled_all[i][int(idx)]
                            for idx in correct_indices
                        ],
                        "syn_label_list": [int(x) for x in syn_labels],
                        "mem_label_list": [int(x) for x in mem_labels],
                        # Kept for downstream compatibility: classifier consumes mem label list.
                        "correct_label_list": [int(x) for x in mem_labels],
                        "is_evaluable": bool(is_evaluable),
                        "is_correct": bool(is_correct)
                        if is_correct is not None
                        else None,
                        "score_mem_X": [float(score * sign_mem) for score in H_layer_mem],
                        "score_syn_Y": [float(score * sign_syn) for score in H_layer_syn],
                        "shuffle_seed": int(i),
                        "shuffled_to_orig_index": shuffled_to_orig_all[i],
                        "orig_to_shuffled_index": orig_to_shuffled_all[i],
                        "choice_uids_shuffled": choice_uids_shuffled_all[i],
                    }
                )

            accuracy = (layer_correct / layer_total) if layer_total > 0 else 0.0
            layer_rank_stats.append(
                {
                    "layer": int(layer),
                    "rank_accuracy": float(accuracy),
                    "correct_predictions": int(layer_correct),
                    "total_questions": int(layer_total),
                }
            )

        return {
            "label_schema": {
                "syn_labels": "scalar_0_1_2",
                "mem_labels": "scalar_0_1_2",
                "binary_eval_policy": "classifier_uses_only_0_1_label2_ignored",
            },
            "all_accuracies": layer_rank_stats,
            "results": detailed_results,
        }

    def fit_2d_classifier(
        self,
        val_results,
        n_splits=5,
        weight_auc=0.5,
        weight_acc=0.25,
        weight_f1=0.25,
        random_state=42,
    ):
        layer_classifiers = {}
        val_data_by_layer = defaultdict(lambda: {"X": [], "y": []})
        per_layer_results = {}

        for item in val_results["results"]:
            layer = item["layer"]
            _ = val_data_by_layer[layer]
            scores_x = item["score_mem_X"]
            scores_y = item["score_syn_Y"]
            labels = item["correct_label_list"]

            for idx, label in enumerate(labels):
                y_val = self._label_to_binary(label)
                if y_val is None:
                    continue
                val_data_by_layer[layer]["X"].append([scores_x[idx], scores_y[idx]])
                val_data_by_layer[layer]["y"].append(y_val)

        best_layer = None
        best_score = -1.0

        for layer, data in val_data_by_layer.items():
            layer_key = int(layer)
            sample_count = len(data["y"])
            class_counts = {
                "0": int(sum(1 for v in data["y"] if v == 0)),
                "1": int(sum(1 for v in data["y"] if v == 1)),
            }
            layer_report = {
                "status": "skipped",
                "num_samples": int(sample_count),
                "class_counts": class_counts,
            }

            if sample_count == 0:
                layer_report["reason"] = "no_valid_binary_labels"
                per_layer_results[layer_key] = layer_report
                continue

            X = np.array(data["X"])
            y = np.array(data["y"])

            if len(set(y)) < 2:
                layer_report["reason"] = "single_class_only"
                per_layer_results[layer_key] = layer_report
                continue

            min_class_count = min(class_counts["0"], class_counts["1"])
            effective_splits = min(n_splits, min_class_count)

            if effective_splits < 2:
                layer_report["reason"] = "insufficient_samples_for_cv"
                per_layer_results[layer_key] = layer_report
                continue

            skf = StratifiedKFold(
                n_splits=effective_splits,
                shuffle=True,
                random_state=random_state,
            )

            fold_aurocs = []
            fold_accs = []
            fold_f1s = []

            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_va = X[train_idx], X[val_idx]
                y_tr, y_va = y[train_idx], y[val_idx]

                clf = LogisticRegression(class_weight="balanced", max_iter=1000)
                clf.fit(X_tr, y_tr)

                y_pred = clf.predict(X_va)
                y_prob = clf.predict_proba(X_va)[:, 1]

                acc = accuracy_score(y_va, y_pred)
                f1 = f1_score(y_va, y_pred, zero_division=0)

                if len(np.unique(y_va)) == 2:
                    auc = roc_auc_score(y_va, y_prob)
                else:
                    auc = float("nan")

                fold_accs.append(acc)
                fold_f1s.append(f1)
                fold_aurocs.append(auc)

            mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.0
            mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else 0.0
            valid_aurocs = [a for a in fold_aurocs if not np.isnan(a)]
            mean_auc = float(np.mean(valid_aurocs)) if valid_aurocs else 0.0

            combined_score = (
                weight_auc * mean_auc + weight_acc * mean_acc + weight_f1 * mean_f1
            )

            final_clf = LogisticRegression(class_weight="balanced", max_iter=1000)
            final_clf.fit(X, y)

            boundary = {
                "w1": float(final_clf.coef_[0][0]),
                "w2": float(final_clf.coef_[0][1]),
                "b": float(final_clf.intercept_[0]),
            }

            layer_classifiers[layer] = final_clf
            layer_report.update(
                {
                    "status": "fitted",
                    "cv_folds": int(effective_splits),
                    "cv_mean_auroc": mean_auc,
                    "cv_mean_accuracy": mean_acc,
                    "cv_mean_f1": mean_f1,
                    "combined_score": float(combined_score),
                    "decision_boundary": boundary,
                }
            )
            per_layer_results[layer_key] = layer_report

            if combined_score > best_score:
                best_score = combined_score
                best_layer = layer

        if best_layer is None:
            return {
                "error": "Failed to fit 2D classifier.",
                "per_layer_results": per_layer_results,
            }

        self.best_clf = layer_classifiers[best_layer]

        boundary = {
            "w1": float(self.best_clf.coef_[0][0]),
            "w2": float(self.best_clf.coef_[0][1]),
            "b": float(self.best_clf.intercept_[0]),
        }

        best_layer_report = per_layer_results[int(best_layer)]

        self.classifier_results = {
            "best_layer": int(best_layer),
            "selection_metric": "0.5*AUROC + 0.25*Accuracy + 0.25*F1",
            "validation_combined_score": float(best_score),
            "validation_auroc": float(best_layer_report["cv_mean_auroc"]),
            "validation_accuracy": float(best_layer_report["cv_mean_accuracy"]),
            "validation_f1": float(best_layer_report["cv_mean_f1"]),
            "decision_boundary": boundary,
            "per_layer_results": per_layer_results,
        }

        return self.classifier_results

    def apply_2d_classifier(self, test_results):
        if self.best_clf is None or self.classifier_results is None:
            return {"error": "Classifier not fitted. Run fit_2d_classifier first."}

        best_layer = self.classifier_results["best_layer"]

        y_true_all = []
        y_pred_all = []
        y_prob_all = []

        re_ranked_results = []
        total_questions = 0
        correct_2d_predictions = 0

        for item in test_results["results"]:
            if item["layer"] != best_layer:
                continue

            scores_x = item["score_mem_X"]
            scores_y = item["score_syn_Y"]
            labels = item["correct_label_list"]

            X_question = []
            y_true_question = []
            valid_choice_indices = []
            correct_indices_question = []

            for idx, label in enumerate(labels):
                y_val = self._label_to_binary(label)
                if y_val is None:
                    continue

                if y_val == 1:
                    correct_indices_question.append(idx)

                X_question.append([scores_x[idx], scores_y[idx]])
                y_true_question.append(y_val)
                valid_choice_indices.append(idx)

            if not X_question:
                continue

            X_question = np.array(X_question)

            y_pred_question = self.best_clf.predict(X_question)
            y_prob_question = self.best_clf.predict_proba(X_question)[:, 1]

            y_true_all.extend(y_true_question)
            y_pred_all.extend(y_pred_question)
            y_prob_all.extend(y_prob_question)

            confidences = self.best_clf.decision_function(X_question)
            ranked_valid_indices = list(np.argsort(confidences)[::-1])
            ranked_indices_2d = [
                int(valid_choice_indices[j]) for j in ranked_valid_indices
            ]
            predicted_label_2d = ranked_indices_2d[0] if ranked_indices_2d else None

            is_correct_2d = (
                predicted_label_2d in correct_indices_question
                if predicted_label_2d is not None
                else False
            )

            if is_correct_2d:
                correct_2d_predictions += 1
            total_questions += 1

            re_ranked_item = item.copy()
            re_ranked_item["predicted_label_2d"] = (
                int(predicted_label_2d) if predicted_label_2d is not None else None
            )
            re_ranked_item["predicted_ranking_2d"] = ranked_indices_2d
            re_ranked_item["is_correct_2d"] = bool(is_correct_2d)
            re_ranked_item["confidence_scores_2d"] = [float(c) for c in confidences]

            re_ranked_results.append(re_ranked_item)

        if len(y_true_all) == 0:
            return {"error": f"No test data found for layer {best_layer}."}

        metrics = {
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "precision": float(
                precision_score(y_true_all, y_pred_all, zero_division=0)
            ),
            "recall": float(recall_score(y_true_all, y_pred_all, zero_division=0)),
            "f1_score": float(f1_score(y_true_all, y_pred_all, zero_division=0)),
            "auroc": float(roc_auc_score(y_true_all, y_prob_all))
            if len(np.unique(y_true_all)) == 2
            else None,
            "ranking_accuracy_2d": float(correct_2d_predictions / total_questions)
            if total_questions > 0
            else 0.0,
        }

        self.classifier_results["test_metrics"] = metrics
        self.classifier_results["test_label_policy"] = "label_in_{0,1}_only"
        self.classifier_results["re_ranked_detailed_results"] = re_ranked_results

        return self.classifier_results
