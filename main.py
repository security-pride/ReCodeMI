# Portions of this file are adapted from code-repe:
# https://github.com/sanadlab/code-repe

import os
import time
import argparse
from collections import defaultdict

import numpy as np

from evaluators import LATEvaluator
from model import Model
from loaders import (
    aligned_shuffle_three_with_index_map,
    load_triplets_jsonl,
    pair_data,
)


def _build_prompt_template(token_tags, concept):
    return (
        f"{token_tags['user_start']}Consider the amount of {concept} in the following code.\n\n"
        + "{target}\n"
        + f"{token_tags['user_end']}"
        + f"{token_tags['assistant_start']}The amount of {concept} in the code is "
    )


def _shuffle_training_triplet(triplet):
    tasks = triplet["tasks"]
    syn_targets_all = triplet["targets"]
    mem_targets_all = triplet["targets_mem"]
    syn_labels_all = triplet.get("syn_labels", triplet.get("labels", []))
    mem_labels_all = triplet.get("mem_labels", triplet.get("labels", []))

    shuffled_targets = []
    shuffled_targets_mem = []
    shuffled_syn_labels = []
    shuffled_mem_labels = []

    for i, (syn_targets, mem_targets, syn_labels, mem_labels) in enumerate(
        zip(syn_targets_all, mem_targets_all, syn_labels_all, mem_labels_all)
    ):
        if not (
            len(syn_targets)
            == len(mem_targets)
            == len(syn_labels)
            == len(mem_labels)
        ):
            raise ValueError(
                f"Training sample {i} length mismatch: "
                f"len(syn_targets)={len(syn_targets)}, "
                f"len(mem_targets)={len(mem_targets)}, "
                f"len(syn_labels)={len(syn_labels)}, "
                f"len(mem_labels)={len(mem_labels)}"
            )

        label_pairs = list(zip(syn_labels, mem_labels))
        (
            syn_targets_s,
            mem_targets_s,
            label_pairs_s,
            _shuffled_to_orig,
            _orig_to_shuffled,
        ) = aligned_shuffle_three_with_index_map(
            list(syn_targets),
            list(mem_targets),
            label_pairs,
            seed=i,
        )

        shuffled_targets.append(syn_targets_s)
        shuffled_targets_mem.append(mem_targets_s)
        shuffled_syn_labels.append([int(p[0]) for p in label_pairs_s])
        shuffled_mem_labels.append([int(p[1]) for p in label_pairs_s])

    return {
        "tasks": tasks,
        "targets": shuffled_targets,
        "targets_mem": shuffled_targets_mem,
        "syn_labels": shuffled_syn_labels,
        "mem_labels": shuffled_mem_labels,
        "sample_ids": triplet.get("sample_ids", list(range(len(tasks)))),
    }


def run_fit_stage(
    model_name,
    concept,
    token_tags,
    stimulus,
    fit_file,
    validate_file,
    model_dir=None,
):
    model = Model(model_name, model_dir=model_dir)

    fit_triplet_raw, val_triplet = LATEvaluator.STIMULUS[stimulus](fit_file, validate_file)
    fit_triplet = _shuffle_training_triplet(fit_triplet_raw)

    raw_template = "{target}"
    prompt_template = _build_prompt_template(token_tags, concept)

    train_pairs_mem = np.concatenate(
        pair_data(raw_template, fit_triplet["tasks"], fit_triplet["targets_mem"])
    ).tolist()
    train_pairs_syn = np.concatenate(
        pair_data(prompt_template, fit_triplet["tasks"], fit_triplet["targets"])
    ).tolist()

    train_labels_mem = fit_triplet["mem_labels"]
    train_labels_syn = fit_triplet["syn_labels"]

    print("Starting Orthogonal PCA fitting...")
    pca_start = time.time()
    model.learn_orthogonal_representations(
        train_pairs_mem,
        train_pairs_syn,
        train_labels_mem,
        train_labels_syn,
    )
    pca_fit_time = time.time() - pca_start
    print(f"PCA fitting completed in {pca_fit_time:.2f} seconds")

    return {
        "model": model,
        "fit_triplet": fit_triplet,
        "val_triplet": val_triplet,
        "timing": {"pca_fit_seconds": pca_fit_time},
    }


def run_validate_stage(
    model,
    concept,
    token_tags,
    val_triplet,
    template,
    output_dir,
    no_wrapper=False,
):
    lat_eval = LATEvaluator(
        model,
        concept,
        template=template,
        target_field_name="code",
        token_tags=token_tags,
        no_wrapper=no_wrapper,
    )

    print("Starting validation for layer selection...")
    validation_start = time.time()
    val_results = lat_eval.evaluate(
        val_triplet["tasks"],
        val_triplet["targets"],
        val_triplet.get("syn_labels", val_triplet.get("labels", [])),
        targets_mem_list=val_triplet["targets_mem"],
        mem_labels_list=val_triplet.get("mem_labels", val_triplet.get("labels", [])),
        sample_ids=val_triplet.get("sample_ids"),
    )

    print("Fitting 2D Linear Classifier on Validation Data...")
    val_clf_info = lat_eval.fit_2d_classifier(val_results)
    if "error" not in val_clf_info:
        print(
            f"-> 2D Classifier Best Layer: {val_clf_info['best_layer']} "
            f"(Val AUROC: {val_clf_info['validation_auroc']:.2%}, "
            f"Val Acc: {val_clf_info['validation_accuracy']:.2%}, "
            f"Val F1: {val_clf_info['validation_f1']:.2%}, "
            f"Combined: {val_clf_info['validation_combined_score']:.4f})"
        )

    val_results["validation_classifier"] = val_clf_info

    validation_time = time.time() - validation_start
    print(f"Validation completed in {validation_time:.2f} seconds")

    val_results["timing"] = {
        "validation_seconds": validation_time,
        "validation_per_sample": validation_time / len(val_triplet["targets"])
        if len(val_triplet["targets"]) > 0
        else 0,
    }

    LATEvaluator.save_to_json(val_results, os.path.join(output_dir, "LAT_val.json"))

    return {
        "evaluator": lat_eval,
        "val_results": val_results,
        "timing": {"validation_seconds": validation_time},
    }


def run_test_stage(
    evaluator,
    tasks,
    targets_syn_list,
    targets_mem_list,
    syn_labels_list,
    mem_labels_list,
    sample_ids,
    output_dir,
):
    print("Starting test evaluation...")
    test_start = time.time()
    results = evaluator.evaluate(
        tasks,
        targets_syn_list,
        syn_labels_list,
        targets_mem_list=targets_mem_list,
        mem_labels_list=mem_labels_list,
        sample_ids=sample_ids,
    )
    test_inference_time = time.time() - test_start
    print(f"Test evaluation completed in {test_inference_time:.2f} seconds")

    print("Applying 2D Linear Classifier to Test Data...")
    final_clf_info = evaluator.apply_2d_classifier(results)

    if "error" not in final_clf_info:
        print("\n--- 2D Classifier Final Results ---")
        print(f"Test Accuracy:  {final_clf_info['test_metrics']['accuracy']:.2%}")
        print(f"Test Precision: {final_clf_info['test_metrics']['precision']:.2%}")
        print(f"Test Recall:    {final_clf_info['test_metrics']['recall']:.2%}")
        print(f"Test F1-Score:  {final_clf_info['test_metrics']['f1_score']:.2%}")
        print(f"Test AUROC:     {final_clf_info['test_metrics']['auroc']:.2%}")
        print("-----------------------------------\n")

    results["linear_classifier"] = final_clf_info

    rank_ks = [1, 2, 3, 4, 5]
    detailed = results["results"]
    layer_groups = defaultdict(list)
    for entry in detailed:
        layer_groups[entry["layer"]].append(entry)

    acc_at_k_by_layer = {}
    for layer, entries in layer_groups.items():
        accs = {}
        for k in rank_ks:
            num = 0
            denom = 0
            for entry in entries:
                if not entry.get("correct_label"):
                    continue
                top_k = entry["predicted_ranking"][:k]
                if any(idx in top_k for idx in entry["correct_label"]):
                    num += 1
                denom += 1
            accs[f"accuracy_at_{k}"] = num / denom if denom > 0 else 0.0
        acc_at_k_by_layer[layer] = accs

    print("Layer-wise accuracy at first correct within rank k:")
    for layer in sorted(acc_at_k_by_layer):
        print(f"Layer {layer}: {acc_at_k_by_layer[layer]}")

    results["accuracy_at_k_by_layer"] = acc_at_k_by_layer
    results["timing"] = {
        "test_inference_seconds": test_inference_time,
        "test_inference_per_sample": test_inference_time / len(tasks)
        if len(tasks) > 0
        else 0,
        "num_test_samples": len(tasks),
    }

    LATEvaluator.save_to_json(results, os.path.join(output_dir, "LAT.json"))

    return {
        "test_results": results,
        "timing": {"test_inference_seconds": test_inference_time},
    }


def LAT_evaluation(
    model_name,
    tasks,
    targets_syn_list,
    targets_mem_list,
    syn_labels_list,
    mem_labels_list,
    template,
    concept,
    output_dir,
    stimulus,
    token_tags=None,
    model_dir=None,
    fit_file=None,
    validate_file=None,
    no_wrapper=False,
    sample_ids=None,
):
    fit_ctx = run_fit_stage(
        model_name=model_name,
        concept=concept,
        token_tags=token_tags,
        stimulus=stimulus,
        fit_file=fit_file,
        validate_file=validate_file,
        model_dir=model_dir,
    )

    validate_ctx = run_validate_stage(
        model=fit_ctx["model"],
        concept=concept,
        token_tags=token_tags,
        val_triplet=fit_ctx["val_triplet"],
        template=template,
        output_dir=output_dir,
        no_wrapper=no_wrapper,
    )

    test_ctx = run_test_stage(
        evaluator=validate_ctx["evaluator"],
        tasks=tasks,
        targets_syn_list=targets_syn_list,
        targets_mem_list=targets_mem_list,
        syn_labels_list=syn_labels_list,
        mem_labels_list=mem_labels_list,
        sample_ids=sample_ids,
        output_dir=output_dir,
    )

    pca_fit_time = fit_ctx["timing"]["pca_fit_seconds"]
    validation_time = validate_ctx["timing"]["validation_seconds"]
    test_inference_time = test_ctx["timing"]["test_inference_seconds"]
    total_lat_time = pca_fit_time + validation_time + test_inference_time

    print("\n" + "=" * 60)
    print("LAT TIMING SUMMARY")
    print("=" * 60)
    print(f"PCA Fitting Time:           {pca_fit_time:>10.2f} seconds")
    print(f"Validation Time:            {validation_time:>10.2f} seconds")
    print(f"Test Inference Time:        {test_inference_time:>10.2f} seconds")
    print(
        f"  Per sample:               {test_inference_time / len(tasks):>10.4f} seconds"
    )
    print(f"Total LAT Time:             {total_lat_time:>10.2f} seconds")
    print("=" * 60 + "\n")



def unescape_newlines(s):
    return s.encode("utf-8").decode("unicode_escape")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LAT evaluations.")
    parser.add_argument("--fit", type=str, required=True, help="Path to the fit .jsonl file")
    parser.add_argument(
        "--validate", type=str, required=True, help="Path to the validate .jsonl"
    )
    parser.add_argument("--test", type=str, required=True, help="Path to the test .jsonl file")
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save evaluation results"
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name to evaluate")
    parser.add_argument(
        "--model-dir", type=str, help="Path to the directory containing model files"
    )
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        help="Concept for LAT evaluation (e.g., 'familiarity')",
    )
    parser.add_argument(
        "-s",
        "--stimulus",
        type=str,
        required=True,
        help="Stimulus type for LAT evaluation data loading",
    )

    parser.add_argument("--user-start", type=str, default="", help="User start tag")
    parser.add_argument("--user-end", type=str, default="", help="User end tag")
    parser.add_argument(
        "--assistant-start", type=str, default="", help="Assistant start tag"
    )
    parser.add_argument("--assistant-end", type=str, default="", help="Assistant end tag")
    parser.add_argument(
        "--no-wrapper",
        dest="no_wrapper",
        action="store_true",
        help="If set, do not wrap prompt in meta_template",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model
    if args.model_dir is not None:
        args.model_dir = os.path.expandvars(args.model_dir)
        if not os.path.exists(args.model_dir):
            raise ValueError(f"Model directory {args.model_dir} does not exist.")
    else:
        args.model_dir = None

    triplet = load_triplets_jsonl(args.test)
    tasks = triplet["tasks"]
    targets_syn_list = triplet["targets"]
    targets_mem_list = triplet["targets_mem"]
    syn_labels_list = triplet.get("syn_labels", triplet.get("labels", []))
    mem_labels_list = triplet.get("mem_labels", triplet.get("labels", []))
    sample_ids = triplet.get("sample_ids", list(range(len(tasks))))

    concept = args.concept
    template = "{target}"
    token_tags = {
        "user_start": args.user_start.replace("\\n", "\n"),
        "user_end": unescape_newlines(args.user_end),
        "assistant_start": unescape_newlines(args.assistant_start),
        "assistant_end": unescape_newlines(args.assistant_end),
    }
    print(token_tags)

    LAT_evaluation(
        model_name,
        tasks,
        targets_syn_list,
        targets_mem_list,
        syn_labels_list,
        mem_labels_list,
        template,
        concept,
        args.output_dir,
        args.stimulus,
        token_tags=token_tags,
        model_dir=args.model_dir,
        fit_file=args.fit,
        validate_file=args.validate,
        no_wrapper=args.no_wrapper,
        sample_ids=sample_ids,
    )
