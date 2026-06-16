import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from repe import repe_pipeline_registry


logger = logging.getLogger(__name__)


def resolve_model_path(path_or_id: str) -> str:
    """Resolve local HF cache root to a concrete snapshot when needed.

    If input is not a local path, return as-is (HF model id).
    """
    p = Path(path_or_id).expanduser()
    if not p.exists():
        return path_or_id
    if (p / "config.json").exists():
        return str(p)
    snapshots = p / "snapshots"
    if snapshots.exists() and snapshots.is_dir():
        candidates = [x for x in snapshots.iterdir() if x.is_dir() and (x / "config.json").exists()]
        if candidates:
            candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(candidates[0])
    return str(p)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def aligned_shuffle_three(list1, list2, list3, seed=0):
    random.seed(seed)
    combined = list(zip(list1, list2, list3))
    random.shuffle(combined)
    if not combined:
        return [], [], []
    shuffled1, shuffled2, shuffled3 = zip(*combined)
    return list(shuffled1), list(shuffled2), list(shuffled3)


def load_triplets_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            mem_labels = item["mem_labels"]
            syn_labels = item["syn_labels"]
            item["labels"] = [[m, s] for m, s in zip(mem_labels, syn_labels)]
            data.append(item)

    return {
        "tasks": [item.get("task", "") for item in data],
        "targets": [item["syn_code"] for item in data],
        "targets_mem": [item["mem_code"] for item in data],
        "labels": [item["labels"] for item in data],
    }


def pair_data(template, tasks, targets_list):
    data = []
    for _task, targets in zip(tasks, targets_list):
        paired_data = []
        for target in targets:
            paired_data.append(template.format(target="```python\n" + target.strip() + "\n```"))
        data.append(paired_data)
    return data


def parse_natcc_line_to_code(line: str) -> str:
    s = line.strip()
    if s.startswith("<s>"):
        s = s[3:].strip()
    if s.endswith("</s>"):
        s = s[:-4].strip()
    s = s.replace(" <EOL> ", "\n")
    s = s.replace(" <EOL>", "\n")
    s = s.replace("<EOL> ", "\n")
    s = s.replace("<EOL>", "\n")
    return s


def load_codegen_line_samples(data_dir):
    out = {}
    for split, label in (("train", 1), ("test", 0)):
        fp = os.path.join(data_dir, f"{split}.txt")
        codes = []
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                codes.append(parse_natcc_line_to_code(line))
        out[split] = {
            "codes": codes,
            "labels": [label for _ in codes],
        }
    return out


def fit_mem_syn_readers(
    model,
    tokenizer,
    fit_jsonl,
    validate_jsonl,
    concept,
    max_length,
    batch_size,
    seed,
):
    _ = validate_jsonl
    train = load_triplets_jsonl(fit_jsonl)

    for i, (targets_syn, targets_mem, labels) in enumerate(
        zip(train["targets"], train["targets_mem"], train["labels"])
    ):
        (
            train["targets"][i],
            train["targets_mem"][i],
            train["labels"][i],
        ) = aligned_shuffle_three(targets_syn, targets_mem, labels, seed=i)

    raw_template = "{target}"
    prompt_template = (
        f"Consider the amount of {concept} in the following code.\n\n"
        + "{target}\n"
        + f"The amount of {concept} in the code is "
    )

    train_pairs_mem = np.concatenate(
        pair_data(raw_template, train["tasks"], train["targets_mem"])
    ).tolist()
    train_pairs_syn = np.concatenate(
        pair_data(prompt_template, train["tasks"], train["targets"])
    ).tolist()

    train_labels_mem = [[lbl[0] for lbl in sublist] for sublist in train["labels"]]
    train_labels_syn = [[lbl[1] for lbl in sublist] for sublist in train["labels"]]

    rep_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))

    rep_reader_mem = rep_pipeline.get_directions(
        train_pairs_mem,
        rep_token=-1,
        hidden_layers=hidden_layers,
        n_difference=1,
        train_labels=train_labels_mem,
        direction_method="pca",
        direction_finder_kwargs={"n_components": 1},
        batch_size=batch_size,
        max_length=max_length,
        truncation=True,
        padding="longest",
    )

    rep_reader_syn = rep_pipeline.get_directions(
        train_pairs_syn,
        rep_token=-1,
        hidden_layers=hidden_layers,
        n_difference=1,
        train_labels=train_labels_syn,
        direction_method="pca",
        direction_finder_kwargs={"n_components": 1},
        batch_size=batch_size,
        max_length=max_length,
        truncation=True,
        padding="longest",
    )

    for layer in hidden_layers:
        v_mem = rep_reader_mem.directions[layer][0]
        v_fam = rep_reader_syn.directions[layer][0]
        proj = (np.dot(v_fam, v_mem) / np.dot(v_mem, v_mem)) * v_mem
        v_syn = v_fam - proj
        v_syn = v_syn / np.linalg.norm(v_syn)
        rep_reader_syn.directions[layer][0] = v_syn

    return rep_pipeline, rep_reader_mem, rep_reader_syn, hidden_layers


def compute_scores_for_codes(
    rep_pipeline,
    rep_reader_mem,
    rep_reader_syn,
    hidden_layers,
    codes,
    concept,
    max_length,
    batch_size,
):
    raw_template = "{scenario}"
    prompt_template = (
        f"Consider the amount of {concept} in the following code.\n\n"
        + "{scenario}\n"
        + f"The amount of {concept} in the code is "
    )

    eval_data_mem = []
    eval_data_syn = []
    for code in codes:
        scenario = "```python\n" + code.strip() + "\n```"
        eval_data_mem.append(raw_template.format(scenario=scenario))
        eval_data_syn.append(prompt_template.format(scenario=scenario))

    h_mem = rep_pipeline(
        eval_data_mem,
        rep_token=-1,
        hidden_layers=hidden_layers,
        rep_reader=rep_reader_mem,
        batch_size=batch_size,
        max_length=max_length,
        truncation=True,
        padding="longest",
    )
    h_syn = rep_pipeline(
        eval_data_syn,
        rep_token=-1,
        hidden_layers=hidden_layers,
        rep_reader=rep_reader_syn,
        batch_size=batch_size,
        max_length=max_length,
        truncation=True,
        padding="longest",
    )

    num_layers = len(hidden_layers)
    scores = np.zeros((len(codes), num_layers, 2), dtype=np.float32)

    for li, layer in enumerate(hidden_layers):
        sign_mem = rep_reader_mem.direction_signs[layer]
        sign_syn = rep_reader_syn.direction_signs[layer]
        for i in range(len(codes)):
            scores[i, li, 0] = float(h_mem[i][layer] * sign_mem)
            scores[i, li, 1] = float(h_syn[i][layer] * sign_syn)

    return scores


def choose_best_layer_with_validate(
    rep_pipeline,
    rep_reader_mem,
    rep_reader_syn,
    hidden_layers,
    validate_jsonl,
    concept,
    max_length,
    batch_size,
    seed,
):
    _ = seed
    val = load_triplets_jsonl(validate_jsonl)

    X_by_layer = defaultdict(list)
    y_by_layer = defaultdict(list)

    for qid, (syn_targets, mem_targets, labels) in enumerate(
        zip(val["targets"], val["targets_mem"], val["labels"])
    ):
        syn_targets, mem_targets, labels = aligned_shuffle_three(syn_targets, mem_targets, labels, seed=qid)

        sample_scores = compute_scores_for_codes(
            rep_pipeline,
            rep_reader_mem,
            rep_reader_syn,
            hidden_layers,
            syn_targets,
            concept,
            max_length,
            batch_size,
        )

        for li, layer in enumerate(hidden_layers):
            for i, lbl in enumerate(labels):
                y = int(lbl[0])
                if y not in (0, 1):
                    continue
                X_by_layer[layer].append(sample_scores[i, li, :].tolist())
                y_by_layer[layer].append(y)

    best_layer = hidden_layers[0]
    best_score = -1.0
    best_stats = {"auc": 0.0, "acc": 0.0, "f1": 0.0}

    for layer in hidden_layers:
        X = np.array(X_by_layer[layer], dtype=np.float32)
        y = np.array(y_by_layer[layer], dtype=np.int64)
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            continue

        clf = LogisticRegression(class_weight="balanced", max_iter=1000)
        clf.fit(X, y)

        pred = clf.predict(X)
        prob = clf.predict_proba(X)[:, 1]

        auc = float(roc_auc_score(y, prob))
        acc = float(accuracy_score(y, pred))
        f1 = float(f1_score(y, pred, zero_division=0))
        score = 0.5 * auc + 0.25 * acc + 0.25 * f1

        if score > best_score:
            best_score = score
            best_layer = layer
            best_stats = {"auc": auc, "acc": acc, "f1": f1}

    logger.info(
        "selected best layer from validate: %s (combined=%.4f, auc=%.4f, acc=%.4f, f1=%.4f)",
        best_layer,
        best_score,
        best_stats["auc"],
        best_stats["acc"],
        best_stats["f1"],
    )
    return best_layer


def dump_mem_syn_pth(output_path, scores, label):
    records = []
    for i in range(scores.shape[0]):
        records.append(
            {
                "mem_score": torch.tensor(float(scores[i, 0])),
                "syn_score": torch.tensor(float(scores[i, 1])),
                "label": torch.tensor(int(label)),
            }
        )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(records, output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", required=True, type=str)
    parser.add_argument("--validate", required=True, type=str)
    parser.add_argument("--data_dir", required=True, type=str,
                        help="membership data root, e.g. datasets/stack2/contrast/1000_original/naturalcc")
    parser.add_argument("--model_name", required=True, type=str,
                        help="HF id, e.g. JetBrains/Mellum-4b-base")
    parser.add_argument("--model_cache", default="", type=str,
                        help="Optional local cache root for HF models")
    parser.add_argument("--concept", default="familiarity", type=str)
    parser.add_argument("--num_shadow", default=10, type=int)
    parser.add_argument("--max_length", default=2048, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--shadow_ranks_subdir",
        default="ranks",
        type=str,
        help="Rank subdir for shadow_i/code-gpt (e.g. ranks).",
    )
    parser.add_argument(
        "--target_ranks_subdir",
        default="ranks",
        type=str,
        help="Rank subdir for target/code-gpt (e.g. ranks_starcoder2).",
    )
    parser.add_argument(
        "--skip_target",
        action="store_true",
        help="Only write shadow mem/syn files; do not compute target mem/syn files.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    set_seed(args.seed)
    repe_pipeline_registry()

    model_name = resolve_model_path(args.model_name)
    cache_dir = args.model_cache if args.model_cache else None
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tok_err = None
    tokenizer = None
    for use_fast in (True, False):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                use_fast=use_fast,
                padding_side="left",
                trust_remote_code=True,
            )
            logger.info("Loaded tokenizer with use_fast=%s", use_fast)
            break
        except Exception as exc:  # noqa: BLE001
            tok_err = exc
            logger.warning("Tokenizer load failed with use_fast=%s: %s", use_fast, exc)
    if tokenizer is None:
        raise tok_err
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rep_pipeline, rep_reader_mem, rep_reader_syn, hidden_layers = fit_mem_syn_readers(
        model=model,
        tokenizer=tokenizer,
        fit_jsonl=args.fit,
        validate_jsonl=args.validate,
        concept=args.concept,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    best_layer = choose_best_layer_with_validate(
        rep_pipeline=rep_pipeline,
        rep_reader_mem=rep_reader_mem,
        rep_reader_syn=rep_reader_syn,
        hidden_layers=hidden_layers,
        validate_jsonl=args.validate,
        concept=args.concept,
        max_length=args.max_length,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    def process_model_split(model_key, ranks_subdir):
        split_dir = os.path.join(args.data_dir, model_key, "code-gpt")
        samples = load_codegen_line_samples(split_dir)

        train_scores = compute_scores_for_codes(
            rep_pipeline,
            rep_reader_mem,
            rep_reader_syn,
            hidden_layers,
            samples["train"]["codes"],
            args.concept,
            args.max_length,
            args.batch_size,
        )
        test_scores = compute_scores_for_codes(
            rep_pipeline,
            rep_reader_mem,
            rep_reader_syn,
            hidden_layers,
            samples["test"]["codes"],
            args.concept,
            args.max_length,
            args.batch_size,
        )

        li = hidden_layers.index(best_layer)
        train_xy = train_scores[:, li, :]
        test_xy = test_scores[:, li, :]

        rank_dir = os.path.join(split_dir, ranks_subdir)
        dump_mem_syn_pth(os.path.join(rank_dir, "member_mem_syn.pth"), train_xy, label=1)
        dump_mem_syn_pth(os.path.join(rank_dir, "non_member_mem_syn.pth"), test_xy, label=0)

        logger.info("saved mem/syn for %s at %s", model_key, rank_dir)

    for i in range(args.num_shadow):
        process_model_split(f"shadow_{i}", args.shadow_ranks_subdir)
    if not args.skip_target:
        process_model_split("target", args.target_ranks_subdir)

    logger.info("all done")


if __name__ == "__main__":
    main()

