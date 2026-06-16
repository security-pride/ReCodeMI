import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluators import LATEvaluator
from loaders import aligned_shuffle_three, pair_data
from model import Model


def unescape_newlines(s):
    return s.encode("utf-8").decode("unicode_escape")


def read_victim_jsonl(path: str):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj["_line_index"] = idx
            records.append(obj)
    return records


def build_code(record, code_field="input_gt"):
    inp = record.get("input", "")
    gt = record.get("gt", "")
    if not isinstance(inp, str) or not isinstance(gt, str):
        return ""
    if code_field == "input":
        return inp
    if code_field == "gt":
        return gt
    if code_field == "input_gt":
        return inp + gt
    raise ValueError(f"Unsupported code_field: {code_field}")


def build_mem_syn_inputs(codes: List[str], prompt_template: str):
    mem_inputs = []
    syn_inputs = []
    for code in codes:
        code = code.strip()
        wrapped = f"```python\n{code}\n```"
        mem_inputs.append(wrapped)
        syn_inputs.append(prompt_template.format(target=wrapped))
    return mem_inputs, syn_inputs


def learn_lat_readers(model, fit_file, validate_file, stimulus, concept, token_tags):
    train, val = LATEvaluator.STIMULUS[stimulus](fit_file, validate_file)

    filtered_tasks = []
    filtered_targets_syn = []
    filtered_targets_mem = []
    filtered_labels_mem = []
    filtered_labels_syn = []

    for i, (task, targets_syn, targets_mem, syn_labels_raw, mem_labels_raw) in enumerate(
        zip(
            train["tasks"],
            train["targets"],
            train["targets_mem"],
            train["syn_labels"],
            train["mem_labels"],
        )
    ):
        label_pairs = list(zip(syn_labels_raw, mem_labels_raw))
        targets_syn, targets_mem, label_pairs = aligned_shuffle_three(
            targets_syn, targets_mem, label_pairs, seed=i
        )
        if not (len(targets_syn) == len(targets_mem) == len(label_pairs)):
            continue

        syn_labels = [pair[0] for pair in label_pairs]
        mem_labels = [pair[1] for pair in label_pairs]

        # repe sign inference requires each question-level label list to contain class 1.
        if 1 not in mem_labels or 1 not in syn_labels:
            continue

        filtered_tasks.append(task)
        filtered_targets_syn.append(targets_syn)
        filtered_targets_mem.append(targets_mem)
        filtered_labels_mem.append(mem_labels)
        filtered_labels_syn.append(syn_labels)

    if not filtered_tasks:
        raise ValueError("No valid training groups remained after label filtering in learn_lat_readers().")

    raw_template = "{target}"
    prompt_template = (
        f"{token_tags['user_start']}Consider the amount of {concept} in the following code.\n\n"
        + "{target}\n"
        + f"{token_tags['user_end']}"
        + f"{token_tags['assistant_start']}The amount of {concept} in the code is "
    )

    # Mem axis uses mem candidates, Syn axis uses syntactic candidates.
    train_pairs_mem = np.concatenate(pair_data(raw_template, filtered_tasks, filtered_targets_mem)).tolist()
    train_pairs_syn = np.concatenate(pair_data(prompt_template, filtered_tasks, filtered_targets_syn)).tolist()

    model.learn_orthogonal_representations(
        train_pairs_mem,
        train_pairs_syn,
        filtered_labels_mem,
        train_labels_syn=filtered_labels_syn,
    )
    return prompt_template


def pick_best_layer(lat_val_path):
    with open(lat_val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    best_layer = None
    if isinstance(val_data, dict):
        vc = val_data.get("validation_classifier")
        if isinstance(vc, dict):
            best_layer = vc.get("best_layer")
    if best_layer is None:
        raise KeyError(f"Cannot find best_layer in {lat_val_path}")
    return int(best_layer)


def score_texts(model, texts, reader, sign, layer, batch_size, max_length, label, log_every, log_fn):
    hidden_layers = list(range(-1, -model.num_hidden_layers(), -1))
    scores = []
    total = len(texts)
    last_print = 0
    start_time = time.time()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        H = model.representation_pipeline(
            chunk,
            hidden_layers=hidden_layers,
            rep_token=-1,
            batch_size=batch_size,
            max_length=max_length,
            padding="longest",
            custom_reader=reader,
        )
        for h in H:
            scores.append(float(h[layer] * sign))
        done = min(start + batch_size, total)
        if done - last_print >= log_every or done == total:
            elapsed = max(time.time() - start_time, 1e-6)
            rate = done / elapsed
            remaining = (total - done) / rate if rate > 0 else 0.0
            msg = f"[{label}] {done}/{total} | {rate:.2f} samples/s | ETA {remaining/60:.1f} min"
            log_fn(msg)
            last_print = done
    return scores


def main():
    parser = argparse.ArgumentParser(description="Compute LAT mem/syn scores for AdvPrompt-MIA victim datasets.")
    parser.add_argument("--fit", type=str, required=True, help="Path to fit.jsonl")
    parser.add_argument("--validate", type=str, required=True, help="Path to validate.jsonl")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g., bigcode/starcoder2-7b")
    parser.add_argument("--model-dir", type=str, help="Optional model cache dir")
    parser.add_argument("--concept", type=str, default="familiarity", help="Concept name")
    parser.add_argument("--stimulus", type=str, default="starcoder2-mia-test", help="Stimulus type")
    parser.add_argument("--user-start", type=str, default="", help="User start tag")
    parser.add_argument("--user-end", type=str, default="", help="User end tag")
    parser.add_argument("--assistant-start", type=str, default="", help="Assistant start tag")
    parser.add_argument("--assistant-end", type=str, default="", help="Assistant end tag")
    parser.add_argument("--lat-val", type=str, help="Path to LAT_val.json to pick best layer")
    parser.add_argument("--layer", type=int, help="Explicit layer to use (overrides --lat-val)")
    parser.add_argument("--train-victim", type=str, default="datasets/AdvPrompt-MIA/train_victim.json")
    parser.add_argument("--test-victim", type=str, default="datasets/AdvPrompt-MIA/test_victim.json")
    parser.add_argument("--group-size", type=int, default=12, help="Group size for victim datasets")
    parser.add_argument("--only-original", action="store_true", help="Only score the original (first) sample in each group")
    parser.add_argument(
        "--code-field",
        choices=["input", "gt", "input_gt"],
        default="input_gt",
        help="Text used for LAT scoring. Default input_gt preserves the original behavior.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--output", type=str, default="datasets/AdvPrompt-MIA/lat_scores_advprompt.jsonl")
    parser.add_argument("--sign-out", type=str, default=None, help="Path to save sign_mem and sign_syn")
    parser.add_argument("--dirs-out", type=str, default=None, help="Path to save mem/syn direction vectors (.npz)")
    parser.add_argument("--log-file", type=str, default="datasets/AdvPrompt-MIA/lat_scores_advprompt.log", help="Path to progress log file")
    args = parser.parse_args()

    token_tags = {
        "user_start": args.user_start.replace("\\n", "\n"),
        "user_end": unescape_newlines(args.user_end),
        "assistant_start": unescape_newlines(args.assistant_start),
        "assistant_end": unescape_newlines(args.assistant_end),
    }

    model = Model(args.model, model_dir=args.model_dir)
    prompt_template = learn_lat_readers(
        model, args.fit, args.validate, args.stimulus, args.concept, token_tags
    )

    if args.layer is not None:
        layer = args.layer
    elif args.lat_val:
        layer = pick_best_layer(args.lat_val)
    else:
        layer = -3

    sign_mem = model.rep_reader_mem.direction_signs[layer]
    sign_syn = model.rep_reader_syn.direction_signs[layer]
    if args.sign_out is None:
        base, ext = os.path.splitext(args.output)
        args.sign_out = f"{base}_signs.txt"
    if args.dirs_out is None:
        base, ext = os.path.splitext(args.output)
        args.dirs_out = f"{base}_directions.npz"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.sign_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.dirs_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    def log_fn(msg):
        print(msg)
        with open(args.log_file, "a", encoding="utf-8") as lf:
            lf.write(msg + "\n")
    with open(args.sign_out, "w", encoding="utf-8") as f:
        f.write(f"layer={layer}\n")
        f.write(f"sign_mem={sign_mem}\n")
        f.write(f"sign_syn={sign_syn}\n")
        f.write("\nall_layers_signs:\n")
        hidden_layers = list(range(-1, -model.num_hidden_layers(), -1))
        for ly in hidden_layers:
            f.write(f"layer={ly}\tsign_mem={model.rep_reader_mem.direction_signs[ly]}\tsign_syn={model.rep_reader_syn.direction_signs[ly]}\n")

    # Save full direction vectors for all layers
    hidden_layers = list(range(-1, -model.num_hidden_layers(), -1))
    mem_dirs = [model.rep_reader_mem.directions[ly][0] for ly in hidden_layers]
    syn_dirs = [model.rep_reader_syn.directions[ly][0] for ly in hidden_layers]
    np.savez_compressed(
        args.dirs_out,
        layers=np.array(hidden_layers, dtype=np.int32),
        mem_dirs=np.stack(mem_dirs, axis=0),
        syn_dirs=np.stack(syn_dirs, axis=0),
    )

    records = []
    for split, path in [("train", args.train_victim), ("test", args.test_victim)]:
        if not os.path.exists(path):
            continue
        recs = read_victim_jsonl(path)
        for r in recs:
            r["_split"] = split
        records.extend(recs)

    filtered = []
    for r in records:
        line_idx = r["_line_index"]
        if args.only_original and (line_idx % args.group_size != 0):
            continue
        filtered.append(r)

    codes = [build_code(r, args.code_field) for r in filtered]
    mem_inputs, syn_inputs = build_mem_syn_inputs(codes, prompt_template)

    mem_scores = score_texts(
        model, mem_inputs, model.rep_reader_mem, sign_mem, layer, args.batch_size, args.max_length, "mem", 100, log_fn
    )
    syn_scores = score_texts(
        model, syn_inputs, model.rep_reader_syn, sign_syn, layer, args.batch_size, args.max_length, "syn", 100, log_fn
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for r, mem_s, syn_s in zip(filtered, mem_scores, syn_scores):
            line_idx = r["_line_index"]
            rec = {
                "split": r["_split"],
                "id": r.get("id"),
                "line_index": int(line_idx),
                "is_original": bool(line_idx % args.group_size == 0),
                "mem_score": float(mem_s),
                "syn_score": float(syn_s),
                "mem_minus_syn": float(mem_s - syn_s),
                "layer": int(layer),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log_fn(f"Wrote {args.output}")
    log_fn(f"Wrote {args.sign_out}")
    log_fn(f"Wrote {args.dirs_out}")


if __name__ == "__main__":
    main()

