#!/usr/bin/env python3
import argparse
import json
import random
import re
from io import BytesIO
from pathlib import Path
from tokenize import COMMENT, ENCODING, ENDMARKER, INDENT, NEWLINE, NL, NUMBER, STRING, tokenize


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc


def load_literals(literals_path: Path):
    if not literals_path.exists():
        return {"str": [], "num": [], "char": []}
    with literals_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def process_string(token: str, literals, special_chars=None):
    if special_chars is None:
        special_chars = {" ": "U+0020", ",": "U+002C"}
    str_quote_options = ["'''", '"""', "'", '"']
    start_quote = ""
    end_quote = ""
    qualifier_regex = r"^[a-zA-Z]+"
    qualifier_match = re.search(qualifier_regex, token)
    qualifier = "" if not qualifier_match else qualifier_match[0]
    token_string = re.sub(qualifier_regex, "", token)
    str_lit = token_string
    for q in str_quote_options:
        if token_string.startswith(q):
            start_quote = q
            str_lit = str_lit[len(q) :]
            if token_string.endswith(q):
                end_quote = q
                str_lit = str_lit[: -len(q)]
            break

    for src, dst in special_chars.items():
        str_lit = str_lit.replace(src, dst)

    if str_lit in literals.get("str", []):
        return f"{qualifier}{start_quote}<STR_LIT:{str_lit}>{end_quote}"
    return f"{qualifier}{start_quote}<STR_LIT>{end_quote}"


def tokenize_code_for_codegen(code: str, literals):
    try:
        token_gen = tokenize(BytesIO(code.encode("utf-8")).readline)
        out_tokens = []
        prev_eol = False
        for toknum, tokval, _, _, _ in token_gen:
            tokval = " ".join(tokval.split())
            if toknum == STRING:
                out_tokens.append(process_string(tokval, literals))
                prev_eol = False
            elif toknum == NUMBER:
                if tokval in literals.get("num", []):
                    out_tokens.append(f"<NUM_LIT:{tokval}>")
                else:
                    out_tokens.append("<NUM_LIT>")
                prev_eol = False
            elif toknum in [NEWLINE, NL]:
                if not prev_eol:
                    out_tokens.append("<EOL>")
                    prev_eol = True
            elif toknum in [COMMENT, INDENT, ENCODING, ENDMARKER] or len(tokval) == 0:
                continue
            else:
                out_tokens.append(tokval)
                prev_eol = False

        if out_tokens and out_tokens[0] == "<EOL>":
            out_tokens = out_tokens[1:]
        if out_tokens and out_tokens[-1] == "<EOL>":
            out_tokens = out_tokens[:-1]
    except Exception:
        # Fallback: keep one-line sequence and mark line breaks with <EOL>.
        cleaned = code.replace("\r\n", "\n").replace("\r", "\n")
        out_tokens = []
        for raw_line in cleaned.split("\n"):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            out_tokens.extend(raw_line.split())
            out_tokens.append("<EOL>")
        if out_tokens and out_tokens[-1] == "<EOL>":
            out_tokens = out_tokens[:-1]

    if not out_tokens:
        out_tokens = ["pass"]
    return "<s> " + " ".join(out_tokens) + " </s>"


def write_text_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            f.write("\n")


def build_one_split(
    split_dir: Path,
    train_files,
    test_files,
    dev_files,
    tokenized_by_file,
):
    split_dir.mkdir(parents=True, exist_ok=True)
    write_text_lines(split_dir / "python_train.txt", train_files)
    write_text_lines(split_dir / "python_test.txt", test_files)
    write_text_lines(split_dir / "python_dev.txt", dev_files)

    codegpt_dir = split_dir / "code-gpt"
    codegpt_dir.mkdir(parents=True, exist_ok=True)
    write_text_lines(codegpt_dir / "train.txt", [tokenized_by_file[x] for x in train_files])
    write_text_lines(codegpt_dir / "test.txt", [tokenized_by_file[x] for x in test_files])
    write_text_lines(codegpt_dir / "dev.txt", [tokenized_by_file[x] for x in dev_files])


def build_lat_sample(index: int, member_name: str, non_member_name: str, source_dir: Path):
    member_code = (source_dir / member_name).read_text(encoding="utf-8")
    non_member_code = (source_dir / non_member_name).read_text(encoding="utf-8")
    return {
        "index": index,
        "sem_code": [member_code, non_member_code],
        "mem_code": [member_code, non_member_code],
        "sem_labels": [1, 0],
        "mem_labels": [1, 0],
        "code_generation": [],
    }


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def build_lat_triplets(output_dir: Path, source_dir: Path, seed: int):
    """Build LAT triplets under output_dir/lat.

    Mapping rule:
    - fit:      shadow_0 train(member) + shadow_0 test(non-member)
    - validate: shadow_0 dev mixed into member/non-member buckets
    - test:     target train(member) + target test(non-member)
    """
    lat_dir = output_dir / "lat"
    rng = random.Random(seed)

    def read_names(path: Path):
        with path.open("r", encoding="utf-8") as f:
            return [x.strip() for x in f if x.strip()]

    def pair_member_nonmember(member_names, non_member_names):
        n = min(len(member_names), len(non_member_names))
        return [
            build_lat_sample(i, member_names[i], non_member_names[i], source_dir)
            for i in range(n)
        ]

    shadow0 = output_dir / "shadow_0"
    target = output_dir / "target"

    fit_member = read_names(shadow0 / "python_train.txt")
    fit_non_member = read_names(shadow0 / "python_test.txt")
    fit_rows = pair_member_nonmember(fit_member, fit_non_member)

    dev_names = read_names(shadow0 / "python_dev.txt")
    val_member = [x for x in dev_names if x.startswith("member_")]
    val_non_member = [x for x in dev_names if x.startswith("non_member_")]
    rng.shuffle(val_member)
    rng.shuffle(val_non_member)
    val_rows = pair_member_nonmember(val_member, val_non_member)

    test_member = read_names(target / "python_train.txt")
    test_non_member = read_names(target / "python_test.txt")
    test_rows = pair_member_nonmember(test_member, test_non_member)

    write_jsonl(lat_dir / "fit.jsonl", fit_rows)
    write_jsonl(lat_dir / "validate.jsonl", val_rows)
    write_jsonl(lat_dir / "test.jsonl", test_rows)

    return {
        "fit_samples": len(fit_rows),
        "validate_samples": len(val_rows),
        "test_samples": len(test_rows),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build NaturalCC-style membership-inference dataset from paired code_1/code_2 JSONL."
    )
    parser.add_argument(
        "--input-jsonl",
        default="datasets/stack2/original_stack2_apps.jsonl",
        help="Input JSONL path containing code_1 (member) and code_2 (non-member).",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/stack2/naturalcc",
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--num-shadow-models",
        type=int,
        default=10,
        help="Number of shadow_i splits to create.",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.1,
        help="Dev set ratio sampled from each class (member/non-member).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_jsonl = Path(args.input_jsonl)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[4]
    literals_path = (
        repo_root
        / "naturalcc/examples/code-membership-inference/CodeGPT/dataset/py150/literals.json"
    )
    literals = load_literals(literals_path)

    pairs = []
    for obj in iter_jsonl(input_jsonl):
        if "code_1" not in obj or "code_2" not in obj:
            raise KeyError("Each JSONL record must contain 'code_1' and 'code_2'.")
        pairs.append((obj["code_1"], obj["code_2"]))

    if not pairs:
        raise ValueError(f"No valid samples found in {input_jsonl}")

    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)

    member_files = []
    non_member_files = []
    tokenized_by_file = {}
    for idx, (member_code, non_member_code) in enumerate(pairs):
        member_name = f"member_{idx:06d}.py"
        non_member_name = f"non_member_{idx:06d}.py"

        (source_dir / member_name).write_text(member_code, encoding="utf-8")
        (source_dir / non_member_name).write_text(non_member_code, encoding="utf-8")

        member_files.append(member_name)
        non_member_files.append(non_member_name)
        tokenized_by_file[member_name] = tokenize_code_for_codegen(member_code, literals)
        tokenized_by_file[non_member_name] = tokenize_code_for_codegen(non_member_code, literals)

    per_class_dev = max(1, int(len(pairs) * args.dev_ratio))
    base_rng = random.Random(args.seed)

    # target split: keep all code_1 as member(train), all code_2 as non-member(test)
    target_member = list(member_files)
    target_non_member = list(non_member_files)
    dev_member = list(member_files)
    dev_non_member = list(non_member_files)
    base_rng.shuffle(dev_member)
    base_rng.shuffle(dev_non_member)
    target_dev = dev_member[:per_class_dev] + dev_non_member[:per_class_dev]
    base_rng.shuffle(target_dev)
    build_one_split(
        output_dir / "target",
        target_member,
        target_non_member,
        target_dev,
        tokenized_by_file,
    )

    # shadow splits: same pools with deterministic shuffles (close to the paper repo style)
    for i in range(args.num_shadow_models):
        rng = random.Random(args.seed + i + 1)
        shadow_member = list(member_files)
        shadow_non_member = list(non_member_files)
        rng.shuffle(shadow_member)
        rng.shuffle(shadow_non_member)
        shadow_dev = shadow_member[:per_class_dev] + shadow_non_member[:per_class_dev]
        rng.shuffle(shadow_dev)
        build_one_split(
            output_dir / f"shadow_{i}",
            shadow_member,
            shadow_non_member,
            shadow_dev,
            tokenized_by_file,
        )

    lat_meta = None
    if args.num_shadow_models >= 1:
        lat_meta = build_lat_triplets(output_dir, source_dir, args.seed)

    meta = {
        "input_jsonl": str(input_jsonl),
        "output_dir": str(output_dir),
        "num_pairs": len(pairs),
        "member_source": "code_1",
        "non_member_source": "code_2",
        "num_shadow_models": args.num_shadow_models,
        "per_class_dev_samples": per_class_dev,
        "generated_subdirs": ["target"] + [f"shadow_{i}" for i in range(args.num_shadow_models)],
        "notes": [
            "python_train.txt / python_test.txt / python_dev.txt list filenames under source/.",
            "code-gpt/*.txt are tokenized one-sample-per-line files for NaturalCC CodeGPT pipeline.",
        ],
    }
    if lat_meta is not None:
        meta["lat_triplets"] = {
            "dir": str(output_dir / "lat"),
            "mapping": {
                "fit": "shadow_0: train(member) + test(non-member)",
                "validate": "shadow_0: dev member/non-member pairs",
                "test": "target: train(member) + test(non-member)",
            },
            **lat_meta,
        }
    (output_dir / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Done. Wrote dataset to: {output_dir}")
    print(f"Pairs: {len(pairs)}, per-class dev size: {per_class_dev}")


if __name__ == "__main__":
    main()
