#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


LEVELS: List[Tuple[str, List[str]]] = [
    ("v1", ["operator_rewrite"]),
    (
        "v2",
        ["renaming", "operator_rewrite"],
    ),
    (
        "v3",
        [
            "renaming",
            "operator_rewrite",
            "structure_rewrite",
        ],
    ),
    (
        "v4",
        [
            "renaming",
            "operator_rewrite",
            "structure_rewrite",
            "formatter_rewrite",
        ],
    )
]


def load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if isinstance(row, dict):
                yield row


def ensure_list_str(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            else:
                out.append("" if item is None else str(item))
        return out
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    return [str(value)]


def first_code(value: Any) -> str:
    arr = ensure_list_str(value)
    return arr[0] if arr else ""


def apply_one_group(
    code: str,
    group_name: str,
    group_ns: Dict[str, Any],
    code_transformer: Any,
    preserved_names: List[str],
) -> str:
    if not code:
        return ""
    ns = group_ns[group_name]
    rw = code_transformer.build_rewriter(ns, preserved_names)
    fmt = code_transformer.build_formatter(ns)
    return code_transformer.transform_with_obfupy(code, rw, fmt)


def order_groups_for_execution(groups: List[str]) -> List[str]:
    # Ensure Formatter runs after all Rewriter groups so formatting effects are kept.
    ordered = [group for group in groups if group != "formatter_rewrite"]
    if "formatter_rewrite" in groups:
        ordered.append("formatter_rewrite")
    return ordered


def apply_group_chain(
    source_code: str,
    groups: List[str],
    group_ns: Dict[str, Any],
    code_transformer: Any,
    preserved_names: List[str],
) -> str:
    current = source_code
    for group in order_groups_for_execution(groups):
        try:
            transformed = apply_one_group(current, group, group_ns, code_transformer, preserved_names)
            if isinstance(transformed, str) and transformed.strip():
                current = transformed
        except Exception:
            # Keep previous output on transform failure for this group.
            pass
    return current


def build_cumulative_variants(
    source_code: str,
    levels: List[Tuple[str, List[str]]],
    group_ns: Dict[str, Any],
    code_transformer: Any,
    preserved_names: List[str],
) -> List[str]:
    variants: List[str] = []
    for _, groups in levels:
        variants.append(
            apply_group_chain(source_code, groups, group_ns, code_transformer, preserved_names)
        )
    return variants


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build RQ2 cumulative transformed datasets. "
            "Transforms are applied via datasets/code_transformer.py groups."
        )
    )
    p.add_argument(
        "--input-root",
        type=Path,
        default=Path("datasets/stack2"),
        help="Directory containing fit.jsonl / validate.jsonl / test.jsonl.",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/stack2/RQ2"),
        help="Output root to create v1..v4 subfolders.",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="fit,validate,test",
        help="Comma separated split names.",
    )
    p.add_argument(
        "--code-transformer-path",
        type=Path,
        default=Path("datasets/code_transformer.py"),
        help="Path to datasets/code_transformer.py",
    )
    p.add_argument(
        "--preserved-names",
        type=str,
        default="__init__,__name__,__main__",
        help="Comma separated preserved names.",
    )
    p.add_argument("--progress-every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.code_transformer_path.is_file():
        raise FileNotFoundError(f"code_transformer.py not found: {args.code_transformer_path}")

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    if not splits:
        raise ValueError("No splits provided")

    code_transformer = load_module(
        args.code_transformer_path.resolve(),
        "stack2_code_transformer_rq2",
    )

    required_groups = ["formatter_rewrite", "operator_rewrite", "renaming", "structure_rewrite"]
    group_ns = code_transformer.build_group_namespaces(required_groups)
    preserved_names = [x.strip() for x in args.preserved_names.split(",") if x.strip()]

    for level_name, _ in LEVELS:
        out_dir = args.output_root / level_name
        out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        in_file = args.input_root / f"{split}.jsonl"
        if not in_file.is_file():
            raise FileNotFoundError(f"Missing split file: {in_file}")

        out_files = {
            level_name: (args.output_root / level_name / f"{split}.jsonl").open("w", encoding="utf-8")
            for level_name, _ in LEVELS
        }

        written = 0
        try:
            for row in iter_jsonl(in_file):
                if split != "test":
                    # fit/validate are copied as-is for every level.
                    for level_name, _ in LEVELS:
                        out_files[level_name].write(json.dumps(row, ensure_ascii=False) + "\n")
                else:
                    # Only test.jsonl is transformed.
                    # The new transformed candidate is appended with label=2.
                    syn_list = ensure_list_str(row.get("syn_code"))
                    mem_list = ensure_list_str(row.get("mem_code"))
                    syn_src = first_code(syn_list)
                    mem_src = first_code(mem_list)

                    syn_variants = build_cumulative_variants(
                        syn_src, LEVELS, group_ns, code_transformer, preserved_names
                    )
                    mem_variants = build_cumulative_variants(
                        mem_src, LEVELS, group_ns, code_transformer, preserved_names
                    )

                    for level_idx, (level_name, _) in enumerate(LEVELS):
                        out = dict(row)
                        out["syn_code"] = syn_list + [syn_variants[level_idx]]
                        out["mem_code"] = mem_list + [mem_variants[level_idx]]
                        out["syn_labels"] = [1, 0, 2]
                        out["mem_labels"] = [1, 0, 2]
                        out_files[level_name].write(json.dumps(out, ensure_ascii=False) + "\n")

                written += 1
                if args.progress_every > 0 and written % args.progress_every == 0:
                    print(f"[{split}] processed={written}")
        finally:
            for f in out_files.values():
                f.close()

        print(f"[{split}] done, rows={written}")

    print("Done. Generated folders:")
    for level_name, groups in LEVELS:
        print(f"- {args.output_root / level_name} (groups={'+'.join(groups)})")


if __name__ == "__main__":
    main()
