#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AdvPrompt-MIA victim completion inference.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-cache")
    parser.add_argument("--file-types", nargs="+", default=["train", "test"])
    parser.add_argument("--max-input-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def load_tokenizer(args):
    last_error = None
    for use_fast in (True, False):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                cache_dir=args.model_cache,
                trust_remote_code=args.trust_remote_code,
                use_fast=use_fast,
                padding_side="left",
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise last_error


def normalize_prompt(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("\\n", "\n")


def run_split(args, model, tokenizer, device, file_type: str) -> None:
    input_path = args.data_dir / f"{file_type}_victim.json"
    output_path = args.data_dir / f"{file_type}_models--{args.model.replace('/', '--')}_victim_infer.txt"
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            prompt = normalize_prompt(record.get("input", ""))
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).to(device)
            torch.manual_seed(args.seed + index)
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_k=args.top_k,
                    temperature=args.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion_ids = generated[0, encoded["input_ids"].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            handle.write(completion.replace("\n", "\\n") + "\n")
            if index % 100 == 0:
                LOGGER.info("%s: %d/%d", file_type, index, len(records))
    LOGGER.info("Saved %s", output_path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.model_cache,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    tokenizer = load_tokenizer(args)
    for file_type in args.file_types:
        run_split(args, model, tokenizer, device, file_type)


if __name__ == "__main__":
    main()
