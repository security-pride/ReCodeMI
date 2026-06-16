#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract NaturalCC-compatible token ranks.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-cache")
    parser.add_argument("--block-size", type=int, default=2048)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_tokenizer(args: argparse.Namespace):
    last_error = None
    for use_fast in (True, False):
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                cache_dir=args.model_cache,
                trust_remote_code=args.trust_remote_code,
                use_fast=use_fast,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as error:  # noqa: BLE001
            last_error = error
    raise last_error


def extract_split(args, model, tokenizer, device, split: str, label: int) -> None:
    input_path = args.data_dir / f"{split}.txt"
    lines = [line.rstrip("\n") for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []

    for index, line in enumerate(lines):
        encoded = tokenizer(
            line,
            return_tensors="pt",
            truncation=True,
            max_length=args.block_size,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[0, :-1]
            probabilities = F.softmax(logits, dim=-1)

        targets = input_ids[0, 1:]
        target_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
        ranks = (logits > target_logits.unsqueeze(1)).sum(dim=1).cpu()
        target_probabilities = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1).cpu()
        predictions = logits.argmax(dim=-1).cpu()

        records.append(
            {
                "gt": targets.cpu(),
                "pred": predictions,
                "rank": ranks,
                "prob": target_probabilities,
                "label": torch.tensor(label),
            }
        )
        if index % args.logging_steps == 0:
            LOGGER.info("%s: %d/%d", split, index, len(lines))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = "member_ranks.pth" if label == 1 else "non_member_ranks.pth"
    torch.save(records, args.output_dir / output_name)
    LOGGER.info("Saved %d records to %s", len(records), args.output_dir / output_name)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    set_seed(args.seed)
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
    extract_split(args, model, tokenizer, device, "train", 1)
    extract_split(args, model, tokenizer, device, "test", 0)


if __name__ == "__main__":
    main()
