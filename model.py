# Portions of this file are adapted from code-repe:
# https://github.com/sanadlab/code-repe

import os
import json
import numpy as np
import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.processors import TemplateProcessing
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
)
from repe import repe_pipeline_registry
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

repe_pipeline_registry()

# Abstracts the usage of both the tokenizer and the model
class Model:
    def __init__(self, name, model_dir=None):
        self.name = name
        self.rep_pipeline = None
        self.rep_reader = None

        if model_dir is not None:
            self.model = AutoModelForCausalLM.from_pretrained(name, cache_dir=model_dir, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        
        if name == "sapienzanlp/Minerva-7B-base-v1.0":
            self.tokenizer = self._load_minerva_tokenizer(name, model_dir=model_dir)
            self.tokenizer.padding_side = "left"
        else:
            use_fast_tokenizer = "LlamaForCausalLM" not in self.model.config.architectures
            if model_dir is not None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name,
                    cache_dir=model_dir,
                    use_fast=use_fast_tokenizer,
                    padding_side="left",
                    legacy=False,
                    trust_remote_code=True,
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name,
                    use_fast=use_fast_tokenizer,
                    padding_side="left",
                    legacy=False,
                    trust_remote_code=True,
                )

        self.tokenizer.pad_token_id = 0

    @staticmethod
    def _load_minerva_tokenizer(name, model_dir=None):
        try:
            snapshot_dir = snapshot_download(
                repo_id=name,
                cache_dir=model_dir,
                local_files_only=True,
            )
        except Exception:
            snapshot_dir = snapshot_download(repo_id=name, cache_dir=model_dir)

        tokenizer_json = os.path.join(snapshot_dir, "tokenizer.json")
        with open(tokenizer_json, "r", encoding="utf-8") as f:
            tok_json = json.load(f)

        model_cfg = tok_json["model"]
        if model_cfg.get("type") != "BPE":
            raise ValueError("Minerva tokenizer fallback only supports BPE tokenizer.json.")

        vocab = model_cfg["vocab"]
        merges = [tuple(m) for m in model_cfg["merges"]]

        core = Tokenizer(
            BPE(
                vocab=vocab,
                merges=merges,
                unk_token=model_cfg.get("unk_token"),
                byte_fallback=model_cfg.get("byte_fallback", False),
                fuse_unk=model_cfg.get("fuse_unk", False),
            )
        )

        core.normalizer = Sequence([NFKC()])
        core.pre_tokenizer = ByteLevel(add_prefix_space=True, use_regex=True)
        core.decoder = ByteLevelDecoder(add_prefix_space=True, use_regex=True)
        core.post_processor = TemplateProcessing(
            single="<s> $A",
            pair="<s> $A <s>:1 $B:1",
            special_tokens=[("<s>", vocab.get("<s>", 0))],
        )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=core,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<s>",
            model_max_length=4096,
        )
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
        return tokenizer

    # Tokenizes and does a forward pass to obtain the outputs
    def forward_pass(self, inputs):
        tokenized_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        
        with torch.no_grad():
            # Forward pass through the model for the entire batch
            outputs = self.model(**tokenized_inputs)

        return tokenized_inputs, outputs

    def encode(self, text):
        return self.tokenizer.encode(text, return_tensors="pt").to(self.model.device)
    
    def num_hidden_layers(self):
        return self.model.config.num_hidden_layers
    
    def _validate_token_ids(self, texts, batch_size=8, max_length=4096, padding="longest"):
        vocab_size = len(self.tokenizer)
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=padding,
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"]
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= vocab_size:
                bad_indices = (input_ids < 0) | (input_ids >= vocab_size)
                first_bad = bad_indices.nonzero(as_tuple=False)[0].tolist()
                sample_idx = start + first_bad[0]
                bad_id = int(input_ids[first_bad[0], first_bad[1]].item())
                raise ValueError(
                    f"Invalid token id detected (id={bad_id}) in sample index {sample_idx}. "
                    f"Valid range is [0, {vocab_size - 1}]."
                )
    
    def _debug_find_cuda_failure(self, texts, max_length=4096, padding="longest"):
        if not torch.cuda.is_available():
            return
        for i, text in enumerate(texts):
            try:
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=padding,
                    truncation=True,
                    max_length=max_length,
                ).to(self.model.device)
                with torch.no_grad():
                    _ = self.model(**enc, output_hidden_states=True)
                for d in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(d)
            except Exception as e:
                preview = text[:200].replace("\n", "\\n")
                raise RuntimeError(
                    f"CUDA failure while processing sample index {i}. "
                    f"Text preview: {preview}"
                ) from e

    def learn_representations(self, train_scenarios, train_labels,
                   n_difference = 1, n_components = 1, batch_size=8, max_length=4096,
                   direction_method = 'pca', padding="longest", truncation=True):
        # Preflight check to catch invalid token ids before GPU execution
        self._validate_token_ids(
            train_scenarios,
            batch_size=batch_size,
            max_length=max_length,
            padding=padding,
        )
        # Optional deep debug to pinpoint the exact failing sample
        if os.getenv("REPE_DEBUG", "0") == "1":
            self._debug_find_cuda_failure(
                train_scenarios,
                max_length=max_length,
                padding=padding,
            )
        self.rep_pipeline = pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
    
        # Set parameters for rep_reader
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))

        # Get directions using train_data
        self.rep_reader = self.rep_pipeline.get_directions(
            train_scenarios,
            rep_token=-1,
            hidden_layers=hidden_layers,
            n_difference=n_difference,
            train_labels=train_labels,
            direction_method=direction_method,
            direction_finder_kwargs={"n_components": n_components},
            batch_size=batch_size,
            max_length=max_length,
            truncation=truncation,
            padding=padding,
        )

    def probs(self, eval_data, max_length=512, padding="longest"):
        probabilities = []

        for sequence in eval_data:
            tokenized = self.tokenizer(
                sequence,
                return_tensors="pt",
                padding=padding,
                truncation=True,
                max_length=max_length
            ).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**tokenized)
            logits = outputs.logits
            input_ids = tokenized["input_ids"]
            # Shift logits and labels so that each prediction is compared with the next token.
            shifted_logits = logits[:, :-1, :]
            shifted_input_ids = input_ids[:, 1:]
            probs = torch.softmax(shifted_logits, dim=-1)
            token_probs = probs.gather(2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
            # The probability of the sequence is the average of the token probabilities.
            seq_prob = token_probs.mean().item()
            probabilities.append(seq_prob)

        return probabilities

    def representation_pipeline(self, eval_data, hidden_layers, rep_token=-1,
                                batch_size=8, max_length=512, padding="longest",
                                truncation=True, custom_reader=None):
        return self.rep_pipeline(
            eval_data,
            rep_token=rep_token,
            hidden_layers=hidden_layers,
            rep_reader=custom_reader if custom_reader is not None else self.rep_reader,
            batch_size=batch_size,
            max_length=max_length,
            truncation=truncation,
            padding=padding
        )

    def get_sign(self, layer):
        return self.rep_reader.direction_signs[layer]

    def learn_orthogonal_representations(self, train_scenarios_mem, train_scenarios_syn, train_labels_mem,
                       train_labels_syn=None, n_difference = 1, n_components = 1, batch_size=8, max_length=4096,
                       direction_method = 'pca', padding="longest", truncation=True):
        
        if train_labels_syn is None:
            train_labels_syn = train_labels_mem
            print("Warning: train_labels_syn is not provided, using train_labels_mem for both memory and syntactic directions. This may lead to suboptimal separation of the two directions.")

        def _effective_label_count(labels):
            if isinstance(labels, list) and labels and isinstance(labels[0], list):
                return sum(len(x) for x in labels)
            return len(labels)

        mem_label_count = _effective_label_count(train_labels_mem)
        syn_label_count = _effective_label_count(train_labels_syn)

        if len(train_scenarios_mem) != mem_label_count:
            raise ValueError(
                "Length mismatch: train_scenarios_mem and train_labels_mem must have the same "
                f"effective length (scenarios={len(train_scenarios_mem)}, labels={mem_label_count})."
            )
        if len(train_scenarios_syn) != syn_label_count:
            raise ValueError(
                "Length mismatch: train_scenarios_syn and train_labels_syn must have the same "
                f"effective length (scenarios={len(train_scenarios_syn)}, labels={syn_label_count})."
            )
        if len(train_scenarios_mem) != len(train_scenarios_syn):
            raise ValueError(
                "Length mismatch: mem and syn scenario lists must have the same length."
            )

        self.rep_pipeline = pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))

        # 1. Fit Mem direction (Mem axis input, loaded from fim_input)
        self.rep_reader_mem = self.rep_pipeline.get_directions(
            train_scenarios_mem, rep_token=-1, hidden_layers=hidden_layers, n_difference=n_difference,
            train_labels=train_labels_mem, direction_method=direction_method, direction_finder_kwargs={"n_components": n_components},
            batch_size=batch_size, max_length=max_length, truncation=truncation, padding=padding,
        )

        # 2. Fit Syn direction (Prompted code input, loaded from code)
        self.rep_reader_syn = self.rep_pipeline.get_directions(
            train_scenarios_syn, rep_token=-1, hidden_layers=hidden_layers, n_difference=n_difference,
            train_labels=train_labels_syn, direction_method=direction_method, direction_finder_kwargs={"n_components": n_components},
            batch_size=batch_size, max_length=max_length, truncation=truncation, padding=padding,
        )

        # 3. Gram-Schmidt Orthogonalization: Syn = Fam - proj_Mem(Fam)
        for layer in hidden_layers:
            v_mem = self.rep_reader_mem.directions[layer][0]
            v_fam = self.rep_reader_syn.directions[layer][0]

            proj = (np.dot(v_fam, v_mem) / np.dot(v_mem, v_mem)) * v_mem
            v_syn = v_fam - proj
            v_syn = v_syn / np.linalg.norm(v_syn)
            self.rep_reader_syn.directions[layer][0] = v_syn

    
    def generate(self, inputs):
        # Store the initial prompt length to avoid stopping on prompt's trailing ```
        initial_length = inputs["input_ids"].shape[-1]

        class StopOnTripleBackticks(StoppingCriteria):
            def __init__(self, tokenizer, initial_length):
                self.tokenizer = tokenizer
                self.initial_length = initial_length

            def __call__(self, input_ids, scores, **kwargs):
                if input_ids.shape[-1] > self.initial_length:
                    # Only decode the newly generated tokens
                    generated_text = self.tokenizer.decode(
                        input_ids[0][self.initial_length:], skip_special_tokens=True
                    )
                    if "```" in generated_text:
                        return True
                return False

        stopping_criteria = StoppingCriteriaList(
            [StopOnTripleBackticks(self.tokenizer, initial_length)]
        )
        return self.model.generate(
            **inputs, max_new_tokens=1024, stopping_criteria=stopping_criteria
        )
