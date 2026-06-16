# Portions of this file are adapted from code-repe:
# https://github.com/sanadlab/code-repe

import json
import random
from typing import Any, Tuple, Dict, List

def pair_data(template: str, tasks: List[str], targets_list: List[List[str]]) -> List[List[str]]:
    data = []
    for task, targets in zip(tasks, targets_list):
        paired_data = []
        for target in targets:
            paired_data.append(template.format(target="```python\n"+target.strip()+"\n```"))
        data.append(paired_data)
    return data

def aligned_shuffle(list1: List[Any], list2: List[Any], seed = 0) -> Tuple[List[Any], List[Any]]:
    random.seed(seed)
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    if not combined:
        return [], []
    shuffled1, shuffled2 = zip(*combined)
    return list(shuffled1), list(shuffled2)

def aligned_shuffle_three(
    list1: List[Any], list2: List[Any], list3: List[Any], seed=0
) -> Tuple[List[Any], List[Any], List[Any]]:
    random.seed(seed)
    combined = list(zip(list1, list2, list3))
    random.shuffle(combined)
    if not combined:
        return [], [], []
    shuffled1, shuffled2, shuffled3 = zip(*combined)
    return list(shuffled1), list(shuffled2), list(shuffled3)


def aligned_shuffle_three_with_index_map(
    list1: List[Any], list2: List[Any], list3: List[Any], seed=0
) -> Tuple[List[Any], List[Any], List[Any], List[int], List[int]]:
    """
    Shuffle three aligned lists and keep index mapping between original and shuffled order.

    Returns:
    - shuffled_list1
    - shuffled_list2
    - shuffled_list3
    - shuffled_to_orig: shuffled_idx -> original_idx
    - orig_to_shuffled: original_idx -> shuffled_idx
    """
    random.seed(seed)
    combined = list(zip(range(len(list1)), list1, list2, list3))
    random.shuffle(combined)
    if not combined:
        return [], [], [], [], []
    shuffled_to_orig, shuffled1, shuffled2, shuffled3 = zip(*combined)
    shuffled_to_orig = list(shuffled_to_orig)
    orig_to_shuffled = [0] * len(shuffled_to_orig)
    for shuffled_idx, orig_idx in enumerate(shuffled_to_orig):
        orig_to_shuffled[int(orig_idx)] = int(shuffled_idx)
    return (
        list(shuffled1),
        list(shuffled2),
        list(shuffled3),
        shuffled_to_orig,
        orig_to_shuffled,
    )

def load_triplets_jsonl(filename: str) -> Dict[str, List[str]]:
    data = []
    with open(filename, "r") as f:
        for line_idx, line in enumerate(f):
            item = json.loads(line)
            mem_labels = item.get("mem_labels")
            syn_labels = item.get("syn_labels")

            if not isinstance(mem_labels, list):
                mem_labels = item.get("labels", [])
            if not isinstance(syn_labels, list):
                syn_labels = item.get("labels", [])

            if not isinstance(mem_labels, list):
                mem_labels = []
            if not isinstance(syn_labels, list):
                syn_labels = []

            item["mem_labels"] = mem_labels
            item["syn_labels"] = syn_labels
            item["sample_id"] = item.get("index", line_idx)
            data.append(item)

    res = { 
            'tasks': [item.get('task', '') for item in data],
            'targets': [item['syn_code'] for item in data],       # Syn axis
            'targets_mem': [item['mem_code'] for item in data],   # Mem axis
            'syn_labels': [item['syn_labels'] for item in data],
            'mem_labels': [item['mem_labels'] for item in data],
            'sample_ids': [item['sample_id'] for item in data],
            # Backward-compatible alias for old callers that only consume one label list.
            'labels': [item['mem_labels'] for item in data],
            }
    
    return res

def load_memorization_jsonl(file_path: str) -> Dict[str, List[str]]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    res = { 
            'seen': [item['seen'] for item in data],
            'mutated_seen': [item['mutated_seen'] for item in data],
            'unseen': [item['unseen'] for item in data]
            }
    
    return res
