import os
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel, LoraConfig, get_peft_model
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
import math
from typing import Dict, Literal, Tuple, List, Optional, Union
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import seaborn as sns
from pathlib import Path
import gc
from datasets import load_dataset
import json
import argparse

# ============================================
# Configuration: Paths (can be overridden via CLI)
# ============================================

# 1. Base model path (Qwen3-8B)
DEFAULT_BASE_MODEL_PATH = "/datas/huggingface/Qwen3-8B"

# 2. Task vector anchor paths
DEFAULT_SAFER_STATE_PATH = "/datas/wangxiao/Grad_Reward/SFT_Qwen7_8B/Qwen3_8b_ckpt/PKU-SafeRlhf-10k-safer_5e-6/checkpoint-7000"
DEFAULT_DANGER_STATE_PATH = "/datas/wangxiao/Grad_Reward/SFT_Qwen7_8B/Qwen3_8b_ckpt/harmful/beavertails_unsafe_random3000_5e-6"

# 3. Intermediate checkpoints to track (training trajectory)
DEFAULT_CHECKPOINT_DIR = "./checkpoint"
DEFAULT_CHECKPOINT_START = 150
DEFAULT_CHECKPOINT_END = 6150
DEFAULT_CHECKPOINT_STEP = 150

# 4. Evaluation metric file
DEFAULT_METRIC_FILE = "./metric.jsonl"

# ============================================
# Core Functions
# ============================================

def load_and_merge_lora(base_model_path: str, lora_path: str) -> Dict[str, torch.Tensor]:
    """
    Load base model and LoRA adapter, extract LoRA parameters, and merge A and B matrices.

    This function encapsulates the entire process:
    1. Load model and LoRA adapter from paths to GPU.
    2. Extract all LoRA-related parameters (A and B matrices).
    3. Compute B @ A matrix product for each submodule.
    4. Return a new dict with submodule names as keys and merged weight delta matrices as values.

    Args:
        base_model_path (str): Path to the base model.
        lora_path (str): Path to the LoRA adapter weights.

    Returns:
        Dict[str, torch.Tensor]:
            A dict with submodule names (without LoRA suffix) as keys,
            and B@A computed result tensors (delta_W) as values.
    """
    # --- Step 1 & 2: Load model and extract LoRA parameters ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not detected, running on CPU which may be slow.")

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )

    # Load LoRA adapter
    lora_model = PeftModel.from_pretrained(base_model, lora_path)

    # Extract LoRA parameters
    lora_params = {
        k: v
        for k, v in lora_model.state_dict().items()
        if "lora" in k
    }

    # Free model memory since we only need lora_params
    del base_model
    del lora_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Step 3: Merge A and B matrices ---
    # Group by submodule name
    submodule_tensors = defaultdict(dict)
    for name, tensor in lora_params.items():
        # Note: peft library LoRA key format may vary, here we adapt to common formats
        # e.g., '...q_proj.lora_A.weight' or '...q_proj.lora_A.default.weight'
        if ".lora_A." in name:
            submodule_name = name.split(".lora_A.")[0]
            submodule_tensors[submodule_name]['A'] = tensor
        elif ".lora_B." in name:
            submodule_name = name.split(".lora_B.")[0]
            submodule_tensors[submodule_name]['B'] = tensor

    merged_weights = {}

    # Iterate grouped tensors, compute B @ A
    for submodule_name, tensors in submodule_tensors.items():
        if 'A' in tensors and 'B' in tensors:
            lora_A = tensors['A']
            lora_B = tensors['B']

            # Matrix multiplication B @ A
            delta_W = lora_B @ lora_A

            merged_weights[submodule_name] = delta_W.to(torch.bfloat16)
        else:
            print(f"Warning: Submodule '{submodule_name}' missing A or B matrix, skipped.")

    return merged_weights


def sum_dict_values(data_dict: Dict[str, Union[int, float]]) -> float:
    """
    Calculate the sum of all values in a dictionary.

    Args:
        data_dict (Dict[str, Union[int, float]]): Data dictionary.

    Returns:
        float: Sum of all values.
    """
    total_sum = 0.0
    for key, value in data_dict.items():
        total_sum += value
    return total_sum


def calculate_lora_delta_metrics(
    lora_delta_1: Dict[str, torch.Tensor],
    lora_delta_2: Dict[str, torch.Tensor],
    metric: Literal["cosine", "l2", "l1", "dot", "projection"] = "cosine"
) -> Dict[str, float]:
    """
    Calculate metrics between tensors of corresponding modules in two LoRA delta dicts.

    Before calculation, each 2D tensor is flattened to a 1D vector.

    Args:
        lora_delta_1 (Dict[str, torch.Tensor]): First LoRA delta dict.
        lora_delta_2 (Dict[str, torch.Tensor]): Second LoRA delta dict.
        metric (Literal["cosine", "l2", "l1", "dot", "projection"]): Metric to calculate.
            - "cosine":     Cosine similarity ([-1, 1], 1 means identical, -1 means opposite).
            - "l2":         L2 distance/Euclidean distance (>=0, 0 means identical).
            - "l1":         L1 distance/Manhattan distance (>=0, 0 means identical).
            - "dot":        Dot product.
            - "projection": Scalar projection of lora_delta_1 onto lora_delta_2.

    Returns:
        Dict[str, float]: A dict with module names as keys and calculated float metrics as values.
    """
    supported_metrics = ["cosine", "l2", "l1", "dot", "projection"]
    if metric not in supported_metrics:
        raise ValueError(f"Unsupported metric: '{metric}'. Available values: {supported_metrics}")

    results = {}

    for key, tensor1 in lora_delta_1.items():
        if key not in lora_delta_2:
            print(f"Warning: Key '{key}' not found in lora_delta_2, skipped.")
            continue

        tensor2 = lora_delta_2[key]

        # Ensure tensors are on the same device
        if tensor1.device != tensor2.device:
            tensor2 = tensor2.to(tensor1.device)

        # Flatten 2D tensor to 1D vector
        v1 = tensor1.flatten().float()
        v2 = tensor2.flatten().float()

        # Calculate based on specified metric
        score = 0.0
        if metric == "cosine":
            score = F.cosine_similarity(v1, v2, dim=0).item()
        elif metric == "l2":
            score = torch.linalg.norm(v1 - v2).item()
        elif metric == "l1":
            score = torch.linalg.norm(v1 - v2, ord=1).item()
        elif metric == "dot":
            score = torch.dot(v1, v2).item()
        elif metric == "projection":
            dot_product = torch.dot(v1, v2)
            norm_v2 = torch.linalg.norm(v2)
            if norm_v2 > 1e-9:
                score = (dot_product / norm_v2).item()
            else:
                score = 0.0

        results[key] = score

    return results


def calculate_module_norms(
    lora_params: Dict[str, torch.Tensor],
    metric: Literal['l2', 'l1'] = 'l2'
) -> Dict[str, float]:
    """
    Calculate the norm of each module in a LoRA params dict.

    Args:
        lora_params (Dict[str, torch.Tensor]): LoRA params dict.
        metric (Literal['l2', 'l1']): Norm type to calculate. Defaults to 'l2'.

    Returns:
        Dict[str, float]: A dict with module names as keys and calculated norm values as values.
    """
    if metric not in ['l2', 'l1']:
        raise ValueError(f"Unsupported metric: '{metric}'. Available values: 'l2', 'l1'")

    calculated_norms = {}

    with torch.no_grad():
        for key, tensor in lora_params.items():
            value = 0.0
            float_tensor = tensor.float()

            if metric == 'l2':
                value = torch.linalg.norm(float_tensor).item()
            elif metric == 'l1':
                value = torch.abs(float_tensor).sum().item()

            calculated_norms[key] = value

    return calculated_norms


def generate_checkpoint_paths(base_dir: str, start: int, end: int, step: int) -> List[str]:
    """Generate checkpoint paths from start to end with given step."""
    return [f"{base_dir}/checkpoint-{str(i)}" for i in range(start, end + 1, step)]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute LoRA task vector metrics"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Path to base model"
    )
    parser.add_argument(
        "--safer_state_path",
        type=str,
        default=DEFAULT_SAFER_STATE_PATH,
        help="Path to safe task vector anchor"
    )
    parser.add_argument(
        "--danger_state_path",
        type=str,
        default=DEFAULT_DANGER_STATE_PATH,
        help="Path to danger task vector anchor"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory containing checkpoint folders"
    )
    parser.add_argument(
        "--checkpoint_start",
        type=int,
        default=DEFAULT_CHECKPOINT_START,
        help="Start checkpoint step"
    )
    parser.add_argument(
        "--checkpoint_end",
        type=int,
        default=DEFAULT_CHECKPOINT_END,
        help="End checkpoint step"
    )
    parser.add_argument(
        "--checkpoint_step",
        type=int,
        default=DEFAULT_CHECKPOINT_STEP,
        help="Checkpoint step interval"
    )
    parser.add_argument(
        "--metric_file",
        type=str,
        default=DEFAULT_METRIC_FILE,
        help="Path to metric.jsonl file"
    )
    parser.add_argument(
        "--cuda_device",
        type=str,
        default="3",
        help="CUDA device ID"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    print("=" * 60)
    print("Configuration:")
    print(f"  Base model: {args.base_model_path}")
    print(f"  Safer state: {args.safer_state_path}")
    print(f"  Danger state: {args.danger_state_path}")
    print(f"  Checkpoint dir: {args.checkpoint_dir}")
    print(f"  Checkpoint range: {args.checkpoint_start} ~ {args.checkpoint_end} (step {args.checkpoint_step})")
    print(f"  Metric file: {args.metric_file}")
    print(f"  CUDA device: {args.cuda_device}")
    print("=" * 60)

    print("Loading anchor task vectors...")
    safer_state = load_and_merge_lora(args.base_model_path, args.safer_state_path)
    danger_state = load_and_merge_lora(args.base_model_path, args.danger_state_path)

    print("Generating checkpoint paths...")
    checkpoint_paths = generate_checkpoint_paths(
        args.checkpoint_dir, args.checkpoint_start, args.checkpoint_end, args.checkpoint_step
    )

    # Initialize metric dictionaries
    safer_metric_dict = {
        "cosine": [],
        "l2": [],
        "l1": [],
        "dot": [],
        "projection": []
    }
    danger_metric_dict = {
        "cosine": [],
        "l2": [],
        "l1": [],
        "dot": [],
        "projection": []
    }
    sum_l1s = []
    sum_l2s = []

    # Compute metrics for each checkpoint
    print("Computing metrics for checkpoints...")
    for checkpoint_path in checkpoint_paths:
        print(f"Processing {checkpoint_path}")
        lora_state = load_and_merge_lora(args.base_model_path, checkpoint_path)

        # Compute distance metrics from checkpoint to anchor task vectors
        for metric in ["cosine", "l2", "l1", "dot", "projection"]:
            safer_value_dict = calculate_lora_delta_metrics(lora_state, safer_state, metric)
            danger_value_dict = calculate_lora_delta_metrics(lora_state, danger_state, metric)
            safer_metric_dict[metric].append(sum_dict_values(safer_value_dict))
            danger_metric_dict[metric].append(sum_dict_values(danger_value_dict))

        # Compute self-norm (magnitude of the task vector itself)
        norm_dict_l1 = calculate_module_norms(lora_state, "l1")
        norm_dict_l2 = calculate_module_norms(lora_state, "l2")
        sum_l1s.append(sum_dict_values(norm_dict_l1))
        sum_l2s.append(sum_dict_values(norm_dict_l2))

        del lora_state

    # Load and enrich metric file
    print("Loading and enriching metric file...")
    dataset = load_dataset("json", data_files=args.metric_file, split="train")
    print(f"Original columns: {dataset.column_names}")

    # Add safe direction metrics
    dataset = dataset.add_column("safe_projection", safer_metric_dict["projection"])
    dataset = dataset.add_column("safe_cosine", safer_metric_dict["cosine"])
    dataset = dataset.add_column("safe_dot", safer_metric_dict["dot"])
    dataset = dataset.add_column("safe_l2", safer_metric_dict["l2"])
    dataset = dataset.add_column("safe_l1", safer_metric_dict["l1"])

    # Add danger direction metrics (beaver_unsafe)
    dataset = dataset.add_column("danger_projection(beaver_unsafe)", danger_metric_dict["projection"])
    dataset = dataset.add_column("danger_cosine(beaver_unsafe)", danger_metric_dict["cosine"])
    dataset = dataset.add_column("danger_dot(beaver_unsafe)", danger_metric_dict["dot"])
    dataset = dataset.add_column("danger_l2(beaver_unsafe)", danger_metric_dict["l2"])
    dataset = dataset.add_column("danger_l1(beaver_unsafe)", danger_metric_dict["l1"])

    # Add self-norm
    dataset = dataset.add_column("self_l1", sum_l1s)
    dataset = dataset.add_column("self_l2", sum_l2s)

    # Save enriched dataset
    print(f"Final columns: {dataset.column_names}")
    dataset.to_json(metric_file, lines=True)
    print(f"Results saved to {args.metric_file}")


if __name__ == "__main__":
    main()