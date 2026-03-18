"""
Ensemble v6.1 — Shared Configuration
=====================================
Single source of truth for labels, model configs, paths, and class weights.
Both ensemble_v6_preprocess.py and ensemble_v6_train.py import from here.

To add a new model or class, edit ONLY this file.

v6.1 fixes:
  - dtype key (not torch_dtype) — compatible with transformers >= 4.40
  - Corrected effective batch comments (batch_size × grad_accum)
  - grad_ckpt=False for all models (H100 80GB has enough VRAM)
"""

import os
import torch

# ═══════════════════════════════════════════════════════════════
# LABELS — 7 classes
# ═══════════════════════════════════════════════════════════════

LABEL_ORDER = ["Human", "Claude", "DeepSeek", "Gemini", "GPT", "Grok", "Mistral"]
LABEL2ID    = {m: i for i, m in enumerate(LABEL_ORDER)}
ID2LABEL    = {i: m for m, i in LABEL2ID.items()}
NUM_LABELS  = len(LABEL_ORDER)

# Label unification map: substring match → canonical label
LABEL_MAP = {
    "Human":    "Human",
    "GPT":      "GPT",
    "DeepSeek": "DeepSeek",
    "Claude":   "Claude",
    "Gemini":   "Gemini",
    "Grok":     "Grok",
    "Mistral":  "Mistral",
}

# ═══════════════════════════════════════════════════════════════
# CLASS WEIGHTS — inverse frequency normalized
# VERIFIED counts from golden_chunks_builder_v2_1.py
# Dataset: 4,462,570 rows, 50 parquets, ZERO empty rows.
# Median of 6 AI classes: (382720 + 382720) / 2 = 382,720
# Weight = median / class_count
# ═══════════════════════════════════════════════════════════════

DATASET_COUNTS = {
    "Human":    2_231_285,   # 49.99% — majority class, downweighted
    "Claude":     382_720,   # 8.57%
    "DeepSeek":   317_683,   # 7.12%  — minority, upweighted
    "Gemini":     382_720,   # 8.57%
    "GPT":        382_720,   # 8.57%
    "Grok":       382_721,   # 8.57%
    "Mistral":    382_721,   # 8.57%
}
DATASET_TOTAL = 4_462_570
_MEDIAN       = 382_720

CLASS_WEIGHTS_DICT = {k: _MEDIAN / v for k, v in DATASET_COUNTS.items()}
CLASS_WEIGHTS = torch.tensor(
    [CLASS_WEIGHTS_DICT[LABEL_ORDER[i]] for i in range(NUM_LABELS)],
    dtype=torch.float32,
)

# ═══════════════════════════════════════════════════════════════
# MODEL CONFIGS
# ═══════════════════════════════════════════════════════════════
# dtype key (not torch_dtype) is required for transformers >= 4.40.
# grad_ckpt=False: H100 80GB has enough VRAM for all three models
# without recomputation overhead (~20% slower with grad_ckpt=True).
# ═══════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "deberta": {
        "name":       "microsoft/deberta-v3-base",
        "short":      "deberta",
        "lr":         1e-5,           # Disentangled attention → needs lower LR
        "max_length": 512,
        "batch_size": 48,             # H100: fits comfortably with MCL hidden states
        "grad_accum": 2,              # Effective batch: 96  (48 × 2)
        "epochs":     2,
        "warmup_ratio": 0.1,
        "attn_impl":  None,           # DeBERTa manages its own attention
        "dtype":      torch.bfloat16, # ← dtype, NOT torch_dtype
        "grad_ckpt":  False,
        "torch_compile": False,       # DeBERTa dynamic shapes → compile unstable
    },
    "electra": {
        "name":       "google/electra-base-discriminator",
        "short":      "electra",
        "lr":         2e-5,
        "max_length": 512,
        "batch_size": 64,
        "grad_accum": 2,              # Effective batch: 128  (64 × 2)
        "epochs":     2,
        "warmup_ratio": 0.1,
        "attn_impl":  None,
        "dtype":      torch.bfloat16, # ← dtype, NOT torch_dtype
        "grad_ckpt":  False,
        "torch_compile": True,        # Static shapes → compile works well
    },
    "modernbert": {
        "name":       "answerdotai/ModernBERT-base",
        "short":      "modernbert",
        "lr":         2e-5,
        "max_length": 512,
        "batch_size": 64,
        "grad_accum": 2,              # Effective batch: 128  (64 × 2)
        "epochs":     2,
        "warmup_ratio": 0.1,
        "attn_impl":  "sdpa",         # Flash Attention 2 via SDPA
        "dtype":      torch.bfloat16, # ← dtype, NOT torch_dtype
        "grad_ckpt":  False,
        "torch_compile": True,        # SDPA + static shapes → compile works well
    },
}

TRAINING_ORDER = ["deberta", "electra", "modernbert"]

# ═══════════════════════════════════════════════════════════════
# PATHS — Override DRIVE_BASE via env var for portability
# export XPLAGIAX_DRIVE_BASE="/content/drive/MyDrive/other_path"
# ═══════════════════════════════════════════════════════════════

DRIVE_BASE = os.environ.get(
    "XPLAGIAX_DRIVE_BASE",
    "/content/drive/MyDrive/grokipedia_parts/Pavel",
)

DRIVE_DATASET_DIR = f"{DRIVE_BASE}/dataset_golden_5GB_chunks_v2/"
LOCAL_DATASET_DIR = "/content/local_golden_chunks/"
CLEAN_DATASET_DIR = f"{DRIVE_BASE}/clean_dataset_v6/"
SPLIT_CACHE_DIR   = f"{DRIVE_BASE}/split_cache_v6/"
ENSEMBLE_DIR      = f"{DRIVE_BASE}/ensemble_v6/"
FK_CACHE_DIR      = f"{DRIVE_BASE}/fk_cache_v6/"
LOCAL_BASE        = "/content/local_training/"
TOKENIZED_BASE    = f"{DRIVE_BASE}/tokenized_v6/"


def model_paths(key: str) -> dict:
    """Return all relevant paths for a given model key."""
    return {
        "tokenized_train": f"{TOKENIZED_BASE}{key}_train/",
        "tokenized_test":  f"{TOKENIZED_BASE}{key}_test/",
        "local_ckpt":      f"{LOCAL_BASE}/{key}_checkpoints/",
        "drive_ckpt":      f"{ENSEMBLE_DIR}/{key}_checkpoints/",
        "final_model":     f"{ENSEMBLE_DIR}/{key}_final/",
    }


# ═══════════════════════════════════════════════════════════════
# AUGMENTATION CONSTANTS
# ═══════════════════════════════════════════════════════════════

AUGMENT_SKIP_PROB    = 0.3    # P(skip augmentation) → 70% of texts get augmented
AUGMENT_CHAR_PROB    = 0.1    # P(character-level noise per word)
AUGMENT_MERGE_PROB   = 0.05   # P(merge word with next) = char_prob / 2
MIN_WORD_LEN_AUGMENT = 5      # Only augment words longer than 4 chars

# FK score constants
FK_MAX_CHARS = 3000           # Max chars for FK computation (performance cap)
FK_CLAMP_MIN = -5.0           # Floor (some texts produce negative FK)
FK_CLAMP_MAX = 30.0           # Ceiling
FK_DEFAULT   = 0.0            # Default for empty / malformed text

# File copy retry
MAX_COPY_RETRIES    = 5
COPY_RETRY_BASE_SEC = 5       # Exponential backoff base (seconds)
