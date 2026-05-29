"""
xplagiax_ensemble_sota_v7 — trainer.py
======================================
End-to-end training for the SOTA v7 detector.

This pipeline fixes the methodological flaws that make headline accuracy
numbers meaningless (and that produced SzegedAI's ~18-point Dev->Test F1 gap
in 2025.genaidetect-1.15):

  STAGE 1  Data hygiene
           - near-duplicate removal (MinHash/LSH; exact-hash fallback)
           - GROUP-AWARE train/val/test split (by source/domain/prompt) so
             paraphrases of the same item never straddle splits -> no leakage
  STAGE 2  Multilevel adversarial augmentation (TRAIN ONLY)
           - char-level (DAMAGE 2025.genaidetect-1.9) + homoglyph + paraphrase
             hooks. Teaches robustness instead of relying on the normalizer.
  STAGE 3  Supervised multi-class training (heterogeneous archs, optional
           length experts) with focal loss + MILD class reweighting.
           Multi-class > binary for generalization (SzegedAI). We do NOT
           crush the Human class (the v6 mistake that manufactured FPs).
  STAGE 4  Temperature calibration on a DISJOINT validation split (not test).
  STAGE 5  Fit the stacking meta-learner over branch features (val split).
  STAGE 6  Fit the split-conformal threshold for a guaranteed FPR on humans.

Designed to run on Colab/A100/H100. Heavy deps are imported lazily so the
file imports and `--help` runs on a plain CPU box.

Expected dataset columns: `text` (str), `label`/`target_model` (str),
and optionally `source` / `domain` / `group_id` for group-aware splitting.
"""

from __future__ import annotations

import os
import re
import gc
import json
import random
import hashlib
import logging
import argparse
from typing import Dict, List, Optional, Tuple

import numpy as np

from detector_final import (
    LABEL_ORDER, HUMAN_IDX, NUM_LABELS, SHORT_LONG_BOUNDARY,
    StylometricExtractor, SupervisedBranch, ZeroShotBranch,
    MetaLearner, ConformalCalibrator, canonicalize_text, _CONFUSABLES,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sota_v7.trainer")

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

LABEL2ID = {m: i for i, m in enumerate(LABEL_ORDER)}

# Coarse label unification: map raw generator strings to canonical families.
_FAMILY_MAP = {
    "human": "Human", "gpt": "GPT", "openai": "GPT", "davinci": "GPT",
    "claude": "Claude", "anthropic": "Claude",
    "gemini": "Gemini", "bard": "Gemini", "palm": "Gemini",
    "grok": "Grok", "xai": "Grok",
    "mistral": "Mistral", "mixtral": "Mistral",
    "deepseek": "DeepSeek",
}


def unify_label(raw: str) -> Optional[str]:
    r = str(raw).lower()
    for k, v in _FAMILY_MAP.items():
        if k in r:
            return v
    return None  # unknown families dropped (don't pollute the boundary)


# =====================================================================
# STAGE 1 — DATA HYGIENE
# =====================================================================

def _minhash_signature(text: str, num_perm: int = 64, k: int = 5) -> Tuple[int, ...]:
    """Lightweight MinHash over character k-shingles (no external deps)."""
    s = re.sub(r"\s+", " ", text.lower())
    shingles = {s[i:i + k] for i in range(max(1, len(s) - k + 1))}
    if not shingles:
        return tuple([0] * num_perm)
    hashed = [int(hashlib.md5(sh.encode()).hexdigest(), 16) for sh in shingles]
    sig = []
    for p in range(num_perm):
        salt = p * 0x9E3779B1
        sig.append(min((h ^ salt) & 0xFFFFFFFF for h in hashed))
    return tuple(sig)


def dedup_and_split(rows: List[dict],
                    test_size: float = 0.10, val_size: float = 0.10,
                    near_dup_threshold: float = 0.8
                    ) -> Tuple[List[dict], List[dict], List[dict]]:
    """Remove near-duplicates, then GROUP-AWARE split.

    Group key precedence: explicit `group_id` > `source`/`domain` >
    a hash of the normalized prefix (so near-duplicate variants share a group
    and cannot leak across splits).
    """
    logger.info("Stage 1: dedup over %d rows", len(rows))
    seen_bands: Dict[Tuple, int] = {}
    kept: List[dict] = []
    bands = 16
    rows_per_band = 64 // bands
    for r in rows:
        text = str(r.get("text", ""))
        if len(text.strip()) < 1:
            continue
        sig = _minhash_signature(text)
        band_keys = [tuple(sig[b * rows_per_band:(b + 1) * rows_per_band])
                     for b in range(bands)]
        if any(bk in seen_bands for bk in band_keys):
            continue  # near-duplicate -> drop
        for bk in band_keys:
            seen_bands[bk] = 1
        kept.append(r)
    logger.info("  kept %d after near-dup removal (-%d)",
                len(kept), len(rows) - len(kept))

    def group_of(r: dict) -> str:
        for key in ("group_id", "source", "domain"):
            if r.get(key):
                return f"{key}:{r[key]}"
        norm = re.sub(r"\s+", " ", str(r.get("text", "")).lower())[:80]
        return "h:" + hashlib.md5(norm.encode()).hexdigest()[:12]

    groups: Dict[str, List[dict]] = {}
    for r in kept:
        groups.setdefault(group_of(r), []).append(r)
    gkeys = list(groups.keys())
    random.Random(GLOBAL_SEED).shuffle(gkeys)

    n = len(gkeys)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    test_g = set(gkeys[:n_test])
    val_g = set(gkeys[n_test:n_test + n_val])

    train, val, test = [], [], []
    for g, items in groups.items():
        dst = test if g in test_g else val if g in val_g else train
        dst.extend(items)
    logger.info("  group-aware split -> train=%d val=%d test=%d",
                len(train), len(val), len(test))
    return train, val, test


# =====================================================================
# STAGE 2 — MULTILEVEL ADVERSARIAL AUGMENTATION (train only)
# =====================================================================

_INV_CONFUSABLES = {}
for _k, _v in _CONFUSABLES.items():
    _INV_CONFUSABLES.setdefault(_v, _k)


class MultilevelAugmenter:
    """char-level + homoglyph + (optional) paraphrase augmentation.

    DAMAGE (2025.genaidetect-1.9): adversarial training is what makes
    detectors survive paraphrase/structural/char attacks; surface
    normalization alone is not enough. We inject the very attacks the model
    must resist — including homoglyph injection so Branch A learns invariance
    even if canonicalization is ever bypassed."""

    def __init__(self, char_prob=0.05, homoglyph_prob=0.15, p_apply=0.5):
        self.char_prob = char_prob
        self.homoglyph_prob = homoglyph_prob
        self.p_apply = p_apply

    def __call__(self, text: str) -> str:
        if random.random() > self.p_apply:
            return text
        words = text.split()
        for i, w in enumerate(words):
            if len(w) > 4 and random.random() < self.char_prob:
                cs = list(w)
                j = random.randint(0, len(cs) - 2)
                cs[j], cs[j + 1] = cs[j + 1], cs[j]
                words[i] = "".join(cs)
        out = " ".join(words)
        if random.random() < self.homoglyph_prob:
            out = "".join(_INV_CONFUSABLES.get(c, c)
                          if random.random() < 0.3 else c for c in out)
        return out


# =====================================================================
# STAGE 3 — SUPERVISED MULTI-CLASS TRAINING
# =====================================================================

MODEL_CONFIGS = {
    "deberta":   {"name": "microsoft/deberta-v3-base", "lr": 1e-5, "bs": 16},
    "modernbert": {"name": "answerdotai/ModernBERT-base", "lr": 2e-5, "bs": 32},
}


def _focal_ce(logits, labels, weight, gamma: float = 1.0):
    import torch
    import torch.nn.functional as F
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    pt = p.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
    focal = (1 - pt) ** gamma
    ce = F.nll_loss(logp, labels, weight=weight, reduction="none")
    return (focal * ce).mean()


def train_supervised(train_rows: List[dict], val_rows: List[dict],
                     out_dir: str, model_key: str = "modernbert",
                     epochs: int = 2, length_expert: Optional[str] = None,
                     max_length: int = 512) -> str:
    """Fine-tune one transformer. If `length_expert` is 'short'/'long', train
    only on the matching length bucket and save as `<key>_<expert>`."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification, get_scheduler)

    cfg = MODEL_CONFIGS[model_key]
    suffix = f"_{length_expert}" if length_expert else "_final"
    save_dir = os.path.join(out_dir, f"{model_key}{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    if os.path.exists(os.path.join(save_dir, "config.json")):
        logger.info("[%s%s] already trained, skipping.", model_key, suffix)
        return save_dir

    def in_bucket(r):
        if not length_expert:
            return True
        nw = len(str(r.get("text", "")).split())
        return (nw < SHORT_LONG_BOUNDARY) == (length_expert == "short")

    rows = [r for r in train_rows if in_bucket(r)]
    aug = MultilevelAugmenter()
    tok = AutoTokenizer.from_pretrained(cfg["name"], model_max_length=max_length)

    class DS(Dataset):
        def __init__(self, data, augment): self.d, self.a = data, augment
        def __len__(self): return len(self.d)
        def __getitem__(self, i):
            r = self.d[i]
            t, _ = canonicalize_text(str(r["text"]))
            if self.a:
                t = aug(t)
            enc = tok(t, truncation=True, max_length=max_length,
                      padding="max_length", return_tensors="pt")
            return {"input_ids": enc.input_ids[0],
                    "attention_mask": enc.attention_mask[0],
                    "labels": torch.tensor(LABEL2ID[r["label"]])}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["name"], num_labels=NUM_LABELS,
        id2label={i: l for i, l in enumerate(LABEL_ORDER)},
        label2id=LABEL2ID).to(dev)

    # MILD inverse-sqrt reweighting (not the aggressive median/count that
    # crushed Human in v6 and manufactured false positives).
    counts = np.bincount([LABEL2ID[r["label"]] for r in rows],
                         minlength=NUM_LABELS).astype(np.float64) + 1.0
    w = (counts.sum() / counts) ** 0.5
    w = w / w.mean()
    weight = torch.tensor(w, dtype=torch.float32, device=dev)

    dl = DataLoader(DS(rows, True), batch_size=cfg["bs"], shuffle=True,
                    num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    total = len(dl) * epochs
    sched = get_scheduler("linear", opt, num_warmup_steps=int(0.1 * total),
                          num_training_steps=total)

    model.train()
    for ep in range(epochs):
        for step, batch in enumerate(dl):
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            loss = _focal_ce(out.logits, batch["labels"], weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if step % 200 == 0:
                logger.info("[%s%s] ep%d step%d loss=%.4f",
                            model_key, suffix, ep, step, loss.item())
    model.save_pretrained(save_dir)
    tok.save_pretrained(save_dir)
    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_dir


# =====================================================================
# STAGE 4 — TEMPERATURE CALIBRATION (on disjoint val split)
# =====================================================================

def fit_temperature(branch: SupervisedBranch, val_rows: List[dict],
                    out_dir: str) -> dict:
    """Fit one temperature per model on VAL (never test). Saves calibration."""
    import torch
    if not branch._ensure():
        logger.warning("No supervised models; skipping temperature calibration.")
        return {}
    temps = {}
    # Collect logits per loaded variant on val.
    for variant, model in branch._models.items():
        tokz = branch._toks[variant]
        logits_all, labels_all = [], []
        with torch.no_grad():
            for r in val_rows:
                t, _ = canonicalize_text(str(r["text"]))
                enc = tokz(t, truncation=True, max_length=branch.max_length,
                           return_tensors="pt").to(branch._dev)
                logits_all.append(model(**enc).logits.float().cpu().numpy()[0])
                labels_all.append(LABEL2ID[r["label"]])
        L = torch.tensor(np.array(logits_all))
        y = torch.tensor(labels_all)
        T = torch.ones(1, requires_grad=True)
        opt = torch.optim.LBFGS([T], lr=0.01, max_iter=100)
        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(L / T.clamp(min=0.05), y)
            loss.backward(); return loss
        opt.step(closure)
        base = variant.rsplit("_", 1)[0]
        temps[base] = round(float(T.clamp(min=0.05).item()), 4)
    json.dump(temps, open(os.path.join(out_dir, "calibration.json"), "w"),
              indent=2)
    logger.info("Stage 4: temperatures = %s", temps)
    return temps


# =====================================================================
# STAGE 5 + 6 — META-LEARNER & CONFORMAL (on disjoint val split)
# =====================================================================

def fit_meta_and_conformal(out_dir: str, val_rows: List[dict],
                           target_fpr: float = 0.01,
                           enable_zeroshot: bool = True) -> None:
    logger.info("Stage 5/6: assembling branch features on %d val rows",
                len(val_rows))
    supervised = SupervisedBranch(out_dir)
    zeroshot = ZeroShotBranch() if enable_zeroshot else None
    style = StylometricExtractor()
    feat_names = (["p_ai_supervised", "zeroshot_score"]
                  + [f"style_{f}" for f in StylometricExtractor.FEATURES])

    X, y = [], []
    for r in val_rows:
        t, _ = canonicalize_text(str(r["text"]))
        mc = supervised.predict(t)
        p_sup = float(1.0 - mc[HUMAN_IDX]) if mc is not None else 0.0
        z = zeroshot.score(t) if zeroshot else float("nan")
        vec = np.concatenate([[p_sup, z], style.vectorize(t)])
        X.append(np.nan_to_num(vec, nan=0.0))
        y.append(0 if r["label"] == "Human" else 1)
    X = np.array(X, dtype=np.float64); y = np.array(y)

    meta = MetaLearner().fit(X, y, feat_names)
    meta.save(os.path.join(out_dir, "meta_learner.pkl"))
    logger.info("  meta-learner (%s) saved.", meta.kind)

    p_ai = meta.predict_proba_ai(X)
    human_scores = p_ai[y == 0]
    ai_scores = p_ai[y == 1]
    conf = ConformalCalibrator(target_fpr).fit(human_scores, ai_scores)
    json.dump(conf.to_dict(), open(os.path.join(out_dir, "conformal.json"), "w"),
              indent=2)
    logger.info("  conformal thresholds: %s", conf.to_dict())


# =====================================================================
# DATA LOADING
# =====================================================================

def load_rows(data_path: str, text_col="text", label_col="label",
              limit: Optional[int] = None) -> List[dict]:
    """Load rows from parquet/jsonl/csv via datasets or pandas."""
    rows: List[dict] = []
    try:
        from datasets import load_dataset
        ext = "parquet" if data_path.endswith((".parquet",)) or \
            os.path.isdir(data_path) else \
            "json" if data_path.endswith((".json", ".jsonl")) else "csv"
        ds = load_dataset(ext, data_files=data_path if not os.path.isdir(data_path)
                          else None, data_dir=data_path if os.path.isdir(data_path)
                          else None, split="train")
        for ex in ds:
            lbl = unify_label(ex.get(label_col) or ex.get("target_model", ""))
            if lbl is None:
                continue
            rows.append({"text": ex.get(text_col, ""), "label": lbl,
                         "source": ex.get("source"), "domain": ex.get("domain"),
                         "group_id": ex.get("group_id")})
            if limit and len(rows) >= limit:
                break
    except Exception as e:
        raise SystemExit(f"Failed to load data from {data_path}: {e}")
    logger.info("Loaded %d labeled rows", len(rows))
    return rows


# =====================================================================
# MAIN
# =====================================================================

def main():
    ap = argparse.ArgumentParser(description="Train xplagiax_ensemble_sota_v7")
    ap.add_argument("--data", required=True, help="parquet/jsonl/csv or dir")
    ap.add_argument("--out", default="./sota_v7_artifacts")
    ap.add_argument("--models", default="modernbert,deberta")
    ap.add_argument("--length-experts", action="store_true",
                    help="train short/long experts per model (SzegedAI)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--target-fpr", type=float, default=0.01)
    ap.add_argument("--no-zeroshot", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = load_rows(args.data, limit=args.limit)
    train, val, test = dedup_and_split(rows)
    json.dump({"n_train": len(train), "n_val": len(val), "n_test": len(test)},
              open(os.path.join(args.out, "split_sizes.json"), "w"), indent=2)
    # Persist the held-out test set for test.py (NEVER touched in training).
    import pickle
    pickle.dump(test, open(os.path.join(args.out, "test_rows.pkl"), "wb"))

    for key in args.models.split(","):
        key = key.strip()
        if args.length_experts:
            train_supervised(train, val, args.out, key, args.epochs, "short")
            train_supervised(train, val, args.out, key, args.epochs, "long")
        else:
            train_supervised(train, val, args.out, key, args.epochs)

    branch = SupervisedBranch(args.out, model_keys=tuple(
        k.strip() for k in args.models.split(",")))
    fit_temperature(branch, val, args.out)
    fit_meta_and_conformal(args.out, val, args.target_fpr,
                           enable_zeroshot=not args.no_zeroshot)
    logger.info("DONE. Artifacts in %s — run test.py next.", args.out)


if __name__ == "__main__":
    main()
