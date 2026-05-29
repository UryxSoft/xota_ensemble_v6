"""
xplagiax_ensemble_sota_v7 — detector_final.py
==============================================
Production inference for the SOTA v7 AI-text detector.

Design goal: beat commercial detectors (GPTZero, Originality.ai, Copyleaks,
Turnitin) on the metrics that actually matter — MCC and TPR at a *fixed,
guaranteed* low False-Positive Rate — by combining signals that fail
INDEPENDENTLY and by abstaining instead of accusing when uncertain.

Why this can outperform the commercial tools
---------------------------------------------
Every major commercial detector is, at its core, a single-mechanism system
(a fine-tuned transformer + some perplexity) that is English-centric and
brittle to (a) paraphrase / "humanizer" tools, (b) non-native English
(their #1 source of false positives), (c) short text, and (d) the newest
models. v7 attacks exactly those failure modes:

  Branch A  Supervised multi-class transformer ensemble (heterogeneous archs).
            Multi-class > binary generalizes better (SzegedAI, GenAIDetect'25,
            2025.genaidetect-1.15 — 41-class test F1 0.826 vs binary 0.795).
            Length-routed experts (their single best config, test F1 0.827).
  Branch Z  Zero-shot cross-perplexity (Binoculars, Hans et al. ICML'24 +
            Fast-DetectGPT, Bao et al. ICLR'24). Robust cross-domain, low-FP,
            no training. Mechanistically uncorrelated with Branch A.
  Branch F  Stylometric + proxy-perplexity features (burstiness, log-rank,
            type-token, n-gram repetition). CPU, fast, catches humanizers.

  Stacking meta-learner over [A, Z, F]  ->  P(AI).
  Split-conformal calibration  ->  decision threshold with a *finite-sample
            guarantee* that FPR <= alpha on the calibration distribution.
  Unicode canonicalization layer  ->  defeats the homoglyph attack that broke
            ALL 7 SOTA detectors in 2025.genaidetect-1.1 (MCC 0.64 -> -0.01).

Runnable surface
----------------
The canonicalizer, stylometric features, conformal calibrator and metrics are
pure-python/numpy and run anywhere. The transformer (Branch A) and zero-shot
(Branch Z) branches lazy-import torch/transformers and degrade gracefully when
unavailable, so this module imports cleanly on a CPU box with no ML stack.
"""

from __future__ import annotations

import os
import re
import json
import math
import logging
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("sota_v7.detector")

# =====================================================================
# 0. CONFIG
# =====================================================================

LABEL_ORDER = ["Human", "GPT", "Claude", "Gemini", "Grok", "Mistral", "DeepSeek"]
HUMAN_IDX = LABEL_ORDER.index("Human")
NUM_LABELS = len(LABEL_ORDER)

# Length routing boundary (word count). SzegedAI's best config used
# length-specialized experts; transformers are unreliable on very short text.
SHORT_LONG_BOUNDARY = 60          # words
MIN_WORDS_FOR_VERDICT = 25        # below this -> INCONCLUSIVE by default

# Default conformal target: at most 1% of genuine humans flagged as AI.
DEFAULT_TARGET_FPR = 0.01


# =====================================================================
# 1. UNICODE CANONICALIZATION  (defense for 2025.genaidetect-1.1)
# =====================================================================
# The homoglyph attack replaces Latin letters with visually identical
# Cyrillic/Greek codepoints, destroying the tokenizer's embeddings. We
# canonicalize BEFORE any model sees the text, and we also *report* the
# tampering: heavy homoglyph use is itself strong evidence of evasion.

# Curated confusables map (Cyrillic / Greek / fullwidth -> ASCII Latin).
# Covers the high-frequency lookalikes used by real evasion tooling.
_CONFUSABLES = {
    # Cyrillic uppercase
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y", "І": "I", "Ј": "J",
    "Ѕ": "S", "Ԛ": "Q", "Ԝ": "W", "Ѓ": "G",
    # Cyrillic lowercase
    "а": "a", "е": "e", "о": "o", "с": "c", "р": "p", "х": "x", "у": "y",
    "к": "k", "м": "m", "т": "t", "н": "h", "в": "b", "і": "i", "ј": "j",
    "ѕ": "s", "ԛ": "q", "ԝ": "w", "ѐ": "e", "г": "r",
    # Greek
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K",
    "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
    "α": "a", "ο": "o", "ν": "v", "ρ": "p", "τ": "t", "υ": "u", "ι": "i",
    "κ": "k", "μ": "u", "χ": "x", "ε": "e", "ϲ": "c", "ѵ": "v",
}

# Zero-width / invisible characters used to break tokenization without
# visible change. Stripped entirely.
_ZERO_WIDTH = dict.fromkeys(map(ord, [
    "​", "‌", "‍", "⁠", "﻿", "᠎",
    "­",  # soft hyphen
]), None)

_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass
class TamperReport:
    """Evidence of obfuscation found during canonicalization."""
    homoglyphs_replaced: int = 0
    zero_width_removed: int = 0
    homoglyph_ratio: float = 0.0      # replaced / total letters
    suspicious: bool = False          # ratio above threshold -> likely evasion

    def to_dict(self) -> dict:
        return {
            "homoglyphs_replaced": self.homoglyphs_replaced,
            "zero_width_removed": self.zero_width_removed,
            "homoglyph_ratio": round(self.homoglyph_ratio, 4),
            "suspicious": self.suspicious,
        }


def canonicalize_text(text: str,
                      suspicious_ratio: float = 0.02) -> Tuple[str, TamperReport]:
    """Return (canonical_text, tamper_report).

    Steps:
      1. NFKC normalization (fold fullwidth / compatibility forms).
      2. Strip zero-width / invisible characters.
      3. Map Cyrillic/Greek confusables to Latin.
      4. Normalize whitespace and hyphenated line breaks (SzegedAI normalizer).
    """
    if not isinstance(text, str) or not text:
        return "", TamperReport()

    zw_before = sum(1 for ch in text if ord(ch) in _ZERO_WIDTH)
    text = text.translate(_ZERO_WIDTH)
    text = unicodedata.normalize("NFKC", text)

    replaced = 0
    if any(ch in _CONFUSABLES for ch in text):
        out = []
        for ch in text:
            mapped = _CONFUSABLES.get(ch)
            if mapped is not None:
                out.append(mapped)
                replaced += 1
            else:
                out.append(ch)
        text = "".join(out)

    # SzegedAI-style normalization: join hyphen line breaks, collapse newlines.
    text = re.sub(r"(\w+)[-‐‑]\s*\n\s*(\w+)", r"\1\2", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text).strip()

    total_letters = sum(1 for _ in _LATIN_RE.finditer(text)) + replaced
    ratio = replaced / total_letters if total_letters else 0.0
    report = TamperReport(
        homoglyphs_replaced=replaced,
        zero_width_removed=zw_before,
        homoglyph_ratio=ratio,
        suspicious=(ratio >= suspicious_ratio or zw_before > 0),
    )
    return text, report


# =====================================================================
# 2. BRANCH F — STYLOMETRIC + PROXY-PERPLEXITY FEATURES (pure python)
# =====================================================================

_WORD_RE = re.compile(r"[A-Za-z']+")
_SENT_RE = re.compile(r"[.!?]+")


class StylometricExtractor:
    """Fast, dependency-free features. Catch humanizers / paraphrase that
    smooth perplexity but leave statistical fingerprints (DAMAGE,
    2025.genaidetect-1.9 — semantic/structural features survive attacks)."""

    FEATURES = (
        "word_count", "avg_word_len", "type_token_ratio", "hapax_ratio",
        "sent_len_mean", "sent_len_var", "burstiness", "bigram_repeat",
        "trigram_repeat", "punct_entropy", "stopword_ratio", "uppercase_ratio",
    )

    _STOP = frozenset((
        "the a an and or but of to in on for with at by from as is are was were "
        "be been being it its this that these those i you he she we they"
    ).split())

    def vectorize(self, text: str) -> np.ndarray:
        words = _WORD_RE.findall(text.lower())
        n = len(words)
        if n == 0:
            return np.zeros(len(self.FEATURES), dtype=np.float32)

        uniq = set(words)
        avg_word_len = float(np.mean([len(w) for w in words]))
        ttr = len(uniq) / n
        hapax = sum(1 for w in uniq if words.count(w) == 1) / n if n < 2000 else \
            len([w for w in uniq if words.count(w) == 1]) / n

        sents = [s for s in _SENT_RE.split(text) if s.strip()]
        sl = [len(_WORD_RE.findall(s)) for s in sents] or [n]
        sl_mean = float(np.mean(sl))
        sl_var = float(np.var(sl))
        # Burstiness: human text is "bursty" (high variance of sentence length);
        # LLM text is more uniform. (Coefficient of variation, sign per Goh'02.)
        burstiness = ((np.std(sl) - sl_mean) / (np.std(sl) + sl_mean)
                      if (np.std(sl) + sl_mean) > 0 else 0.0)

        bigrams = list(zip(words, words[1:]))
        trigrams = list(zip(words, words[1:], words[2:]))
        bi_rep = 1.0 - (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
        tri_rep = 1.0 - (len(set(trigrams)) / len(trigrams)) if trigrams else 0.0

        puncts = [c for c in text if c in ",.;:!?-—\"'()"]
        punct_entropy = self._entropy(puncts)
        stop_ratio = sum(1 for w in words if w in self._STOP) / n
        upper_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))

        return np.array([
            n, avg_word_len, ttr, hapax, sl_mean, sl_var, burstiness,
            bi_rep, tri_rep, punct_entropy, stop_ratio, upper_ratio,
        ], dtype=np.float32)

    @staticmethod
    def _entropy(items: List[str]) -> float:
        if not items:
            return 0.0
        from collections import Counter
        c = Counter(items)
        tot = sum(c.values())
        return float(-sum((v / tot) * math.log2(v / tot) for v in c.values()))


# =====================================================================
# 3. BRANCH Z — ZERO-SHOT CROSS-PERPLEXITY (Binoculars / Fast-DetectGPT)
# =====================================================================

class ZeroShotBranch:
    """Binoculars-style cross-perplexity (Hans et al., ICML 2024).

    score = log_ppl(text | observer) / cross_ppl(text | observer, performer)

    Human text scores HIGH (observer is 'surprised'); machine text scores LOW
    (both models agree). Crucially this is *training-free* and degrades far
    more gracefully under domain shift than supervised classifiers — the
    branch that keeps FPR low when Branch A is out of distribution.

    Lazy-loads two small causal LMs. Returns np.nan if torch unavailable so
    the meta-learner can route around it.
    """

    def __init__(self, observer: str = "gpt2", performer: str = "gpt2-medium",
                 device: str = "auto", max_tokens: int = 512):
        self.observer_name = observer
        self.performer_name = performer
        self._device = device
        self.max_tokens = max_tokens
        self._loaded = False
        self._obs = self._perf = self._tok = None

    def _ensure(self) -> bool:
        if self._loaded:
            return self._obs is not None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            dev = ("cuda" if torch.cuda.is_available() else "cpu") \
                if self._device == "auto" else self._device
            self._torch = torch
            self._tok = AutoTokenizer.from_pretrained(self.observer_name)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            self._obs = AutoModelForCausalLM.from_pretrained(
                self.observer_name).to(dev).eval()
            self._perf = AutoModelForCausalLM.from_pretrained(
                self.performer_name).to(dev).eval()
            self._dev = dev
        except Exception as e:                       # pragma: no cover
            logger.warning("ZeroShotBranch disabled (no torch/models): %s", e)
            self._obs = None
        self._loaded = True
        return self._obs is not None

    def score(self, text: str) -> float:
        """Binoculars cross-perplexity ratio. Lower => more machine-like."""
        if not self._ensure():
            return float("nan")
        torch = self._torch
        with torch.no_grad():
            enc = self._tok(text, return_tensors="pt", truncation=True,
                            max_length=self.max_tokens).to(self._dev)
            ids = enc.input_ids
            if ids.shape[1] < 2:
                return float("nan")
            obs_logits = self._obs(**enc).logits
            perf_logits = self._perf(**enc).logits

            # log perplexity under observer
            ll = torch.nn.functional.cross_entropy(
                obs_logits[:, :-1].reshape(-1, obs_logits.size(-1)),
                ids[:, 1:].reshape(-1), reduction="mean")
            log_ppl = ll.item()

            # cross-perplexity: observer's surprise at performer's distribution
            obs_logp = torch.nn.functional.log_softmax(obs_logits[:, :-1], dim=-1)
            perf_p = torch.nn.functional.softmax(perf_logits[:, :-1], dim=-1)
            x_ce = -(perf_p * obs_logp).sum(-1).mean().item()

            return log_ppl / x_ce if x_ce > 1e-6 else float("nan")


# =====================================================================
# 4. BRANCH A — SUPERVISED MULTI-CLASS ENSEMBLE (length-routed)
# =====================================================================

class _TempScaler:
    """Platt temperature scaling fit during training, loaded here."""
    def __init__(self, t: float = 1.0):
        self.t = max(0.05, float(t))

    def probs(self, logits: np.ndarray) -> np.ndarray:
        z = logits / self.t
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)


class SupervisedBranch:
    """Heterogeneous transformer ensemble producing calibrated multi-class
    probabilities, collapsed to P(AI) = 1 - P(Human).

    Models are loaded from `ensemble_dir/<key>_final`. Optional per-length
    experts (`<key>_short`, `<key>_long`) implement SzegedAI length routing.
    """

    def __init__(self, ensemble_dir: str, model_keys: Tuple[str, ...] =
                 ("deberta", "modernbert"), device: str = "auto",
                 max_length: int = 512):
        self.dir = ensemble_dir
        self.keys = model_keys
        self._device = device
        self.max_length = max_length
        self._loaded = False
        self._models: Dict[str, object] = {}
        self._toks: Dict[str, object] = {}
        self._scalers: Dict[str, _TempScaler] = {}
        self._load_calibration()

    def _load_calibration(self) -> None:
        path = os.path.join(self.dir, "calibration.json")
        if os.path.exists(path):
            try:
                temps = json.load(open(path))
                self._scalers = {k: _TempScaler(v) for k, v in temps.items()}
            except Exception as e:
                logger.warning("calibration load failed: %s", e)

    def _ensure(self) -> bool:
        if self._loaded:
            return bool(self._models)
        try:
            import torch
            from transformers import (AutoTokenizer,
                                      AutoModelForSequenceClassification)
            dev = ("cuda" if torch.cuda.is_available() else "cpu") \
                if self._device == "auto" else self._device
            self._torch, self._dev = torch, dev
            for key in self.keys:
                # prefer length experts if present
                for variant in (f"{key}_short", f"{key}_long", f"{key}_final"):
                    d = os.path.join(self.dir, variant)
                    if os.path.isdir(d):
                        self._toks[variant] = AutoTokenizer.from_pretrained(
                            d, model_max_length=self.max_length)
                        self._models[variant] = \
                            AutoModelForSequenceClassification.from_pretrained(
                                d).to(dev).eval()
        except Exception as e:                       # pragma: no cover
            logger.warning("SupervisedBranch disabled: %s", e)
            self._models = {}
        self._loaded = True
        return bool(self._models)

    def _pick(self, n_words: int) -> List[str]:
        """Route to length experts if available, else *_final."""
        avail = list(self._models.keys())
        suffix = "_short" if n_words < SHORT_LONG_BOUNDARY else "_long"
        routed = [v for v in avail if v.endswith(suffix)]
        return routed if routed else [v for v in avail if v.endswith("_final")] or avail

    def predict(self, text: str) -> Optional[np.ndarray]:
        """Return ensemble multi-class probability vector, or None."""
        if not self._ensure():
            return None
        torch = self._torch
        n_words = len(_WORD_RE.findall(text))
        chosen = self._pick(n_words)
        probs = []
        with torch.no_grad():
            for variant in chosen:
                tok = self._toks[variant](
                    text, truncation=True, max_length=self.max_length,
                    return_tensors="pt").to(self._dev)
                logits = self._models[variant](**tok).logits.float().cpu().numpy()
                base = variant.rsplit("_", 1)[0]
                scaler = self._scalers.get(base) or self._scalers.get(variant) \
                    or _TempScaler(1.0)
                probs.append(scaler.probs(logits)[0])
        return np.mean(probs, axis=0) if probs else None


# =====================================================================
# 5. META-LEARNER (stacking)  +  CONFORMAL FP CONTROL
# =====================================================================

class MetaLearner:
    """Stacking model over branch features. Prefers LightGBM, falls back to
    logistic regression, falls back to a transparent linear rule so the
    detector still produces a sane P(AI) before the meta-model is fit."""

    def __init__(self):
        self.model = None
        self.kind = "heuristic"
        self.feature_names: List[str] = []

    # ---- training ----
    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_names: List[str]) -> "MetaLearner":
        self.feature_names = feature_names
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMClassifier(
                n_estimators=400, learning_rate=0.03, num_leaves=31,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0)
            self.model.fit(X, y)
            self.kind = "lightgbm"
        except Exception:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import make_pipeline
            self.model = make_pipeline(
                StandardScaler(with_mean=True),
                LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
            self.model.fit(X, y)
            self.kind = "logreg"
        return self

    def predict_proba_ai(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:                       # cold-start heuristic
            return self._heuristic(X)
        p = self.model.predict_proba(X)
        return p[:, 1]

    def _heuristic(self, X: np.ndarray) -> np.ndarray:
        # Before fitting: use the supervised P(AI) column if present (idx 0),
        # else neutral 0.5. Keeps the system usable end-to-end out of the box.
        if "p_ai_supervised" in self.feature_names:
            j = self.feature_names.index("p_ai_supervised")
            return np.clip(X[:, j], 0, 1)
        return np.full(X.shape[0], 0.5, dtype=np.float64)

    # ---- persistence ----
    def save(self, path: str) -> None:
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"kind": self.kind, "model": self.model,
                         "feature_names": self.feature_names}, f)

    @classmethod
    def load(cls, path: str) -> "MetaLearner":
        import pickle
        obj = cls()
        if os.path.exists(path):
            d = pickle.load(open(path, "rb"))
            obj.model, obj.kind = d["model"], d["kind"]
            obj.feature_names = d["feature_names"]
        return obj


class ConformalCalibrator:
    """Split-conformal threshold giving a finite-sample FPR guarantee.

    Given P(AI) scores on a held-out HUMAN calibration set, pick the smallest
    threshold tau such that the empirical fraction of humans with score >= tau
    is <= target_fpr. Decisions below tau (but above a human-confidence band)
    fall into INCONCLUSIVE — we ABSTAIN rather than risk a false accusation.
    This is the core of "no false positives": Turnitin-style conservatism made
    explicit and statistically calibrated.
    """

    def __init__(self, target_fpr: float = DEFAULT_TARGET_FPR):
        self.target_fpr = target_fpr
        # Before fitting we ABSTAIN: tau_ai is unreachable so an uncalibrated
        # detector never produces a false accusation; it returns INCONCLUSIVE.
        self.tau_ai: float = 1.01         # >= tau_ai  -> AI_DETECTED
        self.tau_human: float = 0.49      # <= tau_human -> HUMAN
        self._fitted = False

    def fit(self, human_scores: np.ndarray,
            ai_scores: Optional[np.ndarray] = None) -> "ConformalCalibrator":
        """Split-conformal threshold with a finite-sample FPR guarantee.

        tau_ai is the smallest cutoff such that at most `target_fpr` of the
        calibration humans are flagged. We use the conformal (n+1) rank
        correction so the guarantee holds in expectation on exchangeable data.
        NOTE: tau_ai depends ONLY on human scores — never on AI scores —
        otherwise the FPR guarantee is void.
        """
        hs = np.sort(human_scores)
        n = len(hs)
        if n == 0:
            self._fitted = True
            return self
        # rank for the (1 - alpha) conformal quantile
        k = int(np.ceil((n + 1) * (1.0 - self.target_fpr)))
        idx = min(max(k - 1, 0), n - 1)
        self.tau_ai = float(hs[idx])
        self.tau_human = float(np.quantile(hs, 0.50))
        self._fitted = True
        return self

    def decide(self, p_ai: float) -> str:
        if p_ai >= self.tau_ai:
            return "AI_DETECTED"
        if p_ai <= self.tau_human:
            return "HUMAN"
        return "INCONCLUSIVE"

    def to_dict(self) -> dict:
        return {"target_fpr": self.target_fpr, "tau_ai": self.tau_ai,
                "tau_human": self.tau_human}

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalCalibrator":
        o = cls(d.get("target_fpr", DEFAULT_TARGET_FPR))
        o.tau_ai, o.tau_human, o._fitted = d["tau_ai"], d["tau_human"], True
        return o


# =====================================================================
# 6. ORCHESTRATOR
# =====================================================================

@dataclass
class Verdict:
    verdict: str                       # HUMAN | AI_DETECTED | INCONCLUSIVE | TAMPERED
    p_ai: float
    confidence: float
    detected_model: Optional[str]
    tamper: dict
    branch_scores: Dict[str, float]
    multiclass: Dict[str, float] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class SOTADetector:
    """Top-level v7 detector. Assembles branch features, runs the meta-learner,
    and applies conformal FP control. Safe to instantiate with no trained
    artifacts (degrades to the supervised/heuristic path)."""

    def __init__(self, ensemble_dir: str = ".", target_fpr: float = DEFAULT_TARGET_FPR,
                 enable_zeroshot: bool = True, device: str = "auto"):
        self.dir = ensemble_dir
        self.style = StylometricExtractor()
        self.supervised = SupervisedBranch(ensemble_dir, device=device)
        self.zeroshot = ZeroShotBranch(device=device) if enable_zeroshot else None
        self.meta = MetaLearner.load(os.path.join(ensemble_dir, "meta_learner.pkl"))
        cal_path = os.path.join(ensemble_dir, "conformal.json")
        self.conformal = (ConformalCalibrator.from_dict(json.load(open(cal_path)))
                          if os.path.exists(cal_path)
                          else ConformalCalibrator(target_fpr))

    # ---- feature assembly (shared with trainer.py) ----
    def feature_names(self) -> List[str]:
        names = ["p_ai_supervised", "zeroshot_score"]
        names += [f"style_{f}" for f in StylometricExtractor.FEATURES]
        return names

    def assemble_features(self, text: str
                          ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (feature_vector, multiclass_probs_or_None, p_ai_supervised)."""
        mc = self.supervised.predict(text)
        if mc is not None:
            p_ai_sup = float(1.0 - mc[HUMAN_IDX])
        else:
            p_ai_sup = float("nan")
        z = self.zeroshot.score(text) if self.zeroshot else float("nan")
        style = self.style.vectorize(text)
        feats = np.concatenate([[p_ai_sup, z], style]).astype(np.float64)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats, mc, p_ai_sup

    # ---- single text ----
    def classify(self, text: str) -> Verdict:
        clean, tamper = canonicalize_text(text)
        n_words = len(_WORD_RE.findall(clean))

        if n_words < MIN_WORDS_FOR_VERDICT:
            return Verdict("INCONCLUSIVE", 0.5, 0.0, None, tamper.to_dict(),
                           {}, flags=["too_short"])

        feats, mc, p_ai_sup = self.assemble_features(clean)
        p_ai = float(self.meta.predict_proba_ai(feats.reshape(1, -1))[0])

        verdict = self.conformal.decide(p_ai)
        flags: List[str] = []

        # Tampering is itself evidence of evasion: never clear a tampered text
        # as HUMAN; surface it for human review.
        if tamper.suspicious:
            flags.append("obfuscation_detected")
            if verdict == "HUMAN":
                verdict = "TAMPERED"

        detected = None
        mc_dict: Dict[str, float] = {}
        if mc is not None:
            mc_dict = {LABEL_ORDER[i]: round(float(mc[i]), 4)
                       for i in range(NUM_LABELS)}
            if verdict == "AI_DETECTED":
                ai_idx = int(np.argmax([mc[i] if i != HUMAN_IDX else -1
                                        for i in range(NUM_LABELS)]))
                detected = LABEL_ORDER[ai_idx]

        confidence = abs(p_ai - 0.5) * 2.0
        return Verdict(verdict, round(p_ai, 4), round(confidence, 4),
                       detected, tamper.to_dict(),
                       {"p_ai_supervised": round(p_ai_sup, 4)
                        if not math.isnan(p_ai_sup) else None,
                        "zeroshot": round(feats[1], 4)},
                       mc_dict, flags)

    # ---- document level (sliding window + smoothing) ----
    def classify_document(self, text: str, window_words: int = 120,
                          stride_words: int = 80) -> dict:
        clean, tamper = canonicalize_text(text)
        words = clean.split()
        if len(words) < MIN_WORDS_FOR_VERDICT:
            return {"verdict": "INSUFFICIENT_TEXT", "tamper": tamper.to_dict()}

        windows, spans = [], []
        for start in range(0, max(1, len(words) - window_words + 1), stride_words):
            chunk = words[start:start + window_words]
            if len(chunk) >= MIN_WORDS_FOR_VERDICT:
                windows.append(" ".join(chunk))
                spans.append((start, start + len(chunk)))
        if not windows:
            windows, spans = [clean], [(0, len(words))]

        win_p = []
        for w in windows:
            f, _, _ = self.assemble_features(w)
            win_p.append(float(self.meta.predict_proba_ai(f.reshape(1, -1))[0]))
        win_p = np.array(win_p)

        # light temporal smoothing to localize AI spans (mixed/edited docs)
        if len(win_p) >= 3:
            k = np.array([0.25, 0.5, 0.25])
            win_p = np.convolve(win_p, k, mode="same")

        ai_ratio = float(np.mean(win_p >= self.conformal.tau_ai))
        doc_p = float(np.mean(win_p))
        if tamper.suspicious:
            verdict = "TAMPERED"
        elif ai_ratio > 0.70:
            verdict = "AI_DETECTED"
        elif ai_ratio > 0.30:
            verdict = "MIXED_CONTENT"
        elif doc_p <= self.conformal.tau_human:
            verdict = "HUMAN"
        else:
            verdict = "INCONCLUSIVE"

        return {
            "verdict": verdict,
            "doc_p_ai": round(doc_p, 4),
            "ai_window_ratio": round(ai_ratio, 4),
            "n_windows": len(windows),
            "tamper": tamper.to_dict(),
            "windows": [{"span": s, "p_ai": round(float(p), 4)}
                        for s, p in zip(spans, win_p)],
        }


__all__ = [
    "canonicalize_text", "TamperReport", "StylometricExtractor",
    "ZeroShotBranch", "SupervisedBranch", "MetaLearner",
    "ConformalCalibrator", "SOTADetector", "Verdict",
    "LABEL_ORDER", "HUMAN_IDX", "NUM_LABELS",
]


if __name__ == "__main__":
    # Smoke test: runs with zero trained artifacts (heuristic path).
    det = SOTADetector(enable_zeroshot=False)
    demo = ("The mitochondria is the powerhouse of the cell, and it plays a "
            "central role in energy production through oxidative phosphorylation. "
            "This process is essential for sustaining cellular function.")
    v = det.classify(demo)
    print(json.dumps(v.to_dict(), indent=2, ensure_ascii=False))
    # Homoglyph attack demo: Cyrillic lookalikes get canonicalized + flagged.
    attacked = demo.replace("a", "а").replace("e", "е").replace("o", "о")
    va = det.classify(attacked)
    print("homoglyph tamper:", va.tamper)
