"""
SOTA AI Text Detector v4.0.0 — Production
==========================================
Aligned with Ensemble Training Pipeline v6.1.

v5.3 → v6.1 changes
---------------------
  [ARCH]    3-model ensemble: DeBERTa-v3-base + ELECTRA-base + ModernBERT-base.
            Each model has its own tokenizer — _ensemble_inference tokenizes
            per-model instead of sharing one tokenizer.
  [ARCH]    StatisticalFeatureExtractor (distilgpt2 perplexity) replaced by
            StylometricProfiler.compute_stats() — CPU-only, zero GPU overhead,
            richer features (BST formula, MATTR adaptive, hapax legomena).
  [ARCH]    Confidence calibration updated for new stat features:
            ppl removed, BST (burstiness) now uses [-1,+1] range,
            hapax_legomena_ratio added as AI signal.
  [ARCH]    Ensemble weights equal (33/33/33) matching v6.1 soft voting.
            Set ENSEMBLE_WEIGHTS to override (e.g. for v7.0 40/35/25).
  [API]     EnsembleDetector(ensemble_dir=...) replaces SOTAAIDetector.
            SOTAAIDetector kept as backward-compat alias.
  [FIX]     All v3.2.1 fixes preserved (BatchItem, narrow exceptions, PEP8).

External behaviour identical to v3.2.1 except:
  - detect() returns per-model breakdown in raw_scores
  - StatisticalFeatures now include BST, VR, HL, RW instead of ppl/entropy

Requires
--------
  torch, transformers, textstat, numpy  (required)
  stylometric_profiler.py              (required — same directory)
  spacy + en_core_web_sm               (optional — richer BST/DC features)
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import textstat
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from batch_types import BatchItem

# ── StylometricProfiler replaces distilgpt2 StatisticalFeatureExtractor ──
try:
    from stylometric_profiler import StylometricProfiler
    _PROFILER = StylometricProfiler()
    _STYLO_AVAILABLE = True
except ImportError:
    _PROFILER = None
    _STYLO_AVAILABLE = False

logger = logging.getLogger(__name__)

_DEFAULT_MAX_INPUT_CHARS: int = 100_000
_FK_TRUNCATION_CHARS: int     = 3_000

# Ensemble weights — equal for v6.1, override for v7.0 (40/35/25)
ENSEMBLE_WEIGHTS: Dict[str, float] = {
    "deberta":    1 / 3,
    "electra":    1 / 3,
    "modernbert": 1 / 3,
}


# ═══════════════════════════════════════════════════════════════
# Enums & data classes (unchanged from v3.2.1)
# ═══════════════════════════════════════════════════════════════

class ModelFamily(Enum):
    HUMAN              = "human"
    FRONTIER_OPENAI    = "frontier_openai"
    FRONTIER_ANTHROPIC = "frontier_anthropic"
    FRONTIER_GOOGLE    = "frontier_google"
    REASONING          = "reasoning"
    OPEN_WEIGHTS_LARGE = "open_weights_large"
    OPEN_WEIGHTS_MEDIUM = "open_weights_medium"
    LEGACY             = "legacy"


_COARSE_FAMILY_MAP: Dict[str, ModelFamily] = {
    "Human":    ModelFamily.HUMAN,
    "human":    ModelFamily.HUMAN,
    "GPT":      ModelFamily.FRONTIER_OPENAI,
    "Claude":   ModelFamily.FRONTIER_ANTHROPIC,
    "Gemini":   ModelFamily.FRONTIER_GOOGLE,
    "DeepSeek": ModelFamily.REASONING,
    "Grok":     ModelFamily.REASONING,
    "Mistral":  ModelFamily.OPEN_WEIGHTS_LARGE,
    "Unknown AI": ModelFamily.LEGACY,
}


@dataclass
class DetectionResult:
    prediction:             str
    confidence:             float
    calibrated_confidence:  float
    detected_model:         Optional[str]
    model_family:           Optional[ModelFamily]
    statistical_features:   Dict[str, float]
    raw_scores:             Dict[str, float]
    uncertainty_zone:       bool


@dataclass
class LabelMapping:
    id2label:        Dict[int, str]           = field(default_factory=dict)
    label2id:        Dict[str, int]           = field(default_factory=dict)
    model_to_family: Dict[str, ModelFamily]   = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        label2id: Dict[str, int],
        id2label: Optional[Dict[str, str]] = None,
    ) -> "LabelMapping":
        int_id2label = (
            {int(k): v for k, v in id2label.items()}
            if id2label
            else {v: k for k, v in label2id.items()}
        )
        model_to_family = {
            name: _COARSE_FAMILY_MAP.get(name, ModelFamily.LEGACY)
            for name in label2id
        }
        return cls(
            id2label=int_id2label,
            label2id=dict(label2id),
            model_to_family=model_to_family,
        )

    @classmethod
    def from_json_file(cls, path: str) -> "LabelMapping":
        with open(path, "r") as f:
            data = json.load(f)
        mapping = cls.from_dict(
            label2id=data["label2id"],
            id2label=data.get("id2label"),
        )
        logger.info(
            "Loaded %d labels from %s: %s",
            mapping.num_labels, path,
            list(mapping.label2id.keys()),
        )
        return mapping

    @property
    def num_labels(self) -> int:
        return len(self.id2label)

    @property
    def human_index(self) -> int:
        for name in ("Human", "human"):
            if name in self.label2id:
                return self.label2id[name]
        raise KeyError(f"No human label in: {list(self.label2id.keys())}")

    def get_family(self, model_name: str) -> ModelFamily:
        return self.model_to_family.get(
            model_name,
            _COARSE_FAMILY_MAP.get(model_name, ModelFamily.LEGACY),
        )


# ═══════════════════════════════════════════════════════════════
# Statistical feature extractor — StylometricProfiler bridge
#
# Replaces distilgpt2-based StatisticalFeatureExtractor from v3.2.1.
# Uses StylometricProfiler.compute_stats() — pure CPU, no GPU model.
#
# Feature mapping:
#   OLD (distilgpt2)          NEW (StylometricProfiler)
#   ppl                    →  removed (would need per-LLM API calls)
#   entropy = log(ppl)     →  removed
#   burstiness             →  burstiness (sigma-mu)/(sigma+mu) [-1,+1]
#   lexical_diversity      →  vocabulary_richness (MATTR adaptive)
#   avg_sentence_length    →  avg_sentence_length
#   sentence_len_variance  →  sentence_length_variance
#   —                      →  hapax_legomena_ratio (new — strong AI signal)
#   —                      →  rare_word_ratio (new)
#   —                      →  avg_dep_distance (new, spaCy only)
#   —                      →  complex_sentence_ratio (new, spaCy only)
#   fk_score               →  fk_score (unchanged, from textstat)
# ═══════════════════════════════════════════════════════════════

class StatisticalFeatureExtractor:
    """
    CPU-only statistical feature extractor using StylometricProfiler.
    Zero GPU overhead — replaces distilgpt2 perplexity model.
    """

    def __init__(self, device: torch.device) -> None:
        # device kept for API compat — not used (pure CPU)
        self.device = device
        if not _STYLO_AVAILABLE:
            logger.warning(
                "stylometric_profiler.py not found. "
                "Statistical features will be empty."
            )

    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract all statistical features for a single text."""
        if not text.strip():
            return self._empty_features()

        if not _STYLO_AVAILABLE or _PROFILER is None:
            return self._empty_features()

        try:
            stats = _PROFILER.compute_stats(text)
        except Exception as exc:
            logger.debug("StylometricProfiler.compute_stats failed: %s", exc)
            return self._empty_features()

        return {
            # Core features (backward compat keys preserved)
            "burstiness":              stats.get("burstiness", 0.0),
            "lexical_diversity":       stats.get("vocabulary_richness", 0.5),
            "avg_sentence_length":     stats.get("avg_sentence_length", 15.0),
            "sentence_length_variance":stats.get("sentence_length_variance", 0.0),
            # New v6.1 features
            "vocabulary_richness":     stats.get("vocabulary_richness", 0.5),
            "hapax_legomena_ratio":    stats.get("hapax_legomena_ratio", 0.5),
            "rare_word_ratio":         stats.get("rare_word_ratio", 0.3),
            "avg_word_length":         stats.get("avg_word_length", 4.5),
            "avg_dep_distance":        stats.get("avg_dep_distance", 0.0),
            "complex_sentence_ratio":  stats.get("complex_sentence_ratio", 0.0),
        }

    @staticmethod
    def _empty_features() -> Dict[str, float]:
        return {
            "burstiness":               0.0,
            "lexical_diversity":        0.0,
            "avg_sentence_length":      0.0,
            "sentence_length_variance": 0.0,
            "vocabulary_richness":      0.0,
            "hapax_legomena_ratio":     0.0,
            "rare_word_ratio":          0.0,
            "avg_word_length":          0.0,
            "avg_dep_distance":         0.0,
            "complex_sentence_ratio":   0.0,
        }


# ═══════════════════════════════════════════════════════════════
# Model bundle — one tokenizer + model per ensemble member
# ═══════════════════════════════════════════════════════════════

@dataclass
class ModelBundle:
    """Holds tokenizer + model for one ensemble member."""
    key:         str
    model_dir:   str
    tokenizer:   Any
    model:       torch.nn.Module
    max_length:  int
    weight:      float


# ═══════════════════════════════════════════════════════════════
# Main detector
# ═══════════════════════════════════════════════════════════════

class EnsembleDetector:
    """
    3-model ensemble inference engine aligned with v6.1 training pipeline.

    Usage
    -----
    detector = EnsembleDetector(
        ensemble_dir="/content/drive/MyDrive/.../ensemble_v6/",
        use_statistical_features=True,
    )
    result = detector.detect("Text to classify...")
    print(detector.format_result(result))

    ensemble_dir must contain:
        deberta_final/    (config.json + model.safetensors + label_mappings.json)
        electra_final/
        modernbert_final/
    """

    # Model configs matching ensemble_v6_config.py
    _MODEL_KEYS: Tuple[str, ...] = ("deberta", "electra", "modernbert")

    _MODEL_MAX_LENGTHS: Dict[str, int] = {
        "deberta":    512,
        "electra":    512,
        "modernbert": 512,
    }

    _TRUST_REMOTE: Dict[str, bool] = {
        "deberta":    False,
        "electra":    False,
        "modernbert": False,
    }

    def __init__(
        self,
        ensemble_dir: str,
        use_statistical_features: bool = True,
        confidence_threshold: float = 0.60,
        device: Optional[torch.device] = None,
        max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS,
        batch_size: int = 16,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.ensemble_dir        = ensemble_dir.rstrip("/")
        self.confidence_threshold = confidence_threshold
        self.max_input_chars     = max_input_chars
        self.batch_size          = batch_size
        self.device              = device or self._get_optimal_device()
        self._whitespace_pattern = re.compile(r"\s+")

        # Ensemble weights (default equal, override for v7.0)
        self._weights = ensemble_weights or ENSEMBLE_WEIGHTS
        w_sum = sum(self._weights.values())
        if abs(w_sum - 1.0) > 1e-4:
            logger.warning(
                "Ensemble weights sum to %.4f, not 1.0 — normalising.", w_sum
            )
            self._weights = {k: v / w_sum for k, v in self._weights.items()}

        # Load label mappings from deberta (all 3 share the same labels)
        label_path = f"{self.ensemble_dir}/deberta_final/label_mappings.json"
        self.labels = LabelMapping.from_json_file(label_path)

        # Load all 3 model bundles
        self._bundles: List[ModelBundle] = self._load_bundles()

        # Statistical features (StylometricProfiler, CPU-only)
        self.stats_extractor: Optional[StatisticalFeatureExtractor] = (
            StatisticalFeatureExtractor(self.device)
            if use_statistical_features
            else None
        )

        logger.info(
            "EnsembleDetector v6.1 loaded: %d models | device=%s | "
            "labels=%s | stats=%s",
            len(self._bundles),
            self.device,
            list(self.labels.label2id.keys()),
            "on" if use_statistical_features else "off",
        )

    # ── Device ────────────────────────────────────────────────

    @staticmethod
    def _get_optimal_device() -> torch.device:
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if device_count > 1:
                max_memory, best = 0, 0
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    if props.total_memory > max_memory:
                        max_memory, best = props.total_memory, i
                return torch.device(f"cuda:{best}")
            return torch.device("cuda:0")
        return torch.device("cpu")

    # ── Model loading ─────────────────────────────────────────

    def _load_bundles(self) -> List[ModelBundle]:
        dtype = (
            torch.bfloat16
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16 if self.device.type == "cuda"
            else torch.float32
        )
        bundles: List[ModelBundle] = []

        for key in self._MODEL_KEYS:
            model_dir = f"{self.ensemble_dir}/{key}_final"
            trust     = self._TRUST_REMOTE.get(key, False)
            max_len   = self._MODEL_MAX_LENGTHS.get(key, 512)
            weight    = self._weights.get(key, 1 / len(self._MODEL_KEYS))

            logger.info("Loading %s from %s ...", key, model_dir)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_dir,
                    model_max_length=max_len,
                    padding_side="right",
                    truncation_side="right",
                    trust_remote_code=trust,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_dir,
                    num_labels=self.labels.num_labels,
                    torch_dtype=dtype,
                    trust_remote_code=trust,
                    ignore_mismatched_sizes=True,
                )
                model = model.to(self.device).eval()
                for param in model.parameters():
                    param.requires_grad = False

                bundles.append(ModelBundle(
                    key=key, model_dir=model_dir,
                    tokenizer=tokenizer, model=model,
                    max_length=max_len, weight=weight,
                ))
                logger.info("✅ %s loaded (%d labels, weight=%.2f)", key,
                            self.labels.num_labels, weight)

            except (OSError, RuntimeError, ValueError) as exc:
                logger.error("❌ Failed to load %s: %s", key, exc)

        if not bundles:
            raise RuntimeError(
                "No models loaded. Check ensemble_dir contains "
                "deberta_final/, electra_final/, modernbert_final/ ."
            )
        return bundles

    # ── Text utilities ────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        return self._whitespace_pattern.sub(" ", text).strip() if text else ""

    def _validate_input(self, text: str) -> Optional[str]:
        if not text or not text.strip():
            return "empty input"
        if len(text) > self.max_input_chars:
            return f"input too large ({len(text)} > {self.max_input_chars})"
        return None

    # ── Inference ─────────────────────────────────────────────

    @torch.no_grad()
    def _infer_bundle(
        self,
        bundle: ModelBundle,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run one model and return softmax probabilities."""
        outputs = bundle.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return F.softmax(outputs.logits, dim=-1)

    @torch.no_grad()
    def _weighted_ensemble_inference(
        self,
        injected_text: str,
    ) -> torch.Tensor:
        """
        Tokenize per model and compute weighted average probabilities.
        Each model uses its own tokenizer — critical for DeBERTa (SP) vs
        ELECTRA/ModernBERT (WordPiece).
        """
        weighted_probs: Optional[torch.Tensor] = None

        for bundle in self._bundles:
            # Tokenize for this specific model
            token_count = len(bundle.tokenizer(
                injected_text, truncation=False,
                add_special_tokens=False,
            ).input_ids)

            if token_count > bundle.max_length:
                probs = self._sliding_window_bundle(bundle, injected_text)
            else:
                inputs = bundle.tokenizer(
                    injected_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=bundle.max_length,
                    padding=True,
                ).to(self.device)
                batch_probs = self._infer_bundle(
                    bundle, inputs.input_ids, inputs.attention_mask
                )
                probs = batch_probs[0]

            if weighted_probs is None:
                weighted_probs = bundle.weight * probs
            else:
                weighted_probs += bundle.weight * probs

        if weighted_probs is None:
            return self._empty_probs()
        return weighted_probs

    @torch.no_grad()
    def _sliding_window_bundle(
        self,
        bundle: ModelBundle,
        text: str,
        stride_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Sliding window for texts exceeding max_length for one bundle."""
        full_tokens = bundle.tokenizer(
            text, return_tensors="pt",
            truncation=False, add_special_tokens=False,
        ).input_ids[0]
        num_tokens = len(full_tokens)

        stride = int(bundle.max_length * stride_ratio)
        windows: List[torch.Tensor] = []
        for start in range(0, num_tokens, stride):
            end = min(start + bundle.max_length, num_tokens)
            chunk = full_tokens[start:end]
            if len(chunk) < 50:
                break
            windows.append(chunk)
            if end >= num_tokens:
                break

        if not windows:
            return self._empty_probs()

        pad_id  = bundle.tokenizer.pad_token_id or 0
        max_len = max(len(w) for w in windows)
        padded, masks = [], []
        for w in windows:
            pad_len = max_len - len(w)
            padded.append(F.pad(w, (0, pad_len), value=pad_id))
            masks.append(F.pad(
                torch.ones(len(w), dtype=torch.long),
                (0, pad_len), value=0,
            ))

        input_ids      = torch.stack(padded).to(self.device)
        attention_mask = torch.stack(masks).to(self.device)
        window_probs   = self._infer_bundle(bundle, input_ids, attention_mask)
        return torch.mean(window_probs, dim=0)

    @torch.no_grad()
    def _batch_infer_bundle(
        self,
        bundle: ModelBundle,
        texts: List[str],
    ) -> torch.Tensor:
        """Batch inference for short texts using one model bundle."""
        inputs = bundle.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=bundle.max_length,
            padding=True,
        ).to(self.device)
        return self._infer_bundle(
            bundle, inputs.input_ids, inputs.attention_mask
        )

    def _empty_probs(self) -> torch.Tensor:
        return torch.full(
            (self.labels.num_labels,),
            fill_value=1.0 / self.labels.num_labels,
            device=self.device,
        )

    # ── Confidence calibration ────────────────────────────────
    # Updated for v6.1: ppl removed, BST uses [-1,+1] range,
    # hapax_legomena_ratio added as strong AI signal.

    @staticmethod
    def _adjust_confidence_with_stats(
        raw_human_prob: float,
        stats: Dict[str, float],
    ) -> float:
        adjusted = raw_human_prob

        # Burstiness: (sigma-mu)/(sigma+mu) in [-1, +1]
        # Positive = bursty (human pattern), Negative = uniform (AI pattern)
        bst = stats.get("burstiness", 0.0)
        if bst > 0.20:
            # Bursty → boost human probability
            adjusted = min(0.95, adjusted * 1.15)
        elif bst < -0.10:
            # Very uniform → likely AI
            adjusted *= 0.88

        # Vocabulary richness (MATTR adaptive)
        vr = stats.get("vocabulary_richness", 0.5)
        if vr < 0.40:
            # Low diversity → AI repeats vocabulary
            adjusted *= 0.90
        elif vr > 0.75:
            adjusted = min(0.95, adjusted * 1.05)

        # Hapax legomena — words that appear only once
        # AI reuses vocabulary → lower HL than humans
        hl = stats.get("hapax_legomena_ratio", 0.5)
        if hl < 0.25:
            adjusted *= 0.88
        elif hl > 0.65:
            adjusted = min(0.95, adjusted * 1.08)

        # Complex sentence ratio (spaCy only — 0.0 if unavailable)
        dc = stats.get("complex_sentence_ratio", 0.0)
        if dc > 0.0:
            if dc < 0.10:
                # Very few complex structures → AI tends to be simpler
                adjusted *= 0.95
            elif dc > 0.45:
                adjusted = min(0.95, adjusted * 1.05)

        return max(0.01, min(0.99, adjusted))

    # ── Result construction ───────────────────────────────────

    def _probs_to_result(
        self,
        cleaned_text: str,
        probs: torch.Tensor,
        stats: Optional[Dict[str, float]] = None,
        per_model_probs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> DetectionResult:
        human_idx      = self.labels.human_index
        raw_human_prob = probs[human_idx].item()

        adjusted_human_prob = (
            self._adjust_confidence_with_stats(raw_human_prob, stats)
            if stats
            else raw_human_prob
        )
        ai_prob = 1.0 - adjusted_human_prob
        is_ai   = ai_prob > 0.5

        detected_model: Optional[str]     = None
        model_family:   Optional[ModelFamily] = None
        if is_ai:
            ai_probs = probs.clone()
            ai_probs[human_idx] = 0.0
            model_idx     = int(torch.argmax(ai_probs).item())
            detected_model = self.labels.id2label.get(model_idx, "unknown")
            model_family   = self.labels.get_family(detected_model)

        confidence     = (ai_prob if is_ai else adjusted_human_prob) * 100
        uncertainty    = max(adjusted_human_prob, ai_prob) < self.confidence_threshold

        # Build per-class probability dict for raw_scores
        raw_scores: Dict[str, float] = {
            "human":          raw_human_prob * 100,
            "ai":             (1.0 - raw_human_prob) * 100,
            "adjusted_human": adjusted_human_prob * 100,
        }
        # Add per-class scores (all 7 labels)
        for idx, label in self.labels.id2label.items():
            raw_scores[f"class_{label.lower()}"] = probs[idx].item() * 100

        # Per-model breakdown (new in v6.1)
        if per_model_probs:
            for key, m_probs in per_model_probs.items():
                m_human = m_probs[human_idx].item() * 100
                raw_scores[f"{key}_human"] = m_human
                raw_scores[f"{key}_ai"]    = 100 - m_human

        return DetectionResult(
            prediction=            "AI" if is_ai else "Human",
            confidence=            confidence,
            calibrated_confidence= confidence,
            detected_model=        detected_model,
            model_family=          model_family,
            statistical_features=  stats or {},
            raw_scores=            raw_scores,
            uncertainty_zone=      uncertainty,
        )

    @staticmethod
    def _unknown_result(reason: str = "") -> DetectionResult:
        return DetectionResult(
            "Unknown", 0.0, 0.0, None, None,
            {}, {"human": 50.0, "ai": 50.0}, True,
        )

    # ── Single-text detection ─────────────────────────────────

    def detect(self, text: str) -> DetectionResult:
        """Detect whether a single text is AI-generated."""
        error = self._validate_input(text)
        if error:
            return self._unknown_result(error)

        cleaned = self._clean_text(text)
        if len(cleaned.split()) < 10:
            return self._unknown_result("text too short (< 10 words)")

        # FK prefix — same format as v6.1 training
        fk_score      = textstat.flesch_kincaid_grade(
            cleaned[:_FK_TRUNCATION_CHARS])
        injected_text = f"[FK_SCORE: {fk_score:.1f}] {cleaned}"

        # Statistical features (CPU, no GPU model needed)
        stats: Dict[str, float] = {}
        if self.stats_extractor is not None:
            stats = self.stats_extractor.extract_features(cleaned)
        stats["flesch_kincaid_grade"] = fk_score

        # Weighted ensemble inference (per-model tokenization)
        per_model_probs: Dict[str, torch.Tensor] = {}
        weighted: Optional[torch.Tensor] = None
        for bundle in self._bundles:
            with torch.no_grad():
                token_count = len(bundle.tokenizer(
                    injected_text, truncation=False,
                    add_special_tokens=False,
                ).input_ids)
                if token_count > bundle.max_length:
                    p = self._sliding_window_bundle(bundle, injected_text)
                else:
                    inputs = bundle.tokenizer(
                        injected_text, return_tensors="pt",
                        truncation=True, max_length=bundle.max_length,
                        padding=True,
                    ).to(self.device)
                    p = self._infer_bundle(
                        bundle, inputs.input_ids, inputs.attention_mask
                    )[0]
            per_model_probs[bundle.key] = p.detach()
            weighted = (
                bundle.weight * p if weighted is None
                else weighted + bundle.weight * p
            )

        probs = weighted if weighted is not None else self._empty_probs()
        return self._probs_to_result(cleaned, probs, stats, per_model_probs)

    # ── Batch detection ───────────────────────────────────────

    def detect_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[DetectionResult]:
        """Detect AI authorship for a list of texts."""
        if not texts:
            return []
        bs    = batch_size or self.batch_size
        items = self._preprocess_batch(texts)
        self._classify_lengths(items)
        self._batch_infer_short(items, bs)
        self._sliding_window_long(items)
        self._postprocess_results(items)
        return [item.result for item in items]

    def _preprocess_batch(self, texts: List[str]) -> List[BatchItem]:
        items: List[BatchItem] = []
        for i, text in enumerate(texts):
            item  = BatchItem(index=i, original_text=text)
            error = self._validate_input(text)
            if error:
                item.is_valid    = False
                item.skip_reason = error
                item.result      = self._unknown_result(error)
                items.append(item)
                continue

            cleaned = self._clean_text(text)
            if len(cleaned.split()) < 10:
                item.is_valid    = False
                item.skip_reason = "text too short (< 10 words)"
                item.result      = self._unknown_result(item.skip_reason)
                items.append(item)
                continue

            fk_score            = textstat.flesch_kincaid_grade(
                cleaned[:_FK_TRUNCATION_CHARS])
            item.cleaned_text   = cleaned
            item.injected_text  = f"[FK_SCORE: {fk_score:.1f}] {cleaned}"
            item.fk_score       = fk_score
            items.append(item)
        return items

    def _classify_lengths(self, items: List[BatchItem]) -> None:
        """Mark long items using the first bundle's tokenizer."""
        if not self._bundles:
            return
        ref = self._bundles[0]
        for item in items:
            if not item.is_valid or item.injected_text is None:
                continue
            n = len(ref.tokenizer(
                item.injected_text, truncation=False,
                add_special_tokens=False,
            ).input_ids)
            item.is_long = n > ref.max_length

    def _batch_infer_short(
        self, items: List[BatchItem], batch_size: int
    ) -> None:
        """Weighted batch inference for short items — per-model tokenization."""
        short = [
            item for item in items
            if item.is_valid and not item.is_long and item.injected_text
        ]
        if not short:
            return

        for start in range(0, len(short), batch_size):
            batch = short[start : start + batch_size]
            texts = [item.injected_text for item in batch]

            weighted_batch: Optional[torch.Tensor] = None
            for bundle in self._bundles:
                with torch.no_grad():
                    b_probs = self._batch_infer_bundle(bundle, texts)
                if weighted_batch is None:
                    weighted_batch = bundle.weight * b_probs
                else:
                    weighted_batch += bundle.weight * b_probs

            if weighted_batch is None:
                continue
            for j, item in enumerate(batch):
                item.probs = weighted_batch[j].detach()

    def _sliding_window_long(self, items: List[BatchItem]) -> None:
        for item in items:
            if not item.is_valid or not item.is_long:
                continue
            if item.injected_text is None:
                continue
            probs = self._weighted_ensemble_inference(item.injected_text)
            item.probs = probs.detach()

    def _postprocess_results(self, items: List[BatchItem]) -> None:
        for item in items:
            if item.result is not None:
                continue
            if item.probs is None:
                item.result = self._unknown_result("internal error")
                continue
            stats: Dict[str, float] = {}
            if self.stats_extractor is not None and item.cleaned_text:
                stats = self.stats_extractor.extract_features(item.cleaned_text)
            stats["flesch_kincaid_grade"] = item.fk_score
            item.stats  = stats
            item.result = self._probs_to_result(
                item.cleaned_text or "", item.probs, stats
            )

    # ── Formatting ────────────────────────────────────────────

    @staticmethod
    def format_result(result: DetectionResult) -> str:
        """Format a DetectionResult as an HTML snippet."""
        if result.prediction == "Unknown":
            return "---- Text too short or invalid for analysis ----"
        if result.prediction == "Human":
            return (
                f"**The text is** <span class='highlight-human'>"
                f"**{result.confidence:.2f}%** likely "
                f"<b>Human written</b>.</span>"
            )
        family_str = (
            result.model_family.value if result.model_family else "unknown"
        )
        return (
            f"**The text is** <span class='highlight-ai'>"
            f"**{result.confidence:.2f}%** likely "
            f"<b>AI generated</b>.</span>"
            f"\n\n**Identified LLM: {result.detected_model}** ({family_str})"
        )

    def format_result_detailed(self, result: DetectionResult) -> str:
        """Extended format with per-model breakdown and stats."""
        base = self.format_result(result)
        if result.prediction == "Unknown":
            return base

        lines = [base, ""]

        # Per-class probabilities
        class_scores = {
            k.replace("class_", "").title(): v
            for k, v in result.raw_scores.items()
            if k.startswith("class_")
        }
        if class_scores:
            lines.append("**Class probabilities:**")
            for label, score in sorted(
                class_scores.items(), key=lambda x: -x[1]
            ):
                bar = "█" * int(score / 5)
                lines.append(f"  {label:<12} {score:5.1f}%  {bar}")

        # Per-model breakdown
        model_lines = []
        for key in ("deberta", "electra", "modernbert"):
            h = result.raw_scores.get(f"{key}_human")
            a = result.raw_scores.get(f"{key}_ai")
            if h is not None:
                model_lines.append(
                    f"  {key:<12} Human={h:.1f}%  AI={a:.1f}%"
                )
        if model_lines:
            lines.append("\n**Per-model votes:**")
            lines.extend(model_lines)

        # Statistical features
        sf = result.statistical_features
        if sf:
            lines.append("\n**Statistical signals:**")
            bst = sf.get("burstiness", None)
            vr  = sf.get("vocabulary_richness", None)
            hl  = sf.get("hapax_legomena_ratio", None)
            fk  = sf.get("flesch_kincaid_grade", None)
            if bst is not None:
                interp = "bursty↑human" if bst > 0.1 else "uniform↑AI"
                lines.append(f"  Burstiness:   {bst:+.3f}  ({interp})")
            if vr is not None:
                lines.append(f"  Vocab richness:{vr:.3f}")
            if hl is not None:
                lines.append(f"  Hapax ratio:  {hl:.3f}")
            if fk is not None:
                lines.append(f"  FK grade:     {fk:.1f}")

        return "\n".join(lines)


# ── Backward-compat aliases ───────────────────────────────────
# v5.3 code that instantiated SOTAAIDetector still works,
# but passes ensemble_dir instead of model_paths.

class SOTAAIDetector(EnsembleDetector):
    """
    Backward-compatible alias for EnsembleDetector.
    Accepts either ensemble_dir (v6.1) or model_paths (legacy v5.3).
    """
    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        ensemble_dir: Optional[str] = None,
        label_mappings_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if ensemble_dir is None and model_paths:
            # v5.3 usage: infer ensemble_dir from first model path
            import os
            ensemble_dir = os.path.dirname(
                os.path.dirname(model_paths[0])
            )
            logger.warning(
                "SOTAAIDetector(model_paths=...) is deprecated. "
                "Use EnsembleDetector(ensemble_dir=...) instead."
            )
        if ensemble_dir is None:
            raise ValueError(
                "ensemble_dir is required. "
                "Pass ensemble_dir='/path/to/ensemble_v6/'"
            )
        super().__init__(ensemble_dir=ensemble_dir, **kwargs)


AITextClassifier          = EnsembleDetector
OptimizedAITextClassifier = EnsembleDetector
