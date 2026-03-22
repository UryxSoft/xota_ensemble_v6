"""
Watermark Decoder Module  [EXPERIMENTAL]
==============================================
Detects *candidate* statistical watermarks in text by applying the
methods described in Kirchenbauer et al. (2023) and related literature.

Aligned with SOTAAIDetector v3.3 / forensic_reports v3.3.
Provides ``WatermarkSignature.to_forensic_dict()`` for direct integration.

IMPORTANT: Watermark detection is inherently noisy.  A positive result
means "statistically unusual token distribution consistent with a known
watermarking scheme" — NOT proof that a watermark exists.

Refactor changelog (v3.4 → v3.4.1)
-------------------------------------
  [CRITICAL] Per-chunk entropy reduction: softmax/log_softmax/sum now
             execute INSIDE the sliding-window loop.  Only 1D numpy
             entropy vectors are retained.  Eliminates the 12+ GB VRAM
             bomb that occurred when concatenating full [seq_len, 50257]
             logit tensors across windows.  A 20K-token document now
             uses ~4MB instead of ~12GB.
  [BUG]     Vocab validation on lazy tokenizer load: extracted
             ``_validate_vocab_size()`` method called both at __init__
             (eager DI) and after ``_get_tokenizer()`` lazy load.
             Previously, non-injected tokenizers skipped validation.
  [BUG]     CUDA cache cleanup on entropy exception: ``torch.cuda
             .empty_cache()`` called in the except block of
             ``_detect_impl`` to prevent VRAM leak after OOM.
  [PERF]    Explicit ``del logits, probs, log_probs`` inside the
             chunk loop to release GPU memory immediately instead of
             waiting for Python GC.

Prior changelog: see v3.4, v3.3.1, v3.3, v3.2 in git history.

Requires
--------
  torch, transformers, numpy, scipy
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Module metadata (P4-11)
# ═══════════════════════════════════════════════════════════════════════════

__version__: str = "3.4.1"


# ═══════════════════════════════════════════════════════════════════════════
# Custom exception hierarchy (P1-1)
# ═══════════════════════════════════════════════════════════════════════════


class WatermarkError(Exception):
    """Base exception for watermark decoder failures."""


class WatermarkModelError(WatermarkError):
    """Model loading or inference failure."""


class WatermarkTokenizerError(WatermarkError):
    """Tokenizer loading or encoding failure."""


# ═══════════════════════════════════════════════════════════════════════════
# Named constants (zero magic numbers)
# ═══════════════════════════════════════════════════════════════════════════

_MIN_WORDS: int = 20
"""Minimum word count for reliable analysis."""

_MIN_TOKENS_GREEN_RED: int = 11
"""Minimum token count for green/red list analysis (context_width + 10)."""

_MIN_ENTROPY_TOKENS: int = 5
"""Minimum token count for entropy analysis."""

_MIN_AUTOCORR_TOKENS: int = 20
"""Minimum tokens for autocorrelation periodicity detection."""

_GREEN_CACHE_MAX: int = 10_000
"""Maximum entries in green-list LRU cache per hash variant."""

_REFERENCE_ENTROPY: float = 5.2
"""Empirical GPT-2 mean entropy on natural English text (nats)."""

_MAX_P_VALUE_LOG_SCALE: float = 5.0
"""Denominator for log-scale confidence: conf = -log10(p) / this."""

_FLOAT_EPS: float = 1e-9
"""Epsilon for safe floating-point comparisons (P1-2)."""

_WINDOW_OVERLAP: int = 64
"""Token overlap between sliding-window chunks (P2-4)."""


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class WatermarkConfig:
    """
    All tunable thresholds for watermark detection.

    Exposed as frozen config so they can be tuned without
    modifying detection logic.
    """

    # Green/Red list parameters
    vocab_size: int = 50257
    gamma: float = 0.5
    num_hash_variants: int = 10
    context_width: int = 1

    # Detection thresholds
    p_value_threshold: float = 0.05
    z_score_threshold: float = 2.0

    # Scheme classification thresholds
    green_fraction_threshold: float = 0.55
    periodicity_threshold: float = 0.25

    # Entropy confidence bonuses
    skew_threshold: float = -0.5
    skew_bonus: float = 0.1
    range_ratio_threshold: float = 0.5
    range_bonus: float = 0.1
    periodicity_bonus_threshold: float = 0.2
    periodicity_bonus: float = 0.1

    # Entropy analyzer
    entropy_model_id: str = "gpt2"
    entropy_max_length: int = 1024


DEFAULT_WATERMARK_CONFIG = WatermarkConfig()


# ═══════════════════════════════════════════════════════════════════════════
# Data structures (P4-10: suspected_provider removed)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class WatermarkSignature:
    """Result of watermark detection analysis."""

    detected: bool
    confidence: float
    scheme_type: str
    z_score: float
    p_value: float
    green_fraction: float
    entropy_deviation: float
    evidence: Dict[str, object]

    def to_forensic_dict(self) -> Dict[str, object]:
        """
        Return the dict expected by ``forensic_reports.py``'s
        ``additional_analyses["watermark"]``.

        NOTE: confidence is forced to 0.0 when ``detected=False`` so
        that ``forensic_reporter.watermark_score`` is never non-zero
        for a negative result.
        """
        return {
            "detected": self.detected,
            "confidence": self.confidence if self.detected else 0.0,
            "scheme_type": self.scheme_type,
            "z_score": self.z_score,
            "p_value": self.p_value,
            "green_fraction": self.green_fraction,
        }


def _make_error_signature(error_msg: str) -> WatermarkSignature:
    """Construct a safe ``detected=False`` signature for error paths."""
    return WatermarkSignature(
        detected=False,
        confidence=0.0,
        scheme_type="none",
        z_score=0.0,
        p_value=1.0,
        green_fraction=0.5,
        entropy_deviation=0.0,
        evidence={"error": error_msg},
    )


# ═══════════════════════════════════════════════════════════════════════════
# LRU cache (bounded dict for green list masks)
# ═══════════════════════════════════════════════════════════════════════════


class _LRUCache:
    """
    Bounded LRU cache using OrderedDict.

    Prevents unbounded memory growth when processing long texts
    with many unique context hashes.
    """

    __slots__ = ("_maxsize", "_cache")

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def get(self, key: int) -> Optional[np.ndarray]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: int, value: np.ndarray) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._maxsize:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Green/Red List Decoder (Kirchenbauer et al. 2023)
# ═══════════════════════════════════════════════════════════════════════════


class GreenRedListDecoder:
    """
    Implements green/red list watermark detection.

    Tries multiple hash variants (unknown secret key) and reports
    the best z-score with Bonferroni correction.

    Performance: green list stored as boolean mask (np.ndarray[bool])
    instead of Python set.  Lookup is O(1) array index.
    """

    __slots__ = (
        "_vocab_size", "_gamma", "_num_variants",
        "_context_width", "_green_list_size", "_vocab_indices",
    )

    def __init__(self, config: WatermarkConfig) -> None:
        self._vocab_size = config.vocab_size
        self._gamma = config.gamma
        self._num_variants = config.num_hash_variants
        self._context_width = config.context_width
        self._green_list_size = int(config.vocab_size * config.gamma)
        self._vocab_indices = np.arange(config.vocab_size)

    def _hash_context(
        self, context_tokens: List[int], variant: int
    ) -> int:
        """Deterministic hash of context window + variant index."""
        ctx_bytes = (
            f"{variant}:{':'.join(map(str, context_tokens))}".encode()
        )
        return int(hashlib.sha256(ctx_bytes).hexdigest()[:8], 16)

    def _get_green_mask(self, seed: int) -> np.ndarray:
        """
        Return a boolean mask of shape ``(vocab_size,)`` where
        ``True`` = green token.
        """
        rng = np.random.default_rng(seed % (2**31))
        green_indices = rng.choice(
            self._vocab_indices, self._green_list_size, replace=False
        )
        mask = np.zeros(self._vocab_size, dtype=np.bool_)
        mask[green_indices] = True
        return mask

    def _analyze_single_variant(
        self,
        ids_array: np.ndarray,
        token_ids: List[int],
        variant: int,
    ) -> Tuple[float, float]:
        """
        Run green/red analysis for a single hash variant.

        Returns ``(z_score, green_fraction)``.
        Extracted from the outer loop for readability (P1 audit).
        """
        cache = _LRUCache(_GREEN_CACHE_MAX)
        green_count = 0
        total_count = 0

        for i in range(self._context_width, len(ids_array)):
            context_window = token_ids[
                max(0, i - self._context_width) : i
            ]
            ctx_hash = self._hash_context(context_window, variant)

            mask = cache.get(ctx_hash)
            if mask is None:
                mask = self._get_green_mask(ctx_hash)
                cache.put(ctx_hash, mask)

            token_id = ids_array[i]
            if 0 <= token_id < self._vocab_size and mask[token_id]:
                green_count += 1
            total_count += 1

        if total_count == 0:
            return 0.0, 0.5

        green_fraction = green_count / total_count
        std = math.sqrt(
            self._gamma * (1 - self._gamma) / total_count
        )
        z_score = (
            (green_fraction - self._gamma) / std if std > 0 else 0.0
        )
        return z_score, green_fraction

    def detect(
        self, token_ids: List[int]
    ) -> Tuple[float, float, float]:
        """
        Run green/red list detection across all hash variants.

        Returns ``(best_z_score, corrected_p_value, best_green_fraction)``.
        """
        min_tokens = max(
            _MIN_TOKENS_GREEN_RED, self._context_width + 10
        )
        if len(token_ids) < min_tokens:
            return 0.0, 1.0, 0.5

        ids_array = np.array(token_ids, dtype=np.int64)
        best_z_score = 0.0
        best_green_frac = 0.5

        for variant in range(self._num_variants):
            z_score, green_frac = self._analyze_single_variant(
                ids_array, token_ids, variant
            )
            if z_score > best_z_score:
                best_z_score = z_score
                best_green_frac = green_frac

        # Bonferroni correction
        raw_p = float(scipy_stats.norm.sf(best_z_score))
        corrected_p = min(raw_p * self._num_variants, 1.0)

        return best_z_score, corrected_p, best_green_frac


# ═══════════════════════════════════════════════════════════════════════════
# Entropy Analyzer (P2-4: sliding window, P2-6: typed, P3-9: model DI)
# ═══════════════════════════════════════════════════════════════════════════


class EntropyAnalyzer:
    """
    Analyzes token-level entropy patterns using a reference LM.

    Watermarked text can exhibit unusual entropy distributions:
    periodic patterns, skewed distributions, or compressed range.

    DI: ``tokenizer``, ``model``, and ``model_id`` are all injectable.
    Pass a pre-loaded model to avoid redundant GPT-2 loading.

    Parameters
    ----------
    device : torch device for inference.
    model_id : HuggingFace model identifier (used only if ``model``
               is ``None``).
    tokenizer : optional pre-loaded tokenizer.
    model : optional pre-loaded causal LM (P3-9).  When provided,
            ``model_id`` is ignored for model loading.
    max_length : maximum tokens per inference window.
    """

    __slots__ = (
        "_device", "_model_id", "_model", "_tokenizer", "_max_length",
    )

    def __init__(
        self,
        device: torch.device,
        model_id: str = "gpt2",
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model: Optional[PreTrainedModel] = None,
        max_length: int = 1024,
    ) -> None:
        self._device = device
        self._model_id = model_id
        self._model: Optional[PreTrainedModel] = model
        self._tokenizer: Optional[PreTrainedTokenizerBase] = tokenizer
        self._max_length = max_length

        # If model was injected, ensure eval mode + no grad
        if self._model is not None:
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False

    def _lazy_init(self) -> None:
        """Load model + tokenizer on first use.  Raises WatermarkModelError."""
        if self._model is not None:
            return
        logger.info("Loading entropy analyzer model: %s", self._model_id)
        try:
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._model_id
                )
            self._model = (
                AutoModelForCausalLM.from_pretrained(self._model_id)
                .to(self._device)
                .eval()
            )
            for param in self._model.parameters():
                param.requires_grad = False
        except (OSError, RuntimeError, ValueError) as exc:
            raise WatermarkModelError(
                f"Failed to load entropy model '{self._model_id}': {exc}"
            ) from exc

    def preload(self) -> None:
        """
        Eagerly load model + tokenizer (P3-8).

        Call before first ``analyze()`` to avoid first-call latency
        spike in production pipelines.
        """
        self._lazy_init()

    def set_tokenizer(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """
        Set the tokenizer (public API for shared-tokenizer injection).
        """
        self._tokenizer = tokenizer

    def unload(self) -> None:
        """Release model from memory (tokenizer preserved if shared)."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        logger.info("Entropy analyzer model unloaded")

    @torch.no_grad()
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Compute entropy statistics for the text.

        Uses sliding-window inference for texts longer than
        ``max_length``.  Entropy is reduced to a 1D scalar
        **per-token inside each chunk** — only the 1D numpy
        entropy vector is retained across chunks.

        This prevents the catastrophic VRAM explosion that would
        occur from concatenating full ``[seq_len, vocab_size]``
        logit tensors across windows (e.g. 20K tokens × 50K vocab
        × 4 bytes × 3 tensors = 12+ GB for a single text).

        Returns dict with: mean_entropy, entropy_std, entropy_skew,
        entropy_range_ratio, entropy_periodicity.
        """
        self._lazy_init()
        assert self._tokenizer is not None
        assert self._model is not None

        # Tokenize WITHOUT truncation — we chunk manually
        all_ids = self._tokenizer.encode(
            text, add_special_tokens=False
        )
        if len(all_ids) < _MIN_ENTROPY_TOKENS:
            return self._empty_stats()

        # Sliding-window inference with per-chunk entropy reduction.
        # Only 1D numpy arrays are kept — zero logit tensors retained.
        chunk_entropies: List[np.ndarray] = []
        max_len = self._max_length
        stride = max_len - _WINDOW_OVERLAP
        start = 0

        while start < len(all_ids):
            chunk_ids = all_ids[start : start + max_len]
            input_ids = torch.tensor(
                [chunk_ids], dtype=torch.long, device=self._device
            )

            # Forward pass → logits for this chunk only
            logits = self._model(input_ids=input_ids).logits[0, :-1]

            # --- REDUCE TO 1D IMMEDIATELY (the OOM fix) ---
            # softmax + log_softmax + sum happen on this chunk's
            # logits tensor (at most [1024, 50257]) then the result
            # is moved to CPU numpy.  The logits tensor is released
            # at the end of this iteration.
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            chunk_h = (
                -torch.sum(probs * log_probs, dim=-1)
                .cpu().float().numpy()
            )
            # Free GPU memory for this chunk's intermediates
            del logits, probs, log_probs

            if start == 0:
                chunk_entropies.append(chunk_h)
            else:
                # Skip the overlap region to avoid double-counting
                skip = min(_WINDOW_OVERLAP, len(chunk_h))
                if skip < len(chunk_h):
                    chunk_entropies.append(chunk_h[skip:])

            start += stride
            if len(chunk_ids) < max_len:
                break  # last chunk was shorter than window

        if not chunk_entropies:
            return self._empty_stats()

        # Concatenate only the lightweight 1D entropy vectors
        entropies = np.concatenate(chunk_entropies)

        if len(entropies) < _MIN_ENTROPY_TOKENS:
            return self._empty_stats()

        mean_e = float(np.mean(entropies))
        std_e = float(np.std(entropies))
        e_range = float(np.max(entropies) - np.min(entropies))
        range_ratio = e_range / mean_e if mean_e > 0 else 0.0

        periodicity = 0.0
        if len(entropies) > _MIN_AUTOCORR_TOKENS:
            periodicity = self._fft_periodicity(entropies, mean_e)

        return {
            "mean_entropy": mean_e,
            "entropy_std": std_e,
            "entropy_skew": float(scipy_stats.skew(entropies)),
            "entropy_range_ratio": range_ratio,
            "entropy_periodicity": periodicity,
        }

    @staticmethod
    def _fft_periodicity(
        entropies: np.ndarray, mean: float
    ) -> float:
        """
        Compute autocorrelation periodicity via FFT.

        O(n log n) vs O(n²) for np.correlate(mode='full').
        P1-2: uses ``_FLOAT_EPS`` for safe float comparison.
        """
        centered = entropies - mean
        n = len(centered)
        fft_vals = np.fft.rfft(centered, n=2 * n)
        autocorr = np.fft.irfft(fft_vals * np.conj(fft_vals))[:n]
        # P1-2: safe float comparison instead of == 0
        if abs(autocorr[0]) < _FLOAT_EPS:
            return 0.0
        autocorr = autocorr / autocorr[0]
        if len(autocorr) < 3:
            return 0.0
        peaks = np.where(
            (autocorr[1:-1] > autocorr[:-2])
            & (autocorr[1:-1] > autocorr[2:])
        )[0]
        if len(peaks) > 0:
            return float(np.max(autocorr[peaks + 1]))
        return 0.0

    @staticmethod
    def _empty_stats() -> Dict[str, float]:
        return {
            "mean_entropy": 0.0,
            "entropy_std": 0.0,
            "entropy_skew": 0.0,
            "entropy_range_ratio": 0.0,
            "entropy_periodicity": 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Main Decoder (P1-1 error handling, P2-5 vocab validation,
#               P3-7 batch, P3-8 preload, P4-12 confidence extraction)
# ═══════════════════════════════════════════════════════════════════════════


class WatermarkDecoder:
    """
    Main watermark detection interface.

    Combines green/red list analysis with entropy pattern analysis.
    Single shared tokenizer across all components.

    Parameters
    ----------
    device : torch.device (default: auto-detect).
    config : WatermarkConfig with all thresholds.
    tokenizer : optional pre-loaded tokenizer (DI for testing).
    model : optional pre-loaded causal LM for entropy analysis.
    """

    __slots__ = (
        "_device", "_config", "_tokenizer",
        "_green_red_decoder", "_entropy_analyzer",
    )

    def __init__(
        self,
        device: Optional[torch.device] = None,
        config: Optional[WatermarkConfig] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model: Optional[PreTrainedModel] = None,
    ) -> None:
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._config = config or DEFAULT_WATERMARK_CONFIG
        self._tokenizer = tokenizer
        self._green_red_decoder = GreenRedListDecoder(self._config)
        self._entropy_analyzer = EntropyAnalyzer(
            device=self._device,
            model_id=self._config.entropy_model_id,
            tokenizer=tokenizer,
            model=model,
            max_length=self._config.entropy_max_length,
        )

        # Validate vocab contract (works for both eager + lazy paths)
        if tokenizer is not None:
            self._validate_vocab_size(tokenizer)

    def _validate_vocab_size(
        self, tokenizer: PreTrainedTokenizerBase
    ) -> None:
        """
        Validate tokenizer vocab matches config.  Auto-corrects on
        mismatch.  Called both at __init__ (eager DI) and after
        lazy load in _get_tokenizer().
        """
        actual = len(tokenizer)
        if actual != self._config.vocab_size:
            logger.warning(
                "Tokenizer vocab size (%d) != config.vocab_size (%d). "
                "Auto-correcting config to match tokenizer.",
                actual, self._config.vocab_size,
            )
            cfg = self._config
            self._config = WatermarkConfig(
                vocab_size=actual,
                gamma=cfg.gamma,
                num_hash_variants=cfg.num_hash_variants,
                context_width=cfg.context_width,
                p_value_threshold=cfg.p_value_threshold,
                z_score_threshold=cfg.z_score_threshold,
                green_fraction_threshold=cfg.green_fraction_threshold,
                periodicity_threshold=cfg.periodicity_threshold,
                skew_threshold=cfg.skew_threshold,
                skew_bonus=cfg.skew_bonus,
                range_ratio_threshold=cfg.range_ratio_threshold,
                range_bonus=cfg.range_bonus,
                periodicity_bonus_threshold=cfg.periodicity_bonus_threshold,
                periodicity_bonus=cfg.periodicity_bonus,
                entropy_model_id=cfg.entropy_model_id,
                entropy_max_length=cfg.entropy_max_length,
            )
            # Rebuild green/red decoder with corrected vocab
            self._green_red_decoder = GreenRedListDecoder(self._config)

    def _get_tokenizer(self) -> PreTrainedTokenizerBase:
        """Return shared tokenizer, loading on first access."""
        if self._tokenizer is None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self._config.entropy_model_id
                )
            except (OSError, ValueError) as exc:
                raise WatermarkTokenizerError(
                    f"Failed to load tokenizer "
                    f"'{self._config.entropy_model_id}': {exc}"
                ) from exc
            self._entropy_analyzer.set_tokenizer(self._tokenizer)
            # Validate vocab on lazy path too (not just eager DI)
            self._validate_vocab_size(self._tokenizer)
        return self._tokenizer

    def preload(self) -> None:
        """
        Eagerly load tokenizer + model (P3-8).

        Call before first ``detect()`` in production pipelines
        to eliminate first-call latency spike.
        """
        self._get_tokenizer()
        self._entropy_analyzer.preload()

    def unload(self) -> None:
        """Release all models from memory."""
        self._entropy_analyzer.unload()

    def _calculate_confidence(
        self,
        p_value: float,
        entropy_stats: Dict[str, float],
        detected: bool,
    ) -> float:
        """
        Calculate confidence score from p-value + entropy bonuses (P4-12).

        Extracted to private method for readability.
        Enforces state consistency: ``detected=False → 0.0``.
        """
        cfg = self._config

        # Base confidence from p-value (log scale)
        if p_value >= 1.0:
            confidence = 0.0
        else:
            confidence = min(
                1.0,
                -math.log10(max(p_value, 1e-10)) / _MAX_P_VALUE_LOG_SCALE,
            )

        # Entropy-based bonuses
        if entropy_stats["entropy_skew"] < cfg.skew_threshold:
            confidence += cfg.skew_bonus
        if (
            entropy_stats["entropy_range_ratio"]
            < cfg.range_ratio_threshold
        ):
            confidence += cfg.range_bonus
        if (
            entropy_stats["entropy_periodicity"]
            > cfg.periodicity_bonus_threshold
        ):
            confidence += cfg.periodicity_bonus

        confidence = min(1.0, confidence)

        # State consistency: detected=False → confidence=0
        if not detected:
            confidence = 0.0

        return confidence

    def detect(self, text: str) -> WatermarkSignature:
        """
        Analyze text for candidate watermark signals.

        P1-1: All external calls wrapped in error handling.
        Returns deterministic ``detected=False`` on any failure.
        """
        if not text or len(text.split()) < _MIN_WORDS:
            return _make_error_signature(
                "text too short for reliable analysis"
            )

        try:
            return self._detect_impl(text)
        except WatermarkError:
            raise  # re-raise our own exceptions
        except (RuntimeError, ValueError, OSError) as exc:
            logger.exception("Watermark detection failed: %s", exc)
            return _make_error_signature(str(exc))

    def _detect_impl(self, text: str) -> WatermarkSignature:
        """Core detection logic, separated for clean error boundary."""
        cfg = self._config
        token_ids = self._get_tokenizer().encode(
            text, add_special_tokens=False
        )

        if not token_ids:
            return _make_error_signature("empty after tokenization")

        # Green/red list analysis
        z_score, p_value, green_fraction = (
            self._green_red_decoder.detect(token_ids)
        )

        # Entropy analysis (wrapped — failure returns empty stats)
        try:
            entropy_stats = self._entropy_analyzer.analyze(text)
        except (RuntimeError, WatermarkModelError) as exc:
            logger.warning("Entropy analysis failed: %s", exc)
            # Release any CUDA tensors stuck in allocator after OOM
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            entropy_stats = EntropyAnalyzer._empty_stats()

        # Detection decision
        detected = (
            p_value < cfg.p_value_threshold
            and z_score > cfg.z_score_threshold
        )

        # Scheme classification
        if not detected:
            scheme_type = "none"
        elif green_fraction > cfg.green_fraction_threshold:
            scheme_type = "green_red"
        elif (
            entropy_stats["entropy_periodicity"]
            > cfg.periodicity_threshold
        ):
            scheme_type = "exp_minimum"
        else:
            scheme_type = "semantic"

        # Confidence (P4-12: extracted)
        confidence = self._calculate_confidence(
            p_value, entropy_stats, detected
        )

        # Entropy deviation from reference
        mean_e = entropy_stats["mean_entropy"]
        entropy_deviation = (
            abs(mean_e - _REFERENCE_ENTROPY) / _REFERENCE_ENTROPY
            if mean_e > 0
            else 0.0
        )

        return WatermarkSignature(
            detected=detected,
            confidence=confidence,
            scheme_type=scheme_type,
            z_score=z_score,
            p_value=p_value,
            green_fraction=green_fraction,
            entropy_deviation=entropy_deviation,
            evidence={
                "token_count": len(token_ids),
                "entropy_stats": entropy_stats,
            },
        )

    def detect_batch(
        self, texts: List[str]
    ) -> List[WatermarkSignature]:
        """
        Analyze multiple texts, reusing model across all (P3-7).

        Pre-loads model once, then processes sequentially.
        Returns one ``WatermarkSignature`` per text.
        """
        if not texts:
            return []

        # Ensure model is loaded once before the loop
        self.preload()

        return [self.detect(text) for text in texts]


# ═══════════════════════════════════════════════════════════════════════════
# Reporter (presentation layer, P1-3: version from module constant)
# ═══════════════════════════════════════════════════════════════════════════


class WatermarkReporter:
    """Format watermark analysis results for display."""

    @staticmethod
    def format_report(signature: WatermarkSignature) -> str:
        """Render a human-readable watermark analysis report."""
        status = (
            "CANDIDATE WATERMARK DETECTED"
            if signature.detected
            else "NO WATERMARK SIGNAL"
        )

        lines = [
            "=" * 60,
            f"WATERMARK ANALYSIS REPORT v{__version__}  [EXPERIMENTAL]",
            "=" * 60,
            "",
            f"Detection Status: {status}",
            f"Confidence:       {signature.confidence * 100:.1f}%",
            "",
            "Statistical Analysis:",
            f"  Z-Score:              {signature.z_score:.3f}",
            f"  P-Value (Bonf.):      {signature.p_value:.6f}",
            f"  Green Token Fraction: {signature.green_fraction:.3f}",
            "",
            f"Scheme Type:            {signature.scheme_type}",
            "",
            "Entropy Analysis:",
        ]

        es = signature.evidence.get("entropy_stats", {})
        if es:
            lines.extend([
                f"  Mean Entropy:         {es.get('mean_entropy', 0):.3f} nats",
                f"  Entropy Std:          {es.get('entropy_std', 0):.3f}",
                f"  Entropy Skew:         {es.get('entropy_skew', 0):.3f}",
                f"  Range Ratio:          {es.get('entropy_range_ratio', 0):.3f}",
                f"  Periodicity:          {es.get('entropy_periodicity', 0):.3f}",
            ])

        lines.extend([
            "",
            f"Token Count:            "
            f"{signature.evidence.get('token_count', 'N/A')}",
            f"Entropy Deviation:      "
            f"{signature.entropy_deviation:.1%}",
            "",
            "NOTE: This is an EXPERIMENTAL heuristic. A positive result",
            "indicates a statistical pattern consistent with known "
            "watermark",
            "schemes — it is NOT proof of watermarking.",
            "",
            "=" * 60,
        ])
        return "\n".join(lines)
