"""
perplexity_profiler.py  (XplagiaX Plugin — v1.1)
===================================================
Token-level and window-level perplexity analysis for AI-text detection.

Architecture (Hybrid Tier Design)
──────────────────────────────────
  Tier 1 (CPU, always active):
    LLMDet-style proxy perplexity via n-gram frequency dictionaries.
    ~0.2 s per text, zero external models required.

  Tier 2 (GPU/CPU, auto-enabled when torch available):
    GPT-2 Small (117M) token-level perplexity curve.
    Auto-downloads on first use (~500 MB).
    Provides: real token perplexity, Fast-DetectGPT conditional curvature.

  Segmentation (both tiers):
    Sliding window of 5–10 sentences with 2-sentence overlap.
    Minimum viable window: 128 tokens (~100 words).

Research Basis
──────────────
  - DetectGPT (Mitchell et al., ICML 2023): perturbation discrepancy
  - Fast-DetectGPT (Bao et al., ICLR 2024): conditional probability curvature
  - Binoculars (Hans et al., ICML 2024): cross-perplexity ratio
  - LLMDet (Wu et al., 2023): n-gram proxy perplexity for CPU detection

Interface Contract (matches PluginOrchestrator pattern)
───────────────────────────────────────────────────────
  profiler = PerplexityProfiler()
  stats    = profiler.compute_stats(text)     # → Dict[str, float]
  vec      = profiler.vectorize(text)         # → np.ndarray shape (12,)
  names    = profiler.feature_names()         # → tuple of 12 strings

Integration:
  PluginOrchestrator → .compute_stats(text)
  → additional_analyses["perplexity"]
  → ForensicReportGenerator → HTML section

Dependency: CPU-only base. Optional: torch + transformers for Tier 2.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# VECTOR SCHEMA — single source of truth for feature → tensor index mapping
# ============================================================================

PERPLEXITY_VECTOR_DIM: int = 12

_VECTOR_SCHEMA: Tuple[Tuple[str, int], ...] = (
    ("proxy_perplexity_mean",     0),   # LLMDet-style n-gram proxy PPL (mean)
    ("proxy_perplexity_std",      1),   # Std dev of per-window proxy PPL
    ("perplexity_curve_slope",    2),   # Linear slope of PPL across windows
    ("low_perplexity_ratio",      3),   # Fraction of windows below AI threshold
    ("perplexity_valley_count",   4),   # Count of distinct low-PPL valleys
    ("perplexity_range",          5),   # max(window_ppl) - min(window_ppl)
    ("token_entropy_mean",        6),   # Mean per-token entropy
    ("token_entropy_std",         7),   # Std dev of per-token entropy
    ("burstiness_perplexity",     8),   # CV of inter-window PPL (variation coeff)
    ("curvature_score",           9),   # Fast-DetectGPT conditional curvature (Tier 2)
    ("smoothing_factor",         10),   # Length compensation factor 1/L
    ("hybrid_segment_ratio",     11),   # Ratio of AI-flagged segments to total
)

FEATURE_NAMES: Tuple[str, ...] = tuple(name for name, _ in _VECTOR_SCHEMA)

assert len(_VECTOR_SCHEMA) == PERPLEXITY_VECTOR_DIM
assert sorted(idx for _, idx in _VECTOR_SCHEMA) == list(range(PERPLEXITY_VECTOR_DIM))


# ============================================================================
# CONSTANTS & THRESHOLDS (from papers)
# ============================================================================

# Proxy perplexity thresholds calibrated for self-referential mode (Tier 1).
# Scale: [1, 15] where 1 = perfectly uniform (AI-like), 15 = spiky (human-like)
# With external dictionaries, the scale may differ; these are defaults.
_AI_PPL_THRESHOLD_LOW     = 4.0    # Below this → strong AI signal
_AI_PPL_THRESHOLD_MED     = 7.0    # Below this → moderate AI signal
_HUMAN_PPL_TYPICAL        = 10.0   # Typical human proxy PPL

# [NEW v1.2] Tier 2 GPT-2 real perplexity thresholds.
# GPT-2 perplexity scale: AI text ≈ 10-30, Human text ≈ 40-200+
# Calibrated from DetectGPT/Fast-DetectGPT paper distributions.
_T2_AI_PPL_THRESHOLD_LOW  = 25.0   # Below this → strong AI signal
_T2_AI_PPL_THRESHOLD_MED  = 45.0   # Below this → moderate AI signal
_T2_HUMAN_PPL_TYPICAL     = 70.0   # Typical human GPT-2 PPL

# Fast-DetectGPT curvature thresholds (from paper: AI ≈ 3.0, human ≈ 0.0)
_CURVATURE_AI_THRESHOLD   = 1.5    # Above → AI signal
_CURVATURE_HIGH           = 3.0    # Strong AI signal

# Window segmentation parameters (Turnitin-style, documented in survey papers)
_MIN_WINDOW_TOKENS        = 64     # Minimum tokens per window
_TARGET_WINDOW_SENTENCES  = 7      # Target 5-10 sentences per window
_WINDOW_OVERLAP_SENTENCES = 2      # Overlap between consecutive windows
_MIN_SENTENCES_FOR_WINDOWS = 3     # Don't segment if fewer sentences

# N-gram parameters for proxy perplexity
_NGRAM_ORDERS             = (2, 3, 4)   # Bigrams, trigrams, 4-grams
_SMOOTHING_ALPHA          = 0.01        # Laplace smoothing

# Sentence splitting regex (same pattern as reasoning_profiler.py)
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z\d\"'])",
    re.VERBOSE,
)


# ============================================================================
# N-GRAM DICTIONARY (Tier 1 Core)
# ============================================================================

class NgramDictionary:
    """
    LLMDet-style n-gram frequency dictionary for proxy perplexity.

    Two modes:
      1. Embedded: Uses character-level and word-level n-gram frequencies
         computed from the input text itself (self-referential baseline).
      2. Offline: Loads pre-built dictionaries from disk (more accurate).

    The proxy perplexity measures how "predictable" the text's n-gram
    distribution is. AI text produces more uniform/predictable n-gram
    patterns than human text.
    """

    def __init__(self, dict_path: Optional[str] = None) -> None:
        self._external_freqs: Optional[Dict[str, Dict[str, float]]] = None
        if dict_path and os.path.exists(dict_path):
            try:
                with open(dict_path, "r", encoding="utf-8") as f:
                    self._external_freqs = json.load(f)
                logger.info("Loaded n-gram dictionary from %s", dict_path)
            except Exception as exc:
                logger.warning("Failed to load n-gram dict: %s", exc)

    def proxy_perplexity(self, tokens: List[str], orders: Tuple[int, ...] = _NGRAM_ORDERS) -> float:
        """
        Calculate proxy perplexity of a token sequence.

        With external dictionary: Uses stored n-gram frequencies to compute
        how predictable the text is from a corpus perspective (LLMDet method).

        Without external dictionary: Combines three self-referential metrics
        that correlate with true perplexity:
          1. N-gram repetition rate (AI repeats phrases more)
          2. N-gram hapax ratio (human text has more unique n-grams)
          3. Inter-sentence n-gram overlap (AI reuses patterns across sentences)

        Returns
        -------
        float : proxy perplexity score [1, 15] (higher = more human-like)
        """
        if len(tokens) < 3:
            return _HUMAN_PPL_TYPICAL  # insufficient data

        # ── Mode 1: External dictionary (LLMDet-style) ───────────────
        if self._external_freqs:
            total_log_prob = 0.0
            total_count = 0
            for n in orders:
                if len(tokens) < n or str(n) not in self._external_freqs:
                    continue
                freq = self._external_freqs[str(n)]
                vocab_size = max(len(freq), 1)
                total_ngrams = sum(freq.values())
                for i in range(len(tokens) - n + 1):
                    ngram = " ".join(tokens[i:i + n])
                    count = freq.get(ngram, 0)
                    prob = (count + _SMOOTHING_ALPHA) / (total_ngrams + _SMOOTHING_ALPHA * vocab_size)
                    total_log_prob += math.log2(max(prob, 1e-20))
                    total_count += 1
            if total_count > 0:
                avg_log_prob = total_log_prob / total_count
                L = len(tokens)
                ppl = 2.0 ** (-avg_log_prob + (1.0 / L if L > 0 else 0.0))
                return float(np.clip(ppl, 0.1, 15.0))

        # ── Mode 2: Self-referential metrics (no external dict) ───────
        scores = []

        for n in orders:
            if len(tokens) < n + 2:
                continue
            freq = self._build_ngram_freq(tokens, n)
            if not freq:
                continue

            total_ngrams = sum(freq.values())
            unique_ngrams = len(freq)

            # 1. Repetition rate: fraction of n-gram OCCURRENCES that are repeats
            # AI text reuses phrases → higher repetition rate → lower PPL proxy
            repeat_occurrences = sum(c for c in freq.values() if c > 1)
            repetition_rate = repeat_occurrences / total_ngrams if total_ngrams > 0 else 0.0

            # 2. Hapax ratio: fraction of n-gram TYPES appearing exactly once
            # Human text has more unique n-grams → higher hapax → higher PPL proxy
            hapax_count = sum(1 for c in freq.values() if c == 1)
            hapax_ratio = hapax_count / unique_ngrams if unique_ngrams > 0 else 0.0

            # 3. Type-token ratio of n-grams
            # Higher TTR = more diverse = more human-like
            ttr = unique_ngrams / total_ngrams if total_ngrams > 0 else 0.0

            # Combine: higher hapax + higher TTR + lower repetition → more human
            human_signal = hapax_ratio * 0.35 + ttr * 0.35 + (1.0 - repetition_rate) * 0.30
            scores.append(human_signal)

        if not scores:
            return _HUMAN_PPL_TYPICAL

        avg_score = float(np.mean(scores))

        # Map [0, 1] → proxy PPL scale [1, 15]
        ppl = 1.0 + avg_score * 14.0

        # LLMDet smoothing: 1/L correction for short texts
        L = len(tokens)
        ppl += (1.0 / L if L > 0 else 0.0)

        return float(np.clip(ppl, 1.0, 15.0))

    @staticmethod
    def _build_ngram_freq(tokens: List[str], n: int) -> Dict[str, int]:
        """Build frequency dict for n-grams from token list."""
        freq: Dict[str, int] = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            freq[ngram] += 1
        return dict(freq)

    def build_and_save(self, corpus_texts: List[str], output_path: str,
                       orders: Tuple[int, ...] = _NGRAM_ORDERS,
                       top_k: int = 100_000) -> None:
        """
        Build n-gram dictionaries from a corpus and save to disk.

        Parameters
        ----------
        corpus_texts : list of training texts (should include both human & AI)
        output_path  : path to save JSON dictionary
        orders       : n-gram orders to compute
        top_k        : keep only the top_k most frequent n-grams per order
        """
        all_freqs: Dict[str, Dict[str, int]] = {}

        for n in orders:
            merged: Dict[str, int] = Counter()
            for text in corpus_texts:
                tokens = self._tokenize(text)
                for i in range(len(tokens) - n + 1):
                    ngram = " ".join(tokens[i:i + n])
                    merged[ngram] += 1
            top = dict(merged.most_common(top_k))
            all_freqs[str(n)] = top
            logger.info("Built %d-gram dict: %d entries (kept top %d)", n, len(top), top_k)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_freqs, f, ensure_ascii=False)
        logger.info("Saved n-gram dictionary to %s", output_path)

    def build_and_save_streaming(self, text_iterator, output_path: str,
                                  orders: Tuple[int, ...] = _NGRAM_ORDERS,
                                  top_k: int = 200_000,
                                  prune_every: int = 500_000,
                                  prune_keep: int = 500_000,
                                  log_every: int = 100_000) -> None:
        """
        Build n-gram dictionaries from a LARGE corpus using streaming.

        Designed for 1M–100M+ texts. Never loads entire corpus into memory.
        Periodically prunes low-frequency n-grams to control RAM usage.

        Parameters
        ----------
        text_iterator : iterable yielding text strings (generator, file, dataset)
        output_path   : path to save JSON dictionary
        orders        : n-gram orders to compute (default: 2,3,4)
        top_k         : final dict keeps only top_k most frequent per order
        prune_every   : prune counters every N texts to control memory
        prune_keep    : during pruning, keep only the top N n-grams per order
        log_every     : log progress every N texts

        Memory profile
        --------------
          prune_keep=500k with orders=(2,3,4):
            ~500k * 3 orders * ~40 bytes/entry ≈ 60 MB RAM peak
          Without pruning on 37M texts:
            Could reach 10-50 GB RAM → will crash

        Example
        -------
            dict_builder = NgramDictionary()

            # From generator (never loads all texts into memory)
            def text_gen():
                with open("huge_corpus.jsonl") as f:
                    for line in f:
                        yield json.loads(line)["text"]

            dict_builder.build_and_save_streaming(
                text_gen(), "ngram_dict.json",
                prune_every=500_000, prune_keep=500_000
            )
        """
        import time
        t0 = time.time()

        counters: Dict[int, Counter] = {n: Counter() for n in orders}
        processed = 0

        for text in text_iterator:
            tokens = self._tokenize(text)
            if len(tokens) < 3:
                continue

            for n in orders:
                if len(tokens) < n:
                    continue
                for i in range(len(tokens) - n + 1):
                    counters[n][" ".join(tokens[i:i + n])] += 1

            processed += 1

            # Progress logging
            if processed % log_every == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed if elapsed > 0 else 0
                mem_est = sum(len(c) for c in counters.values())
                logger.info(
                    "Progress: %d texts | %.0f texts/sec | %d unique n-grams in memory",
                    processed, rate, mem_est,
                )
                # Also print for non-logging environments
                print(
                    f"  [{processed:,} texts] {rate:,.0f} texts/sec | "
                    f"{mem_est:,} n-grams in memory | "
                    f"{elapsed:.0f}s elapsed",
                    flush=True,
                )

            # Periodic pruning to control memory
            if processed % prune_every == 0:
                for n in orders:
                    before = len(counters[n])
                    if before > prune_keep:
                        counters[n] = Counter(dict(counters[n].most_common(prune_keep)))
                        logger.info(
                            "Pruned %d-gram: %d → %d entries",
                            n, before, len(counters[n]),
                        )

        # Final: keep top_k per order
        all_freqs: Dict[str, Dict[str, int]] = {}
        for n in orders:
            top = dict(counters[n].most_common(top_k))
            all_freqs[str(n)] = top
            logger.info(
                "Final %d-gram dict: %d entries (top %d of %d)",
                n, len(top), top_k, len(counters[n]),
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_freqs, f, ensure_ascii=False)

        elapsed = time.time() - t0
        size = os.path.getsize(output_path)
        print(
            f"\nDictionary built: {output_path}\n"
            f"  Texts processed: {processed:,}\n"
            f"  Time: {elapsed:.0f}s ({processed/elapsed:,.0f} texts/sec)\n"
            f"  File size: {size:,} bytes ({size/1024/1024:.1f} MB)\n"
            f"  N-gram orders: {orders}\n"
            f"  Top-k per order: {top_k:,}",
            flush=True,
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())


# ============================================================================
# TOKEN ENTROPY CALCULATOR (Tier 1)
# ============================================================================

def _token_entropy(tokens: List[str]) -> float:
    """Shannon entropy of token frequency distribution (bits)."""
    if not tokens:
        return 0.0
    freq = Counter(tokens)
    total = len(tokens)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _windowed_token_entropy(tokens: List[str], window_size: int = 50) -> List[float]:
    """Calculate token entropy for sliding windows across the token list."""
    if len(tokens) <= window_size:
        return [_token_entropy(tokens)]
    entropies = []
    step = max(window_size // 2, 1)
    for i in range(0, len(tokens) - window_size + 1, step):
        window = tokens[i:i + window_size]
        entropies.append(_token_entropy(window))
    return entropies


# ============================================================================
# WINDOW SEGMENTER (Turnitin-style, documented in survey papers)
# ============================================================================

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    parts = _SENTENCE_RE.split(text.strip())
    return [s.strip() for s in parts if s.strip()]


def _segment_into_windows(sentences: List[str],
                          target_sents: int = _TARGET_WINDOW_SENTENCES,
                          overlap: int = _WINDOW_OVERLAP_SENTENCES,
                          min_tokens: int = _MIN_WINDOW_TOKENS
                          ) -> List[str]:
    """
    Segment sentences into overlapping windows (Turnitin method).

    Each window is a concatenation of `target_sents` sentences.
    Windows overlap by `overlap` sentences to capture cross-boundary context.

    Returns list of window texts, each guaranteed ≥ min_tokens words.
    """
    if len(sentences) < _MIN_SENTENCES_FOR_WINDOWS:
        return [" ".join(sentences)]

    windows = []
    step = max(target_sents - overlap, 1)

    for i in range(0, len(sentences), step):
        window_sents = sentences[i:i + target_sents]
        window_text = " ".join(window_sents)
        if len(window_text.split()) >= min_tokens:
            windows.append(window_text)

    # Ensure at least one window
    if not windows:
        windows = [" ".join(sentences)]

    return windows


# ============================================================================
# GPT-2 TIER 2 ENGINE (optional, auto-download)
# ============================================================================

class _GPT2Engine:
    """
    Lazy-loading GPT-2 Small engine for token-level perplexity.

    Auto-downloads model on first use (~500 MB).
    Falls back gracefully if torch/transformers unavailable.
    """

    _instance: Optional["_GPT2Engine"] = None
    _initialised: bool = False

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = None
        self._available = False

    @classmethod
    def get(cls) -> Optional["_GPT2Engine"]:
        """Singleton accessor. Returns None if torch unavailable."""
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._try_init()
        return cls._instance if cls._instance._available else None

    def _try_init(self) -> None:
        """Attempt to load GPT-2 Small. Logs warning on failure."""
        if self._initialised:
            return
        self._initialised = True

        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2TokenizerFast

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info("GPT-2 Tier 2: loading model on %s...", self.device)

            self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
            self.model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.model.to(self.device).eval()

            self._available = True
            logger.info("GPT-2 Tier 2: ready (%s)", self.device)

        except ImportError:
            logger.info("GPT-2 Tier 2: torch/transformers not installed — CPU-only mode")
        except Exception as exc:
            logger.warning("GPT-2 Tier 2: failed to load — %s", exc)

    def token_log_probs(self, text: str, max_tokens: int = 512) -> Optional[np.ndarray]:
        """
        Compute per-token log probabilities using GPT-2 Small.

        Returns np.ndarray of shape (num_tokens,) with log2 probabilities,
        or None if model unavailable.
        """
        if not self._available:
            return None

        import torch

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_tokens,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits  # (1, seq_len, vocab_size)

            # Shift: predict token[i+1] from position[i]
            shift_logits = logits[:, :-1, :]
            shift_labels = inputs["input_ids"][:, 1:]

            # Log softmax → gather actual token log probs
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                2, shift_labels.unsqueeze(-1)
            ).squeeze(-1).squeeze(0)

            # Convert to log2 and numpy
            result = (token_log_probs / math.log(2)).cpu().numpy()
            return result

        except Exception as exc:
            logger.warning("GPT-2 token_log_probs failed: %s", exc)
            return None

    def perplexity_per_window(self, windows: List[str], max_tokens: int = 512
                              ) -> List[float]:
        """Compute GPT-2 perplexity for each text window."""
        ppls = []
        for w in windows:
            lp = self.token_log_probs(w, max_tokens)
            if lp is not None and len(lp) > 0:
                avg_neg_lp = -float(np.mean(lp))
                ppl = 2.0 ** avg_neg_lp
                ppls.append(float(np.clip(ppl, 1.0, 10000.0)))
            else:
                ppls.append(_HUMAN_PPL_TYPICAL * 10)  # fallback
        return ppls

    def conditional_curvature(self, text: str, n_samples: int = 100,
                              max_tokens: int = 512) -> float:
        """
        Fast-DetectGPT-style conditional probability curvature.

        Simplified implementation: samples alternative tokens from the model's
        own distribution at each position, computes mean and std of log-probs,
        returns normalised curvature score.

        Higher curvature → more likely AI-generated.
        Typical values: AI ≈ 2-4, Human ≈ 0-1.
        """
        if not self._available:
            return 0.0

        import torch

        try:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=max_tokens,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, :-1, :]   # (1, L-1, V)
                labels = inputs["input_ids"][:, 1:]  # (1, L-1)

            log_probs = torch.log_softmax(logits, dim=-1)
            seq_len = labels.shape[1]

            # Original token log probs
            orig_lp = log_probs.gather(
                2, labels.unsqueeze(-1)
            ).squeeze(-1).squeeze(0)   # (L-1,)

            # [FIX v1.1] Vectorised sampling — no Python for-loop over CUDA.
            # Sample n alternative tokens simultaneously at each position.
            probs = torch.softmax(logits.squeeze(0), dim=-1)  # (L-1, V)
            n_s = min(n_samples, 50)

            # sampled_ids shape: (L-1, n_s)
            sampled_ids = torch.multinomial(probs, num_samples=n_s, replacement=True)

            # Gather log-probs for all sampled tokens in one op. shape: (L-1, n_s)
            sampled_stack = log_probs.squeeze(0).gather(1, sampled_ids)

            # Mean/std across sample dimension (dim=1)
            mu = sampled_stack.mean(dim=1)              # (L-1,)
            sigma = sampled_stack.std(dim=1).clamp(min=1e-8)  # (L-1,)

            # Curvature per token: (original - mean_alternative) / std_alternative
            curvature_per_token = ((orig_lp - mu) / sigma)
            avg_curvature = float(curvature_per_token.mean().cpu())

            return avg_curvature

        except Exception as exc:
            logger.warning("GPT-2 conditional_curvature failed: %s", exc)
            return 0.0


# ============================================================================
# VALLEY DETECTOR (for hybrid document detection)
# ============================================================================

def _count_valleys(values: List[float], threshold: float) -> int:
    """
    Count distinct contiguous regions where values drop below threshold.

    Each valley = a contiguous AI-flagged segment in the perplexity curve.
    Useful for detecting partial AI usage in otherwise human documents.
    """
    if not values:
        return 0
    in_valley = False
    count = 0
    for v in values:
        if v < threshold and not in_valley:
            in_valley = True
            count += 1
        elif v >= threshold:
            in_valley = False
    return count


# ============================================================================
# PERPLEXITY PROFILER — Main Plugin Class
# ============================================================================

class PerplexityProfiler:
    """
    Hybrid perplexity analyser for AI-text detection.

    Tier 1 (always active): N-gram proxy perplexity, token entropy,
    window segmentation — pure CPU, no external models.

    Tier 2 (auto-enabled when torch available): GPT-2 Small token-level
    perplexity, Fast-DetectGPT conditional curvature.

    Usage::

        profiler = PerplexityProfiler()

        # Dict output (for PluginOrchestrator / forensic reports)
        stats = profiler.compute_stats("Some text to analyse...")

        # Vector output (for ML classifiers)
        vec = profiler.vectorize("Some text to analyse...")
        assert vec.shape == (12,)

        # Feature names
        names = profiler.feature_names()

    Offline dictionary mode::

        profiler.build_dictionary(corpus_texts, "ngram_dict.json")
        profiler = PerplexityProfiler(ngram_dict_path="ngram_dict.json")
    """

    __slots__ = ("_ngram_dict", "_gpt2", "_enable_tier2", "_min_tokens")

    def __init__(self, ngram_dict_path: Optional[str] = None,
                 enable_tier2: bool = True,
                 min_tokens: int = 15) -> None:
        """
        Parameters
        ----------
        ngram_dict_path : Path to pre-built n-gram JSON dictionary (optional).
                          If None, uses self-referential n-gram calculation.
        enable_tier2    : Attempt to load GPT-2 for token-level perplexity.
                          Auto-downloads on first use. Falls back to Tier 1
                          if torch/transformers not installed.
        min_tokens      : Minimum word count for meaningful analysis.
        """
        self._ngram_dict = NgramDictionary(ngram_dict_path)
        self._enable_tier2 = enable_tier2
        self._gpt2: Optional[_GPT2Engine] = None
        self._min_tokens = min_tokens

        if enable_tier2:
            self._gpt2 = _GPT2Engine.get()

    @staticmethod
    def feature_names() -> tuple:
        """Return ordered tuple of feature names matching vectorize() output."""
        return FEATURE_NAMES

    def vectorize(self, text: str) -> np.ndarray:
        """
        Extract a 12-dimensional feature vector from raw text.

        Returns zero vector for empty / near-empty inputs.

        Returns
        -------
        np.ndarray : shape (PERPLEXITY_VECTOR_DIM,) = (12,)
        """
        vec = np.zeros(PERPLEXITY_VECTOR_DIM, dtype=np.float64)

        if not text or len(text.split()) < self._min_tokens:
            return vec

        stats = self.compute_stats(text)
        for name, idx in _VECTOR_SCHEMA:
            vec[idx] = stats.get(name, 0.0)

        return vec

    def compute_stats(self, text: str) -> Dict[str, float]:
        """
        Full perplexity analysis returning a feature dictionary.

        This is the primary interface for PluginOrchestrator integration.

        Returns
        -------
        dict : All 12 features plus metadata keys:
            - tier: "tier1" or "tier2" (which engine produced the results)
            - window_count: number of analysis windows
            - window_ppls: list of per-window perplexity values
            - tokens_analysed: total tokens processed
        """
        result: Dict[str, Any] = {name: 0.0 for name, _ in _VECTOR_SCHEMA}
        result["tier"] = "tier1"
        result["window_count"] = 0
        result["window_ppls"] = []
        result["tokens_analysed"] = 0

        if not text or len(text.split()) < self._min_tokens:
            return result

        # ── Tokenize and segment ──────────────────────────────────────
        tokens = re.findall(r"\b\w+\b", text.lower())
        sentences = _split_sentences(text)
        windows = _segment_into_windows(sentences)

        # [FIX v1.1] Pre-tokenize all windows ONCE. Avoids re-running regex
        # on overlapping text that was already tokenized above.
        window_token_lists = [
            re.findall(r"\b\w+\b", w.lower()) for w in windows
        ]

        result["tokens_analysed"] = len(tokens)
        result["window_count"] = len(windows)

        # ── Smoothing factor (LLMDet: 1/L compensation) ──────────────
        L = len(tokens)
        smoothing = 1.0 / L if L > 0 else 0.0
        result["smoothing_factor"] = smoothing

        # ── Tier selection ────────────────────────────────────────────
        use_tier2 = (self._gpt2 is not None and self._gpt2._available)

        if use_tier2:
            result["tier"] = "tier2"
            window_ppls = self._gpt2.perplexity_per_window(windows)
        else:
            # [FIX v1.1] Use pre-tokenized lists — no double regex
            window_ppls = [
                self._ngram_dict.proxy_perplexity(wtokens)
                for wtokens in window_token_lists
            ]

        result["window_ppls"] = window_ppls

        if not window_ppls:
            return result

        # ── Core perplexity features ──────────────────────────────────
        ppl_arr = np.array(window_ppls, dtype=np.float64)

        result["proxy_perplexity_mean"] = float(np.mean(ppl_arr))
        result["proxy_perplexity_std"] = float(np.std(ppl_arr)) if len(ppl_arr) > 1 else 0.0
        result["perplexity_range"] = float(np.ptp(ppl_arr)) if len(ppl_arr) > 1 else 0.0

        # Slope of PPL across windows (positive = getting more human-like)
        if len(ppl_arr) > 1:
            x = np.arange(len(ppl_arr), dtype=np.float64)
            coeffs = np.polyfit(x, ppl_arr, 1)
            result["perplexity_curve_slope"] = float(coeffs[0])
        else:
            result["perplexity_curve_slope"] = 0.0

        # Low perplexity ratio (fraction of windows below AI threshold)
        # [FIX v1.2] Tier-aware threshold: GPT-2 PPL (10-200) vs proxy PPL (1-15)
        ai_threshold = _T2_AI_PPL_THRESHOLD_LOW if use_tier2 else _AI_PPL_THRESHOLD_MED
        low_count = sum(1 for p in window_ppls if p < ai_threshold)
        result["low_perplexity_ratio"] = low_count / len(window_ppls)

        # Valley count (distinct AI-flagged segments)
        result["perplexity_valley_count"] = float(
            _count_valleys(window_ppls, ai_threshold)
        )

        # Burstiness of perplexity (CV = std/mean)
        mean_ppl = result["proxy_perplexity_mean"]
        if mean_ppl > 0:
            result["burstiness_perplexity"] = result["proxy_perplexity_std"] / mean_ppl
        else:
            result["burstiness_perplexity"] = 0.0

        # Hybrid segment ratio
        result["hybrid_segment_ratio"] = result["low_perplexity_ratio"]

        # ── Token entropy features ────────────────────────────────────
        entropy_windows = _windowed_token_entropy(tokens, window_size=50)
        if entropy_windows:
            ent_arr = np.array(entropy_windows, dtype=np.float64)
            result["token_entropy_mean"] = float(np.mean(ent_arr))
            result["token_entropy_std"] = float(np.std(ent_arr)) if len(ent_arr) > 1 else 0.0

        # ── Tier 2: GPT-2 conditional curvature ──────────────────────
        if use_tier2:
            result["curvature_score"] = self._gpt2.conditional_curvature(text)

        return result

    def build_dictionary(self, corpus_texts: List[str],
                         output_path: str) -> None:
        """
        Build n-gram dictionaries from a corpus and save for offline use.

        Parameters
        ----------
        corpus_texts : list of training texts (mix of human + AI recommended)
        output_path  : where to save the JSON dictionary
        """
        self._ngram_dict.build_and_save(corpus_texts, output_path)
        logger.info("Dictionary built and saved to %s", output_path)


# ============================================================================
# PERPLEXITY RISK CLASSIFIER (for ForensicReportGenerator integration)
# ============================================================================

class PerplexityRiskClassifier:
    """
    Severity-profile classifier for the 12-dim vector from PerplexityProfiler.

    DISPLAY/REPORTING LAYER ONLY — produces per-feature severity levels and
    human-readable explanations for ForensicReportGenerator.

    ⚠ IMPORTANT (Late Fusion compliance): The `ai_score` field is derived
    from a simple majority-vote of per-feature severity levels. It is NOT
    a calibrated probability and MUST NOT be used as ML input. For ML
    pipelines (XGBoost, MLP), use `PerplexityProfiler.vectorize()` directly
    and let the meta-classifier learn its own weights.
    """

    # Severity level thresholds for the majority-vote ai_score
    _HIGH   = 0.65
    _MEDIUM = 0.35

    _EXPLANATIONS = {
        "proxy_perplexity_mean": {
            "display": "Proxy Perplexity (N-gram)",
            "high": "Very low proxy perplexity ({v:.2f}) — the text's n-gram patterns are highly predictable, characteristic of AI-generated content.",
            "medium": "Moderate proxy perplexity ({v:.2f}) — some predictable patterns but not conclusive.",
            "low": "High proxy perplexity ({v:.2f}) — n-gram patterns show natural unpredictability consistent with human writing.",
        },
        "low_perplexity_ratio": {
            "display": "Low-Perplexity Window Ratio",
            "high": "A large fraction of text windows ({v:.0%}) show AI-typical low perplexity. Most of the document appears machine-generated.",
            "medium": "Some windows ({v:.0%}) show low perplexity. Possible partial AI involvement or AI-assisted editing.",
            "low": "Few or no windows show AI-typical perplexity patterns ({v:.0%}). Consistent with human authorship.",
        },
        "perplexity_valley_count": {
            "display": "Perplexity Valley Count (Hybrid Detection)",
            "high": "Multiple distinct low-perplexity valleys ({v:.0f}) detected across the document, suggesting AI-generated sections interspersed with human writing.",
            "medium": "Some perplexity valleys ({v:.0f}) present. Could indicate selective AI assistance.",
            "low": "No significant perplexity valleys ({v:.0f}). Uniform authorship pattern.",
        },
        "curvature_score": {
            "display": "Conditional Probability Curvature (GPT-2)",
            "high": "High curvature score ({v:.2f}) — the text sits near a local maximum of the language model's probability landscape, a strong AI generation signature (Fast-DetectGPT method).",
            "medium": "Moderate curvature ({v:.2f}). Consistent with either AI text or highly structured human writing.",
            "low": "Low curvature ({v:.2f}). The text does not occupy the probability peaks characteristic of AI output.",
        },
        "token_entropy_mean": {
            "display": "Mean Token Entropy",
            "high": "Very low token entropy ({v:.3f}) — the word distribution is highly concentrated/predictable, a common AI signature.",
            "medium": "Moderate token entropy ({v:.3f}).",
            "low": "Healthy token entropy ({v:.3f}) — diverse, unpredictable word distribution typical of human writing.",
        },
        "burstiness_perplexity": {
            "display": "Perplexity Burstiness (CV)",
            "high": "Very low perplexity variation ({v:.3f}) across windows — uniform predictability throughout, typical of single-source AI generation.",
            "medium": "Moderate perplexity variation ({v:.3f}).",
            "low": "High perplexity variation ({v:.3f}) — natural fluctuation in text complexity, typical of human writing.",
        },
    }

    def classify(self, stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Produce severity profile + display-only ai_score from perplexity stats.

        Returns
        -------
        dict with keys:
            ai_score          : float [0, 1] — DISPLAY ONLY, not for ML
            risk_level        : str — "HIGH", "MEDIUM", "LOW", "INSUFFICIENT DATA"
            severity_profile  : dict — per-feature severity (for Late Fusion)
            interpretation    : str — human-readable summary
            feature_details   : dict — per-feature breakdown with explanations
            tier              : str — "tier1" or "tier2"
        """
        # Build severity profile first — this is the ground truth
        severity = self._severity_profile(stats)
        score = self._score_from_severity(severity, stats)
        level = self._level(score)

        return {
            "ai_score": score,
            "risk_level": level,
            "severity_profile": severity,
            "interpretation": self._interpretation(score, severity, stats),
            "feature_details": self._feature_details(stats),
            "tier": stats.get("tier", "tier1"),
            "window_count": stats.get("window_count", 0),
        }

    def _severity_profile(self, stats: Dict[str, float]) -> Dict[str, str]:
        """
        Per-feature severity levels. This is the primary output for
        Late Fusion — XGBoost should consume the raw vector, but the
        report can display these levels to explain each dimension.
        """
        tier = stats.get("tier", "tier1")
        profile = {}
        for feat in self._EXPLANATIONS:
            val = stats.get(feat, 0.0)
            profile[feat] = self._feat_level(feat, val, tier)
        return profile

    def _score_from_severity(self, severity: Dict[str, str],
                             stats: Dict[str, float]) -> float:
        """
        Derive display-only ai_score via majority vote of severity levels.

        No magic weights — each feature that flags HIGH contributes 1.0,
        MEDIUM contributes 0.5, LOW contributes 0.0. The score is the
        mean of these votes, scaled by length-confidence.

        ⚠ This is for ForensicReportGenerator display ONLY.
        """
        tokens = stats.get("tokens_analysed", 0)
        if tokens < 15:
            return 0.0

        votes = []
        for feat, level in severity.items():
            # Skip curvature if Tier 2 not active (would always vote LOW)
            if feat == "curvature_score" and stats.get("curvature_score", 0.0) == 0.0:
                continue
            if level == "high":
                votes.append(1.0)
            elif level == "medium":
                votes.append(0.5)
            else:
                votes.append(0.0)

        if not votes:
            return 0.0

        raw_score = float(np.mean(votes))

        # Length-based confidence scaling (LLMDet finding)
        if tokens < 300:
            confidence = min(1.0, tokens / 300.0)
            raw_score = 0.5 + (raw_score - 0.5) * confidence

        return round(min(1.0, max(0.0, raw_score)), 4)

    def _level(self, score: float) -> str:
        if score <= 0.0:
            return "INSUFFICIENT DATA"
        if score >= self._HIGH:
            return "HIGH — AI Perplexity Signature"
        if score >= self._MEDIUM:
            return "MEDIUM — Possible AI Involvement"
        return "LOW — Human-like Perplexity"

    def _interpretation(self, score: float, severity: Dict[str, str],
                        stats: Dict[str, float]) -> str:
        ppl = stats.get("proxy_perplexity_mean", 0.0)
        curv = stats.get("curvature_score", 0.0)
        lpr = stats.get("low_perplexity_ratio", 0.0)
        tier = stats.get("tier", "tier1")
        valleys = int(stats.get("perplexity_valley_count", 0))

        tier_note = ("Analysis includes GPT-2 token-level perplexity and "
                     "conditional curvature (Tier 2)." if tier == "tier2"
                     else "Analysis uses n-gram proxy perplexity only (Tier 1, CPU).")

        # Count severity flags for interpretation
        high_count = sum(1 for v in severity.values() if v == "high")
        med_count = sum(1 for v in severity.values() if v == "medium")

        if score <= 0.0:
            return ("Insufficient text for reliable perplexity analysis. "
                    "At least 15 words are needed for meaningful results. " + tier_note)

        if score >= self._HIGH:
            base = (f"Strong AI perplexity signature ({high_count} features flag HIGH). "
                    f"Mean proxy perplexity is {ppl:.2f} (AI-typical range). "
                    f"{lpr:.0%} of text windows show AI-level predictability.")
            if curv > _CURVATURE_AI_THRESHOLD:
                base += (f" GPT-2 conditional curvature ({curv:.2f}) confirms "
                         f"the text occupies probability peaks characteristic "
                         f"of machine generation.")
            if valleys > 1:
                base += (f" {valleys} distinct low-perplexity regions detected, "
                         f"suggesting possible hybrid authorship.")
        elif score >= self._MEDIUM:
            base = (f"Moderate AI perplexity indicators ({high_count} HIGH, "
                    f"{med_count} MEDIUM features). "
                    f"Mean proxy perplexity={ppl:.2f}, low-PPL window "
                    f"ratio={lpr:.0%}. Compatible with AI-assisted writing, "
                    f"heavily edited AI text, or methodical human composition.")
        else:
            base = (f"Low AI perplexity indicators (most features flag LOW). "
                    f"Mean proxy perplexity={ppl:.2f} — within human-typical "
                    f"range. The text shows natural unpredictability patterns "
                    f"consistent with human authorship.")

        return f"{base} {tier_note}"

    def _feature_details(self, stats: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Per-feature breakdown for forensic report."""
        tier = stats.get("tier", "tier1")
        details = {}
        for feat, info in self._EXPLANATIONS.items():
            val = stats.get(feat, 0.0)
            level = self._feat_level(feat, val, tier)
            template = info.get(level, "")
            details[feat] = {
                "display_name": info["display"],
                "value": round(val, 6),
                "level": level,
                "explanation": template.format(v=val) if "{v" in template else template,
            }
        return details

    def _feat_level(self, feat: str, val: float, tier: str = "tier1") -> str:
        """
        Determine HIGH/MEDIUM/LOW for a single feature.

        [FIX v1.2] Tier-aware: uses GPT-2 scale thresholds (10-200+)
        for Tier 2, and proxy scale thresholds (1-15) for Tier 1.
        """
        # Inverse features: lower value = higher AI risk
        if feat in ("proxy_perplexity_mean", "token_entropy_mean",
                     "burstiness_perplexity"):
            if feat == "proxy_perplexity_mean":
                if tier == "tier2":
                    # GPT-2 real perplexity: AI ≈ 10-25, Human ≈ 45-200+
                    if val < _T2_AI_PPL_THRESHOLD_LOW: return "high"
                    if val < _T2_AI_PPL_THRESHOLD_MED: return "medium"
                    return "low"
                else:
                    # Tier 1 proxy scale: [1, 15]
                    if val < _AI_PPL_THRESHOLD_LOW: return "high"
                    if val < _AI_PPL_THRESHOLD_MED: return "medium"
                    return "low"
            elif feat == "token_entropy_mean":
                if val < 3.0: return "high"
                if val < 4.5: return "medium"
                return "low"
            else:  # burstiness
                if val < 0.05: return "high"
                if val < 0.15: return "medium"
                return "low"

        # Direct features: higher value = higher AI risk
        if feat == "low_perplexity_ratio":
            if val > 0.7: return "high"
            if val > 0.3: return "medium"
            return "low"
        if feat == "perplexity_valley_count":
            if val >= 3: return "high"
            if val >= 1: return "medium"
            return "low"
        if feat == "curvature_score":
            if val > _CURVATURE_HIGH: return "high"
            if val > _CURVATURE_AI_THRESHOLD: return "medium"
            return "low"

        return "low"
