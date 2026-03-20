"""
reasoning_profiler.py
=====================
Zero-Resource Feature Extractor for Reasoning-Model Detection (v5.0)

Extracts a fixed-dimension numerical vector φ(x) from raw text for
downstream classification (XGBoost, PyTorch MLP, ModernBERT head, etc.).
Contains **NO** classification logic, thresholds, or heuristic scores —
pure statistical feature extraction following Late Fusion architecture.

Architecture:
    ParsedText (immutable DTO, computed once)
      └─► StylometricExtractor   → indices 0–5
      └─► DiscourseExtractor     → indices 6–9
      └─► ReasoningMarkerExtractor → indices 10–12
      └─► StructuralExtractor    → indices 13–14

Target: PyTorch / Polars pipelines processing 4M+ documents.
Complexity: O(n) per document where n = token count.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ============================================================================
# VECTOR SCHEMA — single source of truth for feature → tensor index mapping
# ============================================================================

REASONING_VECTOR_DIM: int = 15

_VECTOR_SCHEMA: tuple[tuple[str, int], ...] = (
    # ── Stylometric / Lexical (0–5) ───────────────────────────────────────
    ("type_token_ratio",            0),   # TTR = |V| / N
    ("mean_sentence_length",        1),   # μ(sentence word counts)
    ("std_sentence_length",         2),   # σ(sentence word counts), ddof=1
    ("mean_word_length",            3),   # Σ len(token) / N
    ("punctuation_ratio",           4),   # |punct chars| / |all chars|
    ("stopword_ratio",              5),   # |stopword tokens| / N
    # ── Discourse Connector Densities (6–9) ───────────────────────────────
    ("consequence_density",         6),   # matches / sentence_count
    ("causal_density",              7),
    ("contrast_density",            8),
    ("sequence_density",            9),
    # ── Reasoning-Specific Markers (10–12) ────────────────────────────────
    ("backtracking_density",       10),   # self-correction / sentence_count
    ("cot_scaffold_density",       11),   # CoT scaffolding / sentence_count
    ("intuition_leap_density",     12),   # heuristic leaps / sentence_count
    # ── Structural / Information-Theoretic (13–14) ────────────────────────
    ("paragraph_length_cv",        13),   # CV = σ/μ of paragraph word counts
    ("word_entropy_normalised",    14),   # H(words) / log₂(|V|)
)

FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in _VECTOR_SCHEMA)

# Compile-time assertion: schema covers every index exactly once
assert len(_VECTOR_SCHEMA) == REASONING_VECTOR_DIM
assert sorted(idx for _, idx in _VECTOR_SCHEMA) == list(range(REASONING_VECTOR_DIM))


# ============================================================================
# MODULE-LEVEL COMPILED REGEXES  (word-boundary guarded, re.VERBOSE)
# ============================================================================

# Sentence boundary: split after terminal punctuation + whitespace.
# Avoids splitting on decimals (3.14), abbreviations (U.S.), ellipses.
_SENTENCE_RE: re.Pattern[str] = re.compile(
    r"""
    (?<= [.!?] )   # lookbehind: sentence-ending punctuation
    \s+             # one or more whitespace chars
    (?= [A-Z\d"'] )  # lookahead: likely sentence start
    """,
    re.VERBOSE,
)

# ── Discourse connectors ──────────────────────────────────────────────────
# Multi-word patterns listed FIRST so they match before single-word fallbacks.
# Every alternative is \b-guarded to prevent the substring bug
# (e.g. "also" matching "so", "asset" matching "as").

_CONSEQUENCE_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: as \s+ a \s+ result
         | consequently
         | accordingly
         | therefore
         | thus
         | hence
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CAUSAL_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: due \s+ to
         | owing \s+ to
         | given \s+ that
         | because
         | since
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CONTRAST_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: on \s+ the \s+ other \s+ hand
         | nevertheless
         | nonetheless
         | however
         | although
         | despite
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SEQUENCE_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: first   (?:ly)?
         | second  (?:ly)?
         | third   (?:ly)?
         | finally
         | subsequently
         | initially
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# ── Reasoning markers ─────────────────────────────────────────────────────
# Backtracking / self-correction signatures from o1, DeepSeek-R1, QwQ traces

_BACKTRACK_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: wait \s* [,.\-!…]+ \s* wait          # "wait...wait"
         | but  \s+ wait
         | let  (?:'s | \s+ us | \s+ me) \s+ re-?evaluate
         | on   \s+ (?:the \s+)? second \s+ thought
         | (?:no | actually) [,\s]+ that (?:'s | \s+ is) \s+
           (?:wrong | incorrect | not \s+ (?:right | correct))
         | i \s+ made \s+ (?:an? \s+)? (?:error | mistake)
         | let  \s+ me \s+ reconsider
         | this \s+ (?:reasoning | approach) \s+ is \s+
           (?:not \s+)? correct
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# CoT scaffolding: structural framing typical of step-by-step models

_COT_SCAFFOLD_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: let (?:'s | \s+ me) \s+ think
         | step \s+ by \s+ step
         | let (?:'s | \s+ me) \s+ break \s+ (?:this | it) \s+ down
         | (?:first | now) \s+ (?:i | we) \s+ need \s+ to
         | working  \s+ through
         | to  \s+ solve  \s+ this
         | the \s+ key    \s+ insight
         | step \s+ \d+
         | from \s+ this \s+ we \s+ can \s+ conclude
         | it \s+ follows \s+ that
         | reasoning \s+ through
         | analyzing \s+ this
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Intuition/heuristic leap markers (human-like shortcuts)

_INTUITION_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: obviously
         | clearly
         | of    \s+ course
         | naturally
         | it    \s+ goes \s+ without \s+ saying
         | needless \s+ to \s+ say
         | surely
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Non-word characters (punctuation counter)
_PUNCTUATION_RE: re.Pattern[str] = re.compile(r"[^\w\s]")


# ============================================================================
# ENGLISH STOPWORDS  (compact inline set — zero external dependency)
# ============================================================================

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "not", "no", "nor", "so", "as", "it", "its", "this", "that", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "him",
    "his", "she", "her", "they", "them", "their", "what", "which", "who",
    "whom", "when", "where", "how", "why", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "about", "above", "after", "again", "also", "any", "because",
    "before", "between", "during", "into", "only", "over", "same", "then",
    "there", "through", "under", "until", "up", "while",
})

# Division guard constant
_EPS: float = 1e-9


# ============================================================================
# PARSED TEXT DTO  (immutable, computed EXACTLY ONCE, injected everywhere)
# ============================================================================

@dataclass(frozen=True, slots=True)
class ParsedText:
    """
    Immutable pre-processed representation of a document.

    Every downstream extractor receives this object rather than raw text,
    guaranteeing that tokenisation, lowercasing, and set construction
    happen at most once per document.
    """

    raw: str
    lower: str
    tokens: tuple[str, ...]
    token_count: int
    char_count: int
    sentences: tuple[str, ...]
    sentence_count: int
    sentence_word_counts: np.ndarray          # dtype=float64
    paragraphs: tuple[str, ...]
    paragraph_count: int
    paragraph_word_counts: np.ndarray          # dtype=float64
    word_freq: Counter[str]
    unique_token_count: int
    stopword_count: int
    punctuation_count: int

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_raw(cls, text: str) -> ParsedText:
        raw = text.strip()
        lower = raw.lower()

        # Tokens — single whitespace split on lowered text
        tokens = tuple(lower.split())
        token_count = len(tokens)
        char_count = max(len(raw), 1)  # guard against empty

        # Sentences — split on terminal punctuation + capital lookahead
        raw_sents = _SENTENCE_RE.split(raw)
        sentences = tuple(s.strip() for s in raw_sents if len(s.split()) >= 3)
        sentence_count = max(len(sentences), 1)
        sentence_word_counts = np.array(
            [len(s.split()) for s in sentences], dtype=np.float64
        ) if sentences else np.zeros(1, dtype=np.float64)

        # Paragraphs — double newline split
        raw_paras = raw.split("\n\n")
        paragraphs = tuple(p.strip() for p in raw_paras if p.strip())
        paragraph_count = len(paragraphs)
        paragraph_word_counts = np.array(
            [len(p.split()) for p in paragraphs], dtype=np.float64
        ) if paragraphs else np.zeros(1, dtype=np.float64)

        # Word frequency distribution
        word_freq: Counter[str] = Counter(tokens)
        unique_token_count = len(word_freq)

        # Stopwords — O(|stopwords|) lookup against the freq counter
        stopword_count = sum(word_freq[w] for w in _STOPWORDS if w in word_freq)

        # Punctuation characters
        punctuation_count = len(_PUNCTUATION_RE.findall(raw))

        return cls(
            raw=raw,
            lower=lower,
            tokens=tokens,
            token_count=token_count,
            char_count=char_count,
            sentences=sentences,
            sentence_count=sentence_count,
            sentence_word_counts=sentence_word_counts,
            paragraphs=paragraphs,
            paragraph_count=paragraph_count,
            paragraph_word_counts=paragraph_word_counts,
            word_freq=word_freq,
            unique_token_count=unique_token_count,
            stopword_count=stopword_count,
            punctuation_count=punctuation_count,
        )


# ============================================================================
# SUB-EXTRACTORS  (__slots__, stateless, write directly into the output vector)
# ============================================================================

class StylometricExtractor:
    """
    Indices 0–5: TTR, sentence length μ/σ, word length μ,
    punctuation ratio, stopword ratio.

    All formulas from Stylometric Detectability (SD) literature.
    """

    __slots__ = ()

    @staticmethod
    def extract(pt: ParsedText, vec: np.ndarray) -> None:
        tc = max(pt.token_count, 1)

        # [0] Type-Token Ratio: TTR = |V| / N
        vec[0] = pt.unique_token_count / tc

        # [1] Mean sentence length: μ
        vec[1] = float(pt.sentence_word_counts.mean())

        # [2] Sentence length standard deviation: σ (sample, ddof=1)
        if len(pt.sentence_word_counts) >= 2:
            vec[2] = float(pt.sentence_word_counts.std(ddof=1))
        else:
            vec[2] = 0.0

        # [3] Mean word length (chars per token)
        total_chars = sum(len(t) for t in pt.tokens)
        vec[3] = total_chars / tc

        # [4] Punctuation ratio: |punct| / |chars|
        vec[4] = pt.punctuation_count / pt.char_count

        # [5] Stopword ratio: |stopwords| / N
        vec[5] = pt.stopword_count / tc


class DiscourseExtractor:
    """
    Indices 6–9: consequence / causal / contrast / sequence connector
    densities normalised by sentence count.

    Uses compiled word-boundary regexes to prevent the substring bug
    (e.g. "also" ≠ "so", "asset" ≠ "as").
    """

    __slots__ = ()

    _PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
        (6, _CONSEQUENCE_RE),
        (7, _CAUSAL_RE),
        (8, _CONTRAST_RE),
        (9, _SEQUENCE_RE),
    )

    @staticmethod
    def extract(pt: ParsedText, vec: np.ndarray) -> None:
        sc = pt.sentence_count  # already ≥ 1 from ParsedText
        for idx, pattern in DiscourseExtractor._PATTERNS:
            vec[idx] = len(pattern.findall(pt.lower)) / sc


class ReasoningMarkerExtractor:
    """
    Indices 10–12: backtracking density, CoT scaffolding density,
    intuition leap density.

    Captures the three dominant reasoning signatures documented in
    o1 / DeepSeek-R1 / QwQ trace analysis.
    """

    __slots__ = ()

    @staticmethod
    def extract(pt: ParsedText, vec: np.ndarray) -> None:
        sc = pt.sentence_count
        vec[10] = len(_BACKTRACK_RE.findall(pt.lower)) / sc
        vec[11] = len(_COT_SCAFFOLD_RE.findall(pt.lower)) / sc
        vec[12] = len(_INTUITION_RE.findall(pt.lower)) / sc


class StructuralExtractor:
    """
    Indices 13–14: paragraph length coefficient of variation (CV = σ/μ),
    normalised Shannon entropy of word distribution.

    CV replaces the original's ``1 - var/500`` magic-number formula.
    Shannon entropy replaces all-pairs Jaccard (O(n²)) with O(n)
    information-theoretic redundancy measurement.
    """

    __slots__ = ()

    @staticmethod
    def extract(pt: ParsedText, vec: np.ndarray) -> None:
        # [13] Paragraph-length Coefficient of Variation: CV = σ / μ
        if pt.paragraph_count >= 2:
            mu = pt.paragraph_word_counts.mean()
            sigma = pt.paragraph_word_counts.std(ddof=1)
            vec[13] = sigma / max(mu, _EPS)
        else:
            vec[13] = 0.0

        # [14] Normalised Shannon entropy of the word distribution
        #      H_norm = H(words) / log₂(|V|)
        #      where H = −Σ p(t) log₂ p(t)
        #
        #      Value near 1.0 → high lexical diversity (human-like)
        #      Value near 0.0 → repetitive / low diversity (AI loop)
        if pt.unique_token_count >= 2:
            counts = np.fromiter(
                pt.word_freq.values(), dtype=np.float64, count=pt.unique_token_count,
            )
            probs = counts / counts.sum()
            entropy = -float(np.sum(probs * np.log2(probs + _EPS)))
            max_entropy = math.log2(pt.unique_token_count)
            vec[14] = entropy / max(max_entropy, _EPS)
        else:
            vec[14] = 0.0


# ============================================================================
# MAIN PROFILER — PUBLIC API
# ============================================================================

class ReasoningProfiler:
    """
    Zero-resource feature extractor for reasoning-model detection.

    Produces a fixed-dimension numpy vector φ(x) suitable for any
    downstream classifier. Contains NO classification or thresholding
    logic — pure Late Fusion feature extraction.

    Usage::

        profiler = ReasoningProfiler()

        # Single document
        vec = profiler.vectorize("Some text to analyse...")
        assert vec.shape == (15,)

        # Batch (for Polars / DataLoader integration)
        matrix = profiler.vectorize_batch(list_of_texts)
        assert matrix.shape == (len(list_of_texts), 15)

    Integration with forensic_reporter v3.2::

        vec = profiler.vectorize(text)
        feature_dict = dict(zip(FEATURE_NAMES, vec.tolist()))
        reporter.generate_report(
            text,
            detection_result=result,
            additional_analyses={"reasoning_profile": feature_dict},
        )
    """

    __slots__ = ("_extractors", "_min_tokens")

    def __init__(self, min_tokens: int = 20) -> None:
        self._min_tokens = min_tokens
        self._extractors: tuple[type, ...] = (
            StylometricExtractor,
            DiscourseExtractor,
            ReasoningMarkerExtractor,
            StructuralExtractor,
        )

    def vectorize(self, text: str) -> np.ndarray:
        """
        Extract a feature vector of shape ``(REASONING_VECTOR_DIM,)``
        from raw text.

        Returns a zero vector for empty / near-empty inputs
        (fewer than ``min_tokens`` whitespace-delimited tokens).

        All indices in the output are guaranteed to be written;
        allocation via ``np.empty`` is safe.
        """
        vec = np.empty(REASONING_VECTOR_DIM, dtype=np.float64)

        if not text or len(text.split()) < self._min_tokens:
            vec[:] = 0.0
            return vec

        pt = ParsedText.from_raw(text)

        for extractor_cls in self._extractors:
            extractor_cls.extract(pt, vec)

        return vec

    def vectorize_batch(self, texts: Sequence[str]) -> np.ndarray:
        """
        Vectorize a batch of texts.

        Returns:
            np.ndarray of shape ``(len(texts), REASONING_VECTOR_DIM)``.

        Note:
            For maximum throughput in production, parallelise at the
            Polars / DataLoader level rather than here. This method
            is a convenience wrapper for sequential processing.
        """
        n = len(texts)
        out = np.empty((n, REASONING_VECTOR_DIM), dtype=np.float64)
        for i in range(n):
            out[i] = self.vectorize(texts[i])
        return out

    @staticmethod
    def schema() -> tuple[tuple[str, int], ...]:
        """Return the (feature_name, index) mapping."""
        return _VECTOR_SCHEMA

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        """Return ordered feature names matching tensor indices."""
        return FEATURE_NAMES

    @staticmethod
    def dim() -> int:
        """Return the dimensionality of the output vector."""
        return REASONING_VECTOR_DIM
