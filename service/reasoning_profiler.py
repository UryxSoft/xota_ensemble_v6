"""
reasoning_profiler.py  v5.0.1
==============================
Zero-Resource Feature Extractor for Reasoning-Model Detection.

Changelog (v5.0 -> v5.0.1)
----------------------------
  [FIX P1]  ``ParsedText`` renamed to ``ReasoningParsedText``.
             The original name collided with ``hallucination_profile.ParsedText``
             (7 fields, no __slots__).  Both classes had identical public names
             but incompatible schemas — any shared-namespace import would
             silently shadow one with the other.
             Public API unaffected: ``ReasoningProfiler`` never exposed
             ``ParsedText`` in its method signatures.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# ============================================================================
# VECTOR SCHEMA
# ============================================================================

REASONING_VECTOR_DIM: int = 15

_VECTOR_SCHEMA: tuple[tuple[str, int], ...] = (
    ("type_token_ratio",        0),
    ("mean_sentence_length",    1),
    ("std_sentence_length",     2),
    ("mean_word_length",        3),
    ("punctuation_ratio",       4),
    ("stopword_ratio",          5),
    ("consequence_density",     6),
    ("causal_density",          7),
    ("contrast_density",        8),
    ("sequence_density",        9),
    ("backtracking_density",   10),
    ("cot_scaffold_density",   11),
    ("intuition_leap_density", 12),
    ("paragraph_length_cv",    13),
    ("word_entropy_normalised",14),
)

FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in _VECTOR_SCHEMA)

assert len(_VECTOR_SCHEMA) == REASONING_VECTOR_DIM
assert sorted(idx for _, idx in _VECTOR_SCHEMA) == list(range(REASONING_VECTOR_DIM))


# ============================================================================
# MODULE-LEVEL COMPILED REGEXES
# ============================================================================

_SENTENCE_RE: re.Pattern[str] = re.compile(
    r"""
    (?<= [.!?] )
    \s+
    (?= [A-Z\d"'] )
    """,
    re.VERBOSE,
)

_CONSEQUENCE_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: as \s+ a \s+ result | consequently | accordingly
         | therefore | thus | hence
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CAUSAL_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: due \s+ to | owing \s+ to | given \s+ that
         | because | since
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CONTRAST_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: on \s+ the \s+ other \s+ hand | nevertheless | nonetheless
         | however | although | despite
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SEQUENCE_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: first (?:ly)? | second (?:ly)? | third (?:ly)?
         | finally | subsequently | initially
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_BACKTRACK_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: wait \s* [,.\-!…]+ \s* wait
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

_INTUITION_RE: re.Pattern[str] = re.compile(
    r"""
    \b (?: obviously | clearly | of \s+ course
         | naturally | it \s+ goes \s+ without \s+ saying
         | needless \s+ to \s+ say | surely
    ) \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PUNCTUATION_RE: re.Pattern[str] = re.compile(r"[^\w\s]")


# ============================================================================
# ENGLISH STOPWORDS
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

_EPS: float = 1e-9


# ============================================================================
# PARSED TEXT DTO  ── P1 FIX: renamed from ParsedText → ReasoningParsedText
# ============================================================================

@dataclass(frozen=True, slots=True)
class ReasoningParsedText:
    """
    Immutable pre-processed representation of a document.

    Renamed from ``ParsedText`` (v5.0) to ``ReasoningParsedText`` (v5.0.1)
    to eliminate the namespace collision with
    ``hallucination_profile.ParsedText`` (different schema, no __slots__).
    """

    raw: str
    lower: str
    tokens: tuple[str, ...]
    token_count: int
    char_count: int
    sentences: tuple[str, ...]
    sentence_count: int
    sentence_word_counts: np.ndarray
    paragraphs: tuple[str, ...]
    paragraph_count: int
    paragraph_word_counts: np.ndarray
    word_freq: Counter
    unique_token_count: int
    stopword_count: int
    punctuation_count: int

    @classmethod
    def from_raw(cls, text: str) -> "ReasoningParsedText":
        raw = text.strip()
        lower = raw.lower()
        tokens = tuple(lower.split())
        token_count = len(tokens)
        char_count = max(len(raw), 1)

        raw_sents = _SENTENCE_RE.split(raw)
        sentences = tuple(s.strip() for s in raw_sents if len(s.split()) >= 3)
        sentence_count = max(len(sentences), 1)
        sentence_word_counts = (
            np.array([len(s.split()) for s in sentences], dtype=np.float64)
            if sentences else np.zeros(1, dtype=np.float64)
        )

        raw_paras = raw.split("\n\n")
        paragraphs = tuple(p.strip() for p in raw_paras if p.strip())
        paragraph_count = len(paragraphs)
        paragraph_word_counts = (
            np.array([len(p.split()) for p in paragraphs], dtype=np.float64)
            if paragraphs else np.zeros(1, dtype=np.float64)
        )

        word_freq: Counter = Counter(tokens)
        unique_token_count = len(word_freq)
        stopword_count = sum(word_freq[w] for w in _STOPWORDS if w in word_freq)
        punctuation_count = len(_PUNCTUATION_RE.findall(raw))

        return cls(
            raw=raw, lower=lower, tokens=tokens,
            token_count=token_count, char_count=char_count,
            sentences=sentences, sentence_count=sentence_count,
            sentence_word_counts=sentence_word_counts,
            paragraphs=paragraphs, paragraph_count=paragraph_count,
            paragraph_word_counts=paragraph_word_counts,
            word_freq=word_freq, unique_token_count=unique_token_count,
            stopword_count=stopword_count, punctuation_count=punctuation_count,
        )


# ============================================================================
# SUB-EXTRACTORS
# ============================================================================

class StylometricExtractor:
    __slots__ = ()

    @staticmethod
    def extract(pt: ReasoningParsedText, vec: np.ndarray) -> None:
        tc = max(pt.token_count, 1)
        vec[0] = pt.unique_token_count / tc
        vec[1] = float(pt.sentence_word_counts.mean())
        vec[2] = float(pt.sentence_word_counts.std(ddof=1)) if len(pt.sentence_word_counts) >= 2 else 0.0
        total_chars = sum(len(t) for t in pt.tokens)
        vec[3] = total_chars / tc
        vec[4] = pt.punctuation_count / pt.char_count
        vec[5] = pt.stopword_count / tc


class DiscourseExtractor:
    __slots__ = ()

    _PATTERNS: tuple[tuple[int, re.Pattern[str]], ...] = (
        (6, _CONSEQUENCE_RE),
        (7, _CAUSAL_RE),
        (8, _CONTRAST_RE),
        (9, _SEQUENCE_RE),
    )

    @staticmethod
    def extract(pt: ReasoningParsedText, vec: np.ndarray) -> None:
        sc = pt.sentence_count
        for idx, pattern in DiscourseExtractor._PATTERNS:
            vec[idx] = len(pattern.findall(pt.lower)) / sc


class ReasoningMarkerExtractor:
    __slots__ = ()

    @staticmethod
    def extract(pt: ReasoningParsedText, vec: np.ndarray) -> None:
        sc = pt.sentence_count
        vec[10] = len(_BACKTRACK_RE.findall(pt.lower)) / sc
        vec[11] = len(_COT_SCAFFOLD_RE.findall(pt.lower)) / sc
        vec[12] = len(_INTUITION_RE.findall(pt.lower)) / sc


class StructuralExtractor:
    __slots__ = ()

    @staticmethod
    def extract(pt: ReasoningParsedText, vec: np.ndarray) -> None:
        if pt.paragraph_count >= 2:
            mu = pt.paragraph_word_counts.mean()
            sigma = pt.paragraph_word_counts.std(ddof=1)
            vec[13] = sigma / max(mu, _EPS)
        else:
            vec[13] = 0.0

        if pt.unique_token_count >= 2:
            counts = np.fromiter(
                pt.word_freq.values(), dtype=np.float64, count=pt.unique_token_count
            )
            probs = counts / counts.sum()
            entropy = -float(np.sum(probs * np.log2(probs + _EPS)))
            max_entropy = math.log2(pt.unique_token_count)
            vec[14] = entropy / max(max_entropy, _EPS)
        else:
            vec[14] = 0.0


# ============================================================================
# MAIN PROFILER
# ============================================================================

class ReasoningProfiler:
    """
    Zero-resource feature extractor for reasoning-model detection.

    Produces a fixed-dimension numpy vector φ(x) ∈ ℝ¹⁵.
    Contains NO classification or thresholding logic.

    Usage::

        profiler = ReasoningProfiler()
        vec = profiler.vectorize("Some text to analyse...")
        assert vec.shape == (15,)
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
        Extract a feature vector of shape ``(REASONING_VECTOR_DIM,)`` from raw text.
        Returns a zero vector for empty / near-empty inputs.
        """
        vec = np.empty(REASONING_VECTOR_DIM, dtype=np.float64)

        if not text or len(text.split()) < self._min_tokens:
            vec[:] = 0.0
            return vec

        pt = ReasoningParsedText.from_raw(text)  # renamed class

        for extractor_cls in self._extractors:
            extractor_cls.extract(pt, vec)

        return vec

    def vectorize_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Vectorize a batch. Returns ndarray of shape (len(texts), REASONING_VECTOR_DIM)."""
        n = len(texts)
        out = np.empty((n, REASONING_VECTOR_DIM), dtype=np.float64)
        for i in range(n):
            out[i] = self.vectorize(texts[i])
        return out

    @staticmethod
    def schema() -> tuple[tuple[str, int], ...]:
        return _VECTOR_SCHEMA

    @staticmethod
    def feature_names() -> tuple[str, ...]:
        return FEATURE_NAMES

    @staticmethod
    def dim() -> int:
        return REASONING_VECTOR_DIM

# Backward-compat alias — soft deprecation
# New code should use ReasoningParsedText directly.
ParsedText = ReasoningParsedText
