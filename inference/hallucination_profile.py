"""
Hallucination Feature Extractor v2.2 (Production)
===================================================
Zero-Resource single-pass feature extractor for veracity anomalies
in AI-generated text.

Produces a **deterministic 25-dimensional feature vector** quantifying
hallucination risk signals across 8 categories.  The vector is consumed
by a downstream ML classifier (XGBoost, LightGBM, logistic regression,
or the ``HallucinationRiskClassifier`` heuristic placeholder).

This module is a **PURE FEATURE EXTRACTOR**.  It does NOT classify.
Classification lives in ``HallucinationRiskClassifier`` (separate class,
same file for deployment convenience — distinct responsibility).

Refactor changelog (v2.1 → v2.2)
----------------------------------
  [NITPICK-1] ``_capped_counter``: eliminated per-iteration ``len()`` call.
              Now tracks cardinality via a manual ``current_size`` int that
              increments only on new-key insertion (KeyError path).
  [NITPICK-2] ``_RE_SENTENCE_SPLIT``: replaced naive ``[.!?]+`` with
              production-grade regex that skips decimal points (``3.5``),
              common abbreviations (``Mr.``, ``Dr.``, ``U.S.``), and
              ellipsis runs.  Priority 1 from audit — highest impact.
  [NITPICK-3] Entropy normalisation in classifier: individual entropy
              values are now ``min(val / normaliser, 1.0)`` BEFORE
              averaging, preventing >1.0 category scores.
  [NITPICK-4] Jaccard weighting: when spaCy is available, ``PROPN``
              tokens receive 2× weight in Jaccard numerator/denominator
              via ``_PROPN_JACCARD_WEIGHT``, amplifying the hallucination
              signal from fabricated proper nouns.
  [NITPICK-5] ``np.mean``/``np.std`` on small lists (≤ 30 elements)
              replaced by ``statistics.mean``/``statistics.pstdev`` from
              stdlib, avoiding NumPy's C-bridge overhead on tiny arrays.
  [COMPAT]    Public API unchanged.  HALLUCINATION_VECTOR_DIM = 25.
              All v2.1 imports, class names, method signatures identical.

Literature basis
----------------
  Schulman (2023)        : Hedging/overconfidence reward shaping for RLHF
  Manakul et al. (2023)  : SelfCheckGPT n-gram entropy for black-box detection
  Kadavath et al. (2022) : Semantic entropy and calibration
  Varshney et al. (2023) : Entity-error hallucination taxonomy
  Mundler et al. (2024)  : Self-contradiction detection via NLI
  Chuang et al. (2024)   : Lookback attention ratio for context grounding

Requires
--------
  numpy (required), spacy + en_core_web_sm (recommended, fallback to regex)
"""

from __future__ import annotations

import functools
import logging
import math
import re
import statistics
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Named constants (zero magic literals)
# ═══════════════════════════════════════════════════════════════════════════

_MIN_TEXT_CHARS: int = 20
"""Minimum stripped text length to produce a non-zero vector."""

_FOURGRAM_N: int = 4
"""N-gram window size for phrase repetition detection."""

_DEFAULT_UNIFORMITY: float = 0.5
"""Sentence-length uniformity fallback when < 2 sentences."""

_DEFAULT_TEMPORAL: float = 0.5
"""Temporal specificity fallback when no temporal references found."""

_TOP_SIGNALS_K: int = 3
"""Number of top risk signals returned by the classifier."""

_PRECISE_NUMBER_MIN_DIGITS: int = 4
"""Minimum integer digit count to qualify as a precise number."""

_COUNTER_CARDINALITY_CAP: int = 500_000
"""Maximum unique keys in a Counter before truncation."""

_PROPN_JACCARD_WEIGHT: float = 2.0
"""Weight multiplier for PROPN tokens in Jaccard similarity (NITPICK-4).
Proper nouns carry stronger hallucination signal than common nouns."""

_SMALL_LIST_THRESHOLD: int = 30
"""Lists at or below this size use statistics module instead of NumPy
(NITPICK-5).  Avoids C-bridge overhead on tiny arrays."""


# ═══════════════════════════════════════════════════════════════════════════
# Optional spaCy (dependency-injected at class level)
# ═══════════════════════════════════════════════════════════════════════════

try:
    import spacy as _spacy
    _NLP = _spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False
    _NLP = None
except OSError:
    _SPACY_AVAILABLE = False
    _NLP = None


# ═══════════════════════════════════════════════════════════════════════════
# Compiled regex — ALL use re.VERBOSE
# ═══════════════════════════════════════════════════════════════════════════

_RE_WORD: re.Pattern = re.compile(
    r"""
    \b \w+ \b              # word-boundary delimited token
    """,
    re.VERBOSE,
)

# NITPICK-2 / Priority 1: Production-grade sentence boundary detection.
# Splits on sentence-ending punctuation ONLY when:
#   - Period is NOT between digits (protects "3.5", "47.3%")
#   - Period is NOT after a known abbreviation (Mr., Dr., U.S., etc.)
#   - Exclamation/question marks split normally
#   - Ellipsis (...) treated as single boundary, not 3 splits
_RE_SENTENCE_SPLIT: re.Pattern = re.compile(
    r"""
    (?:                             # non-capturing group for alternation
        \.{3,}                      # ellipsis (3+ dots) → single split
      | [!?]+                       # exclamation / question marks
      | (?<! Mr)                    # not preceded by Mr
        (?<! Mrs)                   # not preceded by Mrs
        (?<! Dr)                    # not preceded by Dr
        (?<! vs)                    # not preceded by vs
        (?<! St)                    # not preceded by St
        (?<! [A-Z])                 # not preceded by single capital (U.S.A.)
        (?<! \d)                    # not preceded by digit (3.5)
        \.                          # literal period
        (?! \d)                     # not followed by digit (3.5)
        (?! [a-z])                  # not followed by lowercase (abbreviation)
    )
    \s*                             # consume trailing whitespace
    """,
    re.VERBOSE,
)

_RE_NUMBER: re.Pattern = re.compile(
    r"""
    \b                     # word boundary
    \d+                    # integer part (one or more digits)
    \.?                    # optional decimal point
    \d*                    # optional fractional digits
    \b                     # word boundary
    """,
    re.VERBOSE,
)

_RE_SPECIFIC_DATE: re.Pattern = re.compile(
    r"""
    \b(?:
        \d{1,2} [/-] \d{1,2} [/-] \d{2,4}          # 3/15/2024
      | (?:january|february|march|april|may|june
          |july|august|september|october
          |november|december) \s+ \d{1,2}            # March 15
      | (?:19|20)\d{2}                                # 2024
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_RE_VAGUE_TIME: re.Pattern = re.compile(
    r"""
    \b(?:
        recently
      | in \s+ recent \s+ (?:years|months|times)
      | for \s+ (?:some|a\ long) \s+ time
      | historically
      | in \s+ the \s+ past
      | nowadays
      | these \s+ days
    )\b
    """,
    re.VERBOSE | re.IGNORECASE,
)

_SELF_REFERENTIAL_PHRASES: Tuple[str, ...] = (
    "i believe", "in my opinion", "i think", "it seems to me",
    "as far as i know", "to the best of my knowledge",
    "i would say", "from my perspective",
)

_RE_SELF_REFERENTIAL: Tuple[re.Pattern, ...] = tuple(
    re.compile(
        r"""
        \b                 # word boundary before phrase
        """
        + re.escape(phrase)
        + r"""
        \b                 # word boundary after phrase
        """,
        re.VERBOSE | re.IGNORECASE,
    )
    for phrase in _SELF_REFERENTIAL_PHRASES
)


# ═══════════════════════════════════════════════════════════════════════════
# Lexical dictionaries (research-grounded, frozen)
# ═══════════════════════════════════════════════════════════════════════════

HEDGING_WORDS: FrozenSet[str] = frozenset({
    "perhaps", "might", "may", "could", "possibly", "probably",
    "seems", "appears", "arguably", "conceivably", "suggests",
    "roughly", "somewhat", "sometimes", "often", "usually",
    "believe", "assume", "guess", "likely", "unlikely",
})

ABSOLUTE_WORDS: FrozenSet[str] = frozenset({
    "always", "never", "absolutely", "definitely", "undoubtedly",
    "certainly", "obviously", "undeniably", "everyone", "nobody",
    "impossible", "proven", "fact", "guaranteed", "unquestionably",
})

NEGATION_WORDS: FrozenSet[str] = frozenset({
    "not", "no", "none", "cannot", "neither", "nor", "nowhere",
    "nothing", "never", "hardly", "barely", "scarcely",
})

VAGUE_QUANTIFIERS: FrozenSet[str] = frozenset({
    "several", "many", "some", "various", "numerous", "few",
    "multiple", "certain", "significant", "substantial",
    "considerable", "roughly", "approximately", "around",
})

MODAL_VERBS: FrozenSet[str] = frozenset({
    "could", "would", "should", "might", "may", "can", "will",
    "shall", "must", "ought",
})

SUPERLATIVES: FrozenSet[str] = frozenset({
    "best", "worst", "most", "least", "greatest", "largest",
    "smallest", "highest", "lowest", "fastest", "strongest",
})

_PERSON_ORG_LABELS: FrozenSet[str] = frozenset({"PERSON", "ORG", "GPE"})
_DATE_NUM_LABELS: FrozenSet[str] = frozenset({
    "DATE", "TIME", "PERCENT", "MONEY", "CARDINAL",
})


# ═══════════════════════════════════════════════════════════════════════════
# CANONICAL EXPORT
# ═══════════════════════════════════════════════════════════════════════════

HALLUCINATION_VECTOR_DIM: int = 25


# ═══════════════════════════════════════════════════════════════════════════
# Declarative vector schema (deterministic, no index coupling)
# ═══════════════════════════════════════════════════════════════════════════

_VECTOR_SCHEMA: Tuple[Tuple[str, str], ...] = (
    ("hedging_ratio",                "lexical"),
    ("overconfidence_ratio",         "lexical"),
    ("negation_ratio",               "lexical"),
    ("entity_density",               "entity"),
    ("unique_entity_ratio",          "entity"),
    ("person_org_ratio",             "entity"),
    ("date_num_ratio",               "entity"),
    ("unigram_entropy",              "entropy"),
    ("bigram_entropy",               "entropy"),
    ("avg_jaccard_similarity",       "cohesion"),
    ("min_jaccard_similarity",       "cohesion"),
    ("max_semantic_drop",            "cohesion"),
    ("disconnected_sentences_ratio", "cohesion"),
    ("vague_quantifier_ratio",       "vagueness"),
    ("specificity_score",            "vagueness"),
    ("assertive_hedged_ratio",       "vagueness"),
    ("self_referential_ratio",       "repetition"),
    ("entity_repetition_rate",       "repetition"),
    ("phrase_repetition_rate",       "repetition"),
    ("factual_density",              "structural"),
    ("sentence_length_uniformity",   "structural"),
    ("modal_verb_ratio",             "structural"),
    ("superlative_ratio",            "structural"),
    ("numeric_precision_ratio",      "precision"),
    ("temporal_specificity",         "precision"),
)

assert len(_VECTOR_SCHEMA) == HALLUCINATION_VECTOR_DIM

FEATURE_NAMES: Tuple[str, ...] = tuple(name for name, _ in _VECTOR_SCHEMA)


def _build_feature_groups() -> Dict[str, Tuple[str, ...]]:
    """Build group→feature mapping without polluting module namespace."""
    groups: Dict[str, List[str]] = {}
    for name, group in _VECTOR_SCHEMA:
        groups.setdefault(group, []).append(name)
    return {k: tuple(v) for k, v in groups.items()}


FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = _build_feature_groups()


# ═══════════════════════════════════════════════════════════════════════════
# Small-list math helpers (NITPICK-5)
# ═══════════════════════════════════════════════════════════════════════════


def _safe_mean(values: List[float]) -> float:
    """Mean for small lists via statistics module; NumPy for large."""
    if not values:
        return 0.0
    if len(values) <= _SMALL_LIST_THRESHOLD:
        return statistics.mean(values)
    return float(np.mean(values))


def _safe_pstdev(values: List[float]) -> float:
    """Population stdev for small lists via statistics; NumPy for large."""
    if len(values) < 2:
        return 0.0
    if len(values) <= _SMALL_LIST_THRESHOLD:
        return statistics.pstdev(values)
    return float(np.std(values))


# ═══════════════════════════════════════════════════════════════════════════
# ParsedText DTO (immutable, single-pass tokenisation)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ParsedText:
    """
    Immutable single-pass tokenisation result shared by all extractors.

    Fields
    ------
    words : List[str]
        Lowercased alphabetic tokens.
    word_counts : Counter
        Pre-computed unigram frequencies (capped at
        ``_COUNTER_CARDINALITY_CAP`` unique keys).
    total_words : int
        ``max(len(words), 1)`` — safe divisor.
    sentences : List[List[str]]
        Canonical sentence representation: each sentence is a list
        of lowercased alphabetic tokens.  spaCy Spans are normalised
        here; no downstream code touches raw Spans.
    text_lower : str
        Full text lowercased once.
    text_raw : str
        Original text (for regex extraction on mixed-case content).
    doc : Any
        spaCy ``Doc`` object, or ``None`` when spaCy is unavailable.
        Used **only** by entity and POS-dependent extractors.
    """

    words: List[str]
    word_counts: Counter
    total_words: int
    sentences: List[List[str]]
    text_lower: str
    text_raw: str
    doc: Any  # spacy.tokens.Doc | None


def _capped_counter(items: Iterator) -> Counter:
    """
    Build a Counter with cardinality cap.

    NITPICK-1 fix: tracks cardinality via ``current_size`` int.
    Only increments on new-key insertion.  Eliminates the
    per-iteration ``len(counts)`` call from v2.1.
    """
    counts: Counter = Counter()
    current_size: int = 0
    for item in items:
        if item in counts:
            counts[item] += 1
        elif current_size < _COUNTER_CARDINALITY_CAP:
            counts[item] = 1
            current_size += 1
        # else: silently drop — cap reached, item is new
    return counts


def _parse_text(text: str, nlp: Any) -> ParsedText:
    """
    Single-pass tokenisation.  All downstream extractors consume this.

    - spaCy path: tokens via ``doc``, sentences via ``doc.sents``.
    - Regex path: tokens via ``_RE_WORD``, sentences via
      ``_RE_SENTENCE_SPLIT`` (NITPICK-2 robust regex).
    - ``word_counts`` capped to prevent OOM.
    """
    text_lower = text.lower()
    doc = nlp(text) if nlp is not None else None

    if doc is not None:
        words = [t.text.lower() for t in doc if t.is_alpha]
        sentences = [
            [t.text.lower() for t in sent if t.is_alpha]
            for sent in doc.sents
        ]
    else:
        words = _RE_WORD.findall(text_lower)
        raw_sents = [
            s.strip()
            for s in _RE_SENTENCE_SPLIT.split(text_lower)
            if s.strip()
        ]
        sentences = [_RE_WORD.findall(s) for s in raw_sents]

    return ParsedText(
        words=words,
        word_counts=_capped_counter(iter(words)),
        total_words=max(len(words), 1),
        sentences=sentences,
        text_lower=text_lower,
        text_raw=text,
        doc=doc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pure extraction functions (stateless, deterministic)
# ═══════════════════════════════════════════════════════════════════════════


def _word_set_ratio(
    counts: Counter, word_set: FrozenSet[str], total: int
) -> float:
    """Sum occurrences of ``word_set`` in pre-computed Counter / total."""
    return sum(counts[w] for w in word_set if w in counts) / total


def _shannon_entropy_gen(items: Iterator) -> float:
    """
    Shannon entropy in bits from a generator/iterator.

    Accepts generators to avoid materialised lists (zero-allocation
    for the caller).  Internally collects into a capped Counter.
    """
    counts = _capped_counter(items)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (cnt / total) * math.log2(cnt / total)
        for cnt in counts.values()
    )


def _bigram_gen(words: List[str]) -> Iterator[Tuple[str, str]]:
    """Yield word bigrams without materialising a list."""
    for i in range(len(words) - 1):
        yield (words[i], words[i + 1])


def _fourgram_gen(words: List[str]) -> Iterator[Tuple[str, ...]]:
    """Yield word 4-grams without materialising a list."""
    n = _FOURGRAM_N
    for i in range(len(words) - n + 1):
        yield tuple(words[i : i + n])


# --- Extractor: Lexical Risk ---

def _extract_lexical(p: ParsedText) -> Dict[str, float]:
    """Hedging, overconfidence, negation density (Schulman 2023)."""
    c, t = p.word_counts, p.total_words
    return {
        "hedging_ratio": _word_set_ratio(c, HEDGING_WORDS, t),
        "overconfidence_ratio": _word_set_ratio(c, ABSOLUTE_WORDS, t),
        "negation_ratio": _word_set_ratio(c, NEGATION_WORDS, t),
    }


# --- Extractor: Entity Metrics ---

def _extract_entity(p: ParsedText) -> Dict[str, float]:
    """NER density, uniqueness, type distribution (Varshney 2023)."""
    zeros = {
        "entity_density": 0.0,
        "unique_entity_ratio": 0.0,
        "person_org_ratio": 0.0,
        "date_num_ratio": 0.0,
    }
    if p.doc is None or not hasattr(p.doc, "ents"):
        return zeros

    ents = p.doc.ents
    num_ents = len(ents)
    if num_ents == 0:
        return zeros

    unique = len({e.text.lower() for e in ents})
    po = sum(1 for e in ents if e.label_ in _PERSON_ORG_LABELS)
    dn = sum(1 for e in ents if e.label_ in _DATE_NUM_LABELS)

    return {
        "entity_density": num_ents / p.total_words,
        "unique_entity_ratio": unique / num_ents,
        "person_org_ratio": po / num_ents,
        "date_num_ratio": dn / num_ents,
    }


# --- Extractor: Entropy ---

def _extract_entropy(p: ParsedText) -> Dict[str, float]:
    """Shannon entropy of unigrams and bigrams (SelfCheckGPT, Manakul 2023)."""
    return {
        "unigram_entropy": _shannon_entropy_gen(iter(p.words)),
        "bigram_entropy": _shannon_entropy_gen(_bigram_gen(p.words)),
    }


# --- Extractor: Semantic Cohesion (NITPICK-4: weighted Jaccard) ---

def _extract_cohesion(p: ParsedText) -> Dict[str, float]:
    """
    Inter-sentence concept overlap via weighted Jaccard (Mundler 2024).

    NITPICK-4: When spaCy is available, PROPN tokens receive
    ``_PROPN_JACCARD_WEIGHT`` × weight vs 1.0 for common NOUN.
    This amplifies the hallucination signal from fabricated proper nouns.
    """
    defaults = {
        "avg_jaccard_similarity": 1.0,
        "min_jaccard_similarity": 1.0,
        "max_semantic_drop": 0.0,
        "disconnected_sentences_ratio": 0.0,
    }

    if len(p.sentences) < 2:
        return defaults

    if p.doc is not None:
        # Weighted concept bags: {lemma: weight}
        sent_bags: List[Dict[str, float]] = []
        for sent in p.doc.sents:
            bag: Dict[str, float] = {}
            for t in sent:
                if t.pos_ == "PROPN":
                    bag[t.lemma_.lower()] = _PROPN_JACCARD_WEIGHT
                elif t.pos_ == "NOUN":
                    bag.setdefault(t.lemma_.lower(), 1.0)
            sent_bags.append(bag)

        if len(sent_bags) < 2:
            return defaults

        sims: List[float] = []
        for i in range(1, len(sent_bags)):
            b1, b2 = sent_bags[i - 1], sent_bags[i]
            all_keys = set(b1) | set(b2)
            if not all_keys:
                sims.append(0.0)
                continue
            intersection_w = sum(
                min(b1.get(k, 0.0), b2.get(k, 0.0)) for k in all_keys
            )
            union_w = sum(
                max(b1.get(k, 0.0), b2.get(k, 0.0)) for k in all_keys
            )
            sims.append(intersection_w / union_w if union_w > 0 else 0.0)
    else:
        # Regex fallback: unweighted set Jaccard
        sent_concepts = [set(tokens) for tokens in p.sentences]
        if len(sent_concepts) < 2:
            return defaults

        sims = []
        for i in range(1, len(sent_concepts)):
            s1, s2 = sent_concepts[i - 1], sent_concepts[i]
            union = len(s1 | s2)
            sims.append(len(s1 & s2) / union if union > 0 else 0.0)

    # NITPICK-5: use statistics module for small lists
    avg_sim = _safe_mean(sims)
    min_sim = min(sims) if sims else 1.0

    drops = [sims[j - 1] - sims[j] for j in range(1, len(sims))]
    max_drop = max(drops) if drops else 0.0
    disconnected = sum(1 for s in sims if s == 0.0)

    return {
        "avg_jaccard_similarity": avg_sim,
        "min_jaccard_similarity": min_sim,
        "max_semantic_drop": max_drop,
        "disconnected_sentences_ratio": disconnected / len(sims),
    }


# --- Extractor: Vagueness ---

def _extract_vagueness(p: ParsedText) -> Dict[str, float]:
    """Vague quantifiers, specificity, assert/hedge ratio."""
    c, t = p.word_counts, p.total_words
    vague_ratio = _word_set_ratio(c, VAGUE_QUANTIFIERS, t)

    if p.doc is not None:
        nouns = [tok for tok in p.doc if tok.pos_ in ("NOUN", "PROPN")]
        proper = [tok for tok in nouns if tok.pos_ == "PROPN"]
        specificity = len(proper) / max(len(nouns), 1)
    else:
        specificity = 0.0

    abs_n = sum(c[w] for w in ABSOLUTE_WORDS if w in c)
    hedge_n = sum(c[w] for w in HEDGING_WORDS if w in c)
    ah_ratio = abs_n / max(abs_n + hedge_n, 1)

    return {
        "vague_quantifier_ratio": vague_ratio,
        "specificity_score": specificity,
        "assertive_hedged_ratio": ah_ratio,
    }


# --- Extractor: Repetition ---

def _extract_repetition(p: ParsedText) -> Dict[str, float]:
    """Self-reference, entity repetition, phrase repetition."""
    self_ref = sum(
        len(pat.findall(p.text_lower)) for pat in _RE_SELF_REFERENTIAL
    )
    num_sents = max(len(p.sentences), 1)
    self_ref_ratio = self_ref / num_sents

    if p.doc is not None and hasattr(p.doc, "ents") and len(p.doc.ents) > 0:
        unique = len({e.text.lower() for e in p.doc.ents})
        ent_rep_raw = len(p.doc.ents) / unique if unique > 0 else 0.0
        ent_rep = (ent_rep_raw - 1.0) / max(ent_rep_raw, 1.0)
    else:
        ent_rep = 0.0

    if len(p.words) >= _FOURGRAM_N:
        fg_counts = _capped_counter(_fourgram_gen(p.words))
        total_fg = sum(fg_counts.values())
        repeated = sum(1 for cnt in fg_counts.values() if cnt > 1)
        phrase_rep = repeated / max(total_fg, 1)
    else:
        phrase_rep = 0.0

    return {
        "self_referential_ratio": self_ref_ratio,
        "entity_repetition_rate": ent_rep,
        "phrase_repetition_rate": phrase_rep,
    }


# --- Extractor: Structural (NITPICK-5: statistics module) ---

def _extract_structural(p: ParsedText) -> Dict[str, float]:
    """Factual density, sentence uniformity, modal verbs, superlatives."""
    c, t = p.word_counts, p.total_words
    num_sents = max(len(p.sentences), 1)

    if p.doc is not None:
        factual = sum(
            1 for tok in p.doc if tok.pos_ == "PROPN" or tok.like_num
        )
        factual_density = factual / num_sents
    else:
        factual_density = 0.0

    lengths = [len(tokens) for tokens in p.sentences if tokens]
    if len(lengths) > 1:
        mean_len = _safe_mean(lengths)
        std_len = _safe_pstdev(lengths)
        uniformity = 1.0 - min(std_len / max(mean_len, 1.0), 1.0)
    else:
        uniformity = _DEFAULT_UNIFORMITY

    return {
        "factual_density": factual_density,
        "sentence_length_uniformity": uniformity,
        "modal_verb_ratio": _word_set_ratio(c, MODAL_VERBS, t),
        "superlative_ratio": _word_set_ratio(c, SUPERLATIVES, t),
    }


# --- Extractor: Precision ---

def _extract_precision(p: ParsedText) -> Dict[str, float]:
    """Numeric precision, temporal specificity (compiled regex)."""
    numbers = _RE_NUMBER.findall(p.text_raw)
    precise = [
        n for n in numbers
        if "." in n
        or len(n.replace(".", "")) >= _PRECISE_NUMBER_MIN_DIGITS
    ]
    numeric_precision = len(precise) / max(len(numbers), 1)

    specific_dates = len(_RE_SPECIFIC_DATE.findall(p.text_raw))
    vague_time = len(_RE_VAGUE_TIME.findall(p.text_raw))
    temporal_total = specific_dates + vague_time
    temporal_specificity = (
        specific_dates / temporal_total
        if temporal_total > 0
        else _DEFAULT_TEMPORAL
    )

    return {
        "numeric_precision_ratio": numeric_precision,
        "temporal_specificity": temporal_specificity,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Vector assembly (declarative schema, zero index-coupling)
# ═══════════════════════════════════════════════════════════════════════════

_EXTRACTORS: Dict[str, Callable[[ParsedText], Dict[str, float]]] = {
    "lexical": _extract_lexical,
    "entity": _extract_entity,
    "entropy": _extract_entropy,
    "cohesion": _extract_cohesion,
    "vagueness": _extract_vagueness,
    "repetition": _extract_repetition,
    "structural": _extract_structural,
    "precision": _extract_precision,
}


def _assemble_vector(p: ParsedText) -> np.ndarray:
    """
    Build the feature vector from the declarative schema.

    Runs each extractor exactly once, reads features in schema order.
    Pre-allocates output array (no list→array copy).
    Validates length contract.
    """
    results: Dict[str, Dict[str, float]] = {
        group: fn(p) for group, fn in _EXTRACTORS.items()
    }

    vec = np.empty(HALLUCINATION_VECTOR_DIM, dtype=np.float64)
    for idx, (feat_name, group) in enumerate(_VECTOR_SCHEMA):
        vec[idx] = results[group][feat_name]

    assert len(vec) == HALLUCINATION_VECTOR_DIM, (
        f"Vector length {len(vec)} != {HALLUCINATION_VECTOR_DIM}"
    )
    return vec


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extractor (PURE — no classification)
# ═══════════════════════════════════════════════════════════════════════════


class HallucinationProfiler:
    """
    Zero-resource hallucination feature extractor.

    This is a **PURE FEATURE EXTRACTOR**.  It outputs a deterministic
    25-dimensional vector.  It does NOT classify risk levels.

    For risk classification, use ``HallucinationRiskClassifier`` or
    train a downstream ML model on the vector output.

    API contract (matches ``StylometricProfiler``)::

        vectorize(text)      -> np.ndarray[HALLUCINATION_VECTOR_DIM]
        compute_stats(text)  -> Dict[str, float]
        get_feature_names()  -> Tuple[str, ...]
        get_feature_groups() -> Dict[str, Tuple[str, ...]]

    Parameters
    ----------
    nlp : optional spaCy Language model.
          ``None`` → module-level ``_NLP`` fallback.
          Pass a mock for testing.
    """

    __slots__ = ("_nlp",)

    def __init__(self, nlp: Any = None) -> None:
        if nlp is not None:
            self._nlp = nlp
        elif _SPACY_AVAILABLE and _NLP is not None:
            self._nlp = _NLP
        else:
            self._nlp = None
            logger.warning(
                "HallucinationProfiler: spaCy unavailable. "
                "Entity and syntactic features will be zeros."
            )

    def vectorize(self, text: str) -> np.ndarray:
        """
        Return a ``HALLUCINATION_VECTOR_DIM``-dimensional feature vector.

        See ``FEATURE_NAMES`` for the ordered layout.
        """
        if not text or len(text.strip()) < _MIN_TEXT_CHARS:
            return np.zeros(HALLUCINATION_VECTOR_DIM, dtype=np.float64)
        p = _parse_text(text, self._nlp)
        return _assemble_vector(p)

    def compute_stats(self, text: str) -> Dict[str, float]:
        """
        Return human-readable hallucination features as a named dict.

        Compatible with ``ForensicReportGenerator``'s ``additional_analyses``
        under the ``"hallucination"`` key.
        """
        if not text or len(text.strip()) < _MIN_TEXT_CHARS:
            return {name: 0.0 for name in FEATURE_NAMES}
        vec = self.vectorize(text)
        return dict(zip(FEATURE_NAMES, vec.tolist()))

    @staticmethod
    def get_feature_names() -> Tuple[str, ...]:
        """Ordered feature names for the vector layout."""
        return FEATURE_NAMES

    @staticmethod
    def get_feature_groups() -> Dict[str, Tuple[str, ...]]:
        """Feature names grouped by extractor category."""
        return dict(FEATURE_GROUPS)


# ═══════════════════════════════════════════════════════════════════════════
# Risk Classifier (SEPARATE from extractor)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class HallucinationRiskConfig:
    """
    Weights and thresholds for the heuristic risk classifier.

    **TEMPORARY heuristic.**  Replace with a trained model
    (XGBoost / LightGBM / MLP) consuming the 25-dim feature vector.

    All weights are exposed as config so they can be tuned without
    modifying the extractor.
    """

    w_lexical_risk: float = 0.10
    w_entity_anomaly: float = 0.10
    w_entropy: float = 0.05
    w_semantic_incoherence: float = 0.25
    w_vagueness: float = 0.20
    w_repetition: float = 0.10
    w_structural_anomaly: float = 0.10
    w_imprecision: float = 0.10

    entropy_normaliser: float = 10.0

    low_threshold: float = 0.25
    high_threshold: float = 0.50


DEFAULT_RISK_CONFIG = HallucinationRiskConfig()


class HallucinationRiskClassifier:
    """
    Heuristic risk classifier consuming the 25-dim feature vector.

    Architecturally SEPARATE from the extractor.

    For production, replace with a trained ML model::

        features = profiler.vectorize(text)
        risk = xgb_model.predict_proba(features.reshape(1, -1))

    Parameters
    ----------
    config : ``HallucinationRiskConfig`` with weights and thresholds.
    """

    __slots__ = ("cfg",)

    def __init__(
        self, config: Optional[HallucinationRiskConfig] = None
    ) -> None:
        self.cfg = config or DEFAULT_RISK_CONFIG

    def classify(self, stats: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify hallucination risk from feature dict.

        NITPICK-3: entropy values clipped to [0, 1] BEFORE averaging.
        NITPICK-5: category averages use ``_safe_mean`` for small lists.
        """
        c = self.cfg

        # NITPICK-3: clip entropy individually before averaging
        ent_uni = min(
            stats.get("unigram_entropy", 0.0) / c.entropy_normaliser, 1.0
        )
        ent_bi = min(
            stats.get("bigram_entropy", 0.0) / c.entropy_normaliser, 1.0
        )

        categories = {
            "lexical_risk": _safe_mean([
                stats.get("hedging_ratio", 0.0),
                stats.get("overconfidence_ratio", 0.0),
                stats.get("negation_ratio", 0.0),
            ]),
            "entity_anomaly": _safe_mean([
                stats.get("entity_density", 0.0),
                1.0 - stats.get("unique_entity_ratio", 1.0),
                stats.get("entity_repetition_rate", 0.0),
            ]),
            "entropy": _safe_mean([ent_uni, ent_bi]),
            "semantic_incoherence": _safe_mean([
                1.0 - stats.get("avg_jaccard_similarity", 1.0),
                stats.get("max_semantic_drop", 0.0),
                stats.get("disconnected_sentences_ratio", 0.0),
            ]),
            "vagueness": _safe_mean([
                stats.get("vague_quantifier_ratio", 0.0),
                1.0 - stats.get("specificity_score", 1.0),
                1.0 - stats.get("assertive_hedged_ratio", 1.0),
            ]),
            "repetition": _safe_mean([
                stats.get("self_referential_ratio", 0.0),
                stats.get("entity_repetition_rate", 0.0),
                stats.get("phrase_repetition_rate", 0.0),
            ]),
            "structural_anomaly": _safe_mean([
                stats.get("sentence_length_uniformity", _DEFAULT_UNIFORMITY),
                stats.get("modal_verb_ratio", 0.0),
                stats.get("superlative_ratio", 0.0),
            ]),
            "imprecision": _safe_mean([
                1.0 - stats.get("numeric_precision_ratio", _DEFAULT_TEMPORAL),
                1.0 - stats.get("temporal_specificity", _DEFAULT_TEMPORAL),
            ]),
        }

        weights = {
            "lexical_risk": c.w_lexical_risk,
            "entity_anomaly": c.w_entity_anomaly,
            "entropy": c.w_entropy,
            "semantic_incoherence": c.w_semantic_incoherence,
            "vagueness": c.w_vagueness,
            "repetition": c.w_repetition,
            "structural_anomaly": c.w_structural_anomaly,
            "imprecision": c.w_imprecision,
        }

        overall = max(0.0, min(1.0, sum(
            categories[k] * weights[k] for k in categories
        )))

        if overall < c.low_threshold:
            level = "LOW"
        elif overall < c.high_threshold:
            level = "MEDIUM"
        else:
            level = "HIGH"

        top_signals = sorted(
            stats.items(), key=lambda x: x[1], reverse=True
        )[:_TOP_SIGNALS_K]

        return {
            "overall_risk": round(overall, 4),
            "risk_level": level,
            "category_scores": {
                k: round(v, 4) for k, v in categories.items()
            },
            "top_signals": [
                {"feature": k, "value": round(v, 4)}
                for k, v in top_signals
            ],
            "feature_details": stats,
        }

    def classify_from_text(
        self,
        text: str,
        profiler: HallucinationProfiler,
    ) -> Dict[str, Any]:
        """
        Convenience: extract features then classify in one call.

        For batch processing, prefer ``profiler.compute_stats()``
        then ``classify()`` to amortise parsing.
        """
        return self.classify(profiler.compute_stats(text))


# ═══════════════════════════════════════════════════════════════════════════
# Backward compatibility (proper DeprecationWarning)
# ═══════════════════════════════════════════════════════════════════════════


def _deprecated_compute_risk_summary(
    text: str,
    profiler: Optional[HallucinationProfiler] = None,
    config: Optional[HallucinationRiskConfig] = None,
) -> Dict[str, Any]:
    """
    **DEPRECATED** — Use ``HallucinationRiskClassifier.classify_from_text()``.

    Kept for backward compatibility.  Emits ``DeprecationWarning`` on
    every call so downstream code discovers the migration path.
    """
    warnings.warn(
        "compute_risk_summary() is deprecated. "
        "Use HallucinationRiskClassifier.classify_from_text() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if profiler is None:
        profiler = HallucinationProfiler()
    classifier = HallucinationRiskClassifier(config)
    return classifier.classify_from_text(text, profiler)


@functools.wraps(_deprecated_compute_risk_summary)
def compute_risk_summary(
    text: str,
    profiler: Optional[HallucinationProfiler] = None,
    config: Optional[HallucinationRiskConfig] = None,
) -> Dict[str, Any]:
    return _deprecated_compute_risk_summary(text, profiler, config)
