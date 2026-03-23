"""
Stylometric Profiler Module  v2.1.1 (Refactored)
==================================================
Creates unique writing fingerprints for individual authors.
Detects when AI has "taken over" a human's writing.

Refactor changelog (v2.1 -> v2.1.1)
-------------------------------------
  [FIX]     Exported VECTOR_DIM constant — eliminates magic-number mismatch.
  [FIX]     spaCy doc parsed ONCE per text, passed to POS + syntactic extractors.
  [FIX]     MATTR uses adaptive stride for O(n) on long texts.
  [FIX]     StyleProfile.schema_version field for forward-compatible serialisation.
  [FIX]     to_dict/from_dict use dataclasses.fields() — auto-picks up new fields.
  [FIX]     StylometricProfiler accepts injectable nlp (testability).
  [API]     Public vectorize(text) method replaces private _build_temp_profile access.
  [STYLE]   PEP 8 strict; narrowed exception handlers; English identifiers.

All v2.1 public APIs remain fully compatible.

Requires
--------
  numpy, scipy              (required)
  spacy                     (optional — POS bigrams + syntactic features)
    pip install spacy && python -m spacy download en_core_web_sm

Do NOT call logging.basicConfig() here — let the application configure logging.
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy.spatial.distance import cosine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional spaCy — module-level fallback (overridable via DI)
# ---------------------------------------------------------------------------
try:
    import spacy as _spacy

    _NLP = _spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
    logger.debug("spaCy loaded — POS tagger + dependency parser active.")
except ImportError:
    _SPACY_AVAILABLE = False
    _NLP = None
    logger.debug("spaCy not installed — regex fallbacks active.")
except OSError:
    _SPACY_AVAILABLE = False
    _NLP = None
    logger.debug("spaCy model not found — regex fallbacks active.")


# ---------------------------------------------------------------------------
# Pluggable protocols
# ---------------------------------------------------------------------------


class EmbeddingProvider(Protocol):
    """
    Protocol for any dense-embedding model.

    Implement this interface and pass an instance to build_profile() /
    compare() to fuse semantic embeddings with the statistical fingerprint.

    Example with sentence-transformers
    ------------------------------------
    from sentence_transformers import SentenceTransformer

    class STEmbedder:
        def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
            self._model = SentenceTransformer(model_name)

        def embed(self, text: str) -> np.ndarray:
            return self._model.encode(text, normalize_embeddings=True)

        @property
        def dim(self) -> int:
            return self._model.get_sentence_embedding_dimension()

    profiler.build_profile("alice", texts, embedding_provider=STEmbedder())
    """

    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D float64 unit vector for the given text."""
        ...

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vector."""
        ...


class SiameseScorer(Protocol):
    """
    Protocol for a learned similarity scorer.

    Replace cosine distance with a trained Siamese / one-class model.
    The scorer receives two vectors (profile centroid + candidate) and
    returns a similarity in [0, 1].
    """

    def score(
        self,
        profile_vec: np.ndarray,
        candidate_vec: np.ndarray,
    ) -> float:
        """Return a similarity in [0, 1] (1 = same author)."""
        ...


# ---------------------------------------------------------------------------
# Word lists
# ---------------------------------------------------------------------------

FUNCTION_WORDS: frozenset = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "that", "this", "these", "those", "it", "its", "i", "you",
        "he", "she", "we", "they", "me", "him", "her", "us", "them",
    }
)

FILLER_WORDS: frozenset = frozenset(
    {
        "like", "basically", "literally", "actually", "honestly",
        "seriously", "anyway", "whatever", "stuff", "things", "right",
        "kind of", "sort of", "you know", "i mean",
    }
)

TRANSITION_WORDS: frozenset = frozenset(
    {
        "however", "therefore", "furthermore", "moreover", "additionally",
        "consequently", "nevertheless", "nonetheless", "thus", "hence",
        "although", "despite", "meanwhile", "subsequently",
    }
)

_MULTI_WORD_FILLERS = frozenset(w for w in FILLER_WORDS if " " in w)
_SINGLE_WORD_FILLERS = FILLER_WORDS - _MULTI_WORD_FILLERS


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class StyleProfile:
    """
    Complete stylometric fingerprint for one author.

    Scalar fields store normalised rates or means.
    Dict fields store relative-frequency distributions.
    The feature_vector is z-score normalised (Delta-of-Burrows).
    embedding_vector is populated only when an EmbeddingProvider is used.
    """

    # -- Schema version (NEW v2.1.1) ------------------------------------
    schema_version: str = "2.1.1"

    # -- Metadata -------------------------------------------------------
    author_id: str = ""
    sample_count: int = 0
    total_words: int = 0
    created_at: str = ""

    # -- Lexical --------------------------------------------------------
    vocabulary_richness: float = 0.0
    avg_word_length: float = 0.0
    rare_word_ratio: float = 0.0
    hapax_legomena_ratio: float = 0.0

    # -- Punctuation ----------------------------------------------------
    comma_rate: float = 0.0
    semicolon_rate: float = 0.0
    exclamation_rate: float = 0.0
    question_rate: float = 0.0
    ellipsis_rate: float = 0.0
    dash_rate: float = 0.0

    # -- Sentence structure ---------------------------------------------
    avg_sentence_length: float = 0.0
    sentence_length_variance: float = 0.0
    avg_paragraph_length: float = 0.0

    # -- Syntactic depth (v2.1) -----------------------------------------
    avg_dep_distance: float = 0.0
    max_dep_distance: float = 0.0
    avg_tree_depth: float = 0.0
    complex_sentence_ratio: float = 0.0

    # -- Burstiness (v2.1) ----------------------------------------------
    burstiness_score: float = 0.0

    # -- Word n-grams ---------------------------------------------------
    function_word_frequencies: Dict[str, float] = field(default_factory=dict)
    bigram_frequencies: Dict[str, float] = field(default_factory=dict)
    trigram_frequencies: Dict[str, float] = field(default_factory=dict)
    filler_word_frequencies: Dict[str, float] = field(default_factory=dict)
    transition_frequencies: Dict[str, float] = field(default_factory=dict)

    # -- Char n-grams ---------------------------------------------------
    char_4gram_frequencies: Dict[str, float] = field(default_factory=dict)
    char_5gram_frequencies: Dict[str, float] = field(default_factory=dict)

    # -- POS bigrams ----------------------------------------------------
    pos_bigram_frequencies: Dict[str, float] = field(default_factory=dict)
    pos_bigrams_source: str = "none"

    # -- Signature words ------------------------------------------------
    signature_words: List[str] = field(default_factory=list)

    # -- Statistical feature vector + normalisation ---------------------
    # NOTE: size is set lazily; VECTOR_DIM is the canonical source.
    feature_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    feature_mean: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )
    feature_std: np.ndarray = field(
        default_factory=lambda: np.ones(0)
    )

    # -- Semantic embedding (populated by EmbeddingProvider) -------------
    embedding_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(0)
    )

    # -- Adaptive threshold ---------------------------------------------
    adaptive_threshold: Optional[float] = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    _NUMPY_FIELDS: frozenset = frozenset(
        {"feature_vector", "feature_mean", "feature_std", "embedding_vector"}
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile to a JSON-safe dictionary."""
        result: Dict[str, Any] = {}
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, np.ndarray):
                result[f.name] = val.tolist()
            else:
                result[f.name] = val
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StyleProfile":
        """Deserialize from dict with backward-compat defaults."""
        data = dict(data)

        # Schema version validation
        version = data.get("schema_version", "1.0")
        major = version.split(".")[0] if isinstance(version, str) else "0"
        if major not in ("1", "2"):
            logger.warning("Unknown schema version: %s — loading anyway.", version)

        # Extract numpy fields before construction
        numpy_data: Dict[str, np.ndarray] = {}
        for key in cls._NUMPY_FIELDS:
            if key in data:
                numpy_data[key] = np.array(data.pop(key), dtype=np.float64)

        # Backward-compat defaults for fields added in v2.0 / v2.1
        _COMPAT_DEFAULTS: Dict[str, Any] = {
            "schema_version": "1.0",
            "char_4gram_frequencies": {},
            "char_5gram_frequencies": {},
            "pos_bigram_frequencies": {},
            "pos_bigrams_source": "none",
            "adaptive_threshold": None,
            "avg_dep_distance": 0.0,
            "max_dep_distance": 0.0,
            "avg_tree_depth": 0.0,
            "complex_sentence_ratio": 0.0,
            "burstiness_score": 0.0,
        }
        for key, default in _COMPAT_DEFAULTS.items():
            data.setdefault(key, default)

        # Filter to known field names only
        known = {f.name for f in fields(cls) if not f.name.startswith("_")}
        filtered = {k: v for k, v in data.items() if k in known}

        profile = cls(**filtered)

        # Restore numpy arrays
        for key, arr in numpy_data.items():
            if arr.size > 0:
                setattr(profile, key, arr)

        return profile


@dataclass
class StyleComparisonResult:
    """Result of comparing a text against an author profile."""

    author_id: str
    similarity_score: float
    is_likely_same_author: bool
    confidence: float
    threshold_used: float
    scorer_used: str  # "cosine" | "siamese" | "hybrid"
    feature_distances: Dict[str, float] = field(default_factory=dict)
    anomalous_features: List[str] = field(default_factory=list)


@dataclass
class WindowSignals:
    """Per-window signal bundle produced by detect_transition_point()."""

    word_offset: int
    stylometric_sim: float
    burstiness_dev: float
    syntactic_dist: float
    combined_score: float
    is_flagged: bool


@dataclass
class HybridDetectionResult:
    """Result of sliding-window hybrid-text analysis (v2.1 extended)."""

    transition_word_index: Optional[int]
    window_signals: List[WindowSignals]
    max_stylometric_drop: float
    max_burstiness_dev: float
    dual_signal_confidence: float
    suspected_ai_start_ratio: Optional[float]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _extract_lexical_features(text: str) -> Dict[str, float]:
    """Extract vocabulary richness, word length, and rarity metrics."""
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {
            "vocabulary_richness": 0.0,
            "avg_word_length": 0.0,
            "rare_word_ratio": 0.0,
            "hapax_legomena_ratio": 0.0,
        }

    counts = Counter(words)
    total = len(words)
    window = 100

    if total >= window:
        # Adaptive stride: O(n) for large texts instead of O(n^2)
        if total < 2000:
            stride = 1
        elif total < 10_000:
            stride = 5
        else:
            stride = 10

        ttrs = [
            len(set(words[i : i + window])) / window
            for i in range(0, total - window + 1, stride)
        ]
        mattr = float(np.mean(ttrs))
    else:
        mattr = len(set(words)) / total

    hapax = sum(1 for v in counts.values() if v == 1)
    rare = sum(1 for v in counts.values() if v < 3)
    return {
        "vocabulary_richness": mattr,
        "avg_word_length": float(np.mean([len(w) for w in words])),
        "rare_word_ratio": rare / total,
        "hapax_legomena_ratio": hapax / total,
    }


def _extract_punctuation_features(
    text: str, num_sentences: int
) -> Dict[str, float]:
    """Extract punctuation rate features normalised by sentence count."""
    d = max(num_sentences, 1)
    return {
        "comma_rate": text.count(",") / d,
        "semicolon_rate": text.count(";") / d,
        "exclamation_rate": text.count("!") / d,
        "question_rate": text.count("?") / d,
        "ellipsis_rate": text.count("...") / d,
        "dash_rate": (
            text.count("\u2014") + text.count("\u2013") + text.count(" - ")
        )
        / d,
    }


def _extract_sentence_features(text: str) -> Dict[str, float]:
    """Extract sentence-level length statistics."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sentences:
        return {"avg_sentence_length": 0.0, "sentence_length_variance": 0.0}
    lengths = [len(s.split()) for s in sentences]
    return {
        "avg_sentence_length": float(np.mean(lengths)),
        "sentence_length_variance": (
            float(np.var(lengths)) if len(lengths) > 1 else 0.0
        ),
    }


def _compute_burstiness(text: str) -> float:
    """
    Burstiness score of sentence-length distribution.

    Formula:  B = (sigma - mu) / (sigma + mu)   range: [-1, +1]

    Human writing is characteristically bursty (B > 0).
    LLMs produce more uniform sentence lengths (B near 0 or negative).
    """
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = np.array([len(s.split()) for s in sentences], dtype=float)

    if len(lengths) < 3:
        return 0.0

    mu = float(np.mean(lengths))
    sigma = float(np.std(lengths))

    if mu + sigma < 1e-9:
        return 0.0

    return float((sigma - mu) / (sigma + mu))


# ---------------------------------------------------------------------------
# Dependency tree helpers
# ---------------------------------------------------------------------------


def _token_depth(tok: Any) -> int:
    """Walk up the dependency tree to compute depth. Guard against cycles."""
    d = 0
    cur = tok
    while cur.head != cur:
        cur = cur.head
        d += 1
        if d > 50:
            break
    return d


def _extract_syntactic_features(
    doc: Any = None,
) -> Dict[str, float]:
    """
    Dependency-tree metrics from a pre-parsed spaCy Doc.

    When doc is None (spaCy unavailable), returns zeros.
    """
    zeros: Dict[str, float] = {
        "avg_dep_distance": 0.0,
        "max_dep_distance": 0.0,
        "avg_tree_depth": 0.0,
        "complex_sentence_ratio": 0.0,
    }

    if doc is None:
        return zeros

    per_sent_avg_dist: List[float] = []
    per_sent_max_dist: List[float] = []
    per_sent_avg_depth: List[float] = []
    complex_count = 0
    sent_count = 0

    for sent in doc.sents:
        sent_count += 1
        tokens = list(sent)
        if not tokens:
            continue

        dists = [abs(tok.i - tok.head.i) for tok in tokens if tok.head != tok]
        if dists:
            per_sent_avg_dist.append(float(np.mean(dists)))
            per_sent_max_dist.append(float(max(dists)))

        depths = [_token_depth(tok) for tok in tokens]
        per_sent_avg_depth.append(float(np.mean(depths)))

        dep_labels = {tok.dep_ for tok in tokens}
        if dep_labels & {"advcl", "relcl", "acl", "csubj", "ccomp"}:
            complex_count += 1

    return {
        "avg_dep_distance": (
            float(np.mean(per_sent_avg_dist)) if per_sent_avg_dist else 0.0
        ),
        "max_dep_distance": (
            float(np.mean(per_sent_max_dist)) if per_sent_max_dist else 0.0
        ),
        "avg_tree_depth": (
            float(np.mean(per_sent_avg_depth)) if per_sent_avg_depth else 0.0
        ),
        "complex_sentence_ratio": complex_count / max(sent_count, 1),
    }


# ---------------------------------------------------------------------------
# Word / char n-gram helpers
# ---------------------------------------------------------------------------


def _word_frequencies(text: str, word_set: frozenset) -> Dict[str, float]:
    """Compute relative frequency of single-token words from a given set."""
    words = re.findall(r"\b\w+\b", text.lower())
    total = max(len(words), 1)
    counts = Counter(words)
    return {
        w: counts[w] / total
        for w in word_set
        if w in counts and " " not in w
    }


def _mixed_frequencies(text: str, word_set: frozenset) -> Dict[str, float]:
    """Handle both single-token and multi-word entries (e.g. 'kind of')."""
    lower = text.lower()
    words = re.findall(r"\b\w+\b", lower)
    total = max(len(words), 1)
    counts = Counter(words)
    result: Dict[str, float] = {}
    for entry in word_set:
        if " " in entry:
            n = len(re.findall(re.escape(entry), lower))
            if n:
                result[entry] = n / total
        elif entry in counts:
            result[entry] = counts[entry] / total
    return result


def _build_word_ngrams(words: List[str], n: int) -> Counter:
    """Build a Counter of word n-grams."""
    return Counter(
        " ".join(words[i : i + n]) for i in range(len(words) - n + 1)
    )


def _extract_char_ngrams(
    text: str, n: int, top_k: int = 80
) -> Dict[str, float]:
    """Extract character n-gram relative frequencies."""
    if len(text) < n:
        return {}
    ngrams = Counter(text[i : i + n] for i in range(len(text) - n + 1))
    total = max(sum(ngrams.values()), 1)
    return {ng: cnt / total for ng, cnt in ngrams.most_common(top_k)}


# ---------------------------------------------------------------------------
# POS bigrams (spaCy or regex fallback)
# ---------------------------------------------------------------------------

_POS_RULES: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^(am|is|are|was|were|be|been|being)$"), "AUX"),
    (re.compile(r"^(will|would|could|should|may|might|shall|can|must)$"), "AUX"),
    (
        re.compile(
            r"^(the|a|an|this|that|these|those|my|your|his|her|its|our|their)$"
        ),
        "DET",
    ),
    (re.compile(r"^(and|but|or|nor|for|yet|so)$"), "CCONJ"),
    (
        re.compile(r"^(although|because|since|while|unless|if|when|as|than)$"),
        "SCONJ",
    ),
    (
        re.compile(
            r"^(in|on|at|to|for|of|with|by|from|about|into|through"
            r"|during|before|after)$"
        ),
        "ADP",
    ),
    (
        re.compile(
            r"^(i|you|he|she|it|we|they|me|him|her|us|them"
            r"|myself|yourself|himself|herself|itself)$"
        ),
        "PRON",
    ),
    (re.compile(r"\w+ly$"), "ADV"),
    (re.compile(r"\w+(ing|ed)$"), "VERB"),
    (re.compile(r"\w+(tion|ness|ment|ity|ism|age|ance|ence|ship)$"), "NOUN"),
    (re.compile(r"\w+(ful|less|ous|ive|able|ible|al|ic)$"), "ADJ"),
    (re.compile(r"^\d+$"), "NUM"),
    (re.compile(r"^[^\w\s]+$"), "PUNCT"),
]


def _pos_tag_fallback(text: str) -> List[str]:
    """Regex-based POS tagging when spaCy is unavailable."""
    tokens = re.findall(r"\b\w+\b|[^\w\s]", text.lower())
    tags: List[str] = []
    for tok in tokens:
        matched = False
        for pattern, tag in _POS_RULES:
            if pattern.fullmatch(tok):
                tags.append(tag)
                matched = True
                break
        if not matched:
            tags.append("NOUN")
    return tags


def _extract_pos_bigrams(
    doc_or_text: Any = None,
    raw_text: Optional[str] = None,
    top_k: int = 40,
) -> Tuple[Dict[str, float], str]:
    """
    Extract POS-tag bigram frequencies.

    Accepts either a pre-parsed spaCy Doc (preferred — avoids double parsing)
    or falls back to regex-based POS tagging on raw_text.
    """
    if doc_or_text is not None and hasattr(doc_or_text, "sents"):
        # spaCy Doc
        tags = [tok.pos_ for tok in doc_or_text if not tok.is_space]
        source = "spacy"
    elif raw_text is not None:
        tags = _pos_tag_fallback(raw_text)
        source = "regex_fallback"
    else:
        return {}, "none"

    if len(tags) < 2:
        return {}, source

    total = max(len(tags) - 1, 1)
    bigrams = Counter(
        f"{tags[i]} {tags[i + 1]}" for i in range(len(tags) - 1)
    )
    return {bg: cnt / total for bg, cnt in bigrams.most_common(top_k)}, source


# ---------------------------------------------------------------------------
# Feature vector layout
# ---------------------------------------------------------------------------

_SCALAR_FEATURES: Tuple[str, ...] = (
    # Lexical (4)
    "vocabulary_richness",
    "avg_word_length",
    "rare_word_ratio",
    "hapax_legomena_ratio",
    # Punctuation (6)
    "comma_rate",
    "semicolon_rate",
    "exclamation_rate",
    "question_rate",
    "ellipsis_rate",
    "dash_rate",
    # Sentence structure (3)
    "avg_sentence_length",
    "sentence_length_variance",
    "avg_paragraph_length",
    # Syntactic — v2.1 (4)
    "avg_dep_distance",
    "max_dep_distance",
    "avg_tree_depth",
    "complex_sentence_ratio",
    # Burstiness — v2.1 (1)
    "burstiness_score",
)

_TOP_FUNCTION_WORDS: Tuple[str, ...] = tuple(sorted(FUNCTION_WORDS))
_TOP_FILLER_WORDS: Tuple[str, ...] = tuple(sorted(FILLER_WORDS))
_TOP_TRANSITIONS: Tuple[str, ...] = tuple(sorted(TRANSITION_WORDS))

_N_BIGRAMS = 30
_N_TRIGRAMS = 20
_N_CHAR4 = 30
_N_CHAR5 = 20
_N_POS_BG = 15

_STAT_VECTOR_SIZE: int = (
    len(_SCALAR_FEATURES)
    + len(_TOP_FUNCTION_WORDS)
    + len(_TOP_FILLER_WORDS)
    + len(_TOP_TRANSITIONS)
    + _N_BIGRAMS
    + _N_TRIGRAMS
    + _N_CHAR4
    + _N_CHAR5
    + _N_POS_BG
)

# ===== CANONICAL EXPORT — use this everywhere, never a magic number =====
VECTOR_DIM: int = _STAT_VECTOR_SIZE


# ---------------------------------------------------------------------------
# Vector construction
# ---------------------------------------------------------------------------

# Declarative schema: (source_attr, ordered_keys | None, count)
_VECTOR_SCHEMA: List[Tuple[str, Optional[Tuple[str, ...]], int]] = [
    # fixed-key segments
    ("function_word_frequencies", _TOP_FUNCTION_WORDS, len(_TOP_FUNCTION_WORDS)),
    ("filler_word_frequencies", _TOP_FILLER_WORDS, len(_TOP_FILLER_WORDS)),
    ("transition_frequencies", _TOP_TRANSITIONS, len(_TOP_TRANSITIONS)),
    # variable-key segments (top-N by insertion order)
    ("bigram_frequencies", None, _N_BIGRAMS),
    ("trigram_frequencies", None, _N_TRIGRAMS),
    ("char_4gram_frequencies", None, _N_CHAR4),
    ("char_5gram_frequencies", None, _N_CHAR5),
    ("pos_bigram_frequencies", None, _N_POS_BG),
]


def _build_raw_stat_vector(profile: StyleProfile) -> np.ndarray:
    """
    Build the un-normalised statistical feature vector.

    Returns a numpy array of exactly VECTOR_DIM elements.
    """
    v: List[float] = []

    # Scalars
    for feat in _SCALAR_FEATURES:
        v.append(float(getattr(profile, feat, 0.0)))

    # Dict segments
    for attr_name, keys, count in _VECTOR_SCHEMA:
        freq_dict: Dict[str, float] = getattr(profile, attr_name, {})
        if keys is not None:
            # Fixed-key: look up each canonical key
            for k in keys:
                v.append(freq_dict.get(k, 0.0))
        else:
            # Variable-key: take first `count` values, pad remainder
            vals = list(freq_dict.values())[:count]
            v.extend(vals)
            v.extend([0.0] * (count - len(vals)))

    arr = np.array(v[:VECTOR_DIM], dtype=np.float64)
    if arr.size < VECTOR_DIM:
        arr = np.pad(arr, (0, VECTOR_DIM - arr.size))

    assert len(arr) == VECTOR_DIM, (
        f"Vector length {len(arr)} != VECTOR_DIM {VECTOR_DIM}"
    )
    return arr


def _normalise_vector(
    raw: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Delta-of-Burrows z-score normalisation."""
    safe_std = np.where(std < 1e-9, 1.0, std)
    return (raw - mean) / safe_std


def _fuse_vectors(
    stat_vec: np.ndarray,
    emb_vec: np.ndarray,
    stat_weight: float = 0.6,
) -> np.ndarray:
    """
    Concatenate statistical + embedding vectors with L2 normalisation.

    Both sub-vectors are independently L2-normalised before concat so that
    neither dimension count dominates due to size.
    """
    if emb_vec.size == 0:
        return stat_vec
    norm_s = stat_vec / (np.linalg.norm(stat_vec) + 1e-9)
    norm_e = emb_vec / (np.linalg.norm(emb_vec) + 1e-9)
    return np.concatenate(
        [norm_s * stat_weight, norm_e * (1.0 - stat_weight)]
    )


# ---------------------------------------------------------------------------
# Adaptive threshold calibration
# ---------------------------------------------------------------------------


def _calibrate_threshold(
    raw_vectors: List[np.ndarray],
) -> Tuple[Optional[float], np.ndarray, np.ndarray]:
    """Compute adaptive similarity threshold from leave-one-out CV."""
    if len(raw_vectors) < 3:
        stacked = (
            np.stack(raw_vectors)
            if raw_vectors
            else np.zeros((1, VECTOR_DIM))
        )
        return None, stacked.mean(axis=0), stacked.std(axis=0) + 1e-9

    stacked = np.stack(raw_vectors)
    global_mean = stacked.mean(axis=0)
    global_std = stacked.std(axis=0) + 1e-9
    scores: List[float] = []

    for i in range(len(raw_vectors)):
        others = [raw_vectors[j] for j in range(len(raw_vectors)) if j != i]
        ref_norm = _normalise_vector(
            np.mean(np.stack(others), axis=0), global_mean, global_std
        )
        tst_norm = _normalise_vector(raw_vectors[i], global_mean, global_std)
        nr, nt = np.linalg.norm(ref_norm), np.linalg.norm(tst_norm)
        if nr > 0 and nt > 0:
            scores.append(
                max(0.0, float(1.0 - cosine(ref_norm, tst_norm)))
            )

    if not scores:
        return None, global_mean, global_std

    thr = float(np.clip(np.mean(scores) - np.std(scores), 0.0, 0.99))
    logger.debug(
        "Adaptive threshold=%.3f (mu=%.3f sigma=%.3f n=%d)",
        thr,
        np.mean(scores),
        np.std(scores),
        len(scores),
    )
    return thr, global_mean, global_std


# ---------------------------------------------------------------------------
# Core profile builder (pure function)
# ---------------------------------------------------------------------------


def _compute_profile_data(
    author_id: str,
    texts: List[str],
    embedding_provider: Optional[EmbeddingProvider] = None,
    nlp: Any = None,
) -> StyleProfile:
    """
    Pure computation of a StyleProfile.

    Does NOT register anywhere — safe to call from compare() / sliding window.
    The `nlp` parameter receives a spaCy Language instance; when provided,
    the text is parsed ONCE and the Doc is shared by both POS-bigram and
    syntactic-feature extractors.
    """
    combined = " ".join(texts)
    sentences = [s.strip() for s in re.split(r"[.!?]+", combined) if s.strip()]
    paragraphs = [p.strip() for p in combined.split("\n\n") if p.strip()]
    words = re.findall(r"\b\w+\b", combined.lower())
    total_words = max(len(words), 1)

    # Parse spaCy doc ONCE
    doc = nlp(combined) if nlp is not None else None

    lex = _extract_lexical_features(combined)
    punct = _extract_punctuation_features(combined, len(sentences))
    sent = _extract_sentence_features(combined)
    syn = _extract_syntactic_features(doc)
    burst = _compute_burstiness(combined)
    pos_bg, pos_src = _extract_pos_bigrams(
        doc_or_text=doc, raw_text=combined
    )

    avg_paragraph_length = (
        float(np.mean([len(p.split()) for p in paragraphs]))
        if paragraphs
        else 0.0
    )

    fw_freq = _word_frequencies(combined, FUNCTION_WORDS)
    filler_freq = _mixed_frequencies(combined, FILLER_WORDS)
    trans_freq = _word_frequencies(combined, TRANSITION_WORDS)

    bigrams = _build_word_ngrams(words, 2)
    trigrams = _build_word_ngrams(words, 3)
    char4 = _extract_char_ngrams(combined, n=4, top_k=80)
    char5 = _extract_char_ngrams(combined, n=5, top_k=60)

    # Optional embedding
    emb_vec = np.zeros(0)
    if embedding_provider is not None:
        try:
            emb_vec = np.array(
                embedding_provider.embed(combined), dtype=np.float64
            )
        except Exception as exc:
            logger.warning("EmbeddingProvider.embed() failed: %s", exc)

    profile = StyleProfile(
        author_id=author_id,
        sample_count=len(texts),
        total_words=len(words),
        created_at=datetime.now().isoformat(),
        vocabulary_richness=lex["vocabulary_richness"],
        avg_word_length=lex["avg_word_length"],
        rare_word_ratio=lex["rare_word_ratio"],
        hapax_legomena_ratio=lex["hapax_legomena_ratio"],
        comma_rate=punct["comma_rate"],
        semicolon_rate=punct["semicolon_rate"],
        exclamation_rate=punct["exclamation_rate"],
        question_rate=punct["question_rate"],
        ellipsis_rate=punct["ellipsis_rate"],
        dash_rate=punct["dash_rate"],
        avg_sentence_length=sent["avg_sentence_length"],
        sentence_length_variance=sent["sentence_length_variance"],
        avg_paragraph_length=avg_paragraph_length,
        avg_dep_distance=syn["avg_dep_distance"],
        max_dep_distance=syn["max_dep_distance"],
        avg_tree_depth=syn["avg_tree_depth"],
        complex_sentence_ratio=syn["complex_sentence_ratio"],
        burstiness_score=burst,
        function_word_frequencies=fw_freq,
        bigram_frequencies={
            bg: cnt / total_words for bg, cnt in bigrams.most_common(50)
        },
        trigram_frequencies={
            tg: cnt / total_words for tg, cnt in trigrams.most_common(30)
        },
        filler_word_frequencies=filler_freq,
        transition_frequencies=trans_freq,
        char_4gram_frequencies=char4,
        char_5gram_frequencies=char5,
        pos_bigram_frequencies=pos_bg,
        pos_bigrams_source=pos_src,
        signature_words=_find_signature_words(words),
        embedding_vector=emb_vec,
    )
    profile.feature_vector = _build_raw_stat_vector(profile)
    return profile


def _find_signature_words(
    words: List[str], top_n: int = 20, min_freq: int = 3
) -> List[str]:
    """Identify distinctive vocabulary for the author."""
    common_english = frozenset(
        {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "it",
            "for", "not", "on", "with", "he", "as", "you", "do", "at", "this",
            "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make",
        }
    )
    counts = Counter(words)
    candidates = [
        (w, c)
        for w, c in counts.items()
        if w not in common_english and c >= min_freq and len(w) > 3
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [w for w, _ in candidates[:top_n]]


# ---------------------------------------------------------------------------
# Syntactic distance helper
# ---------------------------------------------------------------------------

_SYNTACTIC_FEATURES: Tuple[str, ...] = (
    "avg_dep_distance",
    "max_dep_distance",
    "avg_tree_depth",
    "complex_sentence_ratio",
)


def _syntactic_distance(
    profile: StyleProfile, candidate: StyleProfile
) -> float:
    """Normalised L1 distance on syntactic scalar features (0=same, 1=max)."""
    pv = np.array(
        [getattr(profile, f, 0.0) for f in _SYNTACTIC_FEATURES], dtype=float
    )
    cv = np.array(
        [getattr(candidate, f, 0.0) for f in _SYNTACTIC_FEATURES], dtype=float
    )
    denom = np.abs(pv) + 1e-9
    return float(np.mean(np.abs(pv - cv) / denom))


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


class StylometricProfiler:
    """
    Builds and compares writing-style fingerprints for individual authors.

    Parameters
    ----------
    nlp : optional spaCy Language model
        Injected NLP pipeline. When None, the module-level _NLP fallback
        is used (or None if spaCy is unavailable, activating regex mode).
        Pass a mock or alternative model for testing.
    """

    def __init__(self, nlp: Any = None) -> None:
        self._profiles: Dict[str, StyleProfile] = {}

        if nlp is not None:
            self._nlp = nlp
        elif _SPACY_AVAILABLE and _NLP is not None:
            self._nlp = _NLP
        else:
            self._nlp = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def vectorize(self, text: str) -> np.ndarray:
        """
        Return the raw statistical feature vector for a single text.

        This is the **public API** for external vectorization (e.g.
        mass_vectorizer). Returns a numpy array of exactly VECTOR_DIM
        elements. No normalisation is applied (raw frequencies / rates).

        Parameters
        ----------
        text : str
            Input text to profile.

        Returns
        -------
        np.ndarray
            1-D float64 array, length == VECTOR_DIM.
        """
        if not text or len(text.strip()) < 20:
            return np.zeros(VECTOR_DIM, dtype=np.float64)

        profile = _compute_profile_data(
            "_vectorize_", [text], nlp=self._nlp
        )
        vec = profile.feature_vector
        assert len(vec) == VECTOR_DIM, (
            f"vectorize produced {len(vec)} dims, expected {VECTOR_DIM}"
        )
        return vec

    def compute_stats(self, text: str) -> Dict[str, float]:
        """
        Return human-readable statistical features for a single text.

        This replaces the old distilgpt2-based StatisticalFeatureExtractor.
        All metrics are computed on CPU with zero GPU overhead.

        The returned dict is compatible with ForensicReportGenerator's
        ``_bridge_stats_from_result`` and the evaluator's stat display.

        Keys returned
        -------------
        burstiness, lexical_diversity, avg_sentence_length,
        sentence_length_variance, avg_word_length, vocabulary_richness,
        hapax_legomena_ratio, rare_word_ratio, comma_rate,
        avg_dep_distance, complex_sentence_ratio, burstiness_score.
        """
        if not text or len(text.strip()) < 20:
            return {
                "burstiness": 0.0,
                "lexical_diversity": 0.0,
                "avg_sentence_length": 0.0,
                "sentence_length_variance": 0.0,
                "avg_word_length": 0.0,
                "vocabulary_richness": 0.0,
                "hapax_legomena_ratio": 0.0,
                "rare_word_ratio": 0.0,
                "comma_rate": 0.0,
                "avg_dep_distance": 0.0,
                "complex_sentence_ratio": 0.0,
            }

        profile = _compute_profile_data(
            "_stats_", [text], nlp=self._nlp
        )
        return {
            "burstiness": profile.burstiness_score,
            "lexical_diversity": profile.vocabulary_richness,
            "avg_sentence_length": profile.avg_sentence_length,
            "sentence_length_variance": profile.sentence_length_variance,
            "avg_word_length": profile.avg_word_length,
            "vocabulary_richness": profile.vocabulary_richness,
            "hapax_legomena_ratio": profile.hapax_legomena_ratio,
            "rare_word_ratio": profile.rare_word_ratio,
            "comma_rate": profile.comma_rate,
            "avg_dep_distance": profile.avg_dep_distance,
            "complex_sentence_ratio": profile.complex_sentence_ratio,
        }

    def build_profile(
        self,
        author_id: str,
        texts: List[str],
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> StyleProfile:
        """
        Build a stylometric profile from authenticated writing samples.

        Parameters
        ----------
        author_id          : Unique author identifier.
        texts              : Writing samples (5+ recommended).
        embedding_provider : Optional dense-embedding model.
        """
        if not texts:
            raise ValueError("At least one text sample is required.")
        if len(texts) < 2:
            logger.warning(
                "Only %d sample(s) for '%s' — 5+ recommended.",
                len(texts),
                author_id,
            )

        # Per-sample raw vectors for calibration
        raw_vecs = [
            _build_raw_stat_vector(
                _compute_profile_data(author_id, [t], nlp=self._nlp)
            )
            for t in texts
        ]

        adaptive_thr, corpus_mean, corpus_std = _calibrate_threshold(raw_vecs)

        # Combined profile from all samples
        profile = _compute_profile_data(
            author_id, texts, embedding_provider, nlp=self._nlp
        )
        profile.adaptive_threshold = adaptive_thr
        profile.feature_mean = corpus_mean
        profile.feature_std = corpus_std

        # Normalise the merged vector
        norm_stat = _normalise_vector(
            profile.feature_vector, corpus_mean, corpus_std
        )

        # Fuse with embedding if provided
        profile.feature_vector = _fuse_vectors(
            norm_stat, profile.embedding_vector
        )

        self._profiles[author_id] = profile
        logger.info(
            "Profile built for '%s': %d words, %d samples, "
            "B=%.3f, dep_dist=%.2f, complex_ratio=%.2f, "
            "adaptive_thr=%s, pos_src=%s, vec_dim=%d",
            author_id,
            profile.total_words,
            profile.sample_count,
            profile.burstiness_score,
            profile.avg_dep_distance,
            profile.complex_sentence_ratio,
            f"{adaptive_thr:.3f}" if adaptive_thr is not None else "N/A",
            profile.pos_bigrams_source,
            profile.feature_vector.size,
        )
        return profile

    def compare(
        self,
        text: str,
        profile: StyleProfile,
        similarity_threshold: float = 0.80,
        embedding_provider: Optional[EmbeddingProvider] = None,
        siamese_scorer: Optional[SiameseScorer] = None,
    ) -> StyleComparisonResult:
        """Compare a text against an author profile."""
        candidate = self._build_temp_profile(text, profile, embedding_provider)

        c_stat_norm = _normalise_vector(
            candidate.feature_vector,
            profile.feature_mean,
            profile.feature_std,
        )
        c_full = _fuse_vectors(c_stat_norm, candidate.embedding_vector)

        p_vec = profile.feature_vector
        c_vec = c_full

        min_dim = min(p_vec.size, c_vec.size)
        p_vec = p_vec[:min_dim]
        c_vec = c_vec[:min_dim]

        # Similarity: siamese scorer takes priority
        similarity: float = 0.0
        scorer_label: str = "cosine"

        if siamese_scorer is not None:
            try:
                similarity = float(
                    np.clip(siamese_scorer.score(p_vec, c_vec), 0.0, 1.0)
                )
                scorer_label = "siamese"
            except Exception as exc:
                logger.warning(
                    "SiameseScorer.score() failed (%s) — falling back.",
                    exc,
                )
                siamese_scorer = None

        if siamese_scorer is None:
            np_p, np_c = np.linalg.norm(p_vec), np.linalg.norm(c_vec)
            if np_p == 0 or np_c == 0:
                similarity = 0.0
            else:
                similarity = float(
                    np.clip(1.0 - cosine(p_vec, c_vec), 0.0, 1.0)
                )
            scorer_label = (
                "hybrid" if embedding_provider is not None else "cosine"
            )

        threshold = (
            profile.adaptive_threshold
            if profile.adaptive_threshold is not None
            else similarity_threshold
        )

        # Explainability: per-feature distances
        _EXPLAINABLE = (
            "vocabulary_richness",
            "avg_word_length",
            "rare_word_ratio",
            "hapax_legomena_ratio",
            "comma_rate",
            "semicolon_rate",
            "exclamation_rate",
            "question_rate",
            "avg_sentence_length",
            "sentence_length_variance",
            "avg_dep_distance",
            "avg_tree_depth",
            "complex_sentence_ratio",
            "burstiness_score",
        )
        pv = np.array(
            [getattr(profile, f, 0.0) for f in _EXPLAINABLE], dtype=float
        )
        cv = np.array(
            [getattr(candidate, f, 0.0) for f in _EXPLAINABLE], dtype=float
        )
        diffs = np.abs(pv - cv) / (np.abs(pv) + 1e-9)

        feature_distances = dict(zip(_EXPLAINABLE, diffs.tolist()))
        anomalous = [n for n, d in feature_distances.items() if d > 0.30]
        confidence = min(
            1.0,
            similarity
            + 0.1 * (1.0 - len(anomalous) / max(len(_EXPLAINABLE), 1)),
        )

        return StyleComparisonResult(
            author_id=profile.author_id,
            similarity_score=similarity,
            is_likely_same_author=similarity >= threshold,
            confidence=confidence,
            threshold_used=threshold,
            scorer_used=scorer_label,
            feature_distances=feature_distances,
            anomalous_features=anomalous,
        )

    def _generate_window_signals(
        self,
        text: str,
        profile: StyleProfile,
        window_words: int,
        stride_words: int,
        style_drop_threshold: float,
        burst_dev_threshold: float,
        syntax_dist_threshold: float,
    ) -> Iterator[WindowSignals]:
        """
        Generator that yields per-window signal bundles.

        Extracted from detect_transition_point() so the main method
        only consumes the iterator to find the transition.
        """
        words = text.split()
        total = len(words)
        prev_sim: Optional[float] = None

        for start in range(
            0, max(total - window_words + 1, 1), stride_words
        ):
            chunk = " ".join(words[start : start + window_words])
            if not chunk.strip():
                continue

            # Signal 1: stylometric similarity
            cmp_result = self.compare(chunk, profile)
            curr_sim = cmp_result.similarity_score
            style_drop = (
                (prev_sim - curr_sim) if prev_sim is not None else 0.0
            )
            prev_sim = curr_sim

            # Signal 2: burstiness deviation
            w_burst = _compute_burstiness(chunk)
            burst_dev = abs(w_burst - profile.burstiness_score)

            # Signal 3: syntactic distance
            w_profile = self._build_temp_profile(chunk, profile)
            syn_dist = _syntactic_distance(profile, w_profile)

            combined = (
                0.50 * (1.0 - curr_sim)
                + 0.25 * min(burst_dev, 1.0)
                + 0.25 * min(syn_dist, 1.0)
            )

            signals_fired = sum(
                [
                    style_drop > style_drop_threshold,
                    burst_dev > burst_dev_threshold,
                    syn_dist > syntax_dist_threshold,
                ]
            )

            yield WindowSignals(
                word_offset=start,
                stylometric_sim=curr_sim,
                burstiness_dev=burst_dev,
                syntactic_dist=syn_dist,
                combined_score=combined,
                is_flagged=signals_fired >= 2,
            )

    def detect_transition_point(
        self,
        text: str,
        profile: StyleProfile,
        window_words: int = 100,
        stride_words: int = 40,
        style_drop_threshold: float = 0.20,
        burst_dev_threshold: float = 0.25,
        syntax_dist_threshold: float = 0.30,
    ) -> HybridDetectionResult:
        """
        Multi-signal sliding-window hybrid-text detector (v2.1).

        A window is flagged when >= 2 of 3 signals exceed their thresholds.
        """
        words = text.split()
        total = len(words)

        if total < window_words:
            logger.warning(
                "detect_transition_point: text has only %d words "
                "(< window=%d).",
                total,
                window_words,
            )

        window_signals: List[WindowSignals] = list(
            self._generate_window_signals(
                text,
                profile,
                window_words,
                stride_words,
                style_drop_threshold,
                burst_dev_threshold,
                syntax_dist_threshold,
            )
        )

        if not window_signals:
            return HybridDetectionResult(
                transition_word_index=None,
                window_signals=[],
                max_stylometric_drop=0.0,
                max_burstiness_dev=0.0,
                dual_signal_confidence=0.0,
                suspected_ai_start_ratio=None,
            )

        # Transition = first flagged window after >= 1 non-flagged windows
        transition_idx: Optional[int] = None
        seen_clean = False
        for ws in window_signals:
            if not ws.is_flagged:
                seen_clean = True
            elif seen_clean:
                transition_idx = ws.word_offset
                break

        sims = [ws.stylometric_sim for ws in window_signals]
        max_s_drop = max(
            (sims[i - 1] - sims[i] for i in range(1, len(sims))),
            default=0.0,
        )
        max_burst_dev = max(ws.burstiness_dev for ws in window_signals)
        dual_conf = sum(ws.is_flagged for ws in window_signals) / len(
            window_signals
        )

        ratio: Optional[float] = (
            transition_idx / max(total, 1)
            if transition_idx is not None
            else None
        )

        if transition_idx is not None:
            logger.info(
                "Hybrid transition at word ~%d / %d (%.0f%%), "
                "max_drop=%.3f, max_burst_dev=%.3f, dual_conf=%.2f, "
                "author='%s'",
                transition_idx,
                total,
                ratio * 100,
                max_s_drop,
                max_burst_dev,
                dual_conf,
                profile.author_id,
            )
        else:
            logger.debug(
                "No transition detected (dual_signal_confidence=%.2f).",
                dual_conf,
            )

        return HybridDetectionResult(
            transition_word_index=transition_idx,
            window_signals=window_signals,
            max_stylometric_drop=max_s_drop,
            max_burstiness_dev=max_burst_dev,
            dual_signal_confidence=dual_conf,
            suspected_ai_start_ratio=ratio,
        )

    def save_profile(self, profile: StyleProfile, path: str) -> None:
        """Persist a profile to JSON."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(profile.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Profile saved: %s", path)

    def load_profile(self, path: str) -> StyleProfile:
        """Load a profile from JSON."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        profile = StyleProfile.from_dict(data)
        self._profiles[profile.author_id] = profile
        logger.info("Profile loaded: %s", path)
        return profile

    def list_profiles(self) -> List[str]:
        """Return registered author IDs."""
        return list(self._profiles.keys())

    def get_profile(self, author_id: str) -> Optional[StyleProfile]:
        """Retrieve a registered profile by author ID."""
        return self._profiles.get(author_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_temp_profile(
        self,
        text: str,
        reference_profile: Optional[StyleProfile] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> StyleProfile:
        """
        Build a temporary profile WITHOUT registering it.

        Used internally by compare() and detect_transition_point().
        """
        profile = _compute_profile_data(
            "_temp_", [text], embedding_provider, nlp=self._nlp
        )
        if reference_profile is not None:
            profile.feature_mean = reference_profile.feature_mean
            profile.feature_std = reference_profile.feature_std
        return profile


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def quick_compare(
    text: str,
    reference_texts: List[str],
    author_id: str = "reference",
    embedding_provider: Optional[EmbeddingProvider] = None,
    siamese_scorer: Optional[SiameseScorer] = None,
) -> StyleComparisonResult:
    """One-shot helper for scripts and notebooks."""
    profiler = StylometricProfiler()
    profile = profiler.build_profile(
        author_id, reference_texts, embedding_provider
    )
    return profiler.compare(
        text,
        profile,
        embedding_provider=embedding_provider,
        siamese_scorer=siamese_scorer,
    )
