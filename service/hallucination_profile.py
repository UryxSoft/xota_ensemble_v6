"""
Hallucination Feature Extractor v2.2.1
=======================================
Zero-Resource single-pass feature extractor for veracity anomalies
in AI-generated text.  Deterministic 25-dimensional feature vector.

Changelog (v2.2 -> v2.2.1)
----------------------------
  [FIX P1]  ``ParsedText`` renamed to ``HallucinationParsedText``.
             Original name collided with ``reasoning_profiler.ParsedText``
             (15 fields, __slots__=True, frozen).  The two DTOs share no
             fields and serve different extraction pipelines.
             All internal references updated; public API unchanged.
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
    Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Tuple,
)

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Named constants
# ═══════════════════════════════════════════════════════════════════════════

_MIN_TEXT_CHARS: int = 20
_FOURGRAM_N: int = 4
_DEFAULT_UNIFORMITY: float = 0.5
_DEFAULT_TEMPORAL: float = 0.5
_TOP_SIGNALS_K: int = 3
_PRECISE_NUMBER_MIN_DIGITS: int = 4
_COUNTER_CARDINALITY_CAP: int = 500_000
_PROPN_JACCARD_WEIGHT: float = 2.0
_SMALL_LIST_THRESHOLD: int = 30


# ═══════════════════════════════════════════════════════════════════════════
# Optional spaCy
# ═══════════════════════════════════════════════════════════════════════════

try:
    import spacy as _spacy
    _NLP = _spacy.load("en_core_web_sm")
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False; _NLP = None
except OSError:
    _SPACY_AVAILABLE = False; _NLP = None


# ═══════════════════════════════════════════════════════════════════════════
# Compiled regex
# ═══════════════════════════════════════════════════════════════════════════

_RE_WORD: re.Pattern = re.compile(r"\b \w+ \b", re.VERBOSE)

_RE_SENTENCE_SPLIT: re.Pattern = re.compile(
    r"""
    (?:
        \.{3,}
      | [!?]+
      | (?<! Mr) (?<! Mrs) (?<! Dr) (?<! vs) (?<! St) (?<! [A-Z]) (?<! \d)
        \.
        (?! \d) (?! [a-z])
    )
    \s*
    """,
    re.VERBOSE,
)

_RE_NUMBER: re.Pattern = re.compile(r"\b \d+ \.? \d* \b", re.VERBOSE)

_RE_SPECIFIC_DATE: re.Pattern = re.compile(
    r"""
    \b(?:
        \d{1,2} [/-] \d{1,2} [/-] \d{2,4}
      | (?:january|february|march|april|may|june
          |july|august|september|october|november|december) \s+ \d{1,2}
      | (?:19|20)\d{2}
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
      | historically | in \s+ the \s+ past | nowadays | these \s+ days
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
    re.compile(r"\b" + re.escape(p) + r"\b", re.VERBOSE | re.IGNORECASE)
    for p in _SELF_REFERENTIAL_PHRASES
)


# ═══════════════════════════════════════════════════════════════════════════
# Lexical dictionaries
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
_DATE_NUM_LABELS: FrozenSet[str] = frozenset({"DATE", "TIME", "PERCENT", "MONEY", "CARDINAL"})


# ═══════════════════════════════════════════════════════════════════════════
# CANONICAL EXPORT
# ═══════════════════════════════════════════════════════════════════════════

HALLUCINATION_VECTOR_DIM: int = 25

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
    groups: Dict[str, List[str]] = {}
    for name, group in _VECTOR_SCHEMA:
        groups.setdefault(group, []).append(name)
    return {k: tuple(v) for k, v in groups.items()}


FEATURE_GROUPS: Dict[str, Tuple[str, ...]] = _build_feature_groups()


# ═══════════════════════════════════════════════════════════════════════════
# Small-list math helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return statistics.mean(values) if len(values) <= _SMALL_LIST_THRESHOLD else float(np.mean(values))


def _safe_pstdev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.pstdev(values) if len(values) <= _SMALL_LIST_THRESHOLD else float(np.std(values))


# ═══════════════════════════════════════════════════════════════════════════
# HallucinationParsedText DTO  ── P1 FIX: renamed from ParsedText
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HallucinationParsedText:
    """
    Immutable single-pass tokenisation result shared by all extractors.

    Renamed from ``ParsedText`` (v2.2) to ``HallucinationParsedText`` (v2.2.1)
    to eliminate the namespace collision with
    ``reasoning_profiler.ParsedText`` (15 fields, __slots__=True).
    """

    words: List[str]
    word_counts: Counter
    total_words: int
    sentences: List[List[str]]
    text_lower: str
    text_raw: str
    doc: Any  # spacy.tokens.Doc | None


def _capped_counter(items: Iterator) -> Counter:
    counts: Counter = Counter()
    current_size: int = 0
    for item in items:
        if item in counts:
            counts[item] += 1
        elif current_size < _COUNTER_CARDINALITY_CAP:
            counts[item] = 1
            current_size += 1
    return counts


def _parse_text(text: str, nlp: Any) -> HallucinationParsedText:  # returns renamed class
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
        raw_sents = [s.strip() for s in _RE_SENTENCE_SPLIT.split(text_lower) if s.strip()]
        sentences = [_RE_WORD.findall(s) for s in raw_sents]

    return HallucinationParsedText(
        words=words,
        word_counts=_capped_counter(iter(words)),
        total_words=max(len(words), 1),
        sentences=sentences,
        text_lower=text_lower,
        text_raw=text,
        doc=doc,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Pure extraction functions  (all updated to use HallucinationParsedText)
# ═══════════════════════════════════════════════════════════════════════════

def _word_set_ratio(counts: Counter, word_set: FrozenSet[str], total: int) -> float:
    return sum(counts[w] for w in word_set if w in counts) / total


def _shannon_entropy_gen(items: Iterator) -> float:
    counts = _capped_counter(items)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())


def _bigram_gen(words: List[str]) -> Iterator[Tuple[str, str]]:
    for i in range(len(words) - 1):
        yield (words[i], words[i + 1])


def _fourgram_gen(words: List[str]) -> Iterator[Tuple[str, ...]]:
    n = _FOURGRAM_N
    for i in range(len(words) - n + 1):
        yield tuple(words[i : i + n])


def _extract_lexical(p: HallucinationParsedText) -> Dict[str, float]:
    c, t = p.word_counts, p.total_words
    return {
        "hedging_ratio":        _word_set_ratio(c, HEDGING_WORDS, t),
        "overconfidence_ratio": _word_set_ratio(c, ABSOLUTE_WORDS, t),
        "negation_ratio":       _word_set_ratio(c, NEGATION_WORDS, t),
    }


def _extract_entity(p: HallucinationParsedText) -> Dict[str, float]:
    zeros = {"entity_density": 0.0, "unique_entity_ratio": 0.0,
             "person_org_ratio": 0.0, "date_num_ratio": 0.0}
    if p.doc is None or not hasattr(p.doc, "ents"):
        return zeros
    ents = p.doc.ents
    if not ents:
        return zeros
    unique = len({e.text.lower() for e in ents})
    return {
        "entity_density":       len(ents) / p.total_words,
        "unique_entity_ratio":  unique / len(ents),
        "person_org_ratio":     sum(1 for e in ents if e.label_ in _PERSON_ORG_LABELS) / len(ents),
        "date_num_ratio":       sum(1 for e in ents if e.label_ in _DATE_NUM_LABELS) / len(ents),
    }


def _extract_entropy(p: HallucinationParsedText) -> Dict[str, float]:
    return {
        "unigram_entropy": _shannon_entropy_gen(iter(p.words)),
        "bigram_entropy":  _shannon_entropy_gen(_bigram_gen(p.words)),
    }


def _extract_cohesion(p: HallucinationParsedText) -> Dict[str, float]:
    defaults = {"avg_jaccard_similarity": 1.0, "min_jaccard_similarity": 1.0,
                "max_semantic_drop": 0.0, "disconnected_sentences_ratio": 0.0}
    if len(p.sentences) < 2:
        return defaults

    if p.doc is not None:
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
                sims.append(0.0); continue
            iw = sum(min(b1.get(k, 0.0), b2.get(k, 0.0)) for k in all_keys)
            uw = sum(max(b1.get(k, 0.0), b2.get(k, 0.0)) for k in all_keys)
            sims.append(iw / uw if uw > 0 else 0.0)
    else:
        sent_concepts = [set(tokens) for tokens in p.sentences]
        if len(sent_concepts) < 2:
            return defaults
        sims = []
        for i in range(1, len(sent_concepts)):
            s1, s2 = sent_concepts[i - 1], sent_concepts[i]
            union = len(s1 | s2)
            sims.append(len(s1 & s2) / union if union > 0 else 0.0)

    avg_sim = _safe_mean(sims)
    min_sim = min(sims) if sims else 1.0
    drops = [sims[j - 1] - sims[j] for j in range(1, len(sims))]
    max_drop = max(drops) if drops else 0.0
    disconnected = sum(1 for s in sims if s == 0.0)
    return {
        "avg_jaccard_similarity":       avg_sim,
        "min_jaccard_similarity":       min_sim,
        "max_semantic_drop":            max_drop,
        "disconnected_sentences_ratio": disconnected / len(sims),
    }


def _extract_vagueness(p: HallucinationParsedText) -> Dict[str, float]:
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
        "specificity_score":      specificity,
        "assertive_hedged_ratio": ah_ratio,
    }


def _extract_repetition(p: HallucinationParsedText) -> Dict[str, float]:
    self_ref = sum(len(pat.findall(p.text_lower)) for pat in _RE_SELF_REFERENTIAL)
    num_sents = max(len(p.sentences), 1)
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
        "self_referential_ratio": self_ref / num_sents,
        "entity_repetition_rate": ent_rep,
        "phrase_repetition_rate": phrase_rep,
    }


def _extract_structural(p: HallucinationParsedText) -> Dict[str, float]:
    c, t = p.word_counts, p.total_words
    num_sents = max(len(p.sentences), 1)
    if p.doc is not None:
        factual = sum(1 for tok in p.doc if tok.pos_ == "PROPN" or tok.like_num)
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
        "factual_density":            factual_density,
        "sentence_length_uniformity": uniformity,
        "modal_verb_ratio":           _word_set_ratio(c, MODAL_VERBS, t),
        "superlative_ratio":          _word_set_ratio(c, SUPERLATIVES, t),
    }


def _extract_precision(p: HallucinationParsedText) -> Dict[str, float]:
    numbers = _RE_NUMBER.findall(p.text_raw)
    precise = [n for n in numbers if "." in n or len(n.replace(".", "")) >= _PRECISE_NUMBER_MIN_DIGITS]
    numeric_precision = len(precise) / max(len(numbers), 1)
    specific_dates = len(_RE_SPECIFIC_DATE.findall(p.text_raw))
    vague_time = len(_RE_VAGUE_TIME.findall(p.text_raw))
    temporal_total = specific_dates + vague_time
    temporal_specificity = specific_dates / temporal_total if temporal_total > 0 else _DEFAULT_TEMPORAL
    return {
        "numeric_precision_ratio": numeric_precision,
        "temporal_specificity":    temporal_specificity,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Vector assembly
# ═══════════════════════════════════════════════════════════════════════════

_EXTRACTORS: Dict[str, Callable[[HallucinationParsedText], Dict[str, float]]] = {
    "lexical":    _extract_lexical,
    "entity":     _extract_entity,
    "entropy":    _extract_entropy,
    "cohesion":   _extract_cohesion,
    "vagueness":  _extract_vagueness,
    "repetition": _extract_repetition,
    "structural": _extract_structural,
    "precision":  _extract_precision,
}


def _assemble_vector(p: HallucinationParsedText) -> np.ndarray:
    results = {group: fn(p) for group, fn in _EXTRACTORS.items()}
    vec = np.empty(HALLUCINATION_VECTOR_DIM, dtype=np.float64)
    for idx, (feat_name, group) in enumerate(_VECTOR_SCHEMA):
        vec[idx] = results[group][feat_name]
    assert len(vec) == HALLUCINATION_VECTOR_DIM
    return vec


# ═══════════════════════════════════════════════════════════════════════════
# Feature Extractor
# ═══════════════════════════════════════════════════════════════════════════

class HallucinationProfiler:
    """
    Zero-resource hallucination feature extractor.
    Pure feature extractor — no classification.

    API contract::
        vectorize(text)      -> np.ndarray[HALLUCINATION_VECTOR_DIM]
        compute_stats(text)  -> Dict[str, float]
        get_feature_names()  -> Tuple[str, ...]
        get_feature_groups() -> Dict[str, Tuple[str, ...]]
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
                "HallucinationProfiler: spaCy unavailable — "
                "entity and syntactic features will be zeros."
            )

    def vectorize(self, text: str) -> np.ndarray:
        if not text or len(text.strip()) < _MIN_TEXT_CHARS:
            return np.zeros(HALLUCINATION_VECTOR_DIM, dtype=np.float64)
        p = _parse_text(text, self._nlp)
        return _assemble_vector(p)

    def compute_stats(self, text: str) -> Dict[str, float]:
        if not text or len(text.strip()) < _MIN_TEXT_CHARS:
            return {name: 0.0 for name in FEATURE_NAMES}
        vec = self.vectorize(text)
        return dict(zip(FEATURE_NAMES, vec.tolist()))

    @staticmethod
    def get_feature_names() -> Tuple[str, ...]:
        return FEATURE_NAMES

    @staticmethod
    def get_feature_groups() -> Dict[str, Tuple[str, ...]]:
        return dict(FEATURE_GROUPS)


# ═══════════════════════════════════════════════════════════════════════════
# Risk Classifier
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HallucinationRiskConfig:
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
    """Heuristic risk classifier consuming the 25-dim feature vector."""

    __slots__ = ("cfg",)

    def __init__(self, config: Optional[HallucinationRiskConfig] = None) -> None:
        self.cfg = config or DEFAULT_RISK_CONFIG

    def classify(self, stats: Dict[str, float]) -> Dict[str, Any]:
        c = self.cfg
        ent_uni = min(stats.get("unigram_entropy", 0.0) / c.entropy_normaliser, 1.0)
        ent_bi  = min(stats.get("bigram_entropy",  0.0) / c.entropy_normaliser, 1.0)

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
            "lexical_risk": c.w_lexical_risk, "entity_anomaly": c.w_entity_anomaly,
            "entropy": c.w_entropy, "semantic_incoherence": c.w_semantic_incoherence,
            "vagueness": c.w_vagueness, "repetition": c.w_repetition,
            "structural_anomaly": c.w_structural_anomaly, "imprecision": c.w_imprecision,
        }

        overall = max(0.0, min(1.0, sum(categories[k] * weights[k] for k in categories)))
        level = "LOW" if overall < c.low_threshold else ("MEDIUM" if overall < c.high_threshold else "HIGH")

        top_signals = sorted(stats.items(), key=lambda x: x[1], reverse=True)[:_TOP_SIGNALS_K]
        return {
            "overall_risk": round(overall, 4),
            "risk_level": level,
            "category_scores": {k: round(v, 4) for k, v in categories.items()},
            "top_signals": [{"feature": k, "value": round(v, 4)} for k, v in top_signals],
            "feature_details": stats,
        }

    def classify_from_text(self, text: str, profiler: HallucinationProfiler) -> Dict[str, Any]:
        return self.classify(profiler.compute_stats(text))


# ═══════════════════════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════════════════════

def _deprecated_compute_risk_summary(
    text: str,
    profiler: Optional[HallucinationProfiler] = None,
    config: Optional[HallucinationRiskConfig] = None,
) -> Dict[str, Any]:
    warnings.warn(
        "compute_risk_summary() is deprecated. "
        "Use HallucinationRiskClassifier.classify_from_text() instead.",
        DeprecationWarning, stacklevel=2,
    )
    if profiler is None:
        profiler = HallucinationProfiler()
    return HallucinationRiskClassifier(config).classify_from_text(text, profiler)


@functools.wraps(_deprecated_compute_risk_summary)
def compute_risk_summary(
    text: str,
    profiler: Optional[HallucinationProfiler] = None,
    config: Optional[HallucinationRiskConfig] = None,
) -> Dict[str, Any]:
    return _deprecated_compute_risk_summary(text, profiler, config)

# Backward-compat alias — soft deprecation
# New code should use HallucinationParsedText directly.
ParsedText = HallucinationParsedText
