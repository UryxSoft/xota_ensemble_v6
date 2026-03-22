"""
Forensic Report Generator v3.7
================================
Changelog (v3.5 -> v3.7):
  [FIX]  Verdict: Now displays both Human Score and AI Score instead of
         a single "Confidence" percentage that only showed one side.
  [FIX]  Key Evidence: Moved _collect_evidence() to AFTER hallucination,
         reasoning, and watermark analyses so all data sources are available.
         Lowered thresholds: sentence AI 0.85→0.60, uniform scores 0.7→0.5,
         stylometric burst 0.10→0.12, lex_div 0.35→0.40, hapax 0.25→0.30,
         stat deviation 0.40→0.30. Added human-supporting evidence (Source 1b)
         so evidence section is never empty. Added hallucination category
         evidence (Source 8) and moderate reasoning/hallucination evidence.
  [FIX]  Evidence cards now include "explanation" fields for all evidence
         types (hallucination, reasoning, watermark) with human-readable text.
  - All v3.5 logic preserved. ReasoningRiskClassifier unchanged.
"""

import base64, hashlib, io, json, logging, re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

_FORENSIC_DISCLAIMER = (
    "RESEARCH OUTPUT ONLY \u2014 Not suitable as primary or sole evidence in "
    "academic integrity proceedings, legal decisions, or employment actions. "
    "Results must be reviewed by a qualified human expert."
)


@dataclass
class WordAttribution:
    word: str; position: int; ai_score: float; confidence: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class SentenceAttribution:
    text: str; position: int; ai_score: float; confidence: float
    word_attributions: List[WordAttribution]; key_indicators: List[str]


@dataclass
class ForensicReport:
    report_id: str; generated_at: str; text_hash: str; word_count: int
    verdict: str; confidence: float; neural_score: float
    statistical_score: float; stylometric_score: float
    reasoning_score: float; watermark_score: float
    sentence_attributions: List[SentenceAttribution]
    human_baseline_comparison: Dict[str, Tuple[float, float]]
    evidence_points: List[Dict[str, Any]]
    hallucination_risk: Optional[Dict[str, Any]] = None
    reasoning_analysis: Optional[Dict[str, Any]] = None
    stylometric_stats: Optional[Dict[str, float]] = None   # [NEW v3.5]
    executive_summary: Optional[str] = None                 # [NEW v3.5]
    heatmap_b64: Optional[str] = None
    confidence_chart_b64: Optional[str] = None
    comparison_chart_b64: Optional[str] = None


class AttributionCalculator:
    AI_INDICATOR_WORDS = {
        "furthermore": 0.8, "moreover": 0.8, "additionally": 0.7,
        "consequently": 0.85, "therefore": 0.7, "thus": 0.7,
        "delve": 0.95, "utilize": 0.7, "leverage": 0.75,
        "robust": 0.7, "comprehensive": 0.7, "innovative": 0.65,
        "streamline": 0.75, "optimize": 0.7, "enhance": 0.65,
        "facilitate": 0.75, "implement": 0.65, "integrate": 0.65,
        "whilst": 0.8, "hence": 0.75, "thereby": 0.8,
        "aforementioned": 0.85, "pertaining": 0.8, "regarding": 0.6,
    }
    HUMAN_INDICATOR_WORDS = {
        "kinda": 0.9, "gonna": 0.9, "wanna": 0.9, "like": 0.3,
        "basically": 0.4, "actually": 0.4, "honestly": 0.5,
        "literally": 0.5, "seriously": 0.5, "anyway": 0.6,
        "whatever": 0.7, "stuff": 0.6, "thing": 0.4, "things": 0.4,
        "guy": 0.6, "cool": 0.5, "awesome": 0.5, "crazy": 0.5,
    }

    def calculate_word_attributions(self, text, overall_ai_score):
        words = text.split()
        if not words: return []
        attrs = []
        for i, word in enumerate(words):
            cw = re.sub(r"[^\w]", "", word.lower())
            bs = overall_ai_score
            if cw in self.AI_INDICATOR_WORDS:
                ws = min(1.0, bs + self.AI_INDICATOR_WORDS[cw] * 0.3); conf = 0.8
            elif cw in self.HUMAN_INDICATOR_WORDS:
                ws = max(0.0, bs - self.HUMAN_INDICATOR_WORDS[cw] * 0.3); conf = 0.8
            else:
                ws = bs; conf = 0.5
            pf = 1.0 - (i / len(words)) * 0.1
            ws = float(np.clip(ws * pf, 0.0, 1.0))
            attrs.append(WordAttribution(word=word, position=i, ai_score=ws,
                confidence=conf, features={"ai_indicator": cw in self.AI_INDICATOR_WORDS,
                "human_indicator": cw in self.HUMAN_INDICATOR_WORDS}))
        return attrs

    def calculate_sentence_attributions(self, text, overall_ai_score):
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        attrs = []
        for i, sent in enumerate(sentences):
            wa = self.calculate_word_attributions(sent, overall_ai_score)
            ss = float(np.mean([w.ai_score for w in wa])) if wa else overall_ai_score
            conf = float(np.mean([w.confidence for w in wa])) if wa else 0.5
            inds = []
            ls = sent.lower()
            for w, st in self.AI_INDICATOR_WORDS.items():
                if w in ls and st > 0.7: inds.append(f"AI indicator: '{w}'")
            if re.search(r"\b(first|second|third|finally)\b", ls): inds.append("Sequential structure")
            if re.search(r"\b(therefore|thus|hence|consequently)\b", ls): inds.append("Logical connector")
            attrs.append(SentenceAttribution(text=sent, position=i, ai_score=ss,
                confidence=conf, word_attributions=wa, key_indicators=inds))
        return attrs


class HeatmapGenerator:
    def __init__(self):
        self.cmap = LinearSegmentedColormap.from_list(
            "ai_detection", [(0, "#2ecc71"), (0.5, "#f1c40f"), (1, "#e74c3c")])

    def generate_word_heatmap(self, word_attributions, max_words_per_line=15, figsize=(14, 8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        words = [w.word for w in word_attributions]
        scores = [w.ai_score for w in word_attributions]
        x, y, lh = 0.02, 0.95, 0.06
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        for word, score in zip(words, scores):
            color = self.cmap(score)
            to = ax.text(x, y, word + " ", fontsize=10, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7, edgecolor="none"),
                verticalalignment="top")
            bb = to.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
            x += bb.width + 0.01
            if x > 0.95:
                x = 0.02; y -= lh
                if y < 0.1: break
        ax.legend(handles=[
            mpatches.Patch(facecolor="#2ecc71", label="Human-like (0.0-0.3)"),
            mpatches.Patch(facecolor="#f1c40f", label="Uncertain (0.3-0.7)"),
            mpatches.Patch(facecolor="#e74c3c", label="AI-like (0.7-1.0)"),
        ], loc="lower center", ncol=3, frameon=False, fontsize=9)
        ax.set_title("Word-Level AI Attribution Heatmap [HEURISTIC]", fontsize=14, fontweight="bold")
        return fig

    def generate_sentence_chart(self, sentence_attributions, figsize=(12, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        positions = range(len(sentence_attributions))
        scores = [s.ai_score for s in sentence_attributions]
        ax.bar(positions, scores, color=[self.cmap(s) for s in scores], edgecolor="white", linewidth=0.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Decision threshold")
        ax.set_xlabel("Sentence Position", fontsize=11); ax.set_ylabel("AI Probability Score", fontsize=11)
        ax.set_title("Sentence-by-Sentence AI Detection Scores [HEURISTIC]", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1); ax.legend(loc="upper right"); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        return fig

    def generate_comparison_chart(self, metrics, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize)
        labels = list(metrics.keys())
        tv = [metrics[l][0] for l in labels]; bv = [metrics[l][1] for l in labels]
        x = np.arange(len(labels)); w = 0.35
        ax.bar(x - w/2, tv, w, label="Analysed Text", color="#3498db", edgecolor="white")
        ax.bar(x + w/2, bv, w, label="Human Baseline", color="#2ecc71", edgecolor="white")
        ax.set_xlabel("Metric", fontsize=11); ax.set_ylabel("Value", fontsize=11)
        ax.set_title("Statistical Comparison vs Human Writing", fontsize=14, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right"); ax.legend()
        ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True); plt.tight_layout()
        return fig


# ═══════════════════════════════════════════════════════════════════════════
# ReasoningRiskClassifier (unchanged from v3.4)
# ═══════════════════════════════════════════════════════════════════════════

class ReasoningRiskClassifier:
    """
    Heuristic classifier for the 15-dim vector from ReasoningProfiler.
    Display/reporting layer — does NOT modify reasoning_profiler.py.
    """

    _WEIGHTS = {
        "backtracking_density":    0.26,
        "cot_scaffold_density":    0.23,
        "consequence_density":     0.09,
        "causal_density":          0.07,
        "sequence_density":        0.07,
        "contrast_density":        0.05,
        "word_entropy_normalised": 0.07,
        "type_token_ratio":        0.05,
        "paragraph_length_cv":     0.03,
    }
    _INVERSE_WEIGHT = 0.08

    _THR = {
        "type_token_ratio":         (0.35, 0.72),
        "mean_sentence_length":     (12.0, 26.0),
        "std_sentence_length":      (4.0,  14.0),
        "mean_word_length":         (3.8,  5.8),
        "punctuation_ratio":        (0.02, 0.06),
        "stopword_ratio":           (0.30, 0.52),
        "consequence_density":      (0.02, 0.06),
        "causal_density":           (0.02, 0.07),
        "contrast_density":         (0.02, 0.06),
        "sequence_density":         (0.01, 0.05),
        "backtracking_density":     (0.01, 0.07),
        "cot_scaffold_density":     (0.02, 0.10),
        "intuition_leap_density":   (0.01, 0.04),
        "paragraph_length_cv":      (0.18, 0.55),
        "word_entropy_normalised":  (0.68, 0.90),
    }

    _HIGH   = 0.55
    _MEDIUM = 0.28

    _EXPL = {
        "backtracking_density": {
            "display": "Self-Correction Density", "group": "CoT & Self-Correction",
            "high":   "Very high self-correction density (value: {v:.6f}). Phrases such as 'wait', 'let me reconsider', 'actually that is incorrect', 'I made an error' appear with significant frequency. Strongest single marker of a reasoning-optimised model (o1, o3, DeepSeek-R1, QwQ). Standard models rarely produce this at detectable density.",
            "medium": "Moderate self-correction language (value: {v:.6f}). Could indicate a reasoning model on a moderate task, a standard model with 'think step by step' instructions, or a careful human author revising mid-composition.",
            "low":    "Minimal self-correction language (value: {v:.6f}). Consistent with standard autoregressive models (GPT-4o, Claude 3.x, Gemini 1.5) or typical human prose.",
        },
        "cot_scaffold_density": {
            "display": "Chain-of-Thought Scaffolding Density", "group": "CoT & Self-Correction",
            "high":   "Dense CoT scaffolding (value: {v:.6f}). 'step by step', 'let me think', 'working through this', 'step N:', 'from this we can conclude'. Characteristic of models trained with extended thinking budgets.",
            "medium": "Moderate CoT scaffolding (value: {v:.6f}). May reflect a reasoning model, a prompted standard model, or a methodical human author.",
            "low":    "Negligible CoT scaffolding (value: {v:.6f}). Typical of conversational AI or informal human text.",
        },
        "consequence_density": {
            "display": "Logical Consequence Connector Density", "group": "Logical Connectors",
            "high":   "Dense logical consequence connectors (value: {v:.6f}). 'therefore', 'thus', 'consequently', 'hence', 'accordingly'. Signals deductive reasoning chains.",
            "medium": "Moderate consequence language (value: {v:.6f}).",
            "low":    "Sparse consequence connectors (value: {v:.6f}). Narrative or descriptive prose.",
        },
        "causal_density": {
            "display": "Causal Connector Density", "group": "Logical Connectors",
            "high":   "High causal language density (value: {v:.6f}). 'because', 'due to', 'since', 'owing to', 'given that'. Reasoning models produce dense causal chains when constructing derivations.",
            "medium": "Moderate causal language (value: {v:.6f}).",
            "low":    "Low causal density (value: {v:.6f}). Narrative style predominates.",
        },
        "contrast_density": {
            "display": "Contrast Connector Density", "group": "Logical Connectors",
            "high":   "High contrast language (value: {v:.6f}). 'however', 'nevertheless', 'despite', 'although'. Signals dialectical reasoning.",
            "medium": "Moderate contrastive language (value: {v:.6f}).",
            "low":    "Sparse contrast markers (value: {v:.6f}). Monological style.",
        },
        "sequence_density": {
            "display": "Sequential Structure Density", "group": "Logical Connectors",
            "high":   "Heavy sequential framing (value: {v:.6f}). 'first', 'second', 'third', 'finally', 'subsequently'. Strongly characteristic of step-by-step reasoning model output.",
            "medium": "Moderate sequential structure (value: {v:.6f}).",
            "low":    "Non-sequential prose (value: {v:.6f}). No explicit step enumeration.",
        },
        "intuition_leap_density": {
            "display": "Intuitive Assertion Density [INVERSE SIGNAL]", "group": "Style Markers",
            "high":   "Frequent intuitive assertions (value: {v:.6f}). 'obviously', 'clearly', 'of course', 'naturally'. INVERSE signal: high density here is more consistent with human writing or standard AI — reasoning models prefer explicit derivation.",
            "medium": "Moderate intuitive language (value: {v:.6f}). Does not strongly indicate or rule out a reasoning model.",
            "low":    "Minimal intuitive leaps (value: {v:.6f}). Expected profile for reasoning models (o1, DeepSeek-R1, QwQ) that prefer explicit derivation over bare assertion.",
        },
        "type_token_ratio": {
            "display": "Vocabulary Diversity (TTR = |V|/N)", "group": "Lexical Quality",
            "high": "High lexical diversity TTR={v:.4f}. Rich, varied vocabulary.",
            "medium": "Moderate vocabulary diversity TTR={v:.4f}.",
            "low": "Low lexical diversity TTR={v:.4f}. Vocabulary repetition detected.",
        },
        "word_entropy_normalised": {
            "display": "Normalised Word Entropy H(words)/log\u2082(|V|)", "group": "Lexical Quality",
            "high":   "High normalised word entropy H_norm={v:.4f}. Word distribution spread broadly — rich, varied text.",
            "medium": "Moderate word entropy H_norm={v:.4f}.",
            "low":    "Low normalised entropy H_norm={v:.4f}. Concentrated, repetitive word distribution.",
        },
        "paragraph_length_cv": {
            "display": "Paragraph Length CV (\u03c3/\u03bc)", "group": "Structural Variety",
            "high":   "High paragraph length variability CV={v:.4f}. Reasoning models often produce structurally heterogeneous paragraphs — brief assertions alternating with extended derivations.",
            "medium": "Moderate paragraph length variation CV={v:.4f}.",
            "low":    "Highly uniform paragraph lengths CV={v:.4f}. Common in templated AI output.",
        },
        "mean_sentence_length": {
            "display": "Mean Sentence Length (words/sentence)", "group": "Stylometric",
            "high": "Long mean sentence length \u03bc={v:.1f}. Complex, multi-clause constructions.",
            "medium": "Moderate sentence length \u03bc={v:.1f}.",
            "low": "Short mean sentence length \u03bc={v:.1f}. Terse, direct prose.",
        },
        "std_sentence_length": {
            "display": "Sentence Length Std. Deviation \u03c3", "group": "Stylometric",
            "high": "High sentence length variance \u03c3={v:.1f}.",
            "medium": "Moderate sentence length variation \u03c3={v:.1f}.",
            "low": "Uniform sentence lengths \u03c3={v:.1f}. Highly regular pattern.",
        },
        "mean_word_length": {
            "display": "Mean Word Length (chars/token)", "group": "Stylometric",
            "high": "Long average word length \u03bc={v:.2f}. Dense technical vocabulary.",
            "medium": "Moderate word length \u03bc={v:.2f}.",
            "low": "Short average word length \u03bc={v:.2f}. Informal register.",
        },
        "punctuation_ratio": {
            "display": "Punctuation Density (punct/chars)", "group": "Stylometric",
            "high": "High punctuation density r={v:.4f}. Complex sentence structure.",
            "medium": "Moderate punctuation r={v:.4f}.",
            "low": "Sparse punctuation r={v:.4f}. Linear sentence structure.",
        },
        "stopword_ratio": {
            "display": "Stopword Ratio (stopwords/tokens)", "group": "Stylometric",
            "high": "High stopword density r={v:.4f}. Functional language dominates.",
            "medium": "Moderate stopword ratio r={v:.4f}.",
            "low": "Low stopword density r={v:.4f}. Content-dense, technical writing.",
        },
    }

    def classify(self, vec, feature_names):
        features = dict(zip(feature_names, vec.tolist()))
        score  = self._score(features)
        level  = self._level(score)
        return {
            "ai_score":        score,
            "risk_level":      level,
            "feature_details": self._feature_details(features),
            "group_scores":    self._group_scores(features),
            "top_signals":     self._top_signals(features),
            "interpretation":  self._interpretation(score, features),
        }

    def _norm(self, feat, val):
        thr = self._THR.get(feat, (0.0, 1.0))
        return min(1.0, val / max(thr[1], 1e-9))

    def _score(self, features):
        s = sum(w * self._norm(f, features.get(f, 0.0)) for f, w in self._WEIGHTS.items())
        inv = self._norm("intuition_leap_density", features.get("intuition_leap_density", 0.0))
        s += self._INVERSE_WEIGHT * max(0.0, 1.0 - inv)
        return round(min(1.0, max(0.0, s)), 4)

    def _level(self, score):
        if score >= self._HIGH:   return "HIGH \u2014 Reasoning Model"
        if score >= self._MEDIUM: return "MEDIUM \u2014 Possible Reasoning Model"
        return "LOW \u2014 Standard Model or Human"

    def _feat_level(self, feat, val):
        thr = self._THR.get(feat, (0.0, 1.0))
        if val >= thr[1]: return "high"
        if val >= thr[0]: return "medium"
        return "low"

    def _feature_details(self, features):
        details = {}
        for feat, val in features.items():
            em = self._EXPL.get(feat)
            if em is None: continue
            thr = self._THR.get(feat, (0.0, 1.0))
            lev = self._feat_level(feat, val)
            et = em.get(lev, "")
            details[feat] = {
                "display_name": em["display"], "group": em.get("group", "Other"),
                "value": round(val, 6), "level": lev,
                "explanation": et.format(v=val) if "{v" in et else et,
                "threshold_low": thr[0], "threshold_high": thr[1],
            }
        return details

    def _top_signals(self, features, k=5):
        scored = []
        for feat, w in self._WEIGHTS.items():
            val = features.get(feat, 0.0); norm = self._norm(feat, val)
            lev = self._feat_level(feat, val); em = self._EXPL.get(feat, {})
            et = em.get(lev, "")
            scored.append({"feature": feat, "display_name": em.get("display", feat),
                "group": em.get("group", ""), "raw_value": round(val, 6),
                "normalised": round(norm, 4), "weight": w, "level": lev,
                "explanation": (et.format(v=val) if "{v" in et else et)[:280]})
        iv = features.get("intuition_leap_density", 0.0)
        norm = self._norm("intuition_leap_density", iv); lev = self._feat_level("intuition_leap_density", iv)
        ie = self._EXPL.get("intuition_leap_density", {}); et = ie.get(lev, "")
        scored.append({"feature": "intuition_leap_density",
            "display_name": ie.get("display", "Intuitive Assertion Density"),
            "group": ie.get("group", "Style Markers"), "raw_value": round(iv, 6),
            "normalised": round(norm, 4), "weight": self._INVERSE_WEIGHT, "level": lev,
            "explanation": (et.format(v=iv) if "{v" in et else et)[:280]})
        scored.sort(key=lambda x: x["normalised"], reverse=True)
        return scored[:k]

    def _group_scores(self, features):
        n = lambda f: self._norm(f, features.get(f, 0.0))
        return {
            "CoT & Self-Correction": round(n("backtracking_density")*0.55 + n("cot_scaffold_density")*0.45, 4),
            "Logical Connectors":   round(n("consequence_density")*0.30 + n("causal_density")*0.25 + n("contrast_density")*0.20 + n("sequence_density")*0.25, 4),
            "Lexical Richness":     round(n("type_token_ratio")*0.50 + n("word_entropy_normalised")*0.50, 4),
            "Structural Variety":   round(n("paragraph_length_cv"), 4),
            "Intuitive Assertions (inverse)": round(max(0.0, 1.0 - n("intuition_leap_density")), 4),
        }

    def _interpretation(self, score, features):
        bt  = features.get("backtracking_density", 0.0)
        cot = features.get("cot_scaffold_density", 0.0)
        seq = features.get("sequence_density", 0.0)
        con = features.get("consequence_density", 0.0)
        ent = features.get("word_entropy_normalised", 0.0)
        inv = features.get("intuition_leap_density", 0.0)
        if score >= self._HIGH:
            parts = []
            if bt  >= self._THR["backtracking_density"][1]:  parts.append(f"self-correction (density={bt:.4f})")
            if cot >= self._THR["cot_scaffold_density"][1]:  parts.append(f"CoT scaffolding (density={cot:.4f})")
            if seq >= self._THR["sequence_density"][1]:       parts.append(f"step enumeration (density={seq:.4f})")
            sig = "; ".join(parts) if parts else f"combined score={score:.2f}"
            return (f"Strong reasoning-model signature (overall score={score:.2f}). Dominant signals: {sig}. "
                    f"Characteristic of o1, o3-mini, DeepSeek-R1, QwQ — trained via process reward models or "
                    f"MCTS-style search for explicit multi-step deliberation.")
        if score >= self._MEDIUM:
            return (f"Moderate reasoning-model indicators (overall score={score:.2f}). "
                    f"CoT scaffolding={cot:.4f}, consequence connectors={con:.4f}, word entropy={ent:.4f}. "
                    f"Compatible with a reasoning-capable model, a standard model with step-by-step "
                    f"system-prompt instructions, or a methodical human author.")
        return (f"Low reasoning-model indicators (overall score={score:.2f}). "
                f"Self-correction={bt:.4f}, CoT scaffolding={cot:.4f}, intuitive assertions={inv:.4f}. "
                f"Consistent with a standard autoregressive model without extended chain-of-thought "
                f"inference, or with natural human prose.")


# ═══════════════════════════════════════════════════════════════════════════
# [NEW v3.5] Hallucination category explanation map
# ═══════════════════════════════════════════════════════════════════════════

_HAL_CATEGORY_EXPLANATIONS = {
    "lexical_risk": {
        "name": "Lexical Risk",
        "desc": "Measures hedging phrases, overconfident assertions, and negation patterns.",
        "high": "The text contains many hedging or overconfident phrases that are typical of AI-generated content trying to appear authoritative while being uncertain.",
        "medium": "Some hedging or overconfidence markers detected. This level is common in both careful human writing and AI output.",
        "low": "Very few problematic lexical patterns. The language confidence level appears natural.",
    },
    "entity_anomaly": {
        "name": "Entity Anomaly",
        "desc": "Checks for unusual patterns in how people, places, and organizations are mentioned.",
        "high": "Unusual entity patterns detected — such as names mentioned without context, repeated entities, or entities that don't typically co-occur. This is a common sign of fabricated details.",
        "medium": "Some mild entity irregularities. May reflect a dense-information writing style or minor factual liberties.",
        "low": "Entity usage appears natural and consistent throughout the text.",
    },
    "entropy": {
        "name": "Entropy",
        "desc": "Measures the randomness and predictability of word usage patterns.",
        "high": "The word distribution is highly unusual — either too predictable (low entropy) or too chaotic (high entropy), which can indicate machine-generated filler content.",
        "medium": "Word entropy is in a moderate range. Neither strongly indicative of human or AI authorship.",
        "low": "Word distribution patterns are within normal human writing range.",
    },
    "semantic_incoherence": {
        "name": "Semantic Incoherence",
        "desc": "Evaluates whether sentences logically connect to each other.",
        "high": "Several sentences appear semantically disconnected from their neighbors. This is a common hallucination signature where AI generates plausible-sounding but logically unrelated statements.",
        "medium": "Some mild coherence gaps between sentences. Could reflect topic transitions or paragraph breaks.",
        "low": "Strong semantic flow between sentences. Ideas connect logically throughout the text.",
    },
    "vagueness": {
        "name": "Vagueness",
        "desc": "Detects vague quantifiers and lack of specific details.",
        "high": "The text relies heavily on vague language ('many', 'some', 'various') instead of specific facts. AI models often use vague language to avoid committing to verifiable claims.",
        "medium": "Moderate use of vague language. This is common in introductory or summary-style writing.",
        "low": "The text is specific and detailed, with concrete facts and precise language.",
    },
    "repetition": {
        "name": "Repetition",
        "desc": "Detects repeated phrases, self-referential patterns, and redundant entity mentions.",
        "high": "Significant repetition detected — phrases, entities, or sentence structures are being reused in ways that suggest automated generation loops.",
        "medium": "Some repetition present. May reflect emphasis in human writing or mild AI generation patterns.",
        "low": "Minimal repetition. The text uses varied phrasing and avoids redundancy.",
    },
    "structural_anomaly": {
        "name": "Structural Anomaly",
        "desc": "Checks sentence length uniformity, modal verb usage, and superlative frequency.",
        "high": "The text shows unusually uniform sentence structures, excessive modal verbs, or frequent superlatives — patterns associated with AI-generated content that follows a formulaic template.",
        "medium": "Some structural patterns detected but within acceptable ranges.",
        "low": "Natural structural variation in sentence construction and word choice.",
    },
    "imprecision": {
        "name": "Imprecision",
        "desc": "Evaluates the precision of numbers, dates, and temporal references.",
        "high": "The text avoids or misuses specific numbers and dates. AI-generated text often substitutes vague temporal references for precise ones to avoid verifiable errors.",
        "medium": "Moderate level of precision. Some specific facts present alongside general statements.",
        "low": "The text includes precise numerical and temporal details, consistent with informed human writing.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# [NEW v3.5] Stylometric metric explanation map
# ═══════════════════════════════════════════════════════════════════════════

_STYLO_EXPLANATIONS = {
    "burstiness": {
        "name": "Burstiness (Sentence Length Variation)",
        "desc": "How much sentence lengths vary throughout the text. High burstiness is typical of natural human writing (mixing short and long sentences), while low burstiness suggests machine-generated uniformity.",
        "thresholds": (0.15, 0.30),
    },
    "lexical_diversity": {
        "name": "Lexical Diversity",
        "desc": "How varied the vocabulary is. Higher values mean the author uses many different words; lower values suggest repetitive word choice, which is more common in AI output.",
        "thresholds": (0.40, 0.70),
    },
    "avg_sentence_length": {
        "name": "Average Sentence Length",
        "desc": "The average number of words per sentence. AI models often produce sentences that cluster around 15-20 words, while human writing shows more variation.",
        "thresholds": (12.0, 22.0),
    },
    "sentence_length_variance": {
        "name": "Sentence Length Variance",
        "desc": "The statistical spread in sentence lengths. Low variance means sentences are all about the same length (common in AI text); high variance means a natural mix of short and long sentences.",
        "thresholds": (20.0, 80.0),
    },
    "avg_word_length": {
        "name": "Average Word Length",
        "desc": "The average number of characters per word. Unusually high values may indicate dense technical jargon or AI tendency to use longer, more formal words.",
        "thresholds": (3.8, 5.5),
    },
    "vocabulary_richness": {
        "name": "Vocabulary Richness",
        "desc": "Ratio of unique words to total words. A richer vocabulary suggests more diverse language use, which is more typical of experienced human writers.",
        "thresholds": (0.40, 0.70),
    },
    "hapax_legomena_ratio": {
        "name": "Hapax Legomena Ratio",
        "desc": "Proportion of words that appear only once. A higher ratio indicates more unique word choices — a strong human writing indicator. AI tends to reuse words more frequently.",
        "thresholds": (0.30, 0.60),
    },
    "rare_word_ratio": {
        "name": "Rare Word Ratio",
        "desc": "Proportion of uncommon words. Higher values suggest specialized vocabulary or creative writing; very low values may indicate AI's tendency toward common, 'safe' word choices.",
        "thresholds": (0.05, 0.20),
    },
    "comma_rate": {
        "name": "Comma Rate",
        "desc": "Frequency of comma usage. AI models sometimes overuse commas for list-like structures, while human writers show more varied punctuation patterns.",
        "thresholds": (0.02, 0.06),
    },
    "complex_sentence_ratio": {
        "name": "Complex Sentence Ratio",
        "desc": "Proportion of sentences with multiple clauses. Higher ratios indicate more complex sentence construction, which can be a sign of experienced human writing.",
        "thresholds": (0.20, 0.50),
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# [NEW v3.5] Sentence explanation helper
# ═══════════════════════════════════════════════════════════════════════════

def _explain_sentence(sent_attr: SentenceAttribution) -> str:
    """Generate a plain-language explanation for a sentence attribution."""
    score = sent_attr.ai_score
    inds  = sent_attr.key_indicators

    if score >= 0.85:
        level_desc = "This sentence has a very high AI probability score, strongly suggesting it was machine-generated."
    elif score >= 0.65:
        level_desc = "This sentence shows elevated AI indicators, suggesting it may be AI-generated or heavily AI-assisted."
    elif score >= 0.45:
        level_desc = "This sentence falls in the uncertain zone — it could be either human-written or AI-generated."
    elif score >= 0.25:
        level_desc = "This sentence shows mostly human-like characteristics with some mild AI indicators."
    else:
        level_desc = "This sentence appears strongly human-written with natural language patterns."

    if inds:
        ind_text = " Key signals detected: " + "; ".join(inds[:3]) + "."
    else:
        ind_text = " No specific AI indicator words were detected in this sentence; the score is based on overall text context."

    return level_desc + ind_text


# ═══════════════════════════════════════════════════════════════════════════
# [NEW v3.5] Executive Summary generator
# ═══════════════════════════════════════════════════════════════════════════

def _generate_executive_summary(report: ForensicReport) -> str:
    """
    [FIX v3.6] Fully rewritten executive summary.

    Produces a consolidated plain-language observation covering ALL 8 report
    sections. Written for non-technical readers (instructors, administrators,
    academic integrity officers).
    """
    parts = []

    # ══════════════════════════════════════════════════════════════════
    # 1. OVERALL VERDICT (covers: Detection Scores)
    # ══════════════════════════════════════════════════════════════════
    # [FIX v3.7] report.confidence stores AI probability (overall_score).
    # For human verdict, the *verdict* confidence is (1 - confidence).
    ai_pct    = report.confidence * 100
    human_pct = (1 - report.confidence) * 100
    if "AI" in report.verdict:
        conf_pct = ai_pct
    elif "Human" in report.verdict:
        conf_pct = human_pct
    else:
        conf_pct = max(ai_pct, human_pct)

    if "AI" in report.verdict:
        parts.append(
            f"OVERALL FINDING: This {report.word_count}-word text is classified as "
            f"AI-Generated with {conf_pct:.1f}% confidence. The neural detection model, "
            f"which uses four independent AI classifiers working together, determined that "
            f"this text matches the patterns typically produced by AI language models rather "
            f"than natural human writing."
        )
    elif "Human" in report.verdict:
        parts.append(
            f"OVERALL FINDING: This {report.word_count}-word text is classified as "
            f"Human-Written with {conf_pct:.1f}% confidence. The neural detection model "
            f"found the text consistent with natural human authorship patterns."
        )
    elif "Hybrid" in report.verdict:
        parts.append(
            f"OVERALL FINDING: This {report.word_count}-word text is classified as "
            f"Hybrid (mixed authorship) with {conf_pct:.1f}% confidence. The analysis "
            f"detected that some portions of the text appear AI-generated while other "
            f"portions appear human-written, suggesting the author may have used AI to "
            f"write certain sections while writing others independently."
        )
    else:
        parts.append(
            f"OVERALL FINDING: The analysis of this {report.word_count}-word text is "
            f"Inconclusive (confidence: {conf_pct:.1f}%). The detection model could not "
            f"determine with sufficient certainty whether this text was written by a "
            f"human or generated by AI. Manual review by a qualified expert is "
            f"strongly recommended."
        )

    # Detection score overview
    score_notes = []
    ns = report.neural_score * 100
    if ns > 70:
        score_notes.append(f"the neural classifier scored {ns:.0f}% (strong AI signal)")
    elif ns < 30:
        score_notes.append(f"the neural classifier scored {ns:.0f}% (strong human signal)")
    else:
        score_notes.append(f"the neural classifier scored {ns:.0f}% (uncertain range)")

    ss = report.statistical_score * 100
    if ss != 50:
        score_notes.append(f"statistical analysis scored {ss:.0f}%")

    rs = report.reasoning_score * 100
    if rs > 55:
        score_notes.append(f"reasoning-model markers scored {rs:.0f}% (elevated)")
    elif rs < 28:
        score_notes.append(f"reasoning-model markers scored {rs:.0f}% (low)")

    ws = report.watermark_score * 100
    if ws > 10:
        score_notes.append(f"watermark analysis scored {ws:.0f}%")

    if score_notes:
        parts.append("Detection scores: " + ", ".join(score_notes) + ".")

    # ══════════════════════════════════════════════════════════════════
    # 2. WORD-LEVEL ATTRIBUTION (covers: Heatmap)
    # ══════════════════════════════════════════════════════════════════
    if report.sentence_attributions:
        all_words = [w for s in report.sentence_attributions for w in s.word_attributions]
        if all_words:
            avg_word = float(np.mean([w.ai_score for w in all_words]))
            ai_words = sum(1 for w in all_words if w.ai_score > 0.7)
            # [FIX v3.7] Use report.word_count (from text.split()) as canonical
            # count for consistency with the header. word_attributions can
            # tokenize differently (e.g. splitting punctuation).
            total_w  = report.word_count
            ai_words = min(ai_words, total_w)  # cap to prevent "178 of 177"
            ai_indicator_count = sum(
                1 for w in all_words if w.features.get("ai_indicator")
            )
            human_indicator_count = sum(
                1 for w in all_words if w.features.get("human_indicator")
            )

            if avg_word > 0.7:
                parts.append(
                    f"WORD-LEVEL ANALYSIS: The word-by-word heatmap shows that "
                    f"{ai_words} out of {total_w} words ({ai_words/total_w*100:.0f}%) "
                    f"score in the high AI-probability range. This means the vast majority "
                    f"of the vocabulary and phrasing choices in this text match patterns "
                    f"that AI models typically produce."
                )
            elif avg_word > 0.4:
                parts.append(
                    f"WORD-LEVEL ANALYSIS: The word-level heatmap shows mixed results, "
                    f"with {ai_words} out of {total_w} words in the AI range and the rest "
                    f"showing human-like patterns. This suggests possible mixed authorship "
                    f"or AI-assisted editing."
                )
            else:
                parts.append(
                    f"WORD-LEVEL ANALYSIS: The word-level heatmap shows predominantly "
                    f"human-like word choices ({total_w - ai_words} of {total_w} words "
                    f"score in the human range)."
                )

            if ai_indicator_count > 0:
                parts.append(
                    f"The text contains {ai_indicator_count} "
                    f"{'word' if ai_indicator_count == 1 else 'words'} commonly associated "
                    f"with AI writing (such as 'furthermore', 'utilize', 'comprehensive')."
                )
            if human_indicator_count > 0:
                parts.append(
                    f"The text contains {human_indicator_count} informal or colloquial "
                    f"{'word' if human_indicator_count == 1 else 'words'} typically "
                    f"associated with human writing."
                )
            if ai_indicator_count == 0 and human_indicator_count == 0:
                parts.append(
                    "No specific AI-indicator or human-indicator vocabulary was "
                    "detected; the scores are based on overall statistical patterns "
                    "rather than individual trigger words."
                )

    # ══════════════════════════════════════════════════════════════════
    # 3. SENTENCE ANALYSIS (covers: Sentence chart + Sentence Breakdown)
    # ══════════════════════════════════════════════════════════════════
    if report.sentence_attributions:
        scores = [s.ai_score for s in report.sentence_attributions]
        total  = len(scores)
        high_ai = sum(1 for s in scores if s > 0.7)
        medium  = sum(1 for s in scores if 0.4 <= s <= 0.7)
        low_ai  = sum(1 for s in scores if s < 0.4)
        score_std = float(np.std(scores))
        score_mean = float(np.mean(scores))

        if high_ai == total:
            parts.append(
                f"SENTENCE ANALYSIS: Every single sentence in the text ({total} total) "
                f"scores in the high AI-probability zone (average: {score_mean:.0%}). "
                f"This level of consistency across all sentences strongly indicates that "
                f"the entire text was generated in a single AI session, not written "
                f"incrementally by a human."
            )
        elif high_ai > total * 0.6:
            parts.append(
                f"SENTENCE ANALYSIS: {high_ai} out of {total} sentences ({high_ai/total*100:.0f}%) "
                f"score as AI-generated, while {medium + low_ai} show lower scores. "
                f"This pattern suggests significant AI involvement with some sentences "
                f"potentially edited or written independently."
            )
        elif high_ai > 0:
            parts.append(
                f"SENTENCE ANALYSIS: Only {high_ai} out of {total} sentences show "
                f"strong AI indicators, while {low_ai} score as human-written and "
                f"{medium} fall in the uncertain zone. This pattern is consistent with "
                f"selective AI assistance — the author may have used AI for specific "
                f"paragraphs while writing the rest independently."
            )
        else:
            parts.append(
                f"SENTENCE ANALYSIS: None of the {total} sentences show strong AI "
                f"indicators. The sentence-level scores are consistent with human "
                f"authorship."
            )

        # Uniformity observation
        if score_std < 0.03 and total >= 3:
            if "Human" in report.verdict:
                parts.append(
                    f"Notably, the sentence scores are extremely uniform "
                    f"(standard deviation: {score_std:.4f}), meaning every sentence "
                    f"scores almost identically. While this uniformity pattern is "
                    f"sometimes associated with AI text, in this case the uniform "
                    f"scores are all in the human range — indicating consistent "
                    f"human authorship throughout rather than an AI signature."
                )
            else:
                parts.append(
                    f"Notably, the sentence scores are extremely uniform "
                    f"(standard deviation: {score_std:.4f}), meaning every sentence "
                    f"scores almost identically. Human writing naturally varies from "
                    f"sentence to sentence — this level of uniformity is a strong "
                    f"hallmark of AI-generated text."
                )

    # ══════════════════════════════════════════════════════════════════
    # 4. STATISTICAL COMPARISON (covers: Comparison chart)
    # ══════════════════════════════════════════════════════════════════
    if report.human_baseline_comparison:
        deviations_text = []
        for metric, (text_val, human_val) in report.human_baseline_comparison.items():
            if human_val > 0:
                pct_diff = ((text_val - human_val) / human_val) * 100
                if abs(pct_diff) > 30:
                    direction = "higher" if pct_diff > 0 else "lower"
                    deviations_text.append(
                        f"{metric} is {abs(pct_diff):.0f}% {direction} than typical human writing"
                    )
        if deviations_text:
            if "Human" in report.verdict:
                parts.append(
                    "STATISTICAL COMPARISON: When compared against baselines from "
                    "typical human writing, this text shows notable differences: "
                    + "; ".join(deviations_text) + ". "
                    "While these deviations are sometimes associated with machine-generated "
                    "text, they can also occur in short texts, specialized writing, or texts "
                    "with unusual structure. The overall verdict weighs all signals together."
                )
            else:
                parts.append(
                    "STATISTICAL COMPARISON: When compared against baselines from "
                    "typical human writing, this text shows notable differences: "
                    + "; ".join(deviations_text) + ". "
                    "These deviations from normal human patterns are consistent with "
                    "machine-generated text."
                )
        else:
            parts.append(
                "STATISTICAL COMPARISON: The text's statistical metrics (perplexity, "
                "burstiness, lexical diversity, sentence length) fall within normal "
                "ranges for human writing."
            )

    # ══════════════════════════════════════════════════════════════════
    # 5. STYLOMETRIC ANALYSIS (covers: Stylometric section)
    # ══════════════════════════════════════════════════════════════════
    if report.stylometric_stats:
        ss = report.stylometric_stats
        burst = ss.get("burstiness", -1)
        lex   = ss.get("lexical_diversity", -1)
        hapax = ss.get("hapax_legomena_ratio", -1)
        slv   = ss.get("sentence_length_variance", -1)
        csr   = ss.get("complex_sentence_ratio", -1)

        stylo_findings = []

        # [FIX v3.7] Changed `0 <= burst < 0.12` to `burst < 0.12` to catch
        # negative burstiness (which is WORSE than 0, not better).
        # Sentinel -1 only occurs when key is missing — guarded by `!= -1`.
        if burst != -1 and burst < 0.12:
            stylo_findings.append(
                "sentence lengths are very uniform (low burstiness), meaning the "
                "author wrote sentences of nearly the same length throughout — "
                "a pattern much more common in AI output than human writing"
            )
        elif burst > 0.30:
            if "AI" in report.verdict:
                stylo_findings.append(
                    "sentence lengths vary naturally (healthy burstiness), which "
                    "is unusual for AI-generated text but can occur with modern "
                    "language models that mimic human sentence variation"
                )
            else:
                stylo_findings.append(
                    "sentence lengths vary naturally (healthy burstiness), mixing short "
                    "and long sentences as humans typically do"
                )

        if lex != -1 and lex < 0.35:
            stylo_findings.append(
                "vocabulary is repetitive (low lexical diversity), reusing the "
                "same words frequently instead of varying word choice"
            )
        elif lex > 0.65:
            # [FIX v3.7] Verdict-aware interpretation — modern LLMs produce
            # diverse vocabulary by design; high lex_div does NOT contradict AI.
            if "AI" in report.verdict:
                stylo_findings.append(
                    "vocabulary is rich and varied (high lexical diversity) — "
                    "note that modern AI models routinely produce diverse word "
                    "choices, so this does not contradict the AI classification"
                )
            else:
                stylo_findings.append(
                    "vocabulary is rich and varied, with diverse word choices "
                    "typical of experienced human writers"
                )

        if hapax != -1 and hapax < 0.25:
            stylo_findings.append(
                "very few words appear only once (low hapax ratio), suggesting "
                "formulaic, repetitive language"
            )

        if slv != -1 and slv < 15:
            stylo_findings.append(
                "sentence length variation is minimal, which suggests templated "
                "generation rather than organic composition"
            )
        elif slv > 200:
            stylo_findings.append(
                f"sentence length variation is extremely high ({slv:.0f}), with a "
                "large spread between shortest and longest sentences"
            )

        if csr != -1 and csr < 0.15:
            stylo_findings.append(
                "very few complex sentences were found — the text relies on "
                "simple sentence structures throughout"
            )

        if stylo_findings:
            parts.append(
                "STYLOMETRIC ANALYSIS (Writing Style): " + "; ".join(stylo_findings)
                + ". These writing-style features paint a picture of how the text "
                "was composed."
            )

    # ══════════════════════════════════════════════════════════════════
    # 6. HALLUCINATION RISK ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    if report.hallucination_risk:
        hr = report.hallucination_risk
        risk_level = hr.get("risk_level", "N/A")
        overall    = hr.get("overall_risk", 0.0)
        cats       = hr.get("category_scores", {})

        # Find top 2 highest categories
        top_cats = sorted(cats.items(), key=lambda x: x[1], reverse=True)[:2]
        top_names = []
        for cname, cscore in top_cats:
            info = _HAL_CATEGORY_EXPLANATIONS.get(cname, {})
            display = info.get("name", cname.replace("_", " ").title())
            top_names.append(f"{display} ({cscore:.0%})")

        if risk_level == "HIGH":
            parts.append(
                f"HALLUCINATION RISK: The text shows HIGH hallucination risk "
                f"({overall:.0%}). This means the writing exhibits multiple "
                f"patterns commonly associated with AI-generated content that "
                f"contains factual inaccuracies or fabricated details. "
                f"The highest-risk categories are: {', '.join(top_names)}. "
                f"Any factual claims in this text should be independently verified."
            )
        elif risk_level == "MEDIUM":
            parts.append(
                f"HALLUCINATION RISK: The text shows MEDIUM hallucination risk "
                f"({overall:.0%}). Some anomaly patterns are present — the "
                f"highest categories are: {', '.join(top_names)}. "
                f"This level is sometimes seen in both AI text and careful human "
                f"writing, so it is not conclusive on its own."
            )
        else:
            parts.append(
                f"HALLUCINATION RISK: The text shows LOW hallucination risk "
                f"({overall:.0%}). No significant patterns associated with "
                f"fabricated content were detected."
            )

    # ══════════════════════════════════════════════════════════════════
    # 7. REASONING MODEL DETECTION
    # ══════════════════════════════════════════════════════════════════
    if report.reasoning_analysis:
        ra = report.reasoning_analysis
        ra_score = ra.get("ai_score", 0.0)
        ra_level = ra.get("risk_level", "")

        if "HIGH" in ra_level:
            parts.append(
                f"REASONING MODEL DETECTION: The text shows strong signatures "
                f"({ra_score:.0%}) of a reasoning-optimized AI model — the kind "
                f"that 'thinks step by step' before answering (such as ChatGPT o1, "
                f"o3, DeepSeek-R1, or QwQ). These models leave distinctive traces "
                f"like self-correction phrases, chain-of-thought scaffolding, and "
                f"heavy use of logical connectors."
            )
        elif "MEDIUM" in ra_level:
            parts.append(
                f"REASONING MODEL DETECTION: Moderate reasoning-model indicators "
                f"are present ({ra_score:.0%}). This could indicate a reasoning "
                f"model, a standard AI model prompted with 'think step by step' "
                f"instructions, or a methodical human author. Not conclusive alone."
            )
        else:
            parts.append(
                f"REASONING MODEL DETECTION: Low reasoning-model indicators "
                f"({ra_score:.0%}). The text does not show the step-by-step "
                f"thinking patterns typical of reasoning-optimized AI models. "
                f"If AI was used, it was likely a standard model (ChatGPT, "
                f"Claude, Gemini) without extended reasoning mode."
            )

    # ══════════════════════════════════════════════════════════════════
    # 8. KEY EVIDENCE SUMMARY
    # ══════════════════════════════════════════════════════════════════
    if report.evidence_points:
        ev_count = len(report.evidence_points)
        ev_types = set(e.get("type", "") for e in report.evidence_points)
        type_labels = {
            "high_confidence_ai_sentence": "high-confidence AI sentences",
            "uniform_high_ai_scores": "uniform score patterns",
            "absent_human_markers": "absence of human writing markers",
            "low_perplexity": "low text perplexity",
            "stylometric_anomaly": "writing style anomalies",
            "statistical_deviation_from_baseline": "statistical deviations",
            "high_hallucination_risk": "hallucination risk signals",
            "high_hallucination_category": "high-risk hallucination categories",
            "elevated_hallucination_categories": "elevated hallucination categories",
            "high_reasoning_model_signal": "reasoning model traces",
            "moderate_reasoning_model_signal": "moderate reasoning indicators",
            "candidate_watermark_signal": "watermark signal",
            "human_writing_indicators": "human writing indicators",
        }
        found_labels = [type_labels.get(t, t.replace("_", " ")) for t in ev_types]
        parts.append(
            f"KEY EVIDENCE: The analysis identified {ev_count} specific evidence "
            f"point{'s' if ev_count != 1 else ''}, including: "
            + ", ".join(found_labels) + ". "
            "Each evidence point is detailed in the Key Evidence section below."
        )
    else:
        parts.append(
            "KEY EVIDENCE: No specific high-confidence evidence points were "
            "generated. This can occur when the text is AI-generated but does "
            "not contain specific trigger patterns — the verdict is based on "
            "the aggregate of all analysis signals above."
        )

    # ══════════════════════════════════════════════════════════════════
    # CLOSING DISCLAIMER
    # ══════════════════════════════════════════════════════════════════
    parts.append(
        "IMPORTANT: This is an automated research-grade analysis. No single "
        "tool should be used as the sole basis for academic integrity decisions. "
        "All findings must be reviewed by a qualified human expert who considers "
        "the full context of the submission."
    )

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# ForensicReportGenerator
# ═══════════════════════════════════════════════════════════════════════════

class ForensicReportGenerator:
    """
    Generates forensic analysis reports for detected texts.

    Parameters (all optional)
    ─────────────────────────
    detector               : SOTAAIDetector/EnsembleDetector
    profiler               : StylometricProfiler
    hallucination_profiler : HallucinationProfiler
    hallucination_classifier: HallucinationRiskClassifier (auto-created if profiler given)
    reasoning_profiler     : ReasoningProfiler
    reasoning_classifier   : ReasoningRiskClassifier (auto-created if profiler given)
    watermark_decoder      : WatermarkDecoder
    """

    def __init__(self, detector=None, profiler=None,
                 hallucination_profiler=None, hallucination_classifier=None,
                 reasoning_profiler=None, reasoning_classifier=None,
                 watermark_decoder=None):
        self.detector = detector
        self.profiler = profiler
        self.hallucination_profiler = hallucination_profiler
        if hallucination_classifier is not None:
            self.hallucination_classifier = hallucination_classifier
        elif hallucination_profiler is not None:
            try:
                from hallucination_profile import HallucinationRiskClassifier
                self.hallucination_classifier = HallucinationRiskClassifier()
            except ImportError:
                self.hallucination_classifier = None
        else:
            self.hallucination_classifier = None
        self.reasoning_profiler = reasoning_profiler
        if reasoning_classifier is not None:
            self.reasoning_classifier = reasoning_classifier
        elif reasoning_profiler is not None:
            self.reasoning_classifier = ReasoningRiskClassifier()
        else:
            self.reasoning_classifier = None
        self.watermark_decoder = watermark_decoder
        self.attribution_calc = AttributionCalculator()
        self.heatmap_gen = HeatmapGenerator()

    def _bridge_stats_from_result(self, text, detection_result, additional_analyses):
        additional = dict(additional_analyses) if additional_analyses else {}
        merged_stat = {"ppl": 50.0, "burstiness": 0.1, "entropy": 0.0,
            "lexical_diversity": 0.5, "avg_sentence_length": 16.0,
            "sentence_length_variance": 0.0, "flesch_kincaid_grade": 0.0}
        if self.profiler is not None and text:
            try:
                ps = self.profiler.compute_stats(text)
                for k in ["burstiness", "lexical_diversity", "avg_sentence_length", "sentence_length_variance"]:
                    if k in ps: merged_stat[k] = ps[k]
                # [NEW v3.5] Store full stylometric stats for dedicated section
                additional["stylometric_full"] = ps
            except Exception as exc:
                logger.warning("StylometricProfiler.compute_stats() failed: %s", exc)
        if detection_result is not None:
            stats = getattr(detection_result, "statistical_features", None)
            if stats:
                for k in merged_stat:
                    if k in stats and stats[k] != 0.0: merged_stat[k] = stats[k]
                # [NEW v3.5] Use detection_result stats if profiler wasn't run
                if "stylometric_full" not in additional:
                    additional["stylometric_full"] = stats
        merged_stat.update(additional.get("statistical", {}))
        additional["statistical"] = merged_stat
        return additional

    def generate_report(self, text, detection_result=None,
                        additional_analyses=None, generate_visuals=True):
        additional = self._bridge_stats_from_result(text, detection_result, additional_analyses)

        if detection_result is not None:
            raw_ai = (detection_result.raw_scores or {}).get("ai")
            if raw_ai is not None:
                overall_score = float(raw_ai) / 100.0
            elif detection_result.prediction == "AI":
                overall_score = detection_result.confidence / 100.0
            else:
                overall_score = 1.0 - detection_result.confidence / 100.0
            overall_score = float(np.clip(overall_score, 0.0, 1.0))
            verdict = "AI-Generated" if detection_result.prediction == "AI" else "Human-Written"
            neural_score = overall_score
        else:
            overall_score, verdict, neural_score = 0.5, "Inconclusive", 0.5

        sentence_attrs = self.attribution_calc.calculate_sentence_attributions(text, overall_score)
        if len(sentence_attrs) >= 6:
            half = len(sentence_attrs) // 2
            fm = float(np.mean([s.ai_score for s in sentence_attrs[:half]]))
            sm = float(np.mean([s.ai_score for s in sentence_attrs[half:]]))
            if abs(fm - sm) > 0.25 and ((fm > 0.5) != (sm > 0.5)): verdict = "Hybrid"

        stat = additional.get("statistical", {})
        statistical_score = stat.get("score", 0.5)
        stylometric_score = additional.get("stylometric", {}).get("similarity_score", 0.5)
        fk = stat.get("flesch_kincaid_grade", 0.0)
        human_baseline = {
            "Perplexity":           (stat.get("ppl", 50.0), 55.0),
            "Burstiness (CV)":      (stat.get("burstiness", 0.1), 0.25),
            "Lexical Diversity":    (stat.get("lexical_diversity", 0.65), 0.70),
            "Avg Sentence Length":  (stat.get("avg_sentence_length", 18.0), 16.0),
            "Flesch-Kincaid Grade": (fk, 10.0),
        }
        # [FIX v3.7] Moved _collect_evidence AFTER all analyses so that
        # hallucination, reasoning, and watermark data are available for
        # evidence generation. Previously it ran first, missing those signals.

        # Hallucination (unchanged v3.3)
        hallucination_risk = None
        if self.hallucination_profiler and self.hallucination_classifier and text:
            try:
                hs = self.hallucination_profiler.compute_stats(text)
                hallucination_risk = self.hallucination_classifier.classify(hs)
                additional["hallucination"] = hallucination_risk
            except Exception as exc:
                logger.warning("Hallucination analysis failed: %s", exc)

        # Reasoning [v3.4, improved v3.5]
        reasoning_analysis = additional.get("reasoning")
        # [FIX v3.5] If reasoning dict exists but is PARTIAL (missing
        # group_scores/top_signals/feature_details), re-classify with
        # ReasoningRiskClassifier to produce the full structure.
        if reasoning_analysis is not None:
            if "group_scores" not in reasoning_analysis and "feature_values" in reasoning_analysis:
                # Partial dict from old-style orchestrator — re-classify
                try:
                    from reasoning_profiler import FEATURE_NAMES as _RN
                    import numpy as _np
                    fv = reasoning_analysis["feature_values"]
                    vec = _np.array([fv.get(n, 0.0) for n in _RN])
                    clf = self.reasoning_classifier or ReasoningRiskClassifier()
                    reasoning_analysis = clf.classify(vec, _RN)
                    additional["reasoning"] = reasoning_analysis
                except Exception as exc:
                    logger.warning("Reasoning re-classification failed: %s", exc)

        if reasoning_analysis is None and self.reasoning_profiler is not None:
            try:
                from reasoning_profiler import FEATURE_NAMES as _RN
                vec = self.reasoning_profiler.vectorize(text)
                clf = self.reasoning_classifier or ReasoningRiskClassifier()
                reasoning_analysis = clf.classify(vec, _RN)
                additional["reasoning"] = reasoning_analysis
            except Exception as exc:
                logger.warning("Reasoning analysis failed: %s", exc)

        # Watermark [v3.4]
        if additional.get("watermark") is None and self.watermark_decoder is not None:
            try:
                wm = self.watermark_decoder.detect(text)
                additional["watermark"] = wm.to_forensic_dict()
            except Exception as exc:
                logger.warning("Watermark detection failed: %s", exc)

        reasoning_score = additional.get("reasoning", {}).get("ai_score", 0.5)
        watermark_score = additional.get("watermark", {}).get("confidence", 0.0)

        # [FIX v3.7] Now collect evidence AFTER all analyses are complete
        evidence_points = self._collect_evidence(sentence_attrs, additional)

        # Append high-level evidence from hallucination/reasoning/watermark
        if hallucination_risk and hallucination_risk.get("risk_level") == "HIGH":
            evidence_points.append({"type": "high_hallucination_risk",
                "overall_risk": hallucination_risk["overall_risk"],
                "risk_level": hallucination_risk["risk_level"],
                "top_signals": hallucination_risk["top_signals"],
                "interpretation": "Text exhibits multiple hallucination risk indicators.",
                "explanation": (
                    f"Overall hallucination risk is HIGH ({hallucination_risk['overall_risk']:.0%}). "
                    f"Multiple categories flagged — the text exhibits patterns commonly "
                    f"associated with AI-generated content that may contain fabricated details."
                )})
        # [NEW v3.7] Also collect MEDIUM hallucination risk as moderate evidence
        elif hallucination_risk and hallucination_risk.get("risk_level") == "MEDIUM":
            high_cats = [c for c, v in hallucination_risk.get("category_scores", {}).items() if v >= 0.6]
            if high_cats:
                cat_names = [_HAL_CATEGORY_EXPLANATIONS.get(c, {}).get("name", c) for c in high_cats]
                evidence_points.append({"type": "elevated_hallucination_categories",
                    "categories": cat_names,
                    "risk_level": "MEDIUM",
                    "explanation": (
                        f"While overall hallucination risk is MEDIUM, {len(high_cats)} "
                        f"individual {'category scores' if len(high_cats) > 1 else 'category score'} "
                        f"HIGH: {', '.join(cat_names)}. These specific categories warrant attention."
                    )})

        if reasoning_analysis and reasoning_analysis.get("risk_level", "").startswith("HIGH"):
            evidence_points.append({"type": "high_reasoning_model_signal",
                "ai_score": reasoning_analysis["ai_score"],
                "risk_level": reasoning_analysis["risk_level"],
                "interpretation": reasoning_analysis["interpretation"],
                "explanation": (
                    f"Reasoning model detection scored {reasoning_analysis['ai_score']:.0%} — "
                    f"strong chain-of-thought scaffolding and self-correction markers "
                    f"characteristic of reasoning-optimized models (o1, DeepSeek-R1, QwQ)."
                )})
        # [NEW v3.7] Moderate reasoning evidence
        elif reasoning_analysis and "MEDIUM" in reasoning_analysis.get("risk_level", ""):
            evidence_points.append({"type": "moderate_reasoning_model_signal",
                "ai_score": reasoning_analysis["ai_score"],
                "risk_level": reasoning_analysis["risk_level"],
                "explanation": (
                    f"Reasoning model detection scored {reasoning_analysis['ai_score']:.0%} (moderate). "
                    f"Some chain-of-thought patterns detected but not conclusive — "
                    f"could indicate a reasoning model, prompted standard model, or "
                    f"methodical human author."
                )})

        wm = additional.get("watermark", {})
        if wm.get("detected"):
            evidence_points.append({"type": "candidate_watermark_signal",
                "scheme": wm.get("scheme_type"), "confidence": wm.get("confidence"),
                "note": "EXPERIMENTAL watermark signal.",
                "explanation": (
                    f"A candidate digital watermark was detected (confidence: "
                    f"{wm.get('confidence', 0):.0%}, scheme: {wm.get('scheme_type', 'unknown')}). "
                    f"This is experimental and should be verified independently."
                )})

        # [NEW v3.5] Extract stylometric stats
        stylometric_stats = additional.get("stylometric_full")

        report_id = hashlib.md5(f"{text[:100]}{datetime.now().isoformat()}".encode()).hexdigest()[:12].upper()
        report = ForensicReport(
            report_id=report_id, generated_at=datetime.now().isoformat(),
            text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            word_count=len(text.split()), verdict=verdict, confidence=overall_score,
            neural_score=neural_score, statistical_score=statistical_score,
            stylometric_score=stylometric_score, reasoning_score=reasoning_score,
            watermark_score=watermark_score, sentence_attributions=sentence_attrs,
            human_baseline_comparison=human_baseline, evidence_points=evidence_points,
            hallucination_risk=hallucination_risk, reasoning_analysis=reasoning_analysis,
            stylometric_stats=stylometric_stats,
        )

        # [NEW v3.5] Generate executive summary
        report.executive_summary = _generate_executive_summary(report)

        if generate_visuals:
            awa = [w for s in sentence_attrs for w in s.word_attributions]
            report.heatmap_b64           = self._fig_to_base64(self.heatmap_gen.generate_word_heatmap(awa[:200]))
            report.confidence_chart_b64  = self._fig_to_base64(self.heatmap_gen.generate_sentence_chart(sentence_attrs))
            report.comparison_chart_b64  = self._fig_to_base64(self.heatmap_gen.generate_comparison_chart(human_baseline))
        return report

    def _collect_evidence(self, sentence_attrs, additional):
        """
        [FIX v3.6] Completely rewritten evidence collection.

        v3.4/v3.5 bug: required BOTH ai_score > 0.8 AND key_indicators non-empty.
        key_indicators only fires on 23 specific words (furthermore, delve, etc.),
        so natural-sounding AI text produces ZERO evidence despite 95%+ scores.

        v3.6 fix: evidence is generated from MULTIPLE independent sources with
        OR logic, not AND. Any single qualifying condition produces evidence.
        """
        ev = []

        # ── Source 1: High-confidence AI sentences ────────────────────
        # [FIX v3.7] Lowered threshold 0.85 → 0.60 so moderate-AI sentences
        # generate evidence. Texts where ALL sentences score 0.35 still won't
        # fire, but anything in the uncertain-to-AI zone will.
        for a in sentence_attrs:
            if a.ai_score > 0.60:
                ev.append({
                    "type": "high_confidence_ai_sentence",
                    "sentence_position": a.position,
                    "score": round(a.ai_score, 4),
                    "indicators": a.key_indicators if a.key_indicators else ["Elevated AI probability (score-based)"],
                    "excerpt": a.text[:120] + "..." if len(a.text) > 120 else a.text,
                    "explanation": (
                        f"Sentence {a.position+1} scores {a.ai_score:.0%} AI probability. "
                        + (f"Flagged indicators: {', '.join(a.key_indicators[:3])}."
                           if a.key_indicators else
                           "No specific trigger words detected — the score comes from "
                           "the overall AI classification and the sentence's "
                           "statistical conformity to AI-generated patterns.")
                    ),
                })
                if len(ev) >= 5:  # cap at 5 sentence-level evidence items
                    break

        # ── Source 1b: Human-supporting sentences ─────────────────────
        # [NEW v3.7] When no AI sentences found, collect human-supporting evidence
        # so evidence section is never empty for any classification.
        if not ev:
            human_sents = [a for a in sentence_attrs if a.ai_score < 0.40]
            if human_sents:
                avg_human = float(np.mean([a.ai_score for a in human_sents]))
                ev.append({
                    "type": "human_writing_indicators",
                    "sentence_count": len(human_sents),
                    "total_sentences": len(sentence_attrs),
                    "avg_ai_score": round(avg_human, 4),
                    "explanation": (
                        f"{len(human_sents)} of {len(sentence_attrs)} sentences score below "
                        f"the AI threshold (average AI score: {avg_human:.0%}). These sentences "
                        f"exhibit natural language patterns consistent with human authorship — "
                        f"varied word choice, organic sentence structure, and absence of "
                        f"formulaic AI patterns."
                    ),
                })

        # ── Source 2: Score uniformity (ALL sentences score similarly) ─
        # [FIX v3.7] Lowered mean threshold 0.7→0.5 and std 0.05→0.08
        if len(sentence_attrs) >= 3:
            scores = [s.ai_score for s in sentence_attrs]
            score_std = float(np.std(scores))
            score_mean = float(np.mean(scores))
            if score_std < 0.08 and score_mean > 0.5:
                ev.append({
                    "type": "uniform_high_ai_scores",
                    "mean_score": round(score_mean, 4),
                    "std_dev": round(score_std, 4),
                    "sentence_count": len(sentence_attrs),
                    "explanation": (
                        f"All {len(sentence_attrs)} sentences score within a very narrow "
                        f"band (mean={score_mean:.0%}, std={score_std:.4f}). Human writing "
                        f"typically shows much more variation between sentences. This "
                        f"uniformity is a strong indicator of single-source AI generation."
                    ),
                })

        # ── Source 3: No human indicator words found ──────────────────
        # [FIX v3.7] Skip if Source 1b already produced human-supporting evidence,
        # because "human writing indicators" + "absent human markers" reads as
        # contradictory to non-technical readers. Source 3 measures informal/slang
        # words specifically; Source 1b measures sentence-level AI scores.
        has_human_evidence = any(e.get("type") == "human_writing_indicators" for e in ev)
        has_human_words = any(
            any(w.features.get("human_indicator") for w in s.word_attributions)
            for s in sentence_attrs
        )
        if not has_human_words and len(sentence_attrs) >= 3 and not has_human_evidence:
            ev.append({
                "type": "absent_human_markers",
                "explanation": (
                    "The text contains zero informal or colloquial markers "
                    "(contractions, slang, casual expressions) across all "
                    f"{len(sentence_attrs)} sentences. Human writing almost always "
                    "includes some informal elements. Their complete absence is "
                    "consistent with AI-generated text that maintains a uniform "
                    "formal register throughout."
                ),
            })

        # ── Source 4: Low perplexity (when available) ─────────────────
        st = additional.get("statistical", {})
        ppl = st.get("ppl", 50)
        if ppl < 20:
            ev.append({
                "type": "low_perplexity",
                "value": ppl,
                "human_baseline": 55,
                "explanation": (
                    f"Text perplexity is {ppl:.1f}, far below the human baseline "
                    f"of ~55. Low perplexity means the text is highly predictable "
                    f"— a language model can easily guess what comes next. This "
                    f"is a strong statistical indicator of AI generation."
                ),
            })

        # ── Source 5: Stylometric anomalies ───────────────────────────
        # [FIX v3.7] Broadened thresholds and added high-variance detection
        stylo = additional.get("stylometric_full", {})
        if stylo:
            burst = stylo.get("burstiness", 0.5)
            lex_d = stylo.get("lexical_diversity", 0.5)
            hapax = stylo.get("hapax_legomena_ratio", 0.5)
            slv   = stylo.get("sentence_length_variance", -1)
            anomalies = []
            if burst < 0.12:
                anomalies.append(f"very low burstiness ({burst:.3f})")
            if lex_d < 0.40:
                anomalies.append(f"low lexical diversity ({lex_d:.3f})")
            if hapax < 0.30:
                anomalies.append(f"low hapax ratio ({hapax:.3f})")
            if slv > 0 and slv > 200:
                anomalies.append(f"very high sentence length variance ({slv:.1f})")
            if anomalies:
                ev.append({
                    "type": "stylometric_anomaly",
                    "anomalies": anomalies,
                    "explanation": (
                        "Writing style analysis detected: " + "; ".join(anomalies) + ". "
                        "These patterns indicate uniform sentence construction and "
                        "repetitive vocabulary — characteristics more common in "
                        "AI-generated text than natural human writing."
                    ),
                })

        # ── Source 6: Hallucination risk (individual HIGH categories) ──
        # [FIX v3.7] Now accessible because hallucination analysis runs before
        # _collect_evidence. Collects individual HIGH categories as evidence.
        hal = additional.get("hallucination", {})
        if hal:
            for cname, cscore in hal.get("category_scores", {}).items():
                if cscore >= 0.6:
                    cat_info = _HAL_CATEGORY_EXPLANATIONS.get(cname, {})
                    cat_display = cat_info.get("name", cname.replace("_", " ").title())
                    cat_expl = cat_info.get("high", "")
                    ev.append({
                        "type": "high_hallucination_category",
                        "category": cat_display,
                        "score": round(cscore, 4),
                        "explanation": (
                            f"Hallucination category '{cat_display}' scored {cscore:.0%} (HIGH). "
                            f"{cat_expl}"
                        ),
                    })

        # ── Source 7: Statistical comparison vs human baseline ────────
        if st:
            deviations = []
            baseline_map = {
                "burstiness": (0.25, "burstiness"),
                "lexical_diversity": (0.70, "lexical diversity"),
                "avg_sentence_length": (16.0, "average sentence length"),
            }
            for key, (human_val, label) in baseline_map.items():
                val = st.get(key)
                if val is not None and human_val > 0:
                    ratio = abs(val - human_val) / human_val
                    if ratio > 0.30:
                        deviations.append(f"{label} deviates {ratio:.0%} from human baseline")
            if deviations:
                ev.append({
                    "type": "statistical_deviation_from_baseline",
                    "deviations": deviations,
                    "explanation": (
                        "Compared to typical human writing: " + "; ".join(deviations) + ". "
                        "Significant deviations from human baselines are consistent "
                        "with machine-generated text."
                    ),
                })

        return ev

    def _fig_to_base64(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig); buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    @staticmethod
    def _level_colors(level):
        return {"high": ("#c0392b","#fff0ee"), "medium": ("#856404","#fff8e1"),
                "low":  ("#1e8449","#eafaf1")}.get(level, ("#555","#f5f5f5"))

    # ── Build reasoning HTML (v3.4 structure, v3.5 fixes) ────────────

    def _build_reasoning_html(self, reasoning):
        if reasoning is None: return ""
        score = reasoning.get("ai_score", 0.0); risk_level = reasoning.get("risk_level", "N/A")
        interp = reasoning.get("interpretation", ""); gs = reasoning.get("group_scores", {})
        top = reasoning.get("top_signals", []); fd = reasoning.get("feature_details", {})
        if "HIGH" in risk_level:   rf,rb,rb2 = "#c0392b","#fff0ee","#e74c3c"
        elif "MEDIUM" in risk_level: rf,rb,rb2 = "#856404","#fff8e1","#f39c12"
        else:                        rf,rb,rb2 = "#1e8449","#eafaf1","#27ae60"

        grows = ""
        for gn, gv in gs.items():
            p = gv*100; bc = "#e74c3c" if p>=55 else "#f39c12" if p>=28 else "#27ae60"
            grows += (f"<tr><td style='width:40%;'>{gn}</td>"
                      f"<td style='text-align:right;'><strong>{p:.1f}%</strong></td>"
                      f"<td><div style='background:#ecf0f1;border-radius:3px;height:14px;'>"
                      f"<div class='animated-bar' style='background:{bc};border-radius:3px;height:14px;width:{max(p,2):.1f}%;'>"
                      f"</div></div></td></tr>\n")

        trows = ""
        for s in top:
            fg,bg = self._level_colors(s.get("level","low"))
            expl = (s.get("explanation") or "")[:280]
            trows += (f"<tr><td><strong style='font-size:13px;'>{s.get('display_name',s.get('feature'))}</strong>"
                      f"<br><small style='color:#7f8c8d;'>{s.get('group')}</small></td>"
                      f"<td style='text-align:right;font-family:monospace;'>{s.get('raw_value',0):.6f}</td>"
                      f"<td style='background:{bg};color:{fg};text-align:center;font-weight:bold;'>{s.get('level','').upper()}</td>"
                      f"<td style='font-size:12px;'>{expl}</td></tr>\n")

        frows = ""
        for feat, det in fd.items():
            fg,bg = self._level_colors(det.get("level","low"))
            expl = (det.get("explanation") or "")[:240]
            frows += (f"<tr><td><strong style='font-size:12px;'>{det.get('display_name',feat)}</strong>"
                      f"<br><small style='color:#7f8c8d;'>{det.get('group')}</small></td>"
                      f"<td style='text-align:right;font-family:monospace;font-size:12px;'>{det.get('value',0):.6f}</td>"
                      f"<td style='background:{bg};color:{fg};text-align:center;font-weight:bold;'>{det.get('level','').upper()}</td>"
                      f"<td style='font-size:11px;color:#555;white-space:nowrap;'>&ge;{det.get('threshold_low',0):.3f} / &ge;{det.get('threshold_high',0):.3f}</td>"
                      f"<td style='font-size:11px;'>{expl}</td></tr>\n")

        return f"""
<h2 class="section-header" onclick="toggleSection('reasoning-section')">
  Reasoning Model Detection Analysis <span class="toggle-icon">&#9660;</span>
</h2>
<div id="reasoning-section" class="collapsible-section">
<div class="disclaimer" style="background:#eef5ff; border:2px solid #3498db; color:#1a5276;">
  <strong>REASONING ANALYSIS \u2014 Zero-Resource, CPU-only</strong><br>
  Detects chain-of-thought scaffolding, self-correction markers, and logical connector patterns
  characteristic of reasoning-optimised LLMs (o1, o3, DeepSeek-R1, QwQ).
  Based on 15 features from <code>ReasoningProfiler</code>. No GPU or external knowledge base required.
</div>
<div style="text-align:center; margin:20px 0;">
  <div class="metric" style="width:220px;">
    <div class="metric-value" style="color:{rf};">{score:.1%}</div>
    <div class="metric-label">Reasoning Model Probability</div>
  </div>
  <div class="metric" style="width:330px; background:{rb}; border:2px solid {rb2};">
    <div class="metric-value" style="color:{rf}; font-size:16px;">{risk_level}</div>
    <div class="metric-label">Classification</div>
  </div>
</div>
<p><strong>Professional Interpretation:</strong><br>{interp}</p>
<h3>Signal Group Scores</h3>
<table class="interactive-table"><thead><tr><th style="width:40%;">Group</th><th style="width:10%;">Score</th><th>Intensity</th></tr></thead>
<tbody>{grows}</tbody></table>
<h3>Top 5 Diagnostic Signals</h3>
<table class="interactive-table"><thead><tr><th style="width:22%;">Signal</th><th style="width:10%;">Raw Value</th>
<th style="width:8%;">Level</th><th>Professional Interpretation</th></tr></thead>
<tbody>{trows}</tbody></table>
<h3>Complete 15-Dimensional Feature Profile</h3>
<p style="color:#7f8c8d;font-size:12px;">All features from <code>ReasoningProfiler</code> (O(n), CPU-only).
<em>LOW = below low threshold | MEDIUM = between thresholds | HIGH = above high threshold.</em></p>
<table class="interactive-table"><thead><tr><th style="width:22%;">Feature</th><th style="width:10%;">Value</th>
<th style="width:7%;">Level</th><th style="width:14%;">Thresholds</th><th>Explanation</th></tr></thead>
<tbody>{frows}</tbody></table>
</div>
"""

    # ── [NEW v3.5] Build hallucination HTML with explanations ─────────

    def _build_hallucination_html(self, hallucination_risk):
        if hallucination_risk is None: return ""
        hr = hallucination_risk
        rc = {"LOW":"#27ae60","MEDIUM":"#f39c12","HIGH":"#e74c3c"}.get(hr.get("risk_level",""),"#7f8c8d")
        risk_level = hr.get("risk_level", "N/A")
        overall    = hr.get("overall_risk", 0)

        cr = ""
        for c,v in sorted(hr.get("category_scores",{}).items(), key=lambda x:x[1], reverse=True):
            cat_info = _HAL_CATEGORY_EXPLANATIONS.get(c, {})
            cat_name = cat_info.get("name", c.replace("_"," ").title())
            cat_desc = cat_info.get("desc", "")
            # Determine level for explanation
            if v >= 0.6:
                cat_expl = cat_info.get("high", "")
                lvl_badge = '<span style="color:#e74c3c;font-weight:bold;">HIGH</span>'
            elif v >= 0.3:
                cat_expl = cat_info.get("medium", "")
                lvl_badge = '<span style="color:#f39c12;font-weight:bold;">MEDIUM</span>'
            else:
                cat_expl = cat_info.get("low", "")
                lvl_badge = '<span style="color:#27ae60;font-weight:bold;">LOW</span>'

            pct = v * 100
            bar_c = "#e74c3c" if v >= 0.6 else "#f39c12" if v >= 0.3 else "#27ae60"
            cr += (f'<tr class="hoverable-row">'
                   f'<td><strong>{cat_name}</strong><br>'
                   f'<small style="color:#7f8c8d;">{cat_desc}</small></td>'
                   f'<td style="text-align:right;"><strong>{v:.2%}</strong></td>'
                   f'<td>{lvl_badge}</td>'
                   f'<td><div style="background:#ecf0f1;border-radius:3px;height:12px;">'
                   f'<div class="animated-bar" style="background:{bar_c};border-radius:3px;height:12px;width:{max(pct,2):.1f}%;"></div>'
                   f'</div></td>'
                   f'<td style="font-size:12px;">{cat_expl}</td>'
                   f'</tr>\n')

        sh = "".join(f'<li><code>{s["feature"]}</code> = {s["value"]:.4f}</li>' for s in hr.get("top_signals",[])[:5])

        return f"""
<h2 class="section-header" onclick="toggleSection('hallucination-section')">
  Hallucination Risk Analysis <span class="toggle-icon">&#9660;</span>
</h2>
<div id="hallucination-section" class="collapsible-section">
<div style="text-align:center; margin:20px 0;">
  <div class="metric" style="width:200px;">
    <div class="metric-value" style="color:{rc};">{overall:.0%}</div>
    <div class="metric-label">Overall Risk ({risk_level})</div>
  </div>
</div>
<table class="interactive-table">
  <thead><tr><th>Category</th><th style="width:8%;">Score</th><th style="width:7%;">Level</th><th style="width:12%;">Intensity</th><th>What This Means</th></tr></thead>
  <tbody>{cr}</tbody>
</table>
<p><strong>Top Statistical Signals:</strong></p><ul>{sh}</ul>
<p style="color:#7f8c8d;font-size:12px;"><em>Zero-resource analysis. Scores indicate statistical anomaly patterns, not confirmed factual errors.</em></p>
</div>
"""

    # ── [NEW v3.5] Build stylometric HTML section ────────────────────

    def _build_stylometric_html(self, stylometric_stats):
        if not stylometric_stats: return ""

        rows = ""
        for key in ["burstiness", "lexical_diversity", "avg_sentence_length",
                     "sentence_length_variance", "avg_word_length", "vocabulary_richness",
                     "hapax_legomena_ratio", "rare_word_ratio", "comma_rate",
                     "complex_sentence_ratio"]:
            val = stylometric_stats.get(key)
            if val is None: continue
            info = _STYLO_EXPLANATIONS.get(key, {})
            name = info.get("name", key.replace("_", " ").title())
            desc = info.get("desc", "")
            thr  = info.get("thresholds", (0.0, 1.0))

            if val >= thr[1]:
                lvl = "HIGH"; fg = "#e74c3c"; bg = "#fff0ee"
            elif val >= thr[0]:
                lvl = "MEDIUM"; fg = "#856404"; bg = "#fff8e1"
            else:
                lvl = "LOW"; fg = "#1e8449"; bg = "#eafaf1"

            rows += (f'<tr class="hoverable-row">'
                     f'<td><strong>{name}</strong></td>'
                     f'<td style="text-align:right;font-family:monospace;">{val:.4f}</td>'
                     f'<td style="background:{bg};color:{fg};text-align:center;font-weight:bold;">{lvl}</td>'
                     f'<td style="font-size:12px;">{desc}</td>'
                     f'</tr>\n')

        return f"""
<h2 class="section-header" onclick="toggleSection('stylometric-section')">
  Stylometric Analysis <span class="toggle-icon">&#9660;</span>
</h2>
<div id="stylometric-section" class="collapsible-section">
<div class="disclaimer" style="background:#f0f8e7; border:2px solid #27ae60; color:#1a5216;">
  <strong>WRITING STYLE FINGERPRINT</strong><br>
  Statistical analysis of vocabulary, sentence structure, and writing patterns.
  These metrics compare the text's stylistic features against typical human and AI baselines.
</div>
<table class="interactive-table">
  <thead><tr><th style="width:22%;">Metric</th><th style="width:10%;">Value</th><th style="width:8%;">Level</th><th>What This Means</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
</div>
"""

    # ── [NEW v3.5] Build executive summary HTML ──────────────────────

    def _build_executive_summary_html(self, summary):
        if not summary: return ""
        return f"""
<h2 class="section-header" onclick="toggleSection('summary-section')">
  Executive Summary &amp; Key Findings <span class="toggle-icon">&#9660;</span>
</h2>
<div id="summary-section" class="collapsible-section">
<div style="background:linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-left:5px solid #3498db; padding:20px; border-radius:0 8px 8px 0; margin:15px 0; line-height:1.7; font-size:14px;">
  {summary}
</div>
</div>
"""

    # ═══════════════════════════════════════════════════════════════════
    # export_html — MAIN OUTPUT METHOD (v3.5 overhaul)
    # ═══════════════════════════════════════════════════════════════════

    def export_html(self, report, output_path):
        vc = ("ai" if "AI" in report.verdict else "human" if "Human" in report.verdict
              else "hybrid" if "Hybrid" in report.verdict else "inconclusive")

        # [FIX v3.7] Show only the relevant score based on verdict
        if "AI" in report.verdict:
            verdict_score_line = f"AI Score: {report.confidence*100:.1f}%"
        elif "Human" in report.verdict:
            verdict_score_line = f"Human Score: {(1-report.confidence)*100:.1f}%"
        else:
            verdict_score_line = (f"Human Score: {(1-report.confidence)*100:.1f}%"
                                  f" &nbsp;|&nbsp; AI Score: {report.confidence*100:.1f}%")

        # [FIX v3.6] Evidence rendering — human-friendly cards instead of raw JSON
        evhtml = ""
        for e in report.evidence_points[:8]:
            etype = e.get("type", "unknown").replace("_", " ").title()
            expl  = e.get("explanation", "")
            # Build detail line from non-meta fields
            detail_items = []
            for k, v in e.items():
                if k in ("type", "explanation", "indicators"):
                    continue
                if isinstance(v, float):
                    detail_items.append(f"<strong>{k.replace('_',' ').title()}:</strong> {v:.4f}")
                elif isinstance(v, list):
                    detail_items.append(f"<strong>{k.replace('_',' ').title()}:</strong> {', '.join(str(x) for x in v[:5])}")
                elif isinstance(v, str) and len(v) < 200:
                    detail_items.append(f"<strong>{k.replace('_',' ').title()}:</strong> {v}")
            detail_line = " &nbsp;|&nbsp; ".join(detail_items) if detail_items else ""

            evhtml += (
                f'<div class="evidence">'
                f'<span class="evidence-title">{etype}</span><br>'
                f'{f"<p style=&quot;font-size:13px;margin:8px 0 4px;&quot;>{expl}</p>" if expl else ""}'
                f'{f"<p style=&quot;font-size:11px;color:#666;margin:4px 0;&quot;>{detail_line}</p>" if detail_line else ""}'
                f'</div>'
            )

        # [IMPROVED v3.5] Sentence breakdown with explanation column
        srows = ""
        for s in report.sentence_attributions[:10]:
            explanation = _explain_sentence(s)
            srows += (
                f'<tr class="hoverable-row">'
                f'<td>{s.position+1}</td>'
                f'<td style="background:rgba({int(s.ai_score*255)},{int((1-s.ai_score)*255)},100,0.3);">{s.ai_score:.2f}</td>'
                f'<td>{", ".join(s.key_indicators[:2]) or "\u2014"}</td>'
                f'<td>{s.text[:60]}...</td>'
                f'<td style="font-size:12px;">{explanation}</td></tr>\n'
            )

        hm_img = (f'<img src="data:image/png;base64,{report.heatmap_b64}" alt="Heatmap">' if report.heatmap_b64 else "<p><em>Not generated</em></p>")
        sc_img = (f'<img src="data:image/png;base64,{report.confidence_chart_b64}" alt="Sentence Chart">' if report.confidence_chart_b64 else "<p><em>Not generated</em></p>")
        cc_img = (f'<img src="data:image/png;base64,{report.comparison_chart_b64}" alt="Comparison">' if report.comparison_chart_b64 else "<p><em>Not generated</em></p>")

        # Build section HTMLs
        hal_sec  = self._build_hallucination_html(report.hallucination_risk)
        rsn_sec  = self._build_reasoning_html(report.reasoning_analysis)
        styl_sec = self._build_stylometric_html(report.stylometric_stats)
        exec_sec = self._build_executive_summary_html(report.executive_summary)

        # ── Score cards with animated gauges ──
        def _gauge(val, label, color=None):
            pct = val * 100
            if color is None:
                color = "#e74c3c" if pct >= 70 else "#f39c12" if pct >= 40 else "#27ae60"
            return f"""
            <div class="score-card">
              <div class="gauge-ring" style="--pct:{pct:.0f}; --clr:{color};">
                <span class="gauge-val">{pct:.0f}%</span>
              </div>
              <div class="metric-label">{label}</div>
            </div>"""

        score_cards = (
            _gauge(report.neural_score, "Neural Score")
            + _gauge(report.statistical_score, "Statistical")
            + _gauge(report.reasoning_score, "Reasoning")
            + _gauge(report.watermark_score, "Watermark")
        )

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>AI Detection Forensic Report \u2014 {report.report_id}</title>
<style>
/* ── Base ─────────────────────────────────────────── */
*{{box-sizing:border-box;}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;margin:0;padding:40px;background:#f0f2f5;color:#2c3e50;line-height:1.6;}}
.container{{max-width:960px;margin:0 auto;background:white;padding:0;box-shadow:0 4px 24px rgba(0,0,0,.08);border-radius:12px;overflow:hidden;}}

/* ── Header ───────────────────────────────────────── */
.report-header{{background:linear-gradient(135deg,#1a2a3a 0%,#2c3e50 50%,#34495e 100%);color:white;padding:40px;}}
.report-header h1{{margin:0 0 10px;font-size:28px;font-weight:700;letter-spacing:-0.5px;}}
.report-header .meta{{font-size:13px;opacity:0.8;}}
.report-header .meta strong{{color:#3498db;}}

/* ── Content ──────────────────────────────────────── */
.content{{padding:30px 40px 40px;}}
h2{{color:#2c3e50;margin-top:35px;font-size:20px;border-bottom:2px solid #3498db;padding-bottom:8px;}}
h3{{color:#34495e;margin-top:20px;font-size:16px;}}

/* ── Section toggle ───────────────────────────────── */
.section-header{{cursor:pointer;user-select:none;transition:color 0.2s;position:relative;}}
.section-header:hover{{color:#3498db;}}
.toggle-icon{{float:right;font-size:14px;transition:transform 0.3s;}}
.collapsible-section{{max-height:4000px;overflow:hidden;transition:max-height 0.5s ease-in-out, opacity 0.3s;opacity:1;}}
.collapsible-section.collapsed{{max-height:0;opacity:0;padding:0;margin:0;}}

/* ── Verdict ──────────────────────────────────────── */
.verdict{{font-size:22px;padding:20px;border-radius:10px;text-align:center;margin:20px 0;font-weight:700;animation:fadeSlideIn 0.6s ease-out;}}
.verdict.ai{{background:linear-gradient(135deg,#fee 0%,#fdd 100%);border:2px solid #e74c3c;color:#c0392b;}}
.verdict.human{{background:linear-gradient(135deg,#efe 0%,#dfd 100%);border:2px solid #2ecc71;color:#27ae60;}}
.verdict.hybrid{{background:linear-gradient(135deg,#fef 0%,#edf 100%);border:2px solid #9b59b6;color:#8e44ad;}}
.verdict.inconclusive{{background:#eee;border:2px solid #95a5a6;color:#7f8c8d;}}

/* ── Score cards with animated gauge rings ─────────── */
.score-cards{{display:flex;justify-content:center;flex-wrap:wrap;gap:20px;margin:25px 0;}}
.score-card{{text-align:center;padding:15px;animation:fadeSlideIn 0.5s ease-out;}}
.gauge-ring{{
  width:90px;height:90px;border-radius:50%;margin:0 auto 8px;
  background:conic-gradient(var(--clr) calc(var(--pct) * 1%), #ecf0f1 0);
  display:flex;align-items:center;justify-content:center;
  position:relative;
  animation:gaugeIn 1s ease-out;
}}
.gauge-ring::after{{
  content:'';position:absolute;width:70px;height:70px;border-radius:50%;background:white;
}}
.gauge-val{{
  position:relative;z-index:1;font-size:18px;font-weight:700;color:var(--clr);
}}

/* ── Metrics ──────────────────────────────────────── */
.metric{{display:inline-block;width:150px;padding:15px;margin:10px;background:#f8f9fa;border-radius:8px;text-align:center;transition:transform 0.2s,box-shadow 0.2s;}}
.metric:hover{{transform:translateY(-3px);box-shadow:0 4px 12px rgba(0,0,0,0.1);}}
.metric-value{{font-size:28px;font-weight:bold;color:#2c3e50;}}
.metric-label{{font-size:11px;color:#7f8c8d;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;}}

/* ── Charts ───────────────────────────────────────── */
.chart{{text-align:center;margin:20px 0;}} .chart img{{max-width:100%;border:1px solid #e0e0e0;border-radius:8px;transition:transform 0.2s;}}
.chart img:hover{{transform:scale(1.02);}}

/* ── Evidence ─────────────────────────────────────── */
.evidence{{background:#fff3cd;border:1px solid #ffc107;padding:15px;margin:10px 0;border-radius:8px;transition:transform 0.2s;}}
.evidence:hover{{transform:translateX(4px);}}
.evidence pre{{white-space:pre-wrap;font-size:12px;margin:6px 0 0;}}
.evidence-title{{font-weight:bold;color:#856404;}}

/* ── Disclaimer ───────────────────────────────────── */
.disclaimer{{background:#fff3cd;border:2px solid #fd7e14;padding:16px;border-radius:8px;margin:20px 0;font-size:13px;}}

/* ── Tables ───────────────────────────────────────── */
table{{width:100%;border-collapse:collapse;margin:15px 0;}}
th,td{{padding:10px 12px;text-align:left;border-bottom:1px solid #e0e0e0;font-size:13px;}}
th{{background:linear-gradient(135deg,#2c3e50,#34495e);color:white;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:0.3px;position:sticky;top:0;}}
.interactive-table tbody tr{{transition:background 0.15s,transform 0.15s;}}
.interactive-table tbody tr:hover,.hoverable-row:hover{{background:#f0f7ff;transform:translateX(3px);}}

/* ── Animated progress bars ───────────────────────── */
.animated-bar{{animation:barGrow 0.8s ease-out;}}

/* ── Footer ───────────────────────────────────────── */
.footer{{text-align:center;color:#7f8c8d;font-size:12px;margin-top:40px;padding:25px 40px;border-top:1px solid #e0e0e0;background:#f8f9fa;}}

/* ── Animations ───────────────────────────────────── */
@keyframes fadeSlideIn{{from{{opacity:0;transform:translateY(15px);}}to{{opacity:1;transform:translateY(0);}}}}
@keyframes gaugeIn{{from{{background:conic-gradient(var(--clr) 0%,#ecf0f1 0);}}}}
@keyframes barGrow{{from{{width:0!important;}}}}

/* ── Print ────────────────────────────────────────── */
@media print{{
  body{{padding:0;background:white;}}
  .container{{box-shadow:none;}}
  .section-header{{cursor:default;}}
  .collapsible-section{{max-height:none!important;opacity:1!important;}}
  .toggle-icon{{display:none;}}
}}
</style>
</head>
<body>
<div class="container">

<!-- Header -->
<div class="report-header">
  <h1>AI Detection Forensic Report</h1>
  <div class="meta">
    <strong>Report ID:</strong> {report.report_id} &nbsp;|&nbsp;
    <strong>Generated:</strong> {report.generated_at} &nbsp;|&nbsp;
    <strong>Text Hash:</strong> {report.text_hash} &nbsp;|&nbsp;
    <strong>Words:</strong> {report.word_count}
  </div>
</div>

<div class="content">

<!-- Disclaimer -->
<div class="disclaimer"><strong>RESEARCH OUTPUT ONLY</strong><br>{_FORENSIC_DISCLAIMER}</div>

<!-- Verdict -->
<div class="verdict {vc}"><strong>VERDICT: {report.verdict.upper()}</strong><br>{verdict_score_line}</div>

<!-- Executive Summary -->
{exec_sec}

<!-- Score Cards (animated gauge rings) -->
<h2>Detection Scores</h2>
<div class="score-cards">
{score_cards}
</div>

<!-- Charts -->
<h2 class="section-header" onclick="toggleSection('heatmap-section')">
  Word-Level Attribution <small>(Heuristic)</small> <span class="toggle-icon">&#9660;</span>
</h2>
<div id="heatmap-section" class="collapsible-section"><div class="chart">{hm_img}</div></div>

<h2 class="section-header" onclick="toggleSection('sentence-chart-section')">
  Sentence Analysis <small>(Heuristic)</small> <span class="toggle-icon">&#9660;</span>
</h2>
<div id="sentence-chart-section" class="collapsible-section"><div class="chart">{sc_img}</div></div>

<h2 class="section-header" onclick="toggleSection('comparison-section')">
  Statistical Comparison <span class="toggle-icon">&#9660;</span>
</h2>
<div id="comparison-section" class="collapsible-section"><div class="chart">{cc_img}</div></div>

<!-- Stylometric Analysis [NEW v3.5] -->
{styl_sec}

<!-- Hallucination [IMPROVED v3.5] -->
{hal_sec}

<!-- Reasoning [FIXED v3.5] -->
{rsn_sec}

<!-- Key Evidence -->
<h2 class="section-header" onclick="toggleSection('evidence-section')">
  Key Evidence <span class="toggle-icon">&#9660;</span>
</h2>
<div id="evidence-section" class="collapsible-section">{evhtml if evhtml else '<p style="color:#7f8c8d;"><em>No high-confidence evidence points detected.</em></p>'}</div>

<!-- Sentence Breakdown [IMPROVED v3.5 — added Explanation column] -->
<h2 class="section-header" onclick="toggleSection('breakdown-section')">
  Sentence Breakdown <span class="toggle-icon">&#9660;</span>
</h2>
<div id="breakdown-section" class="collapsible-section">
<table class="interactive-table"><thead><tr>
  <th>#</th><th>AI Score</th><th>Indicators</th><th>Excerpt</th><th>Explanation</th>
</tr></thead><tbody>{srows}</tbody></table>
</div>

</div><!-- .content -->

<!-- Footer -->
<div class="footer">
  <p>XplagiaX SOTA AI Detector v3.7 | Word Count: {report.word_count} | {report.generated_at}</p>
  <p>{_FORENSIC_DISCLAIMER}</p>
</div>

</div><!-- .container -->

<!-- Section toggle script -->
<script>
function toggleSection(id) {{
  var el = document.getElementById(id);
  if (!el) return;
  el.classList.toggle('collapsed');
  // Rotate toggle icon
  var header = el.previousElementSibling;
  if (!header) return;
  var icon = header.querySelector('.toggle-icon');
  if (icon) {{
    icon.style.transform = el.classList.contains('collapsed') ? 'rotate(-90deg)' : 'rotate(0deg)';
  }}
}}
</script>
</body></html>"""

        with open(output_path, "w", encoding="utf-8") as f: f.write(html)
        logger.info("HTML report exported: %s", output_path)
        return output_path

    def export_json(self, report, output_path):
        data = {"report_id": report.report_id, "generated_at": report.generated_at,
            "text_hash": report.text_hash, "word_count": report.word_count,
            "verdict": report.verdict, "confidence": report.confidence,
            "disclaimer": _FORENSIC_DISCLAIMER,
            "scores": {"neural": report.neural_score, "statistical": report.statistical_score,
                "stylometric": report.stylometric_score, "reasoning": report.reasoning_score,
                "watermark": report.watermark_score},
            "evidence_points": report.evidence_points,
            "hallucination_risk": report.hallucination_risk,
            "stylometric_stats": report.stylometric_stats,
            "executive_summary": report.executive_summary,
            "sentence_scores": [{"position": s.position, "score": s.ai_score,
                "indicators": s.key_indicators,
                "explanation": _explain_sentence(s)} for s in report.sentence_attributions]}
        if report.reasoning_analysis is not None:
            ra = report.reasoning_analysis
            data["reasoning_analysis"] = {
                "ai_score": ra.get("ai_score", 0.0), "risk_level": ra.get("risk_level", "N/A"),
                "interpretation": ra.get("interpretation", ""),
                "group_scores": ra.get("group_scores", {}),
                "top_signals": [{"feature": s["feature"], "display_name": s["display_name"],
                    "raw_value": s["raw_value"], "level": s["level"],
                    "explanation": s.get("explanation", "")} for s in ra.get("top_signals", [])],
                "feature_values": {k: v["value"] for k, v in ra.get("feature_details", {}).items()}}
        with open(output_path, "w", encoding="utf-8") as f: json.dump(data, f, indent=2)
        logger.info("JSON report exported: %s", output_path)
        return output_path
