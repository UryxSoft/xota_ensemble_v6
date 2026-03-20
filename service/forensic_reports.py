"""
Forensic Report Generator v3.3.1
=================================
Aligned with EnsembleDetector v6.1 / PluginPipeline v1.0.

Changelog (v3.3 -> v3.3.1)
----------------------------
  [FIX P0-B1] Import corrected: ``hallucination_profile`` (not
              ``hallucination_profiler``).  The trailing 'r' caused an
              ImportError at runtime whenever hallucination_profiler was
              injected but hallucination_classifier was None — the
              auto-create branch silently crashed the constructor.
"""

import base64
import hashlib
import io
import json
import logging
import re
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


# ---------------------------------------------------------------------------
# Data classes  (unchanged from v3.3)
# ---------------------------------------------------------------------------

@dataclass
class WordAttribution:
    word: str
    position: int
    ai_score: float
    confidence: float
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class SentenceAttribution:
    text: str
    position: int
    ai_score: float
    confidence: float
    word_attributions: List[WordAttribution]
    key_indicators: List[str]


@dataclass
class ForensicReport:
    report_id: str
    generated_at: str
    text_hash: str
    word_count: int
    verdict: str
    confidence: float
    neural_score: float
    statistical_score: float
    stylometric_score: float
    reasoning_score: float
    watermark_score: float
    sentence_attributions: List[SentenceAttribution]
    human_baseline_comparison: Dict[str, Tuple[float, float]]
    evidence_points: List[Dict[str, Any]]
    hallucination_risk: Optional[Dict[str, Any]] = None
    heatmap_b64: Optional[str] = None
    confidence_chart_b64: Optional[str] = None
    comparison_chart_b64: Optional[str] = None


# ---------------------------------------------------------------------------
# Attribution calculator
# ---------------------------------------------------------------------------

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
        "kinda": 0.9, "gonna": 0.9, "wanna": 0.9,
        "like": 0.3, "basically": 0.4, "actually": 0.4,
        "honestly": 0.5, "literally": 0.5, "seriously": 0.5,
        "anyway": 0.6, "whatever": 0.7, "stuff": 0.6,
        "thing": 0.4, "things": 0.4, "guy": 0.6,
        "cool": 0.5, "awesome": 0.5, "crazy": 0.5,
    }

    def calculate_word_attributions(
        self, text: str, overall_ai_score: float
    ) -> List[WordAttribution]:
        words = text.split()
        if not words:
            return []
        attributions: List[WordAttribution] = []
        for i, word in enumerate(words):
            clean_word = re.sub(r"[^\w]", "", word.lower())
            base_score = overall_ai_score
            if clean_word in self.AI_INDICATOR_WORDS:
                word_score = min(
                    1.0, base_score + self.AI_INDICATOR_WORDS[clean_word] * 0.3
                )
                confidence = 0.8
            elif clean_word in self.HUMAN_INDICATOR_WORDS:
                word_score = max(
                    0.0, base_score - self.HUMAN_INDICATOR_WORDS[clean_word] * 0.3
                )
                confidence = 0.8
            else:
                word_score = base_score
                confidence = 0.5
            position_factor = 1.0 - (i / len(words)) * 0.1
            word_score = float(np.clip(word_score * position_factor, 0.0, 1.0))
            attributions.append(WordAttribution(
                word=word, position=i,
                ai_score=word_score, confidence=confidence,
                features={
                    "ai_indicator": clean_word in self.AI_INDICATOR_WORDS,
                    "human_indicator": clean_word in self.HUMAN_INDICATOR_WORDS,
                },
            ))
        return attributions

    def calculate_sentence_attributions(
        self, text: str, overall_ai_score: float
    ) -> List[SentenceAttribution]:
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        attributions: List[SentenceAttribution] = []
        for i, sentence in enumerate(sentences):
            word_attrs = self.calculate_word_attributions(sentence, overall_ai_score)
            if word_attrs:
                sentence_score = float(np.mean([w.ai_score for w in word_attrs]))
                confidence = float(np.mean([w.confidence for w in word_attrs]))
            else:
                sentence_score = overall_ai_score
                confidence = 0.5
            indicators: List[str] = []
            lower = sentence.lower()
            for word, strength in self.AI_INDICATOR_WORDS.items():
                if word in lower and strength > 0.7:
                    indicators.append(f"AI indicator: '{word}'")
            if re.search(r"\b(first|second|third|finally)\b", lower):
                indicators.append("Sequential structure")
            if re.search(r"\b(therefore|thus|hence|consequently)\b", lower):
                indicators.append("Logical connector")
            attributions.append(SentenceAttribution(
                text=sentence, position=i,
                ai_score=sentence_score, confidence=confidence,
                word_attributions=word_attrs, key_indicators=indicators,
            ))
        return attributions


# ---------------------------------------------------------------------------
# Heatmap generator
# ---------------------------------------------------------------------------

class HeatmapGenerator:
    def __init__(self) -> None:
        self.cmap = LinearSegmentedColormap.from_list(
            "ai_detection",
            [(0, "#2ecc71"), (0.5, "#f1c40f"), (1, "#e74c3c")],
        )

    def generate_word_heatmap(
        self,
        word_attributions: List[WordAttribution],
        max_words_per_line: int = 15,
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        x, y, line_height = 0.02, 0.95, 0.06
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for wa in word_attributions:
            color = self.cmap(wa.ai_score)
            text_obj = ax.text(
                x, y, wa.word + " ", fontsize=10, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7, edgecolor="none"),
                verticalalignment="top",
            )
            bb = text_obj.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
            x += bb.width + 0.01
            if x > 0.95:
                x = 0.02; y -= line_height
                if y < 0.1:
                    break
        ax.legend(
            handles=[
                mpatches.Patch(facecolor="#2ecc71", label="Human-like (0.0-0.3)"),
                mpatches.Patch(facecolor="#f1c40f", label="Uncertain (0.3-0.7)"),
                mpatches.Patch(facecolor="#e74c3c", label="AI-like (0.7-1.0)"),
            ],
            loc="lower center", ncol=3, frameon=False, fontsize=9,
        )
        ax.set_title("Word-Level AI Attribution Heatmap [HEURISTIC]", fontsize=14, fontweight="bold")
        return fig

    def generate_sentence_chart(
        self, sentence_attributions: List[SentenceAttribution], figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        scores = [s.ai_score for s in sentence_attributions]
        ax.bar(range(len(scores)), scores, color=[self.cmap(s) for s in scores], edgecolor="white", linewidth=0.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Decision threshold")
        ax.set_xlabel("Sentence Position", fontsize=11)
        ax.set_ylabel("AI Probability Score", fontsize=11)
        ax.set_title("Sentence-by-Sentence AI Detection Scores [HEURISTIC]", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1); ax.legend(loc="upper right"); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        return fig

    def generate_comparison_chart(
        self, metrics: Dict[str, Tuple[float, float]], figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        labels = list(metrics.keys())
        x = np.arange(len(labels)); width = 0.35
        ax.bar(x - width/2, [metrics[l][0] for l in labels], width, label="Analysed Text", color="#3498db", edgecolor="white")
        ax.bar(x + width/2, [metrics[l][1] for l in labels], width, label="Human Baseline", color="#2ecc71", edgecolor="white")
        ax.set_xlabel("Metric", fontsize=11); ax.set_ylabel("Value", fontsize=11)
        ax.set_title("Statistical Comparison vs Human Writing", fontsize=14, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend(); ax.yaxis.grid(True, alpha=0.3); ax.set_axisbelow(True)
        plt.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

class ForensicReportGenerator:
    """
    Generates forensic analysis reports for detected texts.

    Parameters
    ----------
    detector              : EnsembleDetector / SOTAAIDetector (optional).
    profiler              : StylometricProfiler (optional).
    hallucination_profiler: HallucinationProfiler (optional).
    hallucination_classifier: HallucinationRiskClassifier (optional).
    """

    def __init__(
        self,
        detector: Any = None,
        profiler: Any = None,
        hallucination_profiler: Any = None,
        hallucination_classifier: Any = None,
    ) -> None:
        self.detector = detector
        self.profiler = profiler
        self.hallucination_profiler = hallucination_profiler

        # ── P0-B1 FIX: was 'hallucination_profiler', module is 'hallucination_profile' ──
        if hallucination_classifier is not None:
            self.hallucination_classifier = hallucination_classifier
        elif hallucination_profiler is not None:
            try:
                from hallucination_profile import HallucinationRiskClassifier  # FIXED
                self.hallucination_classifier = HallucinationRiskClassifier()
            except ImportError:
                logger.warning(
                    "hallucination_profile module not found — "
                    "hallucination risk analysis disabled."
                )
                self.hallucination_classifier = None
        else:
            self.hallucination_classifier = None

        self.attribution_calc = AttributionCalculator()
        self.heatmap_gen = HeatmapGenerator()

    # ------------------------------------------------------------------

    def _bridge_stats_from_result(
        self,
        text: str,
        detection_result: Optional[Any],
        additional_analyses: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        additional = dict(additional_analyses) if additional_analyses else {}
        merged_stat: Dict[str, Any] = {
            "ppl": 50.0, "burstiness": 0.1, "entropy": 0.0,
            "lexical_diversity": 0.5, "avg_sentence_length": 16.0,
            "sentence_length_variance": 0.0, "flesch_kincaid_grade": 0.0,
        }
        if self.profiler is not None and text:
            try:
                ps = self.profiler.compute_stats(text)
                for key in ("burstiness", "lexical_diversity", "avg_sentence_length", "sentence_length_variance"):
                    if key in ps:
                        merged_stat[key] = ps[key]
            except Exception as exc:
                logger.warning("StylometricProfiler.compute_stats() failed: %s", exc)
        if detection_result is not None:
            stats = getattr(detection_result, "statistical_features", None)
            if stats:
                for key in merged_stat:
                    if key in stats and stats[key] != 0.0:
                        merged_stat[key] = stats[key]
        merged_stat.update(additional.get("statistical", {}))
        additional["statistical"] = merged_stat
        return additional

    def generate_report(
        self,
        text: str,
        detection_result: Optional[Any] = None,
        additional_analyses: Optional[Dict[str, Any]] = None,
        generate_visuals: bool = True,
    ) -> ForensicReport:
        additional = self._bridge_stats_from_result(text, detection_result, additional_analyses)

        if detection_result is not None:
            raw_ai_pct = (detection_result.raw_scores or {}).get("ai")
            if raw_ai_pct is not None:
                overall_score = float(raw_ai_pct) / 100.0
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
            fdm = float(np.mean([s.ai_score for s in sentence_attrs[:half]]))
            sdm = float(np.mean([s.ai_score for s in sentence_attrs[half:]]))
            if abs(fdm - sdm) > 0.25 and ((fdm > 0.5) != (sdm > 0.5)):
                verdict = "Hybrid"

        stat = additional.get("statistical", {})
        human_baseline: Dict[str, Tuple[float, float]] = {
            "Perplexity": (stat.get("ppl", 50.0), 55.0),
            "Burstiness (CV)": (stat.get("burstiness", 0.1), 0.25),
            "Lexical Diversity": (stat.get("lexical_diversity", 0.65), 0.70),
            "Avg Sentence Length": (stat.get("avg_sentence_length", 18.0), 16.0),
            "Flesch-Kincaid Grade": (stat.get("flesch_kincaid_grade", 0.0), 10.0),
        }

        evidence_points = self._collect_evidence(sentence_attrs, additional)

        hallucination_risk: Optional[Dict[str, Any]] = None
        if self.hallucination_profiler is not None and self.hallucination_classifier is not None and text:
            try:
                hal_stats = self.hallucination_profiler.compute_stats(text)
                hallucination_risk = self.hallucination_classifier.classify(hal_stats)
                if hallucination_risk.get("risk_level") == "HIGH":
                    evidence_points.append({
                        "type": "high_hallucination_risk",
                        "overall_risk": hallucination_risk["overall_risk"],
                        "risk_level": hallucination_risk["risk_level"],
                        "top_signals": hallucination_risk["top_signals"],
                        "interpretation": (
                            "Text exhibits multiple hallucination risk indicators."
                        ),
                    })
            except Exception as exc:
                logger.warning("Hallucination analysis failed: %s", exc)

        report_id = (
            hashlib.md5(f"{text[:100]}{datetime.now().isoformat()}".encode())
            .hexdigest()[:12].upper()
        )

        report = ForensicReport(
            report_id=report_id,
            generated_at=datetime.now().isoformat(),
            text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            word_count=len(text.split()),
            verdict=verdict,
            confidence=overall_score,
            neural_score=neural_score,
            statistical_score=stat.get("score", 0.5),
            stylometric_score=additional.get("stylometric", {}).get("similarity_score", 0.5),
            reasoning_score=additional.get("reasoning", {}).get("ai_score", 0.5),
            watermark_score=additional.get("watermark", {}).get("confidence", 0.0),
            sentence_attributions=sentence_attrs,
            human_baseline_comparison=human_baseline,
            evidence_points=evidence_points,
            hallucination_risk=hallucination_risk,
        )

        if generate_visuals:
            all_word_attrs = [w for s in sentence_attrs for w in s.word_attributions]
            report.heatmap_b64 = self._fig_to_base64(
                self.heatmap_gen.generate_word_heatmap(all_word_attrs[:200])
            )
            report.confidence_chart_b64 = self._fig_to_base64(
                self.heatmap_gen.generate_sentence_chart(sentence_attrs)
            )
            report.comparison_chart_b64 = self._fig_to_base64(
                self.heatmap_gen.generate_comparison_chart(human_baseline)
            )

        return report

    def _collect_evidence(
        self, sentence_attrs: List[SentenceAttribution], additional: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for attr in sentence_attrs:
            if attr.ai_score > 0.8 and attr.key_indicators:
                evidence.append({
                    "type": "high_confidence_ai_sentence",
                    "sentence_position": attr.position,
                    "score": attr.ai_score,
                    "indicators": attr.key_indicators,
                    "excerpt": attr.text[:100] + ("..." if len(attr.text) > 100 else ""),
                })
        stat = additional.get("statistical", {})
        if stat.get("ppl", 50) < 20:
            evidence.append({"type": "low_perplexity", "value": stat.get("ppl"), "human_baseline": 55,
                              "interpretation": "Text is highly predictable \u2014 consistent with AI generation."})
        wm = additional.get("watermark", {})
        if wm.get("detected"):
            evidence.append({"type": "candidate_watermark_signal", "scheme": wm.get("scheme_type"),
                              "confidence": wm.get("confidence"), "note": "EXPERIMENTAL watermark signal."})
        return evidence

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def export_json(self, report: ForensicReport, output_path: str) -> str:
        data = {
            "report_id": report.report_id,
            "generated_at": report.generated_at,
            "text_hash": report.text_hash,
            "word_count": report.word_count,
            "verdict": report.verdict,
            "confidence": report.confidence,
            "disclaimer": _FORENSIC_DISCLAIMER,
            "scores": {
                "neural": report.neural_score,
                "statistical": report.statistical_score,
                "stylometric": report.stylometric_score,
                "reasoning": report.reasoning_score,
                "watermark": report.watermark_score,
            },
            "evidence_points": report.evidence_points,
            "hallucination_risk": report.hallucination_risk,
            "sentence_scores": [
                {"position": s.position, "score": s.ai_score, "indicators": s.key_indicators}
                for s in report.sentence_attributions
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("JSON report exported: %s", output_path)
        return output_path
