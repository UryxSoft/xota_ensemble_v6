"""
Forensic Report Generator v3.3
===============================
Generates detailed forensic reports for AI text detection.
Aligned with SOTAAIDetector v3.3 (8-class coarse, training pipeline v5.3).

Changelog (v3.2 -> v3.3):
  - Accepts optional StylometricProfiler to replace distilgpt2 stats.
    When provided, compute_stats() fills burstiness, lexical diversity,
    avg_sentence_length etc. from the CPU-based profiler.
  - _bridge_stats_from_result gracefully handles sparse
    statistical_features dict (v3.3 detector only provides FK score).
  - Version string updated.
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
# Data classes
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
                    1.0,
                    base_score
                    + self.AI_INDICATOR_WORDS[clean_word] * 0.3,
                )
                confidence = 0.8
            elif clean_word in self.HUMAN_INDICATOR_WORDS:
                word_score = max(
                    0.0,
                    base_score
                    - self.HUMAN_INDICATOR_WORDS[clean_word] * 0.3,
                )
                confidence = 0.8
            else:
                word_score = base_score
                confidence = 0.5
            position_factor = 1.0 - (i / len(words)) * 0.1
            word_score = float(
                np.clip(word_score * position_factor, 0.0, 1.0)
            )
            attributions.append(
                WordAttribution(
                    word=word,
                    position=i,
                    ai_score=word_score,
                    confidence=confidence,
                    features={
                        "ai_indicator": clean_word
                        in self.AI_INDICATOR_WORDS,
                        "human_indicator": clean_word
                        in self.HUMAN_INDICATOR_WORDS,
                    },
                )
            )
        return attributions

    def calculate_sentence_attributions(
        self, text: str, overall_ai_score: float
    ) -> List[SentenceAttribution]:
        sentences = [
            s.strip() for s in re.split(r"[.!?]+", text) if s.strip()
        ]
        attributions: List[SentenceAttribution] = []
        for i, sentence in enumerate(sentences):
            word_attrs = self.calculate_word_attributions(
                sentence, overall_ai_score
            )
            if word_attrs:
                sentence_score = float(
                    np.mean([w.ai_score for w in word_attrs])
                )
                confidence = float(
                    np.mean([w.confidence for w in word_attrs])
                )
            else:
                sentence_score = overall_ai_score
                confidence = 0.5

            indicators: List[str] = []
            lower_sentence = sentence.lower()
            for word, strength in self.AI_INDICATOR_WORDS.items():
                if word in lower_sentence and strength > 0.7:
                    indicators.append(f"AI indicator: '{word}'")
            if re.search(
                r"\b(first|second|third|finally)\b", lower_sentence
            ):
                indicators.append("Sequential structure")
            if re.search(
                r"\b(therefore|thus|hence|consequently)\b", lower_sentence
            ):
                indicators.append("Logical connector")

            attributions.append(
                SentenceAttribution(
                    text=sentence,
                    position=i,
                    ai_score=sentence_score,
                    confidence=confidence,
                    word_attributions=word_attrs,
                    key_indicators=indicators,
                )
            )
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
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        words = [w.word for w in word_attributions]
        scores = [w.ai_score for w in word_attributions]
        x, y, line_height = 0.02, 0.95, 0.06

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        for word, score in zip(words, scores):
            color = self.cmap(score)
            text_obj = ax.text(
                x,
                y,
                word + " ",
                fontsize=10,
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=color,
                    alpha=0.7,
                    edgecolor="none",
                ),
                verticalalignment="top",
            )
            bbox_data = text_obj.get_window_extent(
                renderer=renderer
            ).transformed(ax.transData.inverted())
            x += bbox_data.width + 0.01
            if x > 0.95:
                x = 0.02
                y -= line_height
                if y < 0.1:
                    break

        legend_elements = [
            mpatches.Patch(
                facecolor="#2ecc71", label="Human-like (0.0-0.3)"
            ),
            mpatches.Patch(
                facecolor="#f1c40f", label="Uncertain (0.3-0.7)"
            ),
            mpatches.Patch(
                facecolor="#e74c3c", label="AI-like (0.7-1.0)"
            ),
        ]
        ax.legend(
            handles=legend_elements,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=9,
        )
        ax.set_title(
            "Word-Level AI Attribution Heatmap [HEURISTIC]",
            fontsize=14,
            fontweight="bold",
        )
        return fig

    def generate_sentence_chart(
        self,
        sentence_attributions: List[SentenceAttribution],
        figsize: Tuple[int, int] = (12, 6),
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        positions = range(len(sentence_attributions))
        scores = [s.ai_score for s in sentence_attributions]
        ax.bar(
            positions,
            scores,
            color=[self.cmap(s) for s in scores],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.axhline(
            y=0.5,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Decision threshold",
        )
        ax.set_xlabel("Sentence Position", fontsize=11)
        ax.set_ylabel("AI Probability Score", fontsize=11)
        ax.set_title(
            "Sentence-by-Sentence AI Detection Scores [HEURISTIC]",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right")
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        return fig

    def generate_comparison_chart(
        self,
        metrics: Dict[str, Tuple[float, float]],
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        fig, ax = plt.subplots(figsize=figsize)
        labels = list(metrics.keys())
        text_values = [metrics[l][0] for l in labels]
        baseline_values = [metrics[l][1] for l in labels]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(
            x - width / 2,
            text_values,
            width,
            label="Analysed Text",
            color="#3498db",
            edgecolor="white",
        )
        ax.bar(
            x + width / 2,
            baseline_values,
            width,
            label="Human Baseline",
            color="#2ecc71",
            edgecolor="white",
        )
        ax.set_xlabel("Metric", fontsize=11)
        ax.set_ylabel("Value", fontsize=11)
        ax.set_title(
            "Statistical Comparison vs Human Writing",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
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
    detector : SOTAAIDetector instance (optional if detection_result pre-built)
    profiler : StylometricProfiler instance (optional).
               When provided, compute_stats() populates burstiness, etc.
    hallucination_profiler : HallucinationProfiler instance (optional).
               Feature extractor. When provided, extracts 25-dim vector.
    hallucination_classifier : HallucinationRiskClassifier instance (optional).
               When provided, classifies risk from extracted features.
               When None but profiler is provided, a default classifier is used.
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
        # Auto-create classifier if profiler is provided but classifier isn't
        if hallucination_classifier is not None:
            self.hallucination_classifier = hallucination_classifier
        elif hallucination_profiler is not None:
            try:
                from hallucination_profiler import HallucinationRiskClassifier
                self.hallucination_classifier = HallucinationRiskClassifier()
            except ImportError:
                self.hallucination_classifier = None
        else:
            self.hallucination_classifier = None
        self.attribution_calc = AttributionCalculator()
        self.heatmap_gen = HeatmapGenerator()

    def _bridge_stats_from_result(
        self,
        text: str,
        detection_result: Optional[Any],
        additional_analyses: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Auto-populate additional_analyses["statistical"] from:
          1. StylometricProfiler.compute_stats() (preferred, CPU-based)
          2. detection_result.statistical_features (legacy fallback)
          3. Defaults

        Caller-provided values in additional_analyses always win.
        """
        additional = dict(additional_analyses) if additional_analyses else {}

        # Start with safe defaults
        merged_stat: Dict[str, Any] = {
            "ppl": 50.0,
            "burstiness": 0.1,
            "entropy": 0.0,
            "lexical_diversity": 0.5,
            "avg_sentence_length": 16.0,
            "sentence_length_variance": 0.0,
            "flesch_kincaid_grade": 0.0,
        }

        # Layer 1: StylometricProfiler (CPU, no GPU, replaces distilgpt2)
        if self.profiler is not None and text:
            try:
                profiler_stats = self.profiler.compute_stats(text)
                merged_stat["burstiness"] = profiler_stats.get(
                    "burstiness", merged_stat["burstiness"]
                )
                merged_stat["lexical_diversity"] = profiler_stats.get(
                    "lexical_diversity",
                    merged_stat["lexical_diversity"],
                )
                merged_stat["avg_sentence_length"] = profiler_stats.get(
                    "avg_sentence_length",
                    merged_stat["avg_sentence_length"],
                )
                merged_stat["sentence_length_variance"] = (
                    profiler_stats.get(
                        "sentence_length_variance",
                        merged_stat["sentence_length_variance"],
                    )
                )
            except Exception as exc:
                logger.warning(
                    "StylometricProfiler.compute_stats() failed: %s", exc
                )

        # Layer 2: detection_result.statistical_features (may be sparse in v3.3)
        if detection_result is not None:
            stats = getattr(
                detection_result, "statistical_features", None
            )
            if stats:
                for key in merged_stat:
                    if key in stats and stats[key] != 0.0:
                        merged_stat[key] = stats[key]

        # Layer 3: caller overrides win
        existing_stat = additional.get("statistical", {})
        merged_stat.update(existing_stat)
        additional["statistical"] = merged_stat
        return additional

    def generate_report(
        self,
        text: str,
        detection_result: Optional[Any] = None,
        additional_analyses: Optional[Dict[str, Any]] = None,
        generate_visuals: bool = True,
    ) -> ForensicReport:
        additional = self._bridge_stats_from_result(
            text, detection_result, additional_analyses
        )

        if detection_result is not None:
            raw_ai_pct = (detection_result.raw_scores or {}).get("ai")
            if raw_ai_pct is not None:
                overall_score = float(raw_ai_pct) / 100.0
            elif detection_result.prediction == "AI":
                overall_score = detection_result.confidence / 100.0
            else:
                overall_score = 1.0 - detection_result.confidence / 100.0

            overall_score = float(np.clip(overall_score, 0.0, 1.0))
            verdict = (
                "AI-Generated"
                if detection_result.prediction == "AI"
                else "Human-Written"
            )
            neural_score = overall_score
        else:
            overall_score, verdict, neural_score = 0.5, "Inconclusive", 0.5

        sentence_attrs = (
            self.attribution_calc.calculate_sentence_attributions(
                text, overall_score
            )
        )

        if len(sentence_attrs) >= 6:
            half = len(sentence_attrs) // 2
            first_half_mean = float(
                np.mean([s.ai_score for s in sentence_attrs[:half]])
            )
            second_half_mean = float(
                np.mean([s.ai_score for s in sentence_attrs[half:]])
            )
            delta = abs(first_half_mean - second_half_mean)
            if delta > 0.25 and (
                (first_half_mean > 0.5) != (second_half_mean > 0.5)
            ):
                verdict = "Hybrid"

        stat = additional.get("statistical", {})
        statistical_score = stat.get("score", 0.5)
        stylometric_score = additional.get("stylometric", {}).get(
            "similarity_score", 0.5
        )
        reasoning_score = additional.get("reasoning", {}).get(
            "ai_score", 0.5
        )
        watermark_score = additional.get("watermark", {}).get(
            "confidence", 0.0
        )

        fk_score = stat.get("flesch_kincaid_grade", 0.0)

        human_baseline: Dict[str, Tuple[float, float]] = {
            "Perplexity": (stat.get("ppl", 50.0), 55.0),
            "Burstiness (CV)": (stat.get("burstiness", 0.1), 0.25),
            "Lexical Diversity": (
                stat.get("lexical_diversity", 0.65),
                0.70,
            ),
            "Avg Sentence Length": (
                stat.get("avg_sentence_length", 18.0),
                16.0,
            ),
            "Flesch-Kincaid Grade": (fk_score, 10.0),
        }

        evidence_points = self._collect_evidence(sentence_attrs, additional)

        # Hallucination risk analysis (CPU, zero-resource)
        hallucination_risk: Optional[Dict[str, Any]] = None
        if (
            self.hallucination_profiler is not None
            and self.hallucination_classifier is not None
            and text
        ):
            try:
                hal_stats = self.hallucination_profiler.compute_stats(text)
                hallucination_risk = self.hallucination_classifier.classify(
                    hal_stats
                )
                if hallucination_risk.get("risk_level") == "HIGH":
                    evidence_points.append(
                        {
                            "type": "high_hallucination_risk",
                            "overall_risk": hallucination_risk["overall_risk"],
                            "risk_level": hallucination_risk["risk_level"],
                            "top_signals": hallucination_risk["top_signals"],
                            "interpretation": (
                                "Text exhibits multiple hallucination risk "
                                "indicators including vagueness, low entity "
                                "specificity, and/or semantic incoherence."
                            ),
                        }
                    )
            except Exception as exc:
                logger.warning(
                    "Hallucination analysis failed: %s", exc,
                )

        report_id = (
            hashlib.md5(
                f"{text[:100]}{datetime.now().isoformat()}".encode()
            )
            .hexdigest()[:12]
            .upper()
        )

        report = ForensicReport(
            report_id=report_id,
            generated_at=datetime.now().isoformat(),
            text_hash=hashlib.sha256(text.encode()).hexdigest()[:16],
            word_count=len(text.split()),
            verdict=verdict,
            confidence=overall_score,
            neural_score=neural_score,
            statistical_score=statistical_score,
            stylometric_score=stylometric_score,
            reasoning_score=reasoning_score,
            watermark_score=watermark_score,
            sentence_attributions=sentence_attrs,
            human_baseline_comparison=human_baseline,
            evidence_points=evidence_points,
            hallucination_risk=hallucination_risk,
        )

        if generate_visuals:
            all_word_attrs = [
                w for s in sentence_attrs for w in s.word_attributions
            ]
            report.heatmap_b64 = self._fig_to_base64(
                self.heatmap_gen.generate_word_heatmap(
                    all_word_attrs[:200]
                )
            )
            report.confidence_chart_b64 = self._fig_to_base64(
                self.heatmap_gen.generate_sentence_chart(sentence_attrs)
            )
            report.comparison_chart_b64 = self._fig_to_base64(
                self.heatmap_gen.generate_comparison_chart(human_baseline)
            )

        return report

    def _collect_evidence(
        self,
        sentence_attrs: List[SentenceAttribution],
        additional: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for attr in sentence_attrs:
            if attr.ai_score > 0.8 and attr.key_indicators:
                evidence.append(
                    {
                        "type": "high_confidence_ai_sentence",
                        "sentence_position": attr.position,
                        "score": attr.ai_score,
                        "indicators": attr.key_indicators,
                        "excerpt": (
                            attr.text[:100] + "..."
                            if len(attr.text) > 100
                            else attr.text
                        ),
                    }
                )
        stat = additional.get("statistical", {})
        if stat.get("ppl", 50) < 20:
            evidence.append(
                {
                    "type": "low_perplexity",
                    "value": stat.get("ppl"),
                    "human_baseline": 55,
                    "interpretation": (
                        "Text is highly predictable \u2014 "
                        "consistent with AI generation."
                    ),
                }
            )
        wm = additional.get("watermark", {})
        if wm.get("detected"):
            evidence.append(
                {
                    "type": "candidate_watermark_signal",
                    "scheme": wm.get("scheme_type"),
                    "confidence": wm.get("confidence"),
                    "note": "EXPERIMENTAL watermark signal.",
                }
            )
        return evidence

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    def export_html(
        self, report: ForensicReport, output_path: str
    ) -> str:
        verdict_class = (
            "ai"
            if "AI" in report.verdict
            else "human"
            if "Human" in report.verdict
            else "hybrid"
            if "Hybrid" in report.verdict
            else "inconclusive"
        )

        evidence_html = "".join(
            f'<div class="evidence">'
            f'<span class="evidence-title">'
            f'{e["type"].replace("_", " ").title()}</span>: '
            f"<pre>"
            f'{json.dumps({k: v for k, v in e.items() if k != "type"}, indent=2)}'
            f"</pre></div>"
            for e in report.evidence_points[:5]
        )

        sentence_rows = "".join(
            f"<tr>"
            f"<td>{s.position + 1}</td>"
            f'<td style="background: rgba('
            f"{int(s.ai_score * 255)},"
            f"{int((1 - s.ai_score) * 255)},100,0.3)\">"
            f"{s.ai_score:.2f}</td>"
            f'<td>{", ".join(s.key_indicators[:2]) or "\u2014"}</td>'
            f"<td>{s.text[:60]}...</td>"
            f"</tr>"
            for s in report.sentence_attributions[:10]
        )

        heatmap_img = (
            f'<img src="data:image/png;base64,{report.heatmap_b64}" '
            f'alt="Word Attribution Heatmap">'
            if report.heatmap_b64
            else "<p><em>Not generated</em></p>"
        )
        confidence_img = (
            f'<img src="data:image/png;base64,'
            f'{report.confidence_chart_b64}" '
            f'alt="Sentence Confidence Chart">'
            if report.confidence_chart_b64
            else "<p><em>Not generated</em></p>"
        )
        comparison_img = (
            f'<img src="data:image/png;base64,'
            f'{report.comparison_chart_b64}" '
            f'alt="Comparison Chart">'
            if report.comparison_chart_b64
            else "<p><em>Not generated</em></p>"
        )

        # Hallucination risk section
        if report.hallucination_risk is not None:
            hr = report.hallucination_risk
            risk_color = {
                "LOW": "#27ae60", "MEDIUM": "#f39c12", "HIGH": "#e74c3c",
            }.get(hr.get("risk_level", ""), "#7f8c8d")
            cat_rows = "".join(
                f"<tr><td>{cat.replace('_', ' ').title()}</td>"
                f"<td>{val:.2%}</td></tr>"
                for cat, val in sorted(
                    hr.get("category_scores", {}).items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            signals_html = "".join(
                f"<li><code>{s['feature']}</code> = {s['value']:.4f}</li>"
                for s in hr.get("top_signals", [])
            )
            hallucination_section = f"""
<h2>Hallucination Risk Analysis</h2>
<div style="text-align:center; margin:20px 0;">
<div class="metric" style="width:200px;">
<div class="metric-value" style="color:{risk_color};">{hr.get('overall_risk', 0):.0%}</div>
<div class="metric-label">Overall Risk ({hr.get('risk_level', 'N/A')})</div>
</div>
</div>
<table><tr><th>Category</th><th>Score</th></tr>{cat_rows}</table>
<p><strong>Top Signals:</strong></p><ul>{signals_html}</ul>
<p style="color:#7f8c8d; font-size:12px;"><em>Zero-resource analysis (no external knowledge base). Scores indicate statistical anomaly patterns, not confirmed factual errors.</em></p>
"""
        else:
            hallucination_section = ""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI Detection Forensic Report \u2014 {report.report_id}</title>
<style>
body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
.container {{ max-width: 900px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,.1); }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 30px; }}
.verdict {{ font-size: 24px; padding: 20px; border-radius: 8px; text-align: center; margin: 20px 0; }}
.verdict.ai {{ background: #fee; border: 2px solid #e74c3c; color: #c0392b; }}
.verdict.human {{ background: #efe; border: 2px solid #2ecc71; color: #27ae60; }}
.verdict.hybrid {{ background: #fef; border: 2px solid #9b59b6; color: #8e44ad; }}
.verdict.inconclusive {{ background: #eee; border: 2px solid #95a5a6; color: #7f8c8d; }}
.metric {{ display: inline-block; width: 150px; padding: 15px; margin: 10px; background: #f8f9fa; border-radius: 8px; text-align: center; }}
.metric-value {{ font-size: 28px; font-weight: bold; color: #2c3e50; }}
.metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
.chart {{ text-align: center; margin: 20px 0; }}
.chart img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
.evidence {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 8px; }}
.evidence pre {{ white-space: pre-wrap; font-size: 12px; margin: 6px 0 0; }}
.evidence-title {{ font-weight: bold; color: #856404; }}
.disclaimer {{ background: #fff3cd; border: 2px solid #fd7e14; padding: 16px; border-radius: 8px; margin: 20px 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
th {{ background: #3498db; color: white; }}
.footer {{ text-align: center; color: #7f8c8d; font-size: 12px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; }}
</style>
</head>
<body>
<div class="container">
<h1>AI Detection Forensic Report</h1>
<div class="disclaimer">
<strong>RESEARCH OUTPUT ONLY</strong><br>{_FORENSIC_DISCLAIMER}
</div>
<p>
<strong>Report ID:</strong> {report.report_id} |
<strong>Generated:</strong> {report.generated_at} |
<strong>Text Hash:</strong> {report.text_hash}
</p>
<div class="verdict {verdict_class}">
<strong>VERDICT: {report.verdict.upper()}</strong><br>
Confidence: {report.confidence * 100:.1f}%
</div>
<h2>Detection Scores</h2>
<div style="text-align:center;">
<div class="metric"><div class="metric-value">{report.neural_score * 100:.0f}%</div><div class="metric-label">Neural Score</div></div>
<div class="metric"><div class="metric-value">{report.statistical_score * 100:.0f}%</div><div class="metric-label">Statistical</div></div>
<div class="metric"><div class="metric-value">{report.reasoning_score * 100:.0f}%</div><div class="metric-label">Reasoning</div></div>
<div class="metric"><div class="metric-value">{report.watermark_score * 100:.0f}%</div><div class="metric-label">Watermark</div></div>
</div>
<h2>Word-Level Attribution <small>(Heuristic)</small></h2>
<div class="chart">{heatmap_img}</div>
<h2>Sentence Analysis <small>(Heuristic)</small></h2>
<div class="chart">{confidence_img}</div>
<h2>Statistical Comparison</h2>
<div class="chart">{comparison_img}</div>
{hallucination_section}
<h2>Key Evidence</h2>
{evidence_html}
<h2>Sentence Breakdown</h2>
<table>
<tr><th>#</th><th>AI Score</th><th>Indicators</th><th>Excerpt</th></tr>
{sentence_rows}
</table>
<div class="footer">
<p>SOTA AI Detector v3.3 | Word Count: {report.word_count} | {report.generated_at}</p>
<p>{_FORENSIC_DISCLAIMER}</p>
</div>
</div>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info("HTML report exported: %s", output_path)
        return output_path

    def export_json(
        self, report: ForensicReport, output_path: str
    ) -> str:
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
                {
                    "position": s.position,
                    "score": s.ai_score,
                    "indicators": s.key_indicators,
                }
                for s in report.sentence_attributions
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("JSON report exported: %s", output_path)
        return output_path
