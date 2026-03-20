"""
unified_result.py
==================
``UnifiedDetectionResult`` — the single output object of the full
xota_ensemble_v6 inference pipeline.

It is assembled from a completed ``PluginContext`` by calling
``UnifiedDetectionResult.from_context(context)``.

Design goals
------------
- One object to rule them all: replaces the ad-hoc ``(str, Figure)``
  tuple previously returned by ``classify_text()``.
- JSON-safe: ``to_dict()`` produces a plain dict with no tensors,
  no ndarray, no dataclass instances — ready for ``json.dumps()``.
- Gradio-safe: ``to_markdown()`` produces the same display string
  the old ``classify_text()`` returned.
- Lossless: the live ``ForensicReport`` object is kept in
  ``forensic_report`` for in-process consumers that need chart bytes
  or the full sentence attribution list.

Schema
------
Core (always populated):
    prediction              str         "Human" | "AI" | "Unknown"
    confidence              float       0–100 (raw from detectot_final)
    top_ai_model            str         highest-probability non-human label
    human_probability       float       raw probability (0–1)
    ai_probability          float       raw probability (0–1)

Plugin outputs (None if plugin was unavailable or failed):
    stylometric             Dict[str,float] | None  compute_stats keys
    reasoning_vector        List[float] | None      15-dim raw vector
    reasoning_ai_score      float | None            derived scalar [0,1]
    hallucination_risk      Dict | None             classify() output
    watermark               Dict | None             to_forensic_dict() output
    forensic_verdict        str | None              report.verdict
    forensic_report_id      str | None              report.report_id
    forensic_report         ForensicReport | None   live object

Diagnostics:
    plugin_errors           Dict[str, str]          {name: error_message}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class UnifiedDetectionResult:
    """
    Single output object from the full inference + plugin pipeline.

    Construct via ``UnifiedDetectionResult.from_context(context)``
    after ``PluginPipeline.execute(context)`` has run.
    """

    # ----------------------------------------------------------------
    # Core ensemble output
    # ----------------------------------------------------------------
    prediction: str                     # "Human" | "AI" | "Unknown"
    confidence: float                   # 0–100
    top_ai_model: str
    human_probability: float            # raw (0–1)
    ai_probability: float               # raw (0–1)

    # ----------------------------------------------------------------
    # Plugin outputs
    # ----------------------------------------------------------------
    stylometric: Optional[Dict[str, Any]] = None
    reasoning_vector: Optional[List[float]] = None
    reasoning_ai_score: Optional[float] = None
    hallucination_risk: Optional[Dict[str, Any]] = None
    watermark: Optional[Dict[str, Any]] = None

    # Forensic aggregation
    forensic_verdict: Optional[str] = None
    forensic_report_id: Optional[str] = None
    forensic_report: Optional[Any] = None   # ForensicReport object

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------
    plugin_errors: Dict[str, str] = field(default_factory=dict)

    # ----------------------------------------------------------------
    # Factory
    # ----------------------------------------------------------------

    @classmethod
    def from_context(cls, context: Any) -> "UnifiedDetectionResult":
        """
        Assemble a ``UnifiedDetectionResult`` from a completed
        ``PluginContext`` (after ``PluginPipeline.execute()``).

        Parameters
        ----------
        context : PluginContext
            The context object returned by ``PluginPipeline.execute()``.
            All ``plugin_results`` entries are read here.
        """
        pr = context.plugin_results

        # --- Stylometric ---
        stylometric = pr.get("statistical") or None

        # --- Reasoning ---
        reasoning_raw = pr.get("reasoning")
        reasoning_vector: Optional[List[float]] = None
        reasoning_ai_score: Optional[float] = None
        if reasoning_raw and "error" not in reasoning_raw:
            reasoning_vector = reasoning_raw.get("feature_vector")
            reasoning_ai_score = reasoning_raw.get("ai_score")

        # --- Hallucination ---
        hal_raw = pr.get("hallucination")
        hallucination_risk: Optional[Dict[str, Any]] = None
        if hal_raw and "error" not in hal_raw:
            hallucination_risk = hal_raw

        # --- Watermark ---
        wm_raw = pr.get("watermark")
        watermark: Optional[Dict[str, Any]] = None
        if wm_raw and "error" not in wm_raw:
            # Strip the live torch/numpy objects if any snuck in
            watermark = {
                k: v for k, v in wm_raw.items()
                if isinstance(v, (bool, int, float, str, type(None)))
            }

        # --- Forensic ---
        forensic_raw = pr.get("forensic_report")
        forensic_verdict: Optional[str] = None
        forensic_report_id: Optional[str] = None
        forensic_report_obj: Optional[Any] = None
        if forensic_raw:
            forensic_verdict = forensic_raw.get("verdict")
            forensic_report_id = forensic_raw.get("report_id")
            forensic_report_obj = forensic_raw.get("forensic_report")

        return cls(
            prediction=context.prediction,
            confidence=context.confidence,
            top_ai_model=context.top_ai_model,
            human_probability=context.human_probability,
            ai_probability=context.ai_probability,
            stylometric=stylometric,
            reasoning_vector=reasoning_vector,
            reasoning_ai_score=reasoning_ai_score,
            hallucination_risk=hallucination_risk,
            watermark=watermark,
            forensic_verdict=forensic_verdict,
            forensic_report_id=forensic_report_id,
            forensic_report=forensic_report_obj,
            plugin_errors=dict(context.errors),
        )

    # ----------------------------------------------------------------
    # Serialisation
    # ----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """
        Return a JSON-safe plain dict.

        Excludes ``forensic_report`` (not JSON-safe — contains
        matplotlib figures and numpy arrays).
        """
        hal_safe = None
        if self.hallucination_risk:
            # Drop feature_details if too large; keep summary keys
            hal_safe = {
                k: v for k, v in self.hallucination_risk.items()
                if k != "feature_details"
            }

        return {
            "prediction":          self.prediction,
            "confidence":          round(self.confidence, 2),
            "top_ai_model":        self.top_ai_model,
            "human_probability":   round(self.human_probability * 100, 2),
            "ai_probability":      round(self.ai_probability * 100, 2),
            "stylometric":         self.stylometric,
            "reasoning_ai_score":  (
                round(self.reasoning_ai_score, 4)
                if self.reasoning_ai_score is not None else None
            ),
            "reasoning_vector":    self.reasoning_vector,
            "hallucination_risk":  hal_safe,
            "watermark":           self.watermark,
            "forensic_verdict":    self.forensic_verdict,
            "forensic_report_id":  self.forensic_report_id,
            "plugin_errors":       self.plugin_errors,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ----------------------------------------------------------------
    # Gradio display helpers
    # ----------------------------------------------------------------

    def to_markdown(self) -> str:
        """
        Produce the same Markdown display string as the original
        ``classify_text()`` — backward-compatible Gradio output.
        """
        human_pct = self.human_probability / (
            self.human_probability + self.ai_probability + 1e-9
        ) * 100
        ai_pct = 100.0 - human_pct

        if self.prediction == "Human":
            base = (
                f"**The text is** <span class='highlight-human'>"
                f"**{human_pct:.2f}%** likely <b>Human written</b>.</span>"
            )
        elif self.prediction == "AI":
            base = (
                f"**The text is** <span class='highlight-ai'>"
                f"**{ai_pct:.2f}%** likely <b>AI generated</b>.</span>"
            )
        else:
            return "**Unknown** — text too short or invalid."

        parts = [base]

        # Append plugin signals as an expandable summary
        signals: List[str] = []

        if self.forensic_verdict and self.forensic_verdict != self.prediction:
            signals.append(f"📋 Forensic verdict: **{self.forensic_verdict}**")

        if self.hallucination_risk:
            level = self.hallucination_risk.get("risk_level", "?")
            risk = self.hallucination_risk.get("overall_risk", 0.0)
            emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}.get(level, "⚪")
            signals.append(
                f"{emoji} Hallucination risk: **{level}** ({risk:.0%})"
            )

        if self.watermark and self.watermark.get("detected"):
            scheme = self.watermark.get("scheme_type", "unknown")
            conf = self.watermark.get("confidence", 0.0)
            signals.append(
                f"🔏 Watermark signal detected — scheme: {scheme} "
                f"({conf:.0%} confidence) *(EXPERIMENTAL)*"
            )

        if self.reasoning_ai_score is not None and self.reasoning_ai_score > 0.4:
            signals.append(
                f"🧠 Reasoning markers: AI score {self.reasoning_ai_score:.2f}"
            )

        if self.plugin_errors:
            errs = ", ".join(self.plugin_errors.keys())
            signals.append(f"⚠️ Plugin warnings: {errs}")

        if signals:
            parts.append("\n\n" + "\n\n".join(signals))

        return "".join(parts)

    def to_short_summary(self) -> str:
        """
        One-line summary for logging / batch output files.
        """
        return (
            f"[{self.prediction}] conf={self.confidence:.1f}% "
            f"model={self.top_ai_model} "
            f"forensic={self.forensic_verdict or 'N/A'} "
            f"hal={self.hallucination_risk.get('risk_level', 'N/A') if self.hallucination_risk else 'N/A'} "
            f"wm={'YES' if self.watermark and self.watermark.get('detected') else 'NO'} "
            f"errors={len(self.plugin_errors)}"
        )
