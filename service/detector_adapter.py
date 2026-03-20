"""
detector_adapter.py
===================
Adapters bridging detectot_final.py's raw output to the plugin pipeline.

Two public classes
------------------
``DetectorFinalAdapter``
    Wraps the averaged softmax probability tensor produced by
    detectot_final.py into an object that:
      (a) satisfies the duck-typed interface consumed by
          ``forensic_reports.ForensicReportGenerator`` (attributes:
          ``prediction``, ``confidence``, ``raw_scores``,
          ``statistical_features``);
      (b) populates a ``PluginContext`` ready for pipeline execution.

``UnifiedDetectionResult``
    Single output object that aggregates the neural detection result
    with all plugin outputs after pipeline execution.  Serialisable
    to dict/JSON for API responses and Gradio display.

Fixes applied
-------------
  [FIX P0-B2]  All torch operations go through no_grad context.
               detectot_final.py omits ``torch.no_grad()`` around its
               4× forward passes — calling code should also wrap adapter
               construction to avoid unnecessary autograd overhead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from plugin_base import PluginContext

logger = logging.getLogger(__name__)

# Label mapping from detectot_final.py — duplicated here so the adapter
# is self-contained.  Must stay in sync with detectot_final.LABEL_MAPPING.
_LABEL_MAPPING: Dict[int, str] = {
    0: "13B", 1: "30B", 2: "65B", 3: "7B", 4: "GLM130B",
    5: "bloom_7b", 6: "bloomz", 7: "cohere", 8: "davinci",
    9: "dolly", 10: "dolly-v2-12b", 11: "flan_t5_base",
    12: "flan_t5_large", 13: "flan_t5_small", 14: "flan_t5_xl",
    15: "flan_t5_xxl", 16: "gemma-7b-it", 17: "gemma2-9b-it",
    18: "gpt-3.5-turbo", 19: "gpt-35", 20: "gpt4", 21: "gpt4o",
    22: "gpt_j", 23: "gpt_neox", 24: "human",
    25: "llama3-70b", 26: "llama3-8b", 27: "mixtral-8x7b",
    28: "opt_1.3b", 29: "opt_125m", 30: "opt_13b", 31: "opt_2.7b",
    32: "opt_30b", 33: "opt_350m", 34: "opt_6.7b",
    35: "opt_iml_30b", 36: "opt_iml_max_1.3b",
    37: "t0_11b", 38: "t0_3b",
    39: "text-davinci-002", 40: "text-davinci-003",
}

_HUMAN_IDX: int = 24  # index of "human" class in the 41-class label space


# ---------------------------------------------------------------------------
# DetectorFinalAdapter
# ---------------------------------------------------------------------------

class DetectorFinalAdapter:
    """
    Wraps a (41,) averaged softmax tensor from detectot_final into an object
    that satisfies the ForensicReportGenerator duck-typed interface AND
    can seed a PluginContext.

    Parameters
    ----------
    averaged_probs : torch.Tensor, shape (41,)
        Output of ``(softmax_1 + ... + softmax_4) / 4`` from detectot_final.
    label_mapping  : Dict[int, str]
        Maps class index → model name.  Defaults to the 41-class mapping
        bundled in this module.

    Attributes exposed for ForensicReportGenerator
    -----------------------------------------------
    prediction          : "AI" | "Human"
    confidence          : float 0–100 (binary renormalised)
    raw_scores          : Dict[str, float]  — keys "human", "ai", "class_*"
    statistical_features: Dict[str, float]  — populated by plugins later
    detected_model      : str               — top AI model name
    """

    def __init__(
        self,
        averaged_probs: torch.Tensor,
        label_mapping: Optional[Dict[int, str]] = None,
    ) -> None:
        if label_mapping is None:
            label_mapping = _LABEL_MAPPING

        # Detach and move to CPU — all further ops are numpy/python
        with torch.no_grad():
            probs_cpu = averaged_probs.detach().cpu().float()

        human_prob = probs_cpu[_HUMAN_IDX].item()

        ai_probs = probs_cpu.clone()
        ai_probs[_HUMAN_IDX] = 0.0
        ai_total = ai_probs.sum().item()

        total = human_prob + ai_total
        if total < 1e-9:
            total = 1.0  # guard against all-zero tensor

        # Binary renormalised percentages (matches detectot_final.py logic)
        human_pct = (human_prob / total) * 100.0
        ai_pct    = (ai_total  / total) * 100.0

        self.prediction:   str   = "Human" if human_pct > ai_pct else "AI"
        self.confidence:   float = max(human_pct, ai_pct)

        ai_argmax = int(torch.argmax(ai_probs).item())
        self.detected_model: str = label_mapping.get(ai_argmax, "unknown")

        # raw_scores: compatible with ForensicReportGenerator expectations
        self.raw_scores: Dict[str, float] = {
            "human": human_pct,
            "ai":    ai_pct,
        }
        for idx, name in label_mapping.items():
            self.raw_scores[f"class_{name}"] = probs_cpu[idx].item() * 100.0

        # statistical_features populated externally by pipeline plugins
        self.statistical_features: Dict[str, float] = {}

        # Store raw values for PluginContext
        self._human_probability: float = human_prob
        self._ai_probability:    float = ai_total
        self._probs_tensor: torch.Tensor = probs_cpu

    # ------------------------------------------------------------------
    # PluginContext factory
    # ------------------------------------------------------------------

    def to_context(self, text: str, cleaned_text: str) -> PluginContext:
        """
        Create a PluginContext seeded with neural inference results.

        Parameters
        ----------
        text         : original (uncleaned) input text
        cleaned_text : output of clean_text(text) from detectot_final
        """
        return PluginContext(
            text=text,
            cleaned_text=cleaned_text,
            probabilities=self._probs_tensor,
            human_probability=self._human_probability,
            ai_probability=self._ai_probability,
            top_ai_model=self.detected_model,
            prediction=self.prediction,
            confidence=self.confidence,
        )

    def __repr__(self) -> str:
        return (
            f"DetectorFinalAdapter("
            f"prediction={self.prediction!r}, "
            f"confidence={self.confidence:.1f}%, "
            f"top_ai_model={self.detected_model!r})"
        )


# ---------------------------------------------------------------------------
# UnifiedDetectionResult
# ---------------------------------------------------------------------------

@dataclass
class UnifiedDetectionResult:
    """
    Single aggregated output object produced after the full plugin pipeline.

    Fields
    ------
    prediction          : "AI" | "Human" | "Unknown"
    confidence          : float 0–100
    top_ai_model        : most likely AI model name (if prediction=="AI")
    stylometric         : StylometricPlugin output dict (or None)
    reasoning_vector    : 15-element feature list from ReasoningPlugin
    reasoning_ai_score  : derived scalar [0, 1] from reasoning vector
    hallucination_risk  : HallucinationPlugin classify() output (or None)
    watermark           : WatermarkPlugin to_forensic_dict() output (or None)
    forensic_report_meta: lightweight dict from ForensicPlugin (report_id, verdict)
    plugin_errors       : names → error messages for failed plugins
    """

    prediction:           str
    confidence:           float
    top_ai_model:         str

    stylometric:          Optional[Dict[str, float]] = None
    reasoning_vector:     Optional[List[float]]      = None
    reasoning_ai_score:   Optional[float]            = None
    hallucination_risk:   Optional[Dict[str, Any]]   = None
    watermark:            Optional[Dict[str, Any]]   = None
    forensic_report_meta: Optional[Dict[str, Any]]   = None
    plugin_errors:        Dict[str, str]             = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_context(
        cls,
        adapter: DetectorFinalAdapter,
        context: PluginContext,
    ) -> "UnifiedDetectionResult":
        """
        Build a UnifiedDetectionResult from a completed PluginContext.

        Extracts known plugin outputs by name.  Unknown plugin names are
        silently ignored — the pipeline is open to extension.
        """
        # Statistical plugin (name = "statistical")
        stat = context.plugin_results.get("statistical")

        # Reasoning plugin (name = "reasoning")
        reasoning = context.plugin_results.get("reasoning")
        reasoning_vector    = reasoning.get("feature_vector")   if reasoning else None
        reasoning_ai_score  = reasoning.get("ai_score")         if reasoning else None

        # Hallucination plugin (name = "hallucination")
        hallucination = context.plugin_results.get("hallucination")

        # Watermark plugin (name = "watermark")
        watermark = context.plugin_results.get("watermark")

        # Forensic plugin (name = "forensic_report")
        forensic_meta = context.plugin_results.get("forensic_report")

        return cls(
            prediction=         adapter.prediction,
            confidence=         adapter.confidence,
            top_ai_model=       adapter.detected_model,
            stylometric=        stat,
            reasoning_vector=   reasoning_vector,
            reasoning_ai_score= reasoning_ai_score,
            hallucination_risk= hallucination,
            watermark=          watermark,
            forensic_report_meta=forensic_meta,
            plugin_errors=      dict(context.errors),
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable dict."""
        return {
            "prediction":           self.prediction,
            "confidence":           round(self.confidence, 2),
            "top_ai_model":         self.top_ai_model,
            "stylometric":          self.stylometric,
            "reasoning_vector":     self.reasoning_vector,
            "reasoning_ai_score":   (
                round(self.reasoning_ai_score, 4)
                if self.reasoning_ai_score is not None else None
            ),
            "hallucination_risk":   self.hallucination_risk,
            "watermark":            self.watermark,
            "forensic_report_meta": self.forensic_report_meta,
            "plugin_errors":        self.plugin_errors,
        }

    def to_gradio_markdown(self) -> str:
        """
        Render a rich markdown string for the Gradio interface.
        Extends the original detectot_final.py result_message with plugin signals.
        """
        lines: List[str] = []

        # Core verdict
        if self.prediction == "Human":
            lines.append(
                f"**The text is** "
                f"<span class='highlight-human'>**{self.confidence:.2f}%** likely "
                f"<b>Human written</b>.</span>"
            )
        else:
            lines.append(
                f"**The text is** "
                f"<span class='highlight-ai'>**{self.confidence:.2f}%** likely "
                f"<b>AI generated</b>.</span>"
            )
            if self.top_ai_model and self.top_ai_model != "unknown":
                lines.append(f"\n**Identified model:** `{self.top_ai_model}`")

        # Reasoning signal
        if self.reasoning_ai_score is not None:
            lines.append(
                f"\n**Reasoning signal:** {self.reasoning_ai_score:.3f} "
                f"{'⚠ high CoT pattern' if self.reasoning_ai_score > 0.5 else '✓ low CoT pattern'}"
            )

        # Hallucination risk
        if self.hallucination_risk:
            level = self.hallucination_risk.get("risk_level", "?")
            risk  = self.hallucination_risk.get("overall_risk", 0.0)
            emoji = {"LOW": "✅", "MEDIUM": "⚠️", "HIGH": "🚨"}.get(level, "")
            lines.append(f"\n**Hallucination risk:** {emoji} {level} ({risk:.1%})")

        # Watermark
        if self.watermark:
            if self.watermark.get("detected"):
                scheme = self.watermark.get("scheme_type", "?")
                conf   = self.watermark.get("confidence", 0.0)
                lines.append(
                    f"\n**⚠ Candidate watermark** detected "
                    f"(scheme: `{scheme}`, confidence: {conf:.1%}) — EXPERIMENTAL"
                )

        # Plugin errors
        if self.plugin_errors:
            failed = ", ".join(self.plugin_errors.keys())
            lines.append(f"\n_Plugins with errors: {failed}_")

        return "\n".join(lines)
