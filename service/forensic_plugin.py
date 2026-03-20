"""
plugins/forensic_plugin.py
===========================
Aggregator plugin — assembles all prior plugin results into a
``ForensicReport`` via ``ForensicReportGenerator.generate_report()``.

Position in DAG
---------------
This plugin runs LAST.  It declares dependencies on all four upstream
plugins so the topological sort guarantees they have completed first::

    requires = ["statistical", "reasoning", "hallucination", "watermark"]

If any upstream plugin failed, its result dict contains an ``"error"``
key but still exists in ``context.plugin_results`` — ForensicPlugin
consumes whatever is available and delegates graceful-degradation to
``ForensicReportGenerator._bridge_stats_from_result()``.

Output schema:
    {
        "report_id":         str,
        "verdict":           str,    # "AI-Generated" | "Human-Written" | "Hybrid"
        "confidence":        float,  # 0-1
        "neural_score":      float,
        "hallucination_risk":Dict | None,
        "evidence_count":    int,
        "forensic_report":   ForensicReport,  # live object (not JSON-safe)
    }
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from plugin_base import InferencePlugin, PluginContext

logger = logging.getLogger(__name__)


class ForensicPlugin(InferencePlugin):
    """
    Aggregator plugin that drives ``ForensicReportGenerator``.

    Collects all upstream plugin results from ``PluginContext.plugin_results``
    and builds the final ``ForensicReport``.

    Parameters
    ----------
    generate_visuals : bool
        Whether to render matplotlib heatmaps and charts.
        Default ``False`` for API/batch use; set ``True`` for Gradio UI.
    profiler : StylometricProfiler, optional
        Injected into ``ForensicReportGenerator`` for its internal
        ``_bridge_stats_from_result`` fallback chain.
    hallucination_profiler : HallucinationProfiler, optional
        Injected into ``ForensicReportGenerator``.
    """

    _name = "forensic_report"

    def __init__(
        self,
        generate_visuals: bool = False,
        profiler: Optional[Any] = None,
        hallucination_profiler: Optional[Any] = None,
    ) -> None:
        self._generate_visuals = generate_visuals
        self._profiler = profiler
        self._hallucination_profiler = hallucination_profiler
        self._generator = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> List[str]:
        return ["statistical", "reasoning", "hallucination", "watermark"]

    def is_available(self) -> bool:
        try:
            import forensic_reports  # noqa: F401
            return True
        except ImportError:
            return False

    def warmup(self) -> None:
        from forensic_reports import ForensicReportGenerator
        self._generator = ForensicReportGenerator(
            profiler=self._profiler,
            hallucination_profiler=self._hallucination_profiler,
        )
        logger.info("ForensicPlugin: ForensicReportGenerator initialised")

    def run(self, context: PluginContext) -> Dict[str, Any]:
        if self._generator is None:
            try:
                self.warmup()
            except Exception as exc:
                return {"error": f"ForensicPlugin init failed: {exc}"}

        # Build DetectorFinalAdapter from context
        try:
            from detector_adapter import DetectorFinalAdapter
            adapter = DetectorFinalAdapter(
                probabilities=context.probabilities,
                label_mapping=context.label_mapping,
                human_idx=context.human_idx,
            )
            adapter.statistical_features = context.plugin_results.get(
                "statistical", {}
            )
        except Exception as exc:
            logger.warning("ForensicPlugin: DetectorFinalAdapter failed: %s", exc)
            adapter = None

        # Assemble additional_analyses from all upstream plugin results
        additional_analyses: Dict[str, Any] = {
            k: v for k, v in context.plugin_results.items()
            if k != self._name
        }

        try:
            report = self._generator.generate_report(
                text=context.text,
                detection_result=adapter,
                additional_analyses=additional_analyses,
                generate_visuals=self._generate_visuals,
            )
        except Exception as exc:
            logger.error("ForensicPlugin.generate_report() failed: %s", exc)
            return {
                "error": str(exc),
                "report_id": None,
                "verdict": context.prediction,
                "confidence": context.confidence / 100.0,
            }

        logger.info(
            "ForensicPlugin: report %s — verdict=%s confidence=%.1f%%",
            report.report_id, report.verdict, report.confidence * 100,
        )

        return {
            "report_id":          report.report_id,
            "verdict":            report.verdict,
            "confidence":         report.confidence,
            "neural_score":       report.neural_score,
            "hallucination_risk": report.hallucination_risk,
            "evidence_count":     len(report.evidence_points),
            "forensic_report":    report,  # live object for UnifiedDetectionResult
        }
