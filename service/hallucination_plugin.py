"""
plugins/hallucination_plugin.py
================================
Plugin wrapper for HallucinationProfiler + HallucinationRiskClassifier.

Output key: ``"hallucination"``
Passed through to ForensicReportGenerator as
``additional_analyses["hallucination"]`` and stored on ForensicReport
as ``hallucination_risk``.

Note: ForensicReportGenerator also runs hallucination analysis internally
when hallucination_profiler/classifier are injected via its constructor.
When this plugin is used, pass neither to ForensicReportGenerator — the
ForensicPlugin reads the pre-computed result from PluginContext instead,
avoiding double-computation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from plugin_base import InferencePlugin, PluginContext

logger = logging.getLogger(__name__)


class HallucinationPlugin(InferencePlugin):
    """
    Wraps ``HallucinationProfiler.compute_stats()`` +
    ``HallucinationRiskClassifier.classify()`` into the plugin interface.

    Output schema (``context.plugin_results["hallucination"]``)::

        {
            "overall_risk":    float,       # [0.0, 1.0]
            "risk_level":      str,         # "LOW" | "MEDIUM" | "HIGH"
            "category_scores": Dict[str, float],   # 8 categories
            "top_signals":     List[Dict],  # top 3 feature signals
            "feature_details": Dict[str, float],   # all 25 features
        }

    Parameters
    ----------
    config : HallucinationRiskConfig — weights/thresholds. None = defaults.
    nlp    : optional spaCy Language model for richer entity/syntactic features.
    """

    name = "hallucination"

    def __init__(self, config: Optional[Any] = None, nlp: Optional[Any] = None) -> None:
        self._config = config
        self._nlp = nlp
        self._profiler: Optional[Any]    = None
        self._classifier: Optional[Any]  = None
        self._available: Optional[bool]  = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from hallucination_profile import HallucinationProfiler  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.warning("HallucinationPlugin: hallucination_profile not found — disabled.")
        return self._available  # type: ignore[return-value]

    def warmup(self) -> None:
        if not self.is_available():
            return
        if self._profiler is None:
            from hallucination_profile import (
                HallucinationProfiler,
                HallucinationRiskClassifier,
            )
            self._profiler   = HallucinationProfiler(nlp=self._nlp)
            self._classifier = HallucinationRiskClassifier(self._config)
            logger.info("HallucinationPlugin warmup complete.")

    # ------------------------------------------------------------------

    def run(self, context: PluginContext) -> Dict[str, Any]:
        if self._profiler is None:
            self.warmup()
        if self._profiler is None or self._classifier is None:
            return {"overall_risk": 0.0, "risk_level": "LOW", "top_signals": []}

        try:
            stats = self._profiler.compute_stats(context.text)
            return self._classifier.classify(stats)
        except Exception as exc:
            logger.warning("HallucinationPlugin failed: %s", exc)
            return {"overall_risk": 0.0, "risk_level": "LOW", "top_signals": [], "error": str(exc)}
