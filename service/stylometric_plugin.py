"""
plugins/stylometric_plugin.py
==============================
Plugin wrapper for StylometricProfiler.compute_stats().

Output key: ``"statistical"``
Maps to ``additional_analyses["statistical"]`` consumed by
ForensicReportGenerator._bridge_stats_from_result().

Key bridging
------------
StylometricProfiler.compute_stats() returns 11 keys.
ForensicReportGenerator expects:
  "burstiness", "lexical_diversity", "avg_sentence_length",
  "sentence_length_variance", "ppl", "entropy", "flesch_kincaid_grade"

"ppl" and "entropy" are not produced by StylometricProfiler (they
require a language model); we pass the forensic_reports defaults
(50.0 and 0.0) so the fallback chain in _bridge_stats_from_result
stays consistent.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from plugin_base import InferencePlugin, PluginContext

logger = logging.getLogger(__name__)


class StylometricPlugin(InferencePlugin):
    """
    Wraps ``StylometricProfiler.compute_stats(text)`` into the plugin interface.

    Output schema (``context.plugin_results["statistical"]``)::

        {
            "burstiness":               float,   # BST ∈ [-1, +1]
            "lexical_diversity":         float,   # MATTR adaptive
            "avg_sentence_length":       float,
            "sentence_length_variance":  float,
            "avg_word_length":           float,
            "vocabulary_richness":       float,
            "hapax_legomena_ratio":      float,
            "rare_word_ratio":           float,
            "comma_rate":                float,
            "avg_dep_distance":          float,   # 0.0 if spaCy absent
            "complex_sentence_ratio":    float,   # 0.0 if spaCy absent
            # Placeholders required by ForensicReportGenerator:
            "ppl":                       50.0,    # not computable without LM
            "entropy":                   0.0,     # not computable without LM
            "flesch_kincaid_grade":      0.0,     # populated by EnsembleDetector
        }

    Parameters
    ----------
    nlp : optional spaCy Language model.
          When None, module-level _NLP fallback is used.
    """

    name = "statistical"

    def __init__(self, nlp: Optional[Any] = None) -> None:
        self._nlp = nlp
        self._profiler: Optional[Any] = None
        self._available: Optional[bool] = None

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from stylometric_profiler import StylometricProfiler  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.warning(
                    "StylometricPlugin: stylometric_profiler not found — "
                    "plugin disabled."
                )
        return self._available  # type: ignore[return-value]

    def warmup(self) -> None:
        if not self.is_available():
            return
        if self._profiler is None:
            from stylometric_profiler import StylometricProfiler
            self._profiler = StylometricProfiler(nlp=self._nlp)
            logger.info("StylometricPlugin warmup complete.")

    # ------------------------------------------------------------------

    def run(self, context: PluginContext) -> Dict[str, Any]:
        if self._profiler is None:
            self.warmup()
        if self._profiler is None:
            return {}

        try:
            stats = self._profiler.compute_stats(context.text)
        except Exception as exc:
            logger.warning("StylometricPlugin.compute_stats() failed: %s", exc)
            return {}

        return {
            # Keys consumed by forensic_reports._bridge_stats_from_result
            "burstiness":              stats.get("burstiness", 0.0),
            "lexical_diversity":       stats.get("lexical_diversity", 0.5),
            "avg_sentence_length":     stats.get("avg_sentence_length", 16.0),
            "sentence_length_variance":stats.get("sentence_length_variance", 0.0),
            # Extended keys (passed through to forensic report)
            "avg_word_length":         stats.get("avg_word_length", 4.5),
            "vocabulary_richness":     stats.get("vocabulary_richness", 0.5),
            "hapax_legomena_ratio":    stats.get("hapax_legomena_ratio", 0.5),
            "rare_word_ratio":         stats.get("rare_word_ratio", 0.3),
            "comma_rate":              stats.get("comma_rate", 0.0),
            "avg_dep_distance":        stats.get("avg_dep_distance", 0.0),
            "complex_sentence_ratio":  stats.get("complex_sentence_ratio", 0.0),
            # Placeholders — forensic_reports defaults
            "ppl":                     50.0,
            "entropy":                 0.0,
            "flesch_kincaid_grade":    0.0,
        }
