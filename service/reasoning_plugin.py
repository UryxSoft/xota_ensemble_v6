"""
plugins/reasoning_plugin.py
============================
Plugin wrapper for ReasoningProfiler.vectorize().

Output key: ``"reasoning"``
Maps to ``additional_analyses["reasoning"]`` consumed by
ForensicReportGenerator (expects key ``"ai_score": float``).

Scalar derivation
-----------------
ReasoningProfiler returns a 15-dim vector — ForensicReportGenerator
expects a single ``ai_score`` float in [0, 1].

Formula (calibrated on o1/DeepSeek-R1 trace analysis):
  ai_score = clip(
      0.40 * cot_scaffold_density     [idx 11]  — CoT patterns → AI
    + 0.40 * backtracking_density     [idx 10]  — self-correction → AI
    - 0.20 * intuition_leap_density   [idx 12]  — heuristic leaps → human
    , 0.0, 1.0
  )

The raw densities are scaled by ×10 before clipping because the
densities are typically in [0, 0.1] for normal text — the ×10 maps
them into the [0, 1] display range.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from plugin_base import InferencePlugin, PluginContext

logger = logging.getLogger(__name__)

# Index constants (from REASONING_VECTOR_DIM schema)
_IDX_BACKTRACK    = 10
_IDX_COT_SCAFFOLD = 11
_IDX_INTUITION    = 12
_SCALE            = 10.0   # density → [0,1] scaling factor


class ReasoningPlugin(InferencePlugin):
    """
    Wraps ``ReasoningProfiler.vectorize(text)`` into the plugin interface.

    Output schema (``context.plugin_results["reasoning"]``)::

        {
            "ai_score":        float,           # derived scalar [0, 1]
            "feature_vector":  List[float],     # 15-dim raw vector
            "features":        Dict[str, float],# named feature dict
            "cot_density":     float,
            "backtrack_density": float,
            "intuition_density": float,
        }
    """

    name = "reasoning"

    def __init__(self) -> None:
        self._profiler: Optional[Any] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from reasoning_profiler import ReasoningProfiler  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.warning("ReasoningPlugin: reasoning_profiler not found — disabled.")
        return self._available  # type: ignore[return-value]

    def warmup(self) -> None:
        if not self.is_available():
            return
        if self._profiler is None:
            from reasoning_profiler import ReasoningProfiler
            self._profiler = ReasoningProfiler()
            logger.info("ReasoningPlugin warmup complete.")

    # ------------------------------------------------------------------

    def run(self, context: PluginContext) -> Dict[str, Any]:
        if self._profiler is None:
            self.warmup()
        if self._profiler is None:
            return {"ai_score": 0.5}

        try:
            vec: np.ndarray = self._profiler.vectorize(context.text)
        except Exception as exc:
            logger.warning("ReasoningPlugin.vectorize() failed: %s", exc)
            return {"ai_score": 0.5}

        cot_density       = float(vec[_IDX_COT_SCAFFOLD])
        backtrack_density = float(vec[_IDX_BACKTRACK])
        intuition_density = float(vec[_IDX_INTUITION])

        # Derived scalar: CoT + backtracking push toward AI,
        # intuition leaps push toward human.
        raw_score = (
            0.40 * cot_density * _SCALE
            + 0.40 * backtrack_density * _SCALE
            - 0.20 * intuition_density * _SCALE
        )
        ai_score = float(np.clip(raw_score, 0.0, 1.0))

        # Named feature dict using ReasoningProfiler.feature_names()
        from reasoning_profiler import FEATURE_NAMES
        features = dict(zip(FEATURE_NAMES, vec.tolist()))

        return {
            "ai_score":           ai_score,
            "feature_vector":     vec.tolist(),
            "features":           features,
            "cot_density":        cot_density,
            "backtrack_density":  backtrack_density,
            "intuition_density":  intuition_density,
        }
