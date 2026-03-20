"""
plugins/watermark_plugin.py
============================
Plugin wrapper for WatermarkDecoder.detect().

Output key: ``"watermark"``
Returns ``WatermarkSignature.to_forensic_dict()`` directly — already
compatible with ForensicReportGenerator._collect_evidence() schema.

GPU usage note
--------------
WatermarkDecoder lazy-loads GPT-2 (~500 MB) on first detect() call.
Call ``warmup()`` during pipeline initialisation to front-load this cost.

Vocab note
----------
The green/red list analysis is calibrated for GPT-2's 50,257-token
vocabulary.  The z-scores are meaningful only for that tokeniser.
Entropy analysis (the other signal) is model-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch

from plugin_base import InferencePlugin, PluginContext

logger = logging.getLogger(__name__)


class WatermarkPlugin(InferencePlugin):
    """
    Wraps ``WatermarkDecoder.detect(text)`` into the plugin interface.

    Output schema (``context.plugin_results["watermark"]``)::

        {
            "detected":       bool,
            "confidence":     float,   # 0.0 when detected=False (enforced)
            "scheme_type":    str,     # "none" | "green_red" | "exp_minimum" | "semantic"
            "z_score":        float,
            "p_value":        float,   # Bonferroni-corrected
            "green_fraction": float,
        }

    Parameters
    ----------
    device : torch.device — GPU for entropy analysis, CPU fallback.
    config : WatermarkConfig — thresholds/parameters. None = defaults.
    """

    name = "watermark"

    def __init__(
        self,
        device: Optional[torch.device] = None,
        config: Optional[Any] = None,
    ) -> None:
        self._device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._config = config
        self._decoder: Optional[Any] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from watermark_decoder import WatermarkDecoder  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
                logger.warning("WatermarkPlugin: watermark_decoder not found — disabled.")
        return self._available  # type: ignore[return-value]

    def warmup(self) -> None:
        if not self.is_available():
            return
        if self._decoder is None:
            from watermark_decoder import WatermarkDecoder
            self._decoder = WatermarkDecoder(device=self._device, config=self._config)
            self._decoder.preload()   # eagerly loads GPT-2
            logger.info(
                "WatermarkPlugin warmup complete (device=%s).", self._device
            )

    # ------------------------------------------------------------------

    def run(self, context: PluginContext) -> Dict[str, Any]:
        if self._decoder is None:
            self.warmup()
        if self._decoder is None:
            return {"detected": False, "confidence": 0.0, "scheme_type": "unavailable"}

        try:
            sig = self._decoder.detect(context.text)
            return sig.to_forensic_dict()
        except Exception as exc:
            logger.warning("WatermarkPlugin.detect() failed: %s", exc)
            # Ensure CUDA memory released on failure
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
            return {"detected": False, "confidence": 0.0, "scheme_type": "error", "error": str(exc)}
