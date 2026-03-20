"""
plugins/
========
Inference plugin package for xota_ensemble_v6.

All five plugins are importable from this package::

    from plugins import (
        StylometricPlugin,
        ReasoningPlugin,
        HallucinationPlugin,
        WatermarkPlugin,
        ForensicPlugin,
    )

Execution order (enforced by PluginPipeline topological sort):
    Group A (no deps, CPU):   StylometricPlugin, ReasoningPlugin
    Group B (no deps, GPU):   WatermarkPlugin
    Group B (no deps, CPU):   HallucinationPlugin
    Group C (deps on A+B):    ForensicPlugin
"""

from plugins.forensic_plugin import ForensicPlugin
from plugins.hallucination_plugin import HallucinationPlugin
from plugins.reasoning_plugin import ReasoningPlugin
from plugins.stylometric_plugin import StylometricPlugin
from plugins.watermark_plugin import WatermarkPlugin

__all__ = [
    "StylometricPlugin",
    "ReasoningPlugin",
    "HallucinationPlugin",
    "WatermarkPlugin",
    "ForensicPlugin",
]
