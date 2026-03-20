"""
plugin_base.py
==============
Abstract plugin interface for the xota_ensemble_v6 inference pipeline.

Architecture
------------
Every plugin receives an immutable view of the current ``PluginContext``
and writes its output exclusively into ``context.plugin_results[self.name]``.
No other field of PluginContext is mutated by plugins.

Dependency contract
-------------------
``InferencePlugin.requires`` lists the names of other plugins whose
``plugin_results`` entries must exist before this plugin's ``run()`` is
called.  The ``PluginPipeline`` (plugin_registry.py) enforces ordering via
topological sort (Kahn's algorithm).

Error policy
------------
``run()`` MUST NOT raise.  All exceptions must be caught internally and
logged.  A partial / empty dict is a valid return value.  The pipeline
records failures in ``PluginContext.errors`` and continues.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared execution context
# ---------------------------------------------------------------------------

@dataclass
class PluginContext:
    """
    Execution context created once per inference call and threaded through
    the entire plugin pipeline.

    Fields are populated in two phases:
      1. Before pipeline: ``text``, ``cleaned_text``, and the neural
         inference fields are set by ``DetectorFinalAdapter``.
      2. During pipeline: each plugin appends to ``plugin_results[name]``.

    Design invariant
    ----------------
    Plugins READ ``text`` / ``cleaned_text`` / ``probabilities`` /
    ``prediction`` / ``confidence`` / ``top_ai_model``.
    Plugins WRITE only ``plugin_results[self.name]``.
    The pipeline owns ``errors``.
    """

    # ── Input ────────────────────────────────────────────────────────────
    text: str
    cleaned_text: str

    # ── Neural inference results (from DetectorFinalAdapter) ─────────────
    probabilities: Optional[Any] = None          # torch.Tensor shape (N_LABELS,) | None
    human_probability: float = 0.0               # raw (not renormalised)
    ai_probability: float = 0.0                  # raw sum over all non-human classes
    top_ai_model: str = "unknown"                # label_mapping[argmax(ai_probs)]
    prediction: str = "Unknown"                  # "Human" | "AI" | "Unknown"
    confidence: float = 0.0                      # 0–100, renormalised binary

    # ── Accumulated plugin outputs ────────────────────────────────────────
    # Keyed by InferencePlugin.name.  Each value is the dict returned by run().
    plugin_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ── Pipeline diagnostics ─────────────────────────────────────────────
    errors: Dict[str, str] = field(default_factory=dict)

    # ── Convenience accessors ────────────────────────────────────────────

    def get_plugin(self, name: str) -> Optional[Dict[str, Any]]:
        """Return plugin output by name, or None if absent / failed."""
        return self.plugin_results.get(name)

    def has_plugin(self, name: str) -> bool:
        """True if plugin produced output (even an empty dict)."""
        return name in self.plugin_results

    def is_healthy(self) -> bool:
        """True if no plugin errors were recorded."""
        return len(self.errors) == 0


# ---------------------------------------------------------------------------
# Abstract plugin interface
# ---------------------------------------------------------------------------

class InferencePlugin(ABC):
    """
    Minimal abstract interface for all inference pipeline plugins.

    Implementors must override:
        ``name``  — unique key used in PluginContext.plugin_results
        ``run()`` — execution logic; must never raise

    Optionally override:
        ``requires``      — dependency list (other plugin names)
        ``is_available()``— checks optional dependencies at init time
        ``warmup()``      — eagerly loads lazy resources (models, etc.)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique plugin identifier used as key in PluginContext.plugin_results
        and as the key in ``additional_analyses`` passed to ForensicReportGenerator.

        Must be a valid Python identifier (no spaces or hyphens).
        """
        ...

    @property
    def requires(self) -> List[str]:
        """
        Names of plugins whose results must already be present in
        ``PluginContext.plugin_results`` before this plugin runs.

        Default: no dependencies (independent plugin).
        """
        return []

    @abstractmethod
    def run(self, context: PluginContext) -> Dict[str, Any]:
        """
        Execute plugin logic.

        Contract
        --------
        - Reads from ``context.text``, ``context.cleaned_text``,
          ``context.probabilities``, ``context.plugin_results``.
        - Returns a ``Dict[str, Any]`` that will be stored as
          ``context.plugin_results[self.name]``.
        - MUST NOT raise any exception.  Catch all errors internally,
          log them, and return a partial / empty dict.
        - MUST NOT mutate any field of ``context`` directly.

        The returned dict should be compatible with
        ``forensic_reports.ForensicReportGenerator.generate_report()``'s
        ``additional_analyses[self.name]`` schema when consumed by
        ``ForensicPlugin``.
        """
        ...

    def is_available(self) -> bool:
        """
        Return False if the plugin cannot run due to missing dependencies
        (e.g. spaCy model not installed, GPU out of memory).

        Called once at pipeline startup.  A plugin that returns False is
        skipped silently; its ``name`` will not appear in ``plugin_results``.

        Default: always available.
        """
        return True

    def warmup(self) -> None:
        """
        Eagerly load any lazy resources (model weights, tokenizers, etc.).

        Called once during ``PluginPipeline.warmup()``, before the first
        inference call.  Eliminates first-call latency spikes in production.

        Default: no-op.
        """
        pass

    def __repr__(self) -> str:
        available = "available" if self.is_available() else "unavailable"
        deps = f", requires={self.requires}" if self.requires else ""
        return f"{self.__class__.__name__}(name={self.name!r}, {available}{deps})"
