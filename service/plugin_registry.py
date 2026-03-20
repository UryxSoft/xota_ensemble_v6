"""
plugin_registry.py
==================
PluginPipeline — DAG executor for the xota_ensemble_v6 inference pipeline.

Responsibilities
----------------
1. Accept a list of InferencePlugin instances at construction time.
2. Topologically sort them by their ``.requires`` dependency declarations
   using Kahn's algorithm.  Raise ValueError on cycles.
3. Execute plugins in sorted order: skip unavailable ones, record errors
   for failed ones, never crash the pipeline.
4. Expose ``warmup()`` to eagerly load all plugin resources before serving.
5. Provide ``summary()`` for operator visibility into plugin graph.

Thread safety
-------------
PluginPipeline is NOT thread-safe.  Each inference request should use its
own PluginContext.  The pipeline itself holds no per-request state, so a
single pipeline instance can serve sequential requests without issue.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

from plugin_base import InferencePlugin, PluginContext

logger = logging.getLogger(__name__)


class PluginPipeline:
    """
    DAG-ordered plugin executor.

    Usage
    -----
    ::
        pipeline = PluginPipeline([
            StylometricPlugin(),
            ReasoningPlugin(),
            HallucinationPlugin(),
            WatermarkPlugin(device=device),
            ForensicPlugin(ForensicReportGenerator()),
        ])
        pipeline.warmup()                      # load models once
        context = DetectorFinalAdapter(...).to_context(text)
        context = pipeline.execute(context)    # run all plugins
    """

    def __init__(self, plugins: List[InferencePlugin]) -> None:
        if not plugins:
            raise ValueError("PluginPipeline requires at least one plugin.")

        # Deduplicate by name — last registration wins
        seen: Dict[str, InferencePlugin] = {}
        for p in plugins:
            if p.name in seen:
                logger.warning(
                    "Duplicate plugin name %r — overwriting previous registration.",
                    p.name,
                )
            seen[p.name] = p
        self._plugins: Dict[str, InferencePlugin] = seen

        # Validate dependency references
        all_names = set(self._plugins)
        for plugin in self._plugins.values():
            for dep in plugin.requires:
                if dep not in all_names:
                    raise ValueError(
                        f"Plugin {plugin.name!r} declares dependency {dep!r} "
                        f"which is not registered in this pipeline.  "
                        f"Registered plugins: {sorted(all_names)}"
                    )

        self._order: List[str] = self._topological_sort()
        logger.info(
            "PluginPipeline initialised: %d plugins, execution order: %s",
            len(self._plugins),
            " → ".join(self._order),
        )

    # ------------------------------------------------------------------
    # Topological sort (Kahn's algorithm)
    # ------------------------------------------------------------------

    def _topological_sort(self) -> List[str]:
        """
        Kahn's algorithm over the plugin dependency graph.

        Raises ValueError if a cycle is detected (impossible dependency
        chain).  Ties (zero-in-degree nodes) are broken alphabetically
        for deterministic ordering across Python versions.
        """
        # Build adjacency: dependency → list of plugins that depend on it
        in_degree: Dict[str, int] = {name: 0 for name in self._plugins}
        dependents: Dict[str, List[str]] = {name: [] for name in self._plugins}

        for plugin in self._plugins.values():
            for dep in plugin.requires:
                in_degree[plugin.name] += 1
                dependents[dep].append(plugin.name)

        # Queue: all zero-in-degree nodes (sorted for determinism)
        queue: deque[str] = deque(
            sorted(name for name, deg in in_degree.items() if deg == 0)
        )
        result: List[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for dependent in sorted(dependents[node]):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._plugins):
            cycle_nodes = [n for n in self._plugins if n not in result]
            raise ValueError(
                f"Circular dependency detected in plugin graph.  "
                f"Nodes in cycle: {sorted(cycle_nodes)}"
            )

        return result

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """
        Eagerly load all available plugin resources.

        Call once after constructing the pipeline, before serving
        inference requests.  Unavailable plugins are skipped silently.
        """
        logger.info("PluginPipeline warmup started (%d plugins).", len(self._plugins))
        for name in self._order:
            plugin = self._plugins[name]
            if not plugin.is_available():
                logger.info("  SKIP (unavailable): %s", name)
                continue
            t0 = time.perf_counter()
            try:
                plugin.warmup()
                elapsed = time.perf_counter() - t0
                logger.info("  WARM %s  (%.2fs)", name, elapsed)
            except Exception as exc:
                logger.error("  FAIL warmup %s: %s", name, exc)
        logger.info("PluginPipeline warmup complete.")

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute(self, context: PluginContext) -> PluginContext:
        """
        Run all available plugins in dependency order.

        For each plugin:
          - Skip if ``is_available()`` returns False.
          - Skip and record error if any declared dependency is absent
            from ``context.plugin_results`` (dependency failed earlier).
          - Catch all exceptions from ``run()``, record in ``context.errors``.
          - Store successful result in ``context.plugin_results[name]``.

        Returns the (mutated) ``context`` for chaining.
        """
        for name in self._order:
            plugin = self._plugins[name]

            if not plugin.is_available():
                logger.debug("Plugin %r skipped (unavailable).", name)
                continue

            # Check dependencies satisfied
            missing_deps = [
                dep for dep in plugin.requires
                if dep not in context.plugin_results
            ]
            if missing_deps:
                msg = (
                    f"Plugin {name!r} skipped: dependencies not satisfied: "
                    f"{missing_deps}"
                )
                logger.warning(msg)
                context.errors[name] = msg
                continue

            t0 = time.perf_counter()
            try:
                result = plugin.run(context)
                elapsed = time.perf_counter() - t0
                if not isinstance(result, dict):
                    logger.warning(
                        "Plugin %r returned %s instead of dict — wrapping.",
                        name, type(result).__name__,
                    )
                    result = {"raw_result": result}
                context.plugin_results[name] = result
                logger.debug("Plugin %r OK  (%.3fs).", name, elapsed)
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.error(
                    "Plugin %r raised exception after %.3fs: %s",
                    name, elapsed, exc, exc_info=True,
                )
                context.errors[name] = f"{type(exc).__name__}: {exc}"

        return context

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable summary of the pipeline for logging/debug.

        Example output::
            PluginPipeline (5 plugins)
            Execution order:
              [1] stylometric_stats       (available, no deps)
              [2] reasoning               (available, no deps)
              [3] hallucination           (available, no deps)
              [4] watermark               (available, no deps)
              [5] forensic_report         (available, requires: statistical, reasoning, hallucination, watermark)
        """
        lines = [f"PluginPipeline ({len(self._plugins)} plugins)", "Execution order:"]
        for i, name in enumerate(self._order, start=1):
            plugin = self._plugins[name]
            avail = "available" if plugin.is_available() else "UNAVAILABLE"
            deps_str = (
                f"requires: {', '.join(plugin.requires)}"
                if plugin.requires
                else "no deps"
            )
            lines.append(f"  [{i}] {name:<28} ({avail}, {deps_str})")
        return "\n".join(lines)

    def plugin_names(self) -> List[str]:
        """Return plugin names in execution order."""
        return list(self._order)

    def __len__(self) -> int:
        return len(self._plugins)

    def __repr__(self) -> str:
        return (
            f"PluginPipeline(plugins={self._order!r})"
        )
