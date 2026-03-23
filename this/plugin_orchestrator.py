"""
plugin_orchestrator.py  (xota_ensemble_v6 — inference/)
=========================================================
Thin coordination layer. Contains NO plugin business logic.

Responsibility
--------------
  1. Load plugins once at startup (controlled by PluginConfig flags).
  2. Call each plugin's existing public method(s) exactly as documented.
  3. Assemble the additional_analyses dict for ForensicReportGenerator.
  4. Call ForensicReportGenerator.generate_report() and export.

What this file does NOT do
---------------------------
  * Does not re-implement any plugin algorithm.
  * Does not subclass any plugin.
  * Does not contain classification logic from any plugin module.

The only computation here is _compute_reasoning_score() — a minimal
weighted aggregation converting ReasoningProfiler's raw 15-dim vector
into a single float. This is necessary because ReasoningProfiler is a
pure extractor by design (no classifier). The weights and thresholds
are local constants in this file, not borrowed from any plugin.

Plugin call map
---------------
  Plugin                        Method(s) called
  ─────────────────────────────────────────────────────────────────
  StylometricProfiler           .compute_stats(text)
  HallucinationProfiler         passed to ForensicReportGenerator.__init__
  HallucinationRiskClassifier   passed to ForensicReportGenerator.__init__
  ReasoningProfiler             .vectorize(text), .feature_names()
  PerplexityProfiler            .compute_stats(text)  [NEW v3.7]
  PerplexityRiskClassifier      .classify(stats)      [NEW v3.7]
  WatermarkDecoder              .detect(text) -> .to_forensic_dict()
  ForensicReportGenerator       .generate_report(...) -> .export_html/json()

Changelog (v3.4 -> v3.5)
-------------------------
  [FIX]  ForensicReportGenerator now receives reasoning_profiler via
         __init__ so _build_reasoning_html() can populate group_scores,
         top_signals, and feature_details tables (previously empty).
  [FIX]  Reasoning dict produced by orchestrator now uses
         ReasoningRiskClassifier.classify() (from forensic_reports.py)
         to generate the FULL reasoning analysis structure instead of
         the partial {ai_score, risk_level, feature_values} dict.

Usage
-----
    from plugin_orchestrator import PluginOrchestrator, PluginConfig
    from detector_final import classify_text

    # Full pipeline from raw text
    orch   = PluginOrchestrator(PluginConfig(enable_watermark=True))
    result = orch.run("Paste text here...")

    # Pre-computed detection (avoids re-running the 4 models)
    msg, fig, det = classify_text("Paste text here...")
    result        = orch.run_with_result("Paste text here...", det)

    print(orch.summary(result))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# PluginConfig
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PluginConfig:
    """
    Activation flags for every plugin in the pipeline.

    Parameters
    ----------
    enable_stylometric      : Run StylometricProfiler.compute_stats().
    enable_hallucination    : Pass HallucinationProfiler + Classifier to
                              ForensicReportGenerator (runs internally).
    enable_reasoning        : Call ReasoningProfiler.vectorize() and include
                              results in additional_analyses["reasoning"].
    enable_watermark        : Call WatermarkDecoder.detect() on every call.
                              Loads GPT-2 — disabled by default.
    enable_forensic_report  : Generate and export the forensic report.
    forensic_output_path    : File path for the exported report.
    forensic_output_format  : "html" (default) or "json".
    watermark_device        : Torch device string for WatermarkDecoder.
                              None = auto-detect.
    """
    enable_stylometric:     bool = True
    enable_hallucination:   bool = True
    enable_reasoning:       bool = True
    enable_perplexity:      bool = True     # [NEW v3.7] Perplexity profiler
    enable_watermark:       bool = False
    enable_forensic_report: bool = True
    forensic_output_path:   str  = "forensic_report.html"
    forensic_output_format: str  = "html"
    watermark_device:       Optional[str] = None
    perplexity_dict_path:   Optional[str] = None   # [NEW v3.7] Pre-built n-gram dict
    perplexity_tier2:       bool = True             # [NEW v3.7] Auto-enable GPT-2


# ═══════════════════════════════════════════════════════════════════════════
# PluginOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

class PluginOrchestrator:
    """
    Thin pipeline coordinator. Instantiate once; call run() per text.

    Parameters
    ----------
    config : PluginConfig with activation flags.
    """

    # Weights for _compute_reasoning_score().
    # 9 positive features (sum=0.92) + 1 inverse feature (0.08) = 1.00
    _RSN_WEIGHTS: Dict[str, float] = {
        "backtracking_density":    0.26,
        "cot_scaffold_density":    0.23,
        "consequence_density":     0.09,
        "causal_density":          0.07,
        "sequence_density":        0.07,
        "contrast_density":        0.05,
        "word_entropy_normalised": 0.07,
        "type_token_ratio":        0.05,
        "paragraph_length_cv":     0.03,
    }
    _RSN_HIGH_THRESHOLDS: Dict[str, float] = {
        "backtracking_density":    0.07,
        "cot_scaffold_density":    0.10,
        "consequence_density":     0.06,
        "causal_density":          0.07,
        "sequence_density":        0.05,
        "contrast_density":        0.06,
        "word_entropy_normalised": 0.90,
        "type_token_ratio":        0.72,
        "paragraph_length_cv":     0.55,
        "intuition_leap_density":  0.04,
    }

    def __init__(self, config: Optional[PluginConfig] = None) -> None:
        self.config = config or PluginConfig()
        self._stylometric:              Any = None
        self._hallucination_profiler:   Any = None
        self._hallucination_classifier: Any = None
        self._reasoning_profiler:       Any = None
        self._reasoning_classifier:     Any = None      # [NEW v3.5]
        self._perplexity_profiler:      Any = None      # [NEW v3.7]
        self._perplexity_classifier:    Any = None      # [NEW v3.7]
        self._watermark_decoder:        Any = None
        self._forensic_generator:       Any = None
        self._init_plugins()

    def _init_plugins(self) -> None:
        """Load each enabled plugin once. Import failures are logged and skipped."""
        cfg = self.config

        if cfg.enable_stylometric:
            try:
                from stylometric_profiler import StylometricProfiler
                self._stylometric = StylometricProfiler()
                logger.info("StylometricProfiler loaded")
            except ImportError as exc:
                logger.warning("StylometricProfiler unavailable: %s", exc)

        if cfg.enable_hallucination:
            try:
                from hallucination_profile import (
                    HallucinationProfiler,
                    HallucinationRiskClassifier,
                )
                self._hallucination_profiler   = HallucinationProfiler()
                self._hallucination_classifier = HallucinationRiskClassifier()
                logger.info("HallucinationProfiler + Classifier loaded")
            except ImportError as exc:
                logger.warning("HallucinationProfiler unavailable: %s", exc)

        if cfg.enable_reasoning:
            try:
                from reasoning_profiler import ReasoningProfiler
                self._reasoning_profiler = ReasoningProfiler()
                logger.info("ReasoningProfiler loaded")
            except ImportError as exc:
                logger.warning("ReasoningProfiler unavailable: %s", exc)

            # [NEW v3.5] Load ReasoningRiskClassifier for full analysis
            try:
                from forensic_reports import ReasoningRiskClassifier
                self._reasoning_classifier = ReasoningRiskClassifier()
                logger.info("ReasoningRiskClassifier loaded")
            except ImportError as exc:
                logger.warning("ReasoningRiskClassifier unavailable: %s", exc)

        if cfg.enable_watermark:
            try:
                import torch
                from watermark_decoder import WatermarkDecoder
                device = torch.device(cfg.watermark_device) if cfg.watermark_device else None
                self._watermark_decoder = WatermarkDecoder(device=device)
                logger.info("WatermarkDecoder loaded")
            except ImportError as exc:
                logger.warning("WatermarkDecoder unavailable: %s", exc)

        # [NEW v3.7] PerplexityProfiler + PerplexityRiskClassifier
        if cfg.enable_perplexity:
            try:
                from perplexity_profiler import PerplexityProfiler, PerplexityRiskClassifier
                self._perplexity_profiler = PerplexityProfiler(
                    ngram_dict_path=cfg.perplexity_dict_path,
                    enable_tier2=cfg.perplexity_tier2,
                )
                self._perplexity_classifier = PerplexityRiskClassifier()
                tier = "tier2" if (self._perplexity_profiler._gpt2 is not None
                                   and self._perplexity_profiler._gpt2._available) else "tier1"
                logger.info("PerplexityProfiler loaded (%s)", tier)
            except ImportError as exc:
                logger.warning("PerplexityProfiler unavailable: %s", exc)

        if cfg.enable_forensic_report:
            try:
                from forensic_reports import ForensicReportGenerator
                # [CHANGED v3.5] Now passes reasoning_profiler and
                # reasoning_classifier so the generator can build full
                # reasoning HTML with group_scores, top_signals, feature_details.
                self._forensic_generator = ForensicReportGenerator(
                    profiler=self._stylometric,
                    hallucination_profiler=self._hallucination_profiler,
                    hallucination_classifier=self._hallucination_classifier,
                    reasoning_profiler=self._reasoning_profiler,
                    reasoning_classifier=self._reasoning_classifier,
                )
                logger.info("ForensicReportGenerator loaded")
            except ImportError as exc:
                logger.warning("ForensicReportGenerator unavailable: %s", exc)

    def run(self, text: str) -> Dict[str, Any]:
        """
        Full pipeline: call classify_text() then all enabled plugins.

        Returns dict with keys:
            "detection_result"    : DetectionResult
            "additional_analyses" : dict of plugin outputs
            "forensic_report"     : ForensicReport | None
        """
        from detector_final import classify_text
        _, _, detection_result = classify_text(text)
        return self.run_with_result(text, detection_result)

    def run_with_result(self, text: str, detection_result: Any) -> Dict[str, Any]:
        """
        Run all enabled plugins against a pre-computed DetectionResult.

        Use this when classify_text() has already been called (e.g. in Gradio)
        to avoid re-running the 4-model ensemble.

        detection_result.statistical_features is populated in-place by
        StylometricProfiler so callers can access stats without opening the report.
        """
        additional: Dict[str, Any] = {}

        # ── StylometricProfiler ───────────────────────────────────────
        if self._stylometric is not None:
            try:
                stats = self._stylometric.compute_stats(text)
                detection_result.statistical_features = stats
                logger.debug(
                    "Stylometric: burstiness=%.3f vocab=%.3f hapax=%.3f",
                    stats.get("burstiness", 0.0),
                    stats.get("vocabulary_richness", 0.0),
                    stats.get("hapax_legomena_ratio", 0.0),
                )
            except Exception as exc:
                logger.warning("StylometricProfiler.compute_stats() failed: %s", exc)

        # ── ReasoningProfiler ─────────────────────────────────────────
        # [CHANGED v3.5] Now uses ReasoningRiskClassifier.classify() to
        # produce the FULL reasoning analysis dict (with group_scores,
        # top_signals, feature_details, interpretation) instead of the
        # partial dict that was causing empty tables in the HTML report.
        if self._reasoning_profiler is not None:
            try:
                from reasoning_profiler import FEATURE_NAMES as _RN
                vec        = self._reasoning_profiler.vectorize(text)
                feat_names = self._reasoning_profiler.feature_names()

                if self._reasoning_classifier is not None:
                    # Full classification with group_scores, top_signals, etc.
                    reasoning_analysis = self._reasoning_classifier.classify(
                        vec, feat_names,
                    )
                    additional["reasoning"] = reasoning_analysis
                    logger.debug(
                        "Reasoning (full): score=%.4f level=%s",
                        reasoning_analysis.get("ai_score", 0.0),
                        reasoning_analysis.get("risk_level", "N/A"),
                    )
                else:
                    # Fallback: partial dict (legacy behaviour)
                    feat_values: Dict[str, float] = dict(zip(feat_names, vec.tolist()))
                    ai_score    = self._compute_reasoning_score(feat_values)
                    risk_level  = self._classify_reasoning_risk(ai_score)
                    additional["reasoning"] = {
                        "ai_score":       ai_score,
                        "risk_level":     risk_level,
                        "feature_values": feat_values,
                    }
                    logger.debug("Reasoning (partial): score=%.4f level=%s",
                                 ai_score, risk_level)
            except Exception as exc:
                logger.warning("ReasoningProfiler.vectorize() failed: %s", exc)

        # ── PerplexityProfiler [NEW v3.7] ─────────────────────────────
        if self._perplexity_profiler is not None:
            try:
                ppl_stats = self._perplexity_profiler.compute_stats(text)

                if self._perplexity_classifier is not None:
                    ppl_analysis = self._perplexity_classifier.classify(ppl_stats)
                    # Merge raw stats into the analysis dict
                    ppl_analysis["window_ppls"] = ppl_stats.get("window_ppls", [])
                    ppl_analysis["tokens_analysed"] = ppl_stats.get("tokens_analysed", 0)
                    ppl_analysis["feature_values"] = {
                        k: ppl_stats[k] for k in ppl_stats
                        if isinstance(ppl_stats[k], (int, float))
                    }
                    additional["perplexity"] = ppl_analysis
                else:
                    additional["perplexity"] = ppl_stats

                logger.debug(
                    "Perplexity (%s): score=%.4f level=%s ppl_mean=%.2f",
                    ppl_stats.get("tier", "tier1"),
                    additional["perplexity"].get("ai_score", 0.0),
                    additional["perplexity"].get("risk_level", "N/A"),
                    ppl_stats.get("proxy_perplexity_mean", 0.0),
                )
            except Exception as exc:
                logger.warning("PerplexityProfiler.compute_stats() failed: %s", exc)

        # ── WatermarkDecoder ──────────────────────────────────────────
        if self._watermark_decoder is not None:
            try:
                sig = self._watermark_decoder.detect(text)
                additional["watermark"] = sig.to_forensic_dict()
                logger.debug(
                    "Watermark: detected=%s confidence=%.4f scheme=%s",
                    sig.detected, sig.confidence, sig.scheme_type,
                )
            except Exception as exc:
                logger.warning("WatermarkDecoder.detect() failed: %s", exc)

        # ── ForensicReportGenerator ───────────────────────────────────
        forensic_report = None
        if self._forensic_generator is not None:
            try:
                forensic_report = self._forensic_generator.generate_report(
                    text=text,
                    detection_result=detection_result,
                    additional_analyses=additional,
                    generate_visuals=True,
                )
                path = self.config.forensic_output_path
                if self.config.forensic_output_format == "json":
                    self._forensic_generator.export_json(forensic_report, path)
                else:
                    self._forensic_generator.export_html(forensic_report, path)
                logger.info(
                    "Forensic report -> %s  verdict=%s  confidence=%.1f%%",
                    path, forensic_report.verdict, forensic_report.confidence * 100,
                )
            except Exception as exc:
                logger.warning("ForensicReportGenerator failed: %s", exc)

        return {
            "detection_result":    detection_result,
            "additional_analyses": additional,
            "forensic_report":     forensic_report,
        }

    # ── Reasoning score helpers ────────────────────────────────────────
    # These exist solely because ReasoningProfiler produces a raw 15-dim
    # vector with no aggregated score. The weights are local constants here,
    # not borrowed from any plugin module.

    @classmethod
    def _compute_reasoning_score(cls, features: Dict[str, float]) -> float:
        """Normalised weighted sum of reasoning marker features -> [0, 1]."""
        score = 0.0
        for feat, weight in cls._RSN_WEIGHTS.items():
            val  = features.get(feat, 0.0)
            high = cls._RSN_HIGH_THRESHOLDS.get(feat, 1.0)
            score += weight * min(1.0, val / max(high, 1e-9))
        # Inverse intuition component (weight 0.08)
        inv_val  = features.get("intuition_leap_density", 0.0)
        inv_high = cls._RSN_HIGH_THRESHOLDS["intuition_leap_density"]
        inv_norm = min(1.0, inv_val / max(inv_high, 1e-9))
        score   += 0.08 * max(0.0, 1.0 - inv_norm)
        return round(min(1.0, max(0.0, score)), 4)

    @staticmethod
    def _classify_reasoning_risk(score: float) -> str:
        if score >= 0.55:
            return "HIGH \u2014 Reasoning Model"
        if score >= 0.28:
            return "MEDIUM \u2014 Possible Reasoning Model"
        return "LOW \u2014 Standard Model or Human"

    # ── Utilities ──────────────────────────────────────────────────────

    def active_plugins(self) -> List[str]:
        """Return names of successfully loaded plugins."""
        active: List[str] = []
        if self._stylometric            is not None: active.append("StylometricProfiler")
        if self._hallucination_profiler is not None: active.append("HallucinationProfiler")
        if self._reasoning_profiler     is not None: active.append("ReasoningProfiler")
        if self._reasoning_classifier   is not None: active.append("ReasoningRiskClassifier")
        if self._perplexity_profiler    is not None: active.append("PerplexityProfiler")
        if self._watermark_decoder      is not None: active.append("WatermarkDecoder")
        if self._forensic_generator     is not None: active.append("ForensicReportGenerator")
        return active

    def summary(self, result: Dict[str, Any]) -> str:
        """Plain-text summary of a run() result dict."""
        sep   = "\u2550" * 58
        lines = [sep, "  PLUGIN ORCHESTRATOR \u2014 RESULT SUMMARY", sep]

        det = result.get("detection_result")
        if det is not None:
            unc = "\u26a0 UNCERTAINTY" if det.uncertainty_zone else "\u2713 decisive"
            lines.append(
                f"  Detection  : {det.prediction:6s}  confidence={det.confidence:.1f}%  [{unc}]"
            )
            if det.detected_model:
                lines.append(f"  Likely LLM : {det.detected_model}")
            lines.append(
                f"  Scores     : human={det.raw_scores.get('human',0):.1f}%  "
                f"ai={det.raw_scores.get('ai',0):.1f}%"
            )

        aa = result.get("additional_analyses", {})

        sf = getattr(det, "statistical_features", {}) or {}
        if sf:
            lines += ["", "  Stylometric:",
                f"    burstiness={sf.get('burstiness',0):.3f}  "
                f"vocab={sf.get('vocabulary_richness',0):.3f}  "
                f"hapax={sf.get('hapax_legomena_ratio',0):.3f}"]

        rsn = aa.get("reasoning")
        if rsn:
            lines += ["", "  Reasoning:",
                f"    score={rsn['ai_score']:.4f}  level={rsn['risk_level']}"]
            fv = rsn.get("feature_values") or {}
            if not fv:
                # v3.5 full dict: extract from feature_details
                fd = rsn.get("feature_details", {})
                fv = {k: v.get("value", 0.0) for k, v in fd.items()}
            top = sorted(fv.items(), key=lambda x: x[1], reverse=True)[:3]
            for feat, val in top:
                lines.append(f"    {feat:<36s} = {val:.6f}")

        ppl = aa.get("perplexity")
        if ppl:
            lines += ["", "  Perplexity:",
                f"    score={ppl.get('ai_score',0):.4f}  level={ppl.get('risk_level','N/A')}  "
                f"tier={ppl.get('tier','?')}",
                f"    ppl_mean={ppl.get('feature_values',{}).get('proxy_perplexity_mean',0):.2f}  "
                f"entropy={ppl.get('feature_values',{}).get('token_entropy_mean',0):.2f}  "
                f"windows={ppl.get('window_count',0)}"]

        wm = aa.get("watermark")
        if wm:
            lines += ["", "  Watermark:",
                f"    detected={wm.get('detected')}  "
                f"confidence={wm.get('confidence',0):.4f}  "
                f"scheme={wm.get('scheme_type','none')}"]

        fr = result.get("forensic_report")
        if fr is not None:
            lines += ["", "  Forensic Report:",
                f"    verdict={fr.verdict}  neural={fr.neural_score:.2f}  "
                f"reasoning={fr.reasoning_score:.2f}  watermark={fr.watermark_score:.2f}",
                f"    saved \u2192 {self.config.forensic_output_path}"]

        lines += ["", f"  Active plugins: {', '.join(self.active_plugins())}", sep]
        return "\n".join(lines)
