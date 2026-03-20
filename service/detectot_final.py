"""
detectot_final.py  (v2.0 — plugin-integrated)
=============================================
Gradio inference entrypoint for the xota_ensemble_v6 AI text detector.

Changes from v1.0
-----------------
  [FIX P0-B2]  All model forward passes wrapped in ``torch.no_grad()``.
               Original code tracked gradients through 4× forward passes
               unnecessarily — ~30% memory overhead eliminated.

  [FIX P0-B3]  ``torch.load()`` calls now use ``weights_only=True``.
               Original code used pickle-unsafe torch.load, allowing
               arbitrary code execution via crafted state_dict files.

  [P2/P3/P4]   Plugin pipeline integrated:
               - DetectorFinalAdapter wraps neural output into a structured
                 object compatible with ForensicReportGenerator.
               - PluginPipeline executes StylometricPlugin, ReasoningPlugin,
                 HallucinationPlugin, WatermarkPlugin, ForensicPlugin in
                 dependency order.
               - classify_text() returns (str, plt.Figure, UnifiedDetectionResult).
               - PLUGIN_PIPELINE initialised at module level and warmed up
                 once after model loading.
               - Gradio interface augmented to show UnifiedDetectionResult
                 markdown alongside the original bar chart.

  [COMPAT]     Original classify_text() behaviour preserved.  Bar chart
               still produced.  result_message format unchanged.
               Gradio UI layout extended — not broken.
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tokenizers import Regex
from tokenizers.normalizers import Replace, Sequence, Strip
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from detector_adapter import DetectorFinalAdapter, UnifiedDetectionResult
from forensic_reports import ForensicReportGenerator
from plugin_base import PluginContext
from plugin_registry import PluginPipeline
from plugins import (
    ForensicPlugin,
    HallucinationPlugin,
    ReasoningPlugin,
    StylometricPlugin,
    WatermarkPlugin,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("detectot_final: using device=%s", device)

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------

_MODEL1_PATH = "/content/modernbert.bin"
_MODEL2_URL  = "https://huggingface.co/mihalykiss/modernbert_2/resolve/main/Model_groups_3class_seed12"
_MODEL3_URL  = "https://huggingface.co/mihalykiss/modernbert_2/resolve/main/Model_groups_3class_seed22"
_MODEL4_URL  = "https://huggingface.co/mihalykiss/ModernBERT-MGT/resolve/main/Model_groups_41class_seed44__new"

_BASE_MODEL   = "answerdotai/ModernBERT-base"
_NUM_LABELS   = 41

# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

label_mapping = {
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

# ---------------------------------------------------------------------------
# Tokenizer + models
# ---------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)

# Normalizer pipeline (unchanged from v1.0)
_newline_to_space   = Replace(Regex(r"\s*\n\s*"), " ")
_join_hyphen_break  = Replace(Regex(r"(\w+)[--]\s*\n\s*(\w+)"), r"\1\2")
tokenizer.backend_tokenizer.normalizer = Sequence([
    tokenizer.backend_tokenizer.normalizer,
    _join_hyphen_break,
    _newline_to_space,
    Strip(),
])


def _load_model_local(path: str) -> torch.nn.Module:
    """Load model from a local .bin file.  [FIX P0-B3: weights_only=True]"""
    m = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL, num_labels=_NUM_LABELS
    )
    # [FIX P0-B3] weights_only=True prevents arbitrary code execution
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return m.to(device).eval()


def _load_model_url(url: str) -> torch.nn.Module:
    """Load model from a HuggingFace URL.  [FIX P0-B3: weights_only=True]"""
    m = AutoModelForSequenceClassification.from_pretrained(
        _BASE_MODEL, num_labels=_NUM_LABELS
    )
    # [FIX P0-B3] weights_only=True
    m.load_state_dict(
        torch.hub.load_state_dict_from_url(url, map_location=device, weights_only=True)
    )
    return m.to(device).eval()


logger.info("Loading ModernBERT models …")
model_1 = _load_model_local(_MODEL1_PATH)
model_2 = _load_model_url(_MODEL2_URL)
model_3 = _load_model_url(_MODEL3_URL)
model_4 = _load_model_url(_MODEL4_URL)
logger.info("All models loaded.")

# ---------------------------------------------------------------------------
# Plugin pipeline  (initialised once at module level)
# ---------------------------------------------------------------------------

_forensic_generator = ForensicReportGenerator()  # no profiler/hallucination injected
                                                  # — plugins handle those independently

PLUGIN_PIPELINE = PluginPipeline([
    StylometricPlugin(),
    ReasoningPlugin(),
    HallucinationPlugin(),
    WatermarkPlugin(device=device),
    ForensicPlugin(_forensic_generator, generate_visuals=False),
])
logger.info("Plugin pipeline:\n%s", PLUGIN_PIPELINE.summary())
PLUGIN_PIPELINE.warmup()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_MULTI_SPACE_RE = re.compile(r"\s{2,}")
_PUNCT_SPACE_RE = re.compile(r"\s+([,.;:?!])")


def clean_text(text: str) -> str:
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _PUNCT_SPACE_RE.sub(r"\1", text)
    return text


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def classify_text(
    text: str,
) -> Tuple[str, Optional[plt.Figure], UnifiedDetectionResult]:
    """
    Classify text as Human or AI-generated.

    Returns
    -------
    result_message : str
        Markdown string for Gradio display (unchanged format from v1.0).
    fig : plt.Figure | None
        Bar chart of Human vs AI probability.
    unified_result : UnifiedDetectionResult
        Structured object aggregating neural + all plugin outputs.
        Use ``.to_dict()`` for JSON export or ``.to_gradio_markdown()``
        for the extended Gradio display panel.
    """
    cleaned_text = clean_text(text)

    if not cleaned_text.strip():
        empty_ctx = PluginContext(text=text, cleaned_text="", prediction="Unknown")
        adapter = DetectorFinalAdapter(
            torch.full((41,), 1.0 / 41, device=device), label_mapping
        )
        unified = UnifiedDetectionResult.from_context(adapter, empty_ctx)
        return "", None, unified

    inputs = tokenizer(
        cleaned_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
    ).to(device)

    # [FIX P0-B2] Wrap all forward passes in no_grad
    with torch.no_grad():
        logits_1 = model_1(**inputs).logits
        logits_2 = model_2(**inputs).logits
        logits_3 = model_3(**inputs).logits
        logits_4 = model_4(**inputs).logits

        softmax_1 = F.softmax(logits_1, dim=1)
        softmax_2 = F.softmax(logits_2, dim=1)
        softmax_3 = F.softmax(logits_3, dim=1)
        softmax_4 = F.softmax(logits_4, dim=1)

        averaged_probabilities = (softmax_1 + softmax_2 + softmax_3 + softmax_4) / 4
        probabilities = averaged_probabilities[0]  # shape (41,)

    # ── Neural result adapter ────────────────────────────────────────────
    adapter = DetectorFinalAdapter(probabilities, label_mapping)
    context = adapter.to_context(text=text, cleaned_text=cleaned_text)

    human_percentage = adapter.raw_scores["human"]
    ai_percentage    = adapter.raw_scores["ai"]
    ai_argmax_model  = adapter.detected_model

    # ── Original result_message format (preserved) ──────────────────────
    if human_percentage > ai_percentage:
        result_message = (
            f"**The text is** <span class='highlight-human'>"
            f"**{human_percentage:.2f}%** likely <b>Human written</b>.</span>"
        )
    else:
        result_message = (
            f"**The text is** <span class='highlight-ai'>"
            f"**{ai_percentage:.2f}%** likely <b>AI generated</b>.</span>\n\n"
            f"**Identified model:** `{ai_argmax_model}`"
        )

    # ── Bar chart (preserved) ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(
        ["Human", "AI"],
        [human_percentage, ai_percentage],
        color=["#4CAF50", "#FF5733"],
        alpha=0.8,
    )
    ax.set_ylabel("Probability (%)", fontsize=12)
    ax.set_title("Human vs AI Probability", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.2f}%",
            ha="center",
        )
    ax.set_ylim(0, 100)
    plt.tight_layout()

    # ── Plugin pipeline ──────────────────────────────────────────────────
    context = PLUGIN_PIPELINE.execute(context)
    unified = UnifiedDetectionResult.from_context(adapter, context)

    return result_message, fig, unified


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

try:
    import gradio as gr

    _TITLE       = "SOTA AI Text Detector — xota_ensemble_v6"
    _DESCRIPTION = (
        "Paste any text to detect whether it was written by a human or an AI. "
        "The ensemble of 4 ModernBERT models is augmented with stylometric, "
        "reasoning-pattern, hallucination-risk, and watermark analysis."
    )
    _BOTTOM_TEXT = (
        "⚠ **Research tool only.** Results are probabilistic and must not be "
        "used as sole evidence in academic integrity or legal proceedings."
    )

    def _gradio_classify(text: str):
        """Wrapper for Gradio — returns (markdown, markdown, figure)."""
        result_message, fig, unified = classify_text(text)
        plugin_markdown = unified.to_gradio_markdown()
        return result_message, plugin_markdown, fig

    with gr.Blocks(title=_TITLE) as iface:
        gr.Markdown(f"# {_TITLE}")
        gr.Markdown(_DESCRIPTION)

        text_input = gr.Textbox(
            label="",
            placeholder="Type or paste your content here…",
            elem_id="text_input_box",
            lines=5,
        )

        with gr.Row():
            result_output      = gr.Markdown("", elem_id="result_output_box")
            plugin_output      = gr.Markdown("", elem_id="plugin_output_box")

        plot_output = gr.Plot(label="Human vs AI Probability")

        text_input.change(
            _gradio_classify,
            inputs=text_input,
            outputs=[result_output, plugin_output, plot_output],
        )

        gr.Markdown(_BOTTOM_TEXT, elem_id="bottom_text")

except ImportError:
    logger.warning("Gradio not installed — interactive UI unavailable.")
    iface = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Module-level public API (for programmatic use without Gradio)
# ---------------------------------------------------------------------------

__all__ = [
    "classify_text",
    "clean_text",
    "PLUGIN_PIPELINE",
    "label_mapping",
    "device",
    "iface",
]
