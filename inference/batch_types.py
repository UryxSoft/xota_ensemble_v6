"""
Batch processing types for SOTA AI Text Detector.
==================================================
Replaces fragile parallel-list tracking in detect_batch()
with a single, typed dataclass per item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class BatchItem:
    """
    Tracks a single text through the detect_batch pipeline.

    Fields are populated progressively by:
        _preprocess_batch  -> cleaned_text, injected_text, fk_score, is_valid
        _classify_lengths  -> is_long
        _batch_infer_short / _sliding_window_long -> probs
        _postprocess_results -> result
    """

    index: int
    original_text: str
    cleaned_text: Optional[str] = None
    injected_text: Optional[str] = None
    fk_score: float = 0.0
    is_valid: bool = True
    skip_reason: str = ""
    is_long: bool = False
    probs: Any = None  # torch.Tensor at runtime
    stats: Dict[str, float] = field(default_factory=dict)
    result: Any = None  # DetectionResult at runtime
