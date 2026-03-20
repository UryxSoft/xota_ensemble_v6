"""
Batch processing types for SOTA AI Text Detector.
Unchanged from original — included for completeness.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class BatchItem:
    index: int
    original_text: str
    cleaned_text: Optional[str] = None
    injected_text: Optional[str] = None
    fk_score: float = 0.0
    is_valid: bool = True
    skip_reason: str = ""
    is_long: bool = False
    probs: Any = None
    stats: Dict[str, float] = field(default_factory=dict)
    result: Any = None
