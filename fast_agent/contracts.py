"""Cross-cutting Architectural Contracts and Telemetry Types.

Enforces:
1. Normalized [0.0, 1.0] coordinate system for all internal reasoning.
2. ViT patch-grid aspect ratio rounding (multiples of 28).
3. Nanosecond-resolution per-stage telemetry schemas.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class NormalizedPoint:
    """A point strictly bounded in [0.0, 1.0] screen space."""
    x: float
    y: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.x <= 1.0 and 0.0 <= self.y <= 1.0):
            # Clamp gracefully with a tiny boundary tolerance for floating-point imprecision
            clamped_x = max(0.0, min(1.0, float(self.x)))
            clamped_y = max(0.0, min(1.0, float(self.y)))
            object.__setattr__(self, "x", clamped_x)
            object.__setattr__(self, "y", clamped_y)

    def to_physical(self, screen_w: int, screen_h: int) -> Tuple[int, int]:
        """Convert normalized point to physical device pixel coordinates."""
        px = int(round(self.x * (screen_w - 1)))
        py = int(round(self.y * (screen_h - 1)))
        return px, py

    @classmethod
    def from_physical(cls, px: int, py: int, screen_w: int, screen_h: int) -> NormalizedPoint:
        return cls(x=px / float(screen_w), y=py / float(screen_h))


@dataclass(frozen=True)
class NormalizedRect:
    """A bounding box [x1, y1, x2, y2] strictly bounded in [0.0, 1.0] space."""
    x1: float
    y1: float
    x2: float
    y2: float

    def centroid(self) -> NormalizedPoint:
        return NormalizedPoint(x=(self.x1 + self.x2) / 2.0, y=(self.y1 + self.y2) / 2.0)

    def to_physical(self, screen_w: int, screen_h: int) -> Tuple[int, int, int, int]:
        px1 = int(round(self.x1 * (screen_w - 1)))
        py1 = int(round(self.y1 * (screen_h - 1)))
        px2 = int(round(self.x2 * (screen_w - 1)))
        py2 = int(round(self.y2 * (screen_h - 1)))
        return px1, py1, px2, py2


def align_to_patch_grid(
    orig_w: int,
    orig_h: int,
    factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 320000,
) -> Tuple[int, int]:
    """Calculate resized dimensions that maintain aspect ratio and are exact multiples of `factor`.

    Guarantees that ViT patch extraction divides cleanly without interpolation artifacts.
    """
    scale = math.sqrt(max_pixels / float(orig_w * orig_h))
    if orig_w * orig_h * (scale**2) < min_pixels:
        scale = math.sqrt(min_pixels / float(orig_w * orig_h))

    # Scale and round to nearest factor
    target_w = max(factor, int(round((orig_w * scale) / factor)) * factor)
    target_h = max(factor, int(round((orig_h * scale) / factor)) * factor)

    # Ensure max_pixels constraint is strictly respected
    while target_w * target_h > max_pixels:
        if target_w >= target_h and target_w > factor:
            target_w -= factor
        elif target_h > factor:
            target_h -= factor
        else:
            break

    return target_w, target_h


@dataclass
class StepTiming:
    """High-resolution telemetry for a single agent action cycle."""
    step_index: int
    capture_ms: float = 0.0
    image_encode_ms: float = 0.0
    settle_ms: float = 0.0
    vit_prefill_ms: float = 0.0  # TTFT (Time To First Token)
    decode_ms: float = 0.0
    input_inject_ms: float = 0.0
    step_total_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    actions_executed: List[str] = field(default_factory=list)
    vlm_bypassed: bool = False
    speculative_time_saved_ms: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    timestamp_ns: int = field(default_factory=time.perf_counter_ns)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TelemetryLogger:
    """Thread-safe, append-only JSONL logger for per-stage latency metrics."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log_step(self, timing: StepTiming) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(timing.to_dict()) + "\n")
