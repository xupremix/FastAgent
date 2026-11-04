"""Adaptive Settling Detector (CPU-Optimized via NumPy).

Replaces static sleeps and arbitrary agent wait times (1.5s - 3.0s) with
real-time visual stabilization detection.
Compares successive downscaled frame deltas on the Xeon CPU:
- When screen is actively animating / loading: delta > threshold (1.0%)
- When screen stabilizes: delta < threshold for 2 consecutive samples
- Exits dynamically in 60ms - 150ms instead of 1500ms - 3000ms.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image


class AdaptiveSettler:
    """Detects UI stabilization via low-overhead pixel delta on downscaled thumbnails."""

    def __init__(
        self,
        thumbnail_size: Tuple[int, int] = (160, 320),
        delta_threshold: float = 0.008,
        min_wait_s: float = 0.04,
        max_wait_s: float = 1.0,
        poll_interval_s: float = 0.03,
    ) -> None:
        self.thumb_w, self.thumb_h = thumbnail_size
        self.delta_threshold = delta_threshold
        self.min_wait_s = min_wait_s
        self.max_wait_s = max_wait_s
        self.poll_interval_s = poll_interval_s
        self.last_delta_mask: Optional[np.ndarray[Any, Any]] = None

    def _to_thumbnail(self, img: Image.Image) -> np.ndarray[Any, Any]:
        gray = img.convert("L").resize((self.thumb_w, self.thumb_h), Image.Resampling.BILINEAR)
        return np.asarray(gray, dtype=np.float32) / 255.0

    def compute_delta(self, thumb_a: np.ndarray[Any, Any], thumb_b: np.ndarray[Any, Any]) -> float:
        """Mean absolute pixel difference normalized to [0.0, 1.0]."""
        return float(np.mean(np.abs(thumb_a - thumb_b)))

    def wait_until_settled(
        self,
        frame_supplier: Callable[[], Optional[Image.Image]],
        timeout_s: Optional[float] = None,
    ) -> float:
        """Poll frames until pixel motion falls below threshold or timeout is reached.

        Returns:
            settle_ms: Total time elapsed waiting for stability.
        """
        t0 = time.perf_counter()
        timeout = timeout_s or self.max_wait_s

        # Get initial frame
        initial_img = frame_supplier()
        if initial_img is None:
            time.sleep(self.min_wait_s)
            return (time.perf_counter() - t0) * 1000.0

        prev_thumb = self._to_thumbnail(initial_img)
        stable_count = 0

        while (time.perf_counter() - t0) < timeout:
            time.sleep(self.poll_interval_s)
            curr_img = frame_supplier()
            if curr_img is None:
                continue

            curr_thumb = self._to_thumbnail(curr_img)
            diff = np.abs(prev_thumb - curr_thumb)
            self.last_delta_mask = (diff > 0.05).astype(np.uint8)
            delta = float(np.mean(diff))
            prev_thumb = curr_thumb

            if delta < self.delta_threshold:
                stable_count += 1
                if stable_count >= 2:
                    # Verified stable over 2 consecutive samples
                    break
            else:
                stable_count = 0

        # Enforce min_wait_s
        elapsed = time.perf_counter() - t0
        if elapsed < self.min_wait_s:
            time.sleep(self.min_wait_s - elapsed)

        return (time.perf_counter() - t0) * 1000.0
