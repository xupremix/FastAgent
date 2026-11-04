"""Phase 3b: Dynamic Foveated ROI Cropping & ViT Patch Alignment.

Features:
- Reduces visual tokens from ~660 down to ~140-160 by cropping the active Region of Interest (ROI).
- Enforces multiples of 28 on both width and height for ViT patch-grid alignment.
- Generates a lightweight 64x128 global context thumbnail.
- Provides strict bidirectional coordinate translation mapping local ROI coordinates
  back to global normalized [0.0, 1.0] screen coordinates.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
from PIL import Image

from fast_agent.contracts import NormalizedPoint, NormalizedRect, align_to_patch_grid


@dataclass
class FoveatedCropResult:
    """Output of foveated ROI cropping with spatial mapping."""
    roi_image: Image.Image
    thumbnail_image: Image.Image
    crop_rect: NormalizedRect  # Global coordinates of the cropped ROI [0.0, 1.0]
    original_size: Tuple[int, int]
    roi_size: Tuple[int, int]
    estimated_tokens: int

    def map_roi_to_global(self, norm_x: float, norm_y: float) -> NormalizedPoint:
        """Map normalized [0.0, 1.0] point within the ROI back to full-screen coordinates."""
        global_x = self.crop_rect.x1 + norm_x * (self.crop_rect.x2 - self.crop_rect.x1)
        global_y = self.crop_rect.y1 + norm_y * (self.crop_rect.y2 - self.crop_rect.y1)
        return NormalizedPoint(global_x, global_y)

    def to_roi_png_bytes(self) -> bytes:
        """Encode the cropped ROI to PNG bytes."""
        bio = io.BytesIO()
        self.roi_image.save(bio, format="PNG")
        return bio.getvalue()

    def to_thumbnail_png_bytes(self) -> bytes:
        """Encode the thumbnail to PNG bytes."""
        bio = io.BytesIO()
        self.thumbnail_image.save(bio, format="PNG")
        return bio.getvalue()


class FoveatedVisualCompressor:
    """Manages foveated cropping, ViT patch alignment, and coordinate translation."""

    def __init__(
        self,
        target_pixels: int = 80000,
        patch_size: int = 28,
        thumbnail_size: Tuple[int, int] = (64, 128),
        padding_ratio: float = 0.08,
    ) -> None:
        self.target_pixels = target_pixels
        self.patch_size = patch_size
        self.thumbnail_size = thumbnail_size
        self.padding_ratio = padding_ratio

    def crop_from_rect(
        self,
        full_image: Image.Image,
        roi_rect: NormalizedRect,
    ) -> FoveatedCropResult:
        """Crop and align an explicit normalized bounding box."""
        orig_w, orig_h = full_image.size

        # Apply padding ratio around the target ROI
        pad_x = (roi_rect.x2 - roi_rect.x1) * self.padding_ratio
        pad_y = (roi_rect.y2 - roi_rect.y1) * self.padding_ratio

        x1 = max(0.0, roi_rect.x1 - pad_x)
        y1 = max(0.0, roi_rect.y1 - pad_y)
        x2 = min(1.0, roi_rect.x2 + pad_x)
        y2 = min(1.0, roi_rect.y2 + pad_y)
        padded_rect = NormalizedRect(x1, y1, x2, y2)

        # Convert to physical pixel coordinates
        px1 = int(padded_rect.x1 * (orig_w - 1))
        py1 = int(padded_rect.y1 * (orig_h - 1))
        px2 = int(padded_rect.x2 * (orig_w - 1))
        py2 = int(padded_rect.y2 * (orig_h - 1))

        # Ensure minimal dimensions
        if px2 - px1 < self.patch_size:
            px2 = min(orig_w, px1 + self.patch_size)
        if py2 - py1 < self.patch_size:
            py2 = min(orig_h, py1 + self.patch_size)

        cropped = full_image.crop((px1, py1, px2, py2))
        crop_w, crop_h = cropped.size

        # Align to ViT patch grid targeting ~80k pixels
        target_w, target_h = align_to_patch_grid(
            crop_w,
            crop_h,
            factor=self.patch_size,
            max_pixels=self.target_pixels,
        )
        resized_roi = cropped.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # Generate lightweight context thumbnail
        thumbnail = full_image.resize(self.thumbnail_size, Image.Resampling.BILINEAR)

        # Token count is approximately (w/patch_size) * (h/patch_size)
        estimated_tokens = (target_w // self.patch_size) * (target_h // self.patch_size)

        return FoveatedCropResult(
            roi_image=resized_roi,
            thumbnail_image=thumbnail,
            crop_rect=padded_rect,
            original_size=(orig_w, orig_h),
            roi_size=(target_w, target_h),
            estimated_tokens=estimated_tokens,
        )

    def crop_from_delta_mask(
        self,
        full_image: Image.Image,
        delta_mask: np.ndarray[Any, Any],
    ) -> Optional[FoveatedCropResult]:
        """Compute bounding box from settling delta heatmap and crop."""
        ys, xs = np.where(delta_mask > 0)
        if len(ys) == 0 or len(xs) == 0:
            return None

        h, w = delta_mask.shape[:2]
        x1 = float(np.min(xs)) / float(w)
        y1 = float(np.min(ys)) / float(h)
        x2 = float(np.max(xs)) / float(w)
        y2 = float(np.max(ys)) / float(h)

        # Discard if the entire screen changed (> 80% coverage)
        area = (x2 - x1) * (y2 - y1)
        if area > 0.85 or area < 0.005:
            return None

        return self.crop_from_rect(full_image, NormalizedRect(x1, y1, x2, y2))
