"""Zero-VLM CPU Micro-Verifier & Element Snapper (Phase 2b & 2c).

Runs RapidOCR on the host Xeon E5 CPU to:
1. Verify deterministic UI text targets (e.g. 'Filtri', 'Prezzo', 'Note', 'Cerca') in ~900ms,
   bypassing the 5.6s GPU ViT prefill and LLM decode turns.
2. Snap arbitrary agent click coordinates to the true touchable text/button centroids,
   preventing edge-clipping and touch misalignments.
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-untyped]

from fast_agent.contracts import NormalizedPoint, NormalizedRect


class CPUGrounder:
    """CPU-based perception and grounding engine."""

    def __init__(self) -> None:
        self._ocr: Optional[RapidOCR] = None

    def _ensure_ocr(self) -> RapidOCR:
        if self._ocr is None:
            self._ocr = RapidOCR()
        return self._ocr

    def extract_elements(self, img: Image.Image) -> List[Tuple[str, NormalizedRect, float]]:
        """Run OCR over image and return list of (text, NormalizedRect, confidence)."""
        ocr = self._ensure_ocr()
        arr = np.array(img)
        h, w = arr.shape[:2]
        raw_res, _ = ocr(arr)
        if not raw_res:
            return []

        elements: List[Tuple[str, NormalizedRect, float]] = []
        for item in raw_res:
            box = item[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            text = str(item[1]).strip()
            score = float(item[2])

            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            norm_rect = NormalizedRect(
                x1=max(0.0, min(1.0, float(min_x) / float(w))),
                y1=max(0.0, min(1.0, float(min_y) / float(h))),
                x2=max(0.0, min(1.0, float(max_x) / float(w))),
                y2=max(0.0, min(1.0, float(max_y) / float(h))),
            )
            elements.append((text, norm_rect, score))

        return elements

    def find_text(
        self,
        img: Image.Image,
        text_query: str,
        min_score: float = 0.5,
    ) -> Optional[NormalizedPoint]:
        """Find text target on screen and return its normalized centroid."""
        elements = self.extract_elements(img)
        query_clean = text_query.lower().strip()

        best_match: Optional[NormalizedPoint] = None
        best_score = 0.0

        for text, rect, score in elements:
            if score < min_score:
                continue
            text_clean = text.lower()
            if query_clean in text_clean or text_clean in query_clean:
                if score > best_score:
                    best_score = score
                    best_match = rect.centroid()

        return best_match

    def snap_to_element(
        self,
        img: Image.Image,
        click_point: NormalizedPoint,
        max_distance: float = 0.08,
    ) -> NormalizedPoint:
        """Snap a normalized coordinate to the nearest detected element centroid.

        Prevents clicking near element margins, borders, or dead zones.
        """
        elements = self.extract_elements(img)
        if not elements:
            return click_point

        closest_centroid = click_point
        min_dist = max_distance

        for _, rect, _ in elements:
            centroid = rect.centroid()
            dist = math.hypot(click_point.x - centroid.x, click_point.y - centroid.y)
            if dist < min_dist:
                min_dist = dist
                closest_centroid = centroid

        return closest_centroid
