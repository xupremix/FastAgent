"""Phase 5c: Prior-Guided Set-of-Mark (SoM) Visual Tagger.

Features:
- Queries the Knowledge Graph for Bayesian coordinate priors on known UIStateNodes.
- Draws high-contrast numbered tags ([1], [2], [3]) onto the BGR frame buffer via OpenCV in <2ms.
- Allows the VLM to emit CLICK_TAG(id) instead of noisy floating-point coordinates,
  eliminating coordinate drift and snapping directly to verified control centroids.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from fast_agent.contracts import NormalizedPoint, NormalizedRect
from fast_agent.ui_knowledge_graph import FrequentElement


@dataclass
class TaggedElement:
    """A visually tagged control mapped to a numbered marker."""
    tag_id: int
    label: str
    centroid: NormalizedPoint
    bounding_rect: NormalizedRect


class SetOfMarkTagger:
    """Draws low-overhead visual markers on UI frames using OpenCV."""

    def __init__(self, font_scale: float = 0.6, tag_padding: int = 4) -> None:
        self.font_scale = font_scale
        self.tag_padding = tag_padding

    def tag_frame(
        self,
        image: Image.Image,
        elements: List[FrequentElement],
    ) -> Tuple[Image.Image, Dict[int, TaggedElement], float]:
        """Draw numbered tags onto image and return the tagged image and mapping dictionary.

        Args:
            image: Input screen frame.
            elements: Known interactive elements with Bayesian coordinate priors.

        Returns:
            (tagged_pil_image, tag_map, elapsed_ms)
        """
        t0 = time.perf_counter()
        orig_w, orig_h = image.size
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        tag_map: Dict[int, TaggedElement] = {}

        for idx, el in enumerate(elements, start=1):
            px = int(el.mean_x * (orig_w - 1))
            py = int(el.mean_y * (orig_h - 1))

            tag_text = f"[{idx}]"
            (text_w, text_h), baseline = cv2.getTextSize(
                tag_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 2
            )

            # Draw high-contrast dark badge with bright border
            x1 = max(0, px - text_w // 2 - self.tag_padding)
            y1 = max(0, py - text_h // 2 - self.tag_padding)
            x2 = min(orig_w, px + text_w // 2 + self.tag_padding)
            y2 = min(orig_h, py + text_h // 2 + self.tag_padding)

            # Background rectangle: solid black with yellow border
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (20, 20, 20), -1)
            cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 215, 255), 2)  # Bright Gold
            cv2.putText(
                cv_img,
                tag_text,
                (x1 + self.tag_padding, y2 - self.tag_padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                2,
            )

            norm_rect = NormalizedRect(
                x1=float(x1) / float(orig_w),
                y1=float(y1) / float(orig_h),
                x2=float(x2) / float(orig_w),
                y2=float(y2) / float(orig_h),
            )
            tag_map[idx] = TaggedElement(
                tag_id=idx,
                label=el.label,
                centroid=NormalizedPoint(el.mean_x, el.mean_y),
                bounding_rect=norm_rect,
            )

        rgb_result = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return Image.fromarray(rgb_result), tag_map, elapsed_ms

    def resolve_tag(
        self,
        tag_id: int,
        tag_map: Dict[int, TaggedElement],
    ) -> Optional[NormalizedPoint]:
        """Convert a tag ID into physical normalized screen centroid."""
        tagged = tag_map.get(tag_id)
        if not tagged:
            return None
        return tagged.centroid
