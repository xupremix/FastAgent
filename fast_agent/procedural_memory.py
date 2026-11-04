"""Phase 3a: Self-Evolving Noise-Tolerant Procedural Memory.

Features:
- SQLite-backed graph store of verified UI state transitions and macros.
- Theme-invariant Sobel edge gradient dHash64 for local ROIs (survives Dark/Light mode toggles).
- Vertical sliding-window micro-search [-6%, +9%] to compensate for dynamic ad banners/shifts.
- Safe macro replay: executes only when anchor Hamming distance <= threshold and EMA confidence > 0.85.
- Self-repair: automatic confidence decay, mid-flight abort on mismatch, and VLM recovery trigger.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from fast_agent.contracts import NormalizedPoint, NormalizedRect


def compute_sobel_edge_dhash(img_gray: np.ndarray[Any, Any]) -> int:
    """Compute 64-bit difference hash over Sobel gradient magnitude.
    
    Invariant to contrast inversion (light vs dark mode) because edge gradients
    have the same spatial positions regardless of background polarity.
    """
    if img_gray.size == 0:
        return 0
    # Downscale to 64x64 for sub-millisecond gradient computation
    tiny = cv2.resize(img_gray, (64, 64), interpolation=cv2.INTER_LINEAR)
    gx = cv2.Sobel(tiny, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tiny, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Downsample magnitude to 9x8 for dHash
    small = cv2.resize(mag, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = 0
    for r in range(8):
        for c in range(8):
            if small[r, c] > small[r, c + 1]:
                dhash |= 1 << (r * 8 + c)
    return int(dhash)


def hamming_distance(h1: int, h2: int) -> int:
    """Calculate Hamming bit distance between two 64-bit integer hashes."""
    return (h1 ^ h2).bit_count()


@dataclass
class ScreenAnchor:
    """Perceptual and semantic fingerprint of a UI screen state."""
    package: str
    activity: str
    roi_box: NormalizedRect
    sobel_edge_dhash: int
    landmark_tokens: List[str] = field(default_factory=list)


@dataclass
class CachedMacro:
    """Stored macro action chain with verification criteria."""
    macro_id: str
    task_id: str
    subgoal_signature: str
    step_index: int
    anchor: ScreenAnchor
    action_chain: List[Dict[str, Any]]
    postcondition: Dict[str, Any]
    ema_confidence: float = 1.0
    execution_count: int = 1
    last_success_ts: float = field(default_factory=time.time)


class ProceduralMemory:
    """SQLite-backed procedural memory manager for fast macro replay."""

    def __init__(
        self,
        db_path: str = "~/.fast_agent/procedural_memory.db",
        dhash_threshold: int = 5,
        min_confidence: float = 0.85,
    ) -> None:
        self.db_path = os.path.expanduser(db_path)
        dir_name = os.path.dirname(self.db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        self.dhash_threshold = dhash_threshold
        self.min_confidence = min_confidence
        self._shared_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:")
            self._shared_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS macros (
                    macro_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    subgoal_signature TEXT NOT NULL,
                    step_index INTEGER NOT NULL,
                    package TEXT NOT NULL,
                    activity TEXT NOT NULL,
                    roi_x1 REAL NOT NULL,
                    roi_y1 REAL NOT NULL,
                    roi_x2 REAL NOT NULL,
                    roi_y2 REAL NOT NULL,
                    sobel_edge_dhash TEXT NOT NULL,
                    landmark_tokens TEXT NOT NULL,
                    action_chain TEXT NOT NULL,
                    postcondition TEXT NOT NULL,
                    ema_confidence REAL NOT NULL,
                    execution_count INTEGER NOT NULL,
                    last_success_ts REAL NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_task_step ON macros (task_id, step_index)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pkg ON macros (package)"
            )
            conn.commit()

    def clear_all(self) -> None:
        """Purge all stored macros (used for test teardown)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM macros")
            conn.commit()

    def sliding_window_search(
        self,
        screen_gray: np.ndarray[Any, Any],
        roi_box: NormalizedRect,
        target_dhash: int,
    ) -> Tuple[int, float]:
        """Perform vertical micro-search [-6%, +9%] to detect banner shifts.
        
        Returns:
            (best_hamming_distance, best_dy_normalized)
        """
        h, w = screen_gray.shape[:2]
        best_dist = 64
        best_dy = 0.0

        # Priority order: search 0% first, then expand outward (+/- 1%, +/- 2%, ...)
        search_order = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, 8, 9]

        for dy_pct in search_order:
            dy = dy_pct / 100.0
            y1 = max(0.0, min(1.0, roi_box.y1 + dy))
            y2 = max(0.0, min(1.0, roi_box.y2 + dy))
            if y2 - y1 < 0.02:
                continue

            py1 = int(y1 * h)
            py2 = int(y2 * h)
            px1 = int(roi_box.x1 * w)
            px2 = int(roi_box.x2 * w)

            crop = screen_gray[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            crop_dhash = compute_sobel_edge_dhash(crop)
            dist = hamming_distance(target_dhash, crop_dhash)
            if dist < best_dist:
                best_dist = dist
                best_dy = dy
                if best_dist == 0 and dy == 0.0:
                    break

        return best_dist, best_dy

    def query_macro(
        self,
        task_id: str,
        step_index: int,
        package: str,
        screen_img: Image.Image,
        landmark_tokens: Optional[List[str]] = None,
    ) -> Optional[Tuple[List[Dict[str, Any]], str, float, float]]:
        """Query for a matching verified macro.
        
        Returns:
            (adjusted_action_chain, macro_id, confidence, vertical_shift_dy) or None
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM macros 
                WHERE task_id = ? AND step_index = ? AND package = ?
                """,
                (task_id, step_index, package),
            )
            rows = cursor.fetchall()

        if not rows:
            return None

        # Convert screen to grayscale numpy array
        screen_arr = np.array(screen_img.convert("L"))

        best_match: Optional[Tuple[List[Dict[str, Any]], str, float, float]] = None
        min_dist = 64

        for row in rows:
            conf = float(row["ema_confidence"])
            if conf < self.min_confidence:
                continue

            roi = NormalizedRect(
                x1=float(row["roi_x1"]),
                y1=float(row["roi_y1"]),
                x2=float(row["roi_x2"]),
                y2=float(row["roi_y2"]),
            )
            raw_dhash = row["sobel_edge_dhash"]
            stored_dhash = int(str(raw_dhash), 16) if isinstance(raw_dhash, str) else int(raw_dhash)

            # Search with vertical micro-shift
            dist, dy = self.sliding_window_search(screen_arr, roi, stored_dhash)

            if dist <= self.dhash_threshold and dist < min_dist:
                min_dist = dist
                raw_actions = json.loads(str(row["action_chain"]))
                # Adjust any coordinate actions with the vertical offset dy
                adjusted_actions: List[Dict[str, Any]] = []
                for act in raw_actions:
                    act_copy = dict(act)
                    if "coordinate" in act_copy and dy != 0.0:
                        coords = list(act_copy["coordinate"])
                        # Coordinate is in 0..1000 scale
                        coords[1] = max(0, min(1000, int(coords[1] + dy * 1000.0)))
                        act_copy["coordinate"] = coords
                    adjusted_actions.append(act_copy)

                best_match = (adjusted_actions, str(row["macro_id"]), conf, dy)

        return best_match

    def store_macro(
        self,
        task_id: str,
        subgoal_signature: str,
        step_index: int,
        package: str,
        activity: str,
        screen_img: Image.Image,
        roi_box: NormalizedRect,
        action_chain: List[Dict[str, Any]],
        postcondition: Dict[str, Any],
        landmark_tokens: Optional[List[str]] = None,
    ) -> str:
        """Store or update a verified macro sequence."""
        macro_id = f"{task_id}_{step_index}_{int(time.time() * 1000)}"

        # Compute Sobel edge dHash on ROI
        screen_arr = np.array(screen_img.convert("L"))
        h, w = screen_arr.shape[:2]
        py1 = int(roi_box.y1 * h)
        py2 = int(roi_box.y2 * h)
        px1 = int(roi_box.x1 * w)
        px2 = int(roi_box.x2 * w)
        crop = screen_arr[py1:py2, px1:px2]
        dhash = compute_sobel_edge_dhash(crop)
        dhash_hex = f"{dhash:016x}"

        tokens_json = json.dumps(landmark_tokens or [])
        actions_json = json.dumps(action_chain)
        postcond_json = json.dumps(postcondition)

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO macros (
                    macro_id, task_id, subgoal_signature, step_index,
                    package, activity, roi_x1, roi_y1, roi_x2, roi_y2,
                    sobel_edge_dhash, landmark_tokens, action_chain,
                    postcondition, ema_confidence, execution_count, last_success_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 1, ?)
                """,
                (
                    macro_id,
                    task_id,
                    subgoal_signature,
                    step_index,
                    package,
                    activity,
                    roi_box.x1,
                    roi_box.y1,
                    roi_box.x2,
                    roi_box.y2,
                    dhash_hex,
                    tokens_json,
                    actions_json,
                    postcond_json,
                    time.time(),
                ),
            )
            conn.commit()

        return macro_id

    def record_feedback(self, macro_id: str, success: bool) -> float:
        """Update EMA confidence: alpha=0.2."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT ema_confidence, execution_count, last_success_ts FROM macros WHERE macro_id = ?",
                (macro_id,),
            )
            row = cursor.fetchone()
            if not row:
                return 0.0

            old_conf = float(row["ema_confidence"])
            exec_count = int(row["execution_count"]) + 1

            alpha = 0.2
            reward = 1.0 if success else 0.0
            new_conf = (1.0 - alpha) * old_conf + alpha * reward

            conn.execute(
                """
                UPDATE macros 
                SET ema_confidence = ?, execution_count = ?, last_success_ts = ?
                WHERE macro_id = ?
                """,
                (
                    new_conf,
                    exec_count,
                    time.time() if success else float(row["last_success_ts"] or 0.0),
                    macro_id,
                ),
            )
            conn.commit()

        return new_conf
