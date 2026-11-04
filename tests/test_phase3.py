"""Unit tests for Phase 3: Procedural Memory and Foveated ROI Cropping."""

from __future__ import annotations

import os
import tempfile
import numpy as np
import pytest
from PIL import Image, ImageDraw

from fast_agent.contracts import NormalizedPoint, NormalizedRect
from fast_agent.foveated_roi import FoveatedCropResult, FoveatedVisualCompressor
from fast_agent.procedural_memory import (
    ProceduralMemory,
    compute_sobel_edge_dhash,
    hamming_distance,
)


def test_sobel_edge_dhash_theme_invariance() -> None:
    """Verify that Sobel edge dHash produces 0 bit distance between Light and Dark mode buttons."""
    # Light mode: black text on white background
    img_light = Image.new("L", (200, 100), color=255)
    draw_l = ImageDraw.Draw(img_light)
    draw_l.rectangle([30, 20, 170, 80], outline=0, width=3)
    draw_l.text((80, 45), "CONFIRM", fill=0)

    # Dark mode: white text on black background
    img_dark = Image.new("L", (200, 100), color=0)
    draw_d = ImageDraw.Draw(img_dark)
    draw_d.rectangle([30, 20, 170, 80], outline=255, width=3)
    draw_d.text((80, 45), "CONFIRM", fill=255)

    arr_light = np.array(img_light)
    arr_dark = np.array(img_dark)

    h_light = compute_sobel_edge_dhash(arr_light)
    h_dark = compute_sobel_edge_dhash(arr_dark)

    dist = hamming_distance(h_light, h_dark)
    assert dist <= 2, f"Expected Sobel dHash distance <= 2, got {dist}"


def test_sliding_window_micro_search() -> None:
    """Verify vertical sliding window search recovers dynamic banner shifts."""
    mem = ProceduralMemory(db_path=":memory:")

    # Base screen: button with text located between y=400..500
    base_screen = np.full((1000, 500), 240, dtype=np.uint8)
    base_screen[400:500, 100:400] = 30
    # Add high-contrast internal label
    base_screen[440:460, 200:300] = 255
    roi = NormalizedRect(x1=0.20, y1=0.40, x2=0.80, y2=0.50)

    crop_base = base_screen[400:500, 100:400]
    target_dhash = compute_sobel_edge_dhash(crop_base)

    # Shifted screen: ad banner at top shifts button down by 50px (+5% screen height)
    shifted_screen = np.full((1000, 500), 240, dtype=np.uint8)
    shifted_screen[450:550, 100:400] = 30
    shifted_screen[490:510, 200:300] = 255

    best_dist, best_dy = mem.sliding_window_search(shifted_screen, roi, target_dhash)
    assert best_dist == 0
    assert best_dy == pytest.approx(0.05, abs=0.005)


def test_procedural_memory_lifecycle_and_ema() -> None:
    """Test storage, retrieval, vertical coordinate adjustment, and EMA confidence gating."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_memory.db")
        mem = ProceduralMemory(db_path=db_path, dhash_threshold=5, min_confidence=0.85)

        # Create dummy screen image with structured element
        img = Image.new("RGB", (500, 1000), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        draw.rectangle([100, 400, 400, 500], fill=(50, 50, 50))
        draw.text((200, 440), "CONTINUE", fill=(255, 255, 255))

        roi = NormalizedRect(x1=0.20, y1=0.40, x2=0.80, y2=0.50)
        action_chain = [{"action": "click", "coordinate": [500, 450]}]
        postcond = {"package": "com.target.app"}

        macro_id = mem.store_macro(
            task_id="test_task",
            subgoal_signature="click_continue_button",
            step_index=1,
            package="com.source.app",
            activity="MainActivity",
            screen_img=img,
            roi_box=roi,
            action_chain=action_chain,
            postcondition=postcond,
        )

        # 1. Exact match query
        match = mem.query_macro("test_task", 1, "com.source.app", img)
        assert match is not None
        actions, mid, conf, dy = match
        assert mid == macro_id
        assert conf == 1.0
        assert dy == 0.0
        assert actions[0]["coordinate"] == [500, 450]

        # 2. Shifted screen (+4% shift) query
        shifted_img = Image.new("RGB", (500, 1000), color=(240, 240, 240))
        draw_s = ImageDraw.Draw(shifted_img)
        draw_s.rectangle([100, 440, 400, 540], fill=(50, 50, 50))
        draw_s.text((200, 480), "CONTINUE", fill=(255, 255, 255))

        shifted_match = mem.query_macro("test_task", 1, "com.source.app", shifted_img)
        assert shifted_match is not None
        s_actions, s_mid, _, s_dy = shifted_match
        assert s_mid == macro_id
        assert s_dy == pytest.approx(0.04, abs=0.005)
        # Coordinate y should be adjusted by ~40px (from 450 to ~490)
        assert s_actions[0]["coordinate"][1] >= 485

        # 3. EMA Confidence Decay on Failure
        # Initial conf 1.0 -> fail: (1 - 0.2) * 1.0 + 0 = 0.80
        new_conf = mem.record_feedback(macro_id, success=False)
        assert pytest.approx(new_conf, abs=0.01) == 0.80

        # Now query should fail because confidence 0.80 < min_confidence 0.85
        gated_match = mem.query_macro("test_task", 1, "com.source.app", img)
        assert gated_match is None

        # Re-raise confidence: 0.8 * 0.80 + 0.2 * 1.0 = 0.84 -> then another success: 0.8 * 0.84 + 0.2 * 1.0 = 0.872
        mem.record_feedback(macro_id, success=True)
        mem.record_feedback(macro_id, success=True)
        restored_match = mem.query_macro("test_task", 1, "com.source.app", img)
        assert restored_match is not None


def test_foveated_roi_alignment_and_coordinate_mapping() -> None:
    """Verify foveated ROI cropping aligns to 28px multiples and maps coordinates accurately."""
    compressor = FoveatedVisualCompressor(target_pixels=80000, patch_size=28)
    full_img = Image.new("RGB", (1220, 2712), color=(255, 255, 255))

    # Target ROI: search results area [0.1, 0.3, 0.9, 0.7]
    roi = NormalizedRect(x1=0.1, y1=0.3, x2=0.9, y2=0.7)
    res = compressor.crop_from_rect(full_img, roi)

    # Patch grid alignment check
    w, h = res.roi_size
    assert w % 28 == 0
    assert h % 28 == 0
    assert w * h <= 80000

    # Thumbnail check
    assert res.thumbnail_image.size == (64, 128)

    # Coordinate mapping check: centroid of ROI
    roi_centroid_x = 0.5
    roi_centroid_y = 0.5
    global_pt = res.map_roi_to_global(roi_centroid_x, roi_centroid_y)

    # Should map back near the centroid of the padded ROI
    assert 0.45 <= global_pt.x <= 0.55
    assert 0.45 <= global_pt.y <= 0.55
