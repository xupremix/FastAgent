"""Unit tests for Phase 4: Scroll Dead-Reckoning, Speculative Dispatch, and Trajectory Distillation."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import pytest
from PIL import Image

from fast_agent.distillation import (
    NGramDraftPredictor,
    TrajectoryDistiller,
    TrajectoryTurn,
)
from fast_agent.scroll_dead_reckoning import ScrollDeadReckoner
from fast_agent.speculative_execution import StreamingActionDispatcher


def test_scroll_dead_reckoning_displacement_and_boundary() -> None:
    """Verify Fourier phase correlation calculates vertical displacement and flags boundaries."""
    reckoner = ScrollDeadReckoner(thumbnail_size=(128, 256), min_displacement_threshold_px=6.0)

    # Base frame (1220x2712) with horizontal texture
    arr1 = np.zeros((2712, 1220), dtype=np.uint8)
    for y in range(0, 2712, 40):
        arr1[y : y + 10, :] = 200
    img1 = Image.fromarray(arr1)

    # Frame 2 shifted down by 160 physical pixels
    arr2 = np.roll(arr1, 160, axis=0)
    img2 = Image.fromarray(arr2)

    # 1. Normal scroll measurement
    res = reckoner.measure_displacement(img1, img2, commanded_dy_px=160.0)
    assert res.computation_ms < 20.0  # Must be fast (< 20ms on CPU)
    assert pytest.approx(res.dy_px, abs=15.0) == 160.0
    assert not res.is_end_of_scroll

    # 2. End-of-scroll boundary detection: commanded swipe of 400px but screen did not move
    static_res = reckoner.measure_displacement(img1, img1, commanded_dy_px=400.0)
    assert static_res.dy_px == 0.0
    assert static_res.is_end_of_scroll


def test_streaming_action_dispatcher() -> None:
    """Verify incremental token streaming dispatches actions immediately without double-firing."""
    dispatched: List[Dict[str, Any]] = []

    def on_dispatch(act: Dict[str, Any]) -> None:
        dispatched.append(act)

    dispatcher = StreamingActionDispatcher(dispatch_callback=on_dispatch)

    # Feed tokens token-by-token
    tokens = [
        "T:Open",
        " App|A:",
        "OPN(",
        '"Wallapop"',
        ")>",
        "WAT(",
        "1.0)",
    ]

    for tok in tokens:
        dispatcher.feed_token(tok)

    # After full token stream, both actions must have dispatched
    assert len(dispatched) == 2
    assert dispatched[0]["action"] == "open"
    assert dispatched[0]["text"] == "Wallapop"
    assert dispatched[1]["action"] == "wait"
    assert dispatched[1]["time"] == 1.0

    # Finalize stream and verify time saved is positive
    t_end = time.perf_counter() + 0.1
    full_text = "".join(tokens)
    saved = dispatcher.finalize_stream(full_text, decode_end_ts=t_end, success=True)
    assert saved > 0.0
    assert dispatcher.stats.total_dispatched == 2
    assert dispatcher.stats.wasted_work_count == 0
    assert dispatcher.stats.wasted_work_rate == 0.0


def test_trajectory_distiller_and_draft_predictor() -> None:
    """Verify ShareGPT dataset export and N-gram speculative draft generation."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset_path = os.path.join(tmp_dir, "trajectories.jsonl")
        distiller = TrajectoryDistiller(dataset_path=dataset_path)

        turn = TrajectoryTurn(
            task_id="test_task",
            run_index=1,
            step_index=1,
            instruction="Open settings and check battery",
            package="com.miui.home",
            action_dsl='OPN("Settings")>WAT(1.0)',
            raw_response='T:Open Settings|A:OPN("Settings")>WAT(1.0)',
            success=True,
            step_total_ms=1250.0,
        )

        distiller.record_turn(turn)
        assert distiller.count_samples() == 1

        # Check N-gram draft predictor
        predictor = NGramDraftPredictor()
        predictor.train_on_turn("com.miui.home", 1, 'OPN("Settings")>WAT(1.0)')
        predictor.train_on_turn("com.miui.home", 1, 'OPN("Settings")>WAT(1.0)')
        predictor.train_on_turn("com.miui.home", 1, 'CLK(500, 500)')

        draft = predictor.predict_draft("com.miui.home", 1)
        assert draft == 'OPN("Settings")>WAT(1.0)'
        assert predictor.predict_draft("unknown.pkg", 1) is None
