"""Unit tests for Phase 0: Contracts, Config, Telemetry, and Device Mocking."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import pytest

from fast_agent.benchmark import (
    BenchmarkRunner,
    BenchmarkTask,
    get_standard_tasks,
    verify_wallapop_to_notes,
    verify_settings_battery,
    verify_telegram_saved,
    verify_gallery_inspect,
    verify_calculator_eval,
)
from fast_agent.config import AgentConfig
from fast_agent.contracts import (
    NormalizedPoint,
    NormalizedRect,
    StepTiming,
    TelemetryLogger,
    align_to_patch_grid,
)
from fast_agent.device import MockAndroidDevice
from fast_agent.vlm import parse_tool_calls


def test_normalized_point_clamping() -> None:
    p = NormalizedPoint(x=1.5, y=-0.2)
    assert p.x == 1.0
    assert p.y == 0.0

    p_valid = NormalizedPoint(x=0.5, y=0.5)
    assert p_valid.x == 0.5
    assert p_valid.y == 0.5


def test_normalized_point_to_physical() -> None:
    p = NormalizedPoint(x=0.5, y=0.5)
    # Physical screen 1220x2712 -> (0.5 * 1219, 0.5 * 2711) = (610, 1356)
    px, py = p.to_physical(1220, 2712)
    assert px == 610
    assert py == 1356

    p_origin = NormalizedPoint(x=0.0, y=0.0)
    assert p_origin.to_physical(1220, 2712) == (0, 0)

    p_max = NormalizedPoint(x=1.0, y=1.0)
    assert p_max.to_physical(1220, 2712) == (1219, 2711)


def test_normalized_rect_centroid() -> None:
    rect = NormalizedRect(x1=0.2, y1=0.4, x2=0.6, y2=0.8)
    centroid = rect.centroid()
    assert pytest.approx(centroid.x, 1e-4) == 0.4
    assert pytest.approx(centroid.y, 1e-4) == 0.6


def test_align_to_patch_grid() -> None:
    orig_w, orig_h = 1220, 2712
    target_w, target_h = align_to_patch_grid(orig_w, orig_h, factor=28, max_pixels=320000)
    # Check divisible by 28
    assert target_w % 28 == 0
    assert target_h % 28 == 0
    # Check max pixels bounded
    assert target_w * target_h <= 320000
    # Check approximate aspect ratio preserved (1220/2712 = 0.4498)
    aspect_orig = orig_w / float(orig_h)
    aspect_target = target_w / float(target_h)
    assert abs(aspect_orig - aspect_target) < 0.05


def test_agent_config_defaults() -> None:
    cfg = AgentConfig()
    # In Phase 0 Baseline, all optimization flags MUST be False
    assert not cfg.enable_os_tuning
    assert not cfg.enable_scrcpy_socket
    assert not cfg.enable_fast_clipboard
    assert not cfg.enable_stream_capture
    assert not cfg.enable_adaptive_settling
    assert not cfg.enable_prefix_caching
    assert not cfg.enable_compact_dsl
    assert not cfg.enable_intent_teleport
    assert not cfg.enable_cpu_ocr_verifier
    assert not cfg.enable_coordinate_snapping
    assert not cfg.enable_procedural_memory
    assert not cfg.enable_foveated_roi
    assert not cfg.enable_scroll_dead_reckoning
    assert not cfg.enable_speculative_execution


def test_telemetry_logger() -> None:
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        logger = TelemetryLogger(tmp_path)
        timing = StepTiming(
            step_index=1,
            capture_ms=850.5,
            vit_prefill_ms=4200.0,
            decode_ms=2300.0,
            input_inject_ms=12.0,
            step_total_ms=7362.5,
            prompt_tokens=650,
            completion_tokens=90,
            actions_executed=["tap", "type"],
            success=True,
        )
        logger.log_step(timing)

        with open(tmp_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 1
        data: Dict[str, Any] = lines[0]
        assert data["step_index"] == 1
        assert data["capture_ms"] == 850.5
        assert data["vit_prefill_ms"] == 4200.0
        assert data["prompt_tokens"] == 650
        assert data["actions_executed"] == ["tap", "type"]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_mock_android_device() -> None:
    dev = MockAndroidDevice(width=1220, height=2712)
    dev.pre_run_reset(["com.example.app"])
    assert ("pre_run_reset", ["com.example.app"]) in dev.action_history

    point = NormalizedPoint(0.5, 0.5)
    dev.tap(point)
    px, py = point.to_physical(1220, 2712)
    assert ("tap", (point, px, py)) in dev.action_history

    dev.input_text("Test input")
    assert ("input_text", "Test input") in dev.action_history

    dev.launch_package("com.example.app")
    assert dev.get_foreground_package() == "com.example.app"

    screenshot_bytes = dev.screencap()
    assert len(screenshot_bytes) > 0
    assert screenshot_bytes.startswith(b"\x89PNG")


def test_parse_tool_calls_single_and_chained() -> None:
    # Single action
    text1 = '<tool_call>{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 80]}}</tool_call>'
    actions1 = parse_tool_calls(text1)
    assert len(actions1) == 1
    assert actions1[0]["action"] == "click"
    assert actions1[0]["coordinate"] == [500, 80]

    # Chained actions
    text2 = '''Observation: Found search bar
<tool_call>
{"name": "mobile_use", "arguments": [
  {"action": "click", "coordinate": [500, 80]},
  {"action": "type", "text": "3060 12gb"}
]}
</tool_call>'''
    actions2 = parse_tool_calls(text2)
    assert len(actions2) == 2
    assert actions2[0]["action"] == "click"
    assert actions2[1]["action"] == "type"
    assert actions2[1]["text"] == "3060 12gb"


def test_benchmark_verifiers() -> None:
    dev = MockAndroidDevice()
    timings: List[StepTiming] = []

    # Test Task 1
    dev.launch_package("com.miui.notes")
    assert verify_wallapop_to_notes(dev, timings, [])

    # Test Task 2
    dev.launch_package("com.android.settings")
    assert verify_settings_battery(dev, timings, [])

    # Test Task 3
    dev.launch_package("org.telegram.messenger")
    assert verify_telegram_saved(dev, timings, [])

    # Test Task 4
    dev.launch_package("com.miui.gallery")
    assert verify_gallery_inspect(dev, timings, [])

    # Test Task 5
    dev.launch_package("com.miui.calculator")
    assert verify_calculator_eval(dev, timings, [])
