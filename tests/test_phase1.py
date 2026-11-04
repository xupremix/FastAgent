"""Unit tests for Phase 1: Systems & Transport Optimizations."""

from __future__ import annotations

import numpy as np
from PIL import Image

from fast_agent.adaptive_settle import AdaptiveSettler
from fast_agent.config import AgentConfig
from fast_agent.contracts import NormalizedPoint
from fast_agent.device import MockAndroidDevice
from fast_agent.vlm import parse_dsl_actions, parse_tool_calls


def test_dsl_action_parsing_all_variants() -> None:
    dsl_text = (
        'CLK(500,80)>TYP("RTX 3060 12GB")>SWP(100,200,300,400)>'
        'KEY(enter)>OPN("Wallapop")>WAT(1.5)>TRM(success)'
    )
    actions = parse_dsl_actions(dsl_text)
    assert len(actions) == 7

    assert actions[0] == {"action": "click", "coordinate": [500, 80]}
    assert actions[1] == {"action": "type", "text": "RTX 3060 12GB"}
    assert actions[2] == {
        "action": "swipe",
        "coordinate": [100, 200],
        "coordinate2": [300, 400],
    }
    assert actions[3] == {"action": "key", "text": "enter"}
    assert actions[4] == {"action": "open", "text": "Wallapop"}
    assert actions[5] == {"action": "wait", "time": 1.5}
    assert actions[6] == {"action": "terminate", "status": "success"}


def test_parse_tool_calls_with_dsl_fallback() -> None:
    wrapped_dsl = "<tool_call>CLK(120,450)>KEY(back)</tool_call>"
    actions = parse_tool_calls(wrapped_dsl)
    assert len(actions) == 2
    assert actions[0]["action"] == "click"
    assert actions[1]["action"] == "key"


def test_adaptive_settler_pixel_delta() -> None:
    settler = AdaptiveSettler()

    # Identical frames -> delta == 0.0
    img_a = Image.new("RGB", (200, 400), color=(100, 100, 100))
    thumb_a = settler._to_thumbnail(img_a)
    delta_zero = settler.compute_delta(thumb_a, thumb_a)
    assert delta_zero == 0.0

    # Radically different frames -> delta > 0.4
    img_b = Image.new("RGB", (200, 400), color=(250, 250, 250))
    thumb_b = settler._to_thumbnail(img_b)
    delta_diff = settler.compute_delta(thumb_a, thumb_b)
    assert delta_diff > 0.4

    # Test wait_until_settled with static supplier
    settle_ms = settler.wait_until_settled(lambda: img_a, timeout_s=0.2)
    assert settle_ms > 0.0
    assert settle_ms < 200.0


def test_phase1_feature_flags() -> None:
    cfg = AgentConfig(
        enable_os_tuning=True,
        enable_scrcpy_socket=True,
        enable_fast_clipboard=True,
        enable_stream_capture=True,
        enable_adaptive_settling=True,
        enable_prefix_caching=True,
        enable_compact_dsl=True,
    )
    assert cfg.enable_os_tuning
    assert cfg.enable_scrcpy_socket
    assert cfg.enable_fast_clipboard
    assert cfg.enable_stream_capture
    assert cfg.enable_adaptive_settling
    assert cfg.enable_prefix_caching
    assert cfg.enable_compact_dsl


def test_mock_device_settle_contract() -> None:
    dev = MockAndroidDevice()
    ms = dev.settle(0.1)
    assert ms == 50.0
    assert ("settle", 0.1) in dev.action_history
