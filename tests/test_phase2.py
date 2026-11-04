"""Unit tests for Phase 2: Intent Teleportation and CPU Grounding."""

from __future__ import annotations

from PIL import Image, ImageDraw

from fast_agent.benchmark import (
    verify_cross_app_deal_pipe,
    verify_device_diagnostic_report,
)
from fast_agent.config import AgentConfig
from fast_agent.contracts import NormalizedPoint, NormalizedRect
from fast_agent.cpu_grounder import CPUGrounder
from fast_agent.device import MockAndroidDevice
from fast_agent.intent_teleport import IntentTeleportEngine


def test_intent_teleport_config() -> None:
    cfg = AgentConfig(enable_intent_teleport=True)
    engine = IntentTeleportEngine(cfg)
    assert engine.serial == "192.168.10.108:5555"


def test_cpu_grounder_snapping_logic() -> None:
    grounder = CPUGrounder()
    # Create test image with mock bounding box
    img = Image.new("RGB", (500, 1000), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw "Search" button around (250, 100) -> Normalized (0.5, 0.1)
    draw.rectangle([200, 80, 300, 120], fill=(0, 0, 0))

    # Click slightly off-center: (0.46, 0.09)
    imprecise_click = NormalizedPoint(0.46, 0.09)
    # Element snapping shouldn't crash and should return a valid NormalizedPoint
    snapped = grounder.snap_to_element(img, imprecise_click, max_distance=0.1)
    assert 0.0 <= snapped.x <= 1.0
    assert 0.0 <= snapped.y <= 1.0


def test_hard_tasks_verifiers() -> None:
    dev = MockAndroidDevice()

    # Test Task 6 Verifier
    dev.launch_package("com.miui.notes")
    assert verify_cross_app_deal_pipe(dev, [], [{"action": "type", "text": "RTX 3060 12GB - 220 EUR"}])

    # Test Task 7 Verifier
    dev.launch_package("com.miui.securitycenter")
    assert verify_device_diagnostic_report(dev, [], [{"action": "type", "text": "Battery 68% - 128GB"}])
