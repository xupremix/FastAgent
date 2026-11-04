"""Configuration and Feature Flags for FastAgent."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Master configuration controlling execution drivers and optimization flags.

    Every optimization is guarded by an explicit feature flag.
    In Phase 0, all optimization flags default to False (Baseline mode).
    """

    # --- Target Hardware & Transport ---
    device_serial: str = "192.168.10.108:5555"
    physical_width: int = 1220
    physical_height: int = 2712
    adb_path: str = "/home/xupremix/Android/Sdk/platform-tools/adb"
    scrcpy_host: str = "127.0.0.1"
    scrcpy_port: int = 8010

    # --- Inference Endpoint ---
    vllm_base_url: str = "http://192.168.10.132:8000/v1"
    vllm_api_key: str = "sk-strata"
    model_name: Optional[str] = None  # Auto-detected from /v1/models if None
    vit_patch_size: int = 28
    max_pixels: int = 320000
    temperature: float = 0.0
    max_tokens: int = 1024

    # --- Telemetry & Logging ---
    log_dir: str = "/tmp/fast_agent_logs"
    telemetry_jsonl: str = "/tmp/fast_agent_logs/telemetry.jsonl"

    # --- Phase 1 Flags: Systems & Transport ---
    enable_os_tuning: bool = False
    enable_scrcpy_socket: bool = False
    enable_fast_clipboard: bool = False
    enable_stream_capture: bool = False
    enable_adaptive_settling: bool = False
    enable_prefix_caching: bool = False
    enable_compact_dsl: bool = False

    # --- Phase 2 Flags: Cheap Perception & CPU Grounding ---
    enable_intent_teleport: bool = False
    enable_cpu_ocr_verifier: bool = False
    enable_coordinate_snapping: bool = False
    enable_hybrid_vision_structure: bool = False

    # --- Phase 3 Flags: Procedural Memory & Foveated ROI ---
    enable_procedural_memory: bool = False
    enable_foveated_roi: bool = False
    procedural_memory_path: str = os.path.expanduser("~/.fast_agent/procedural_memory.db")
    procedural_dhash_threshold: int = 5
    procedural_min_confidence: float = 0.85
    foveated_roi_pixels: int = 80000

    # --- Phase 4 Flags: Speculative & Closed-Loop ---
    enable_scroll_dead_reckoning: bool = False
    enable_speculative_execution: bool = False
    distillation_dataset_path: str = os.path.expanduser("~/.fast_agent/trajectories_distill.jsonl")
    scroll_min_displacement_px: float = 6.0
    enable_trajectory_distillation: bool = True

    def __post_init__(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
