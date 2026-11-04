"""Phase 5: Full Ablation Engine & Performance Profiler.

Executes comparative benchmarks across optimization layers:
1. Phase 0: Baseline (Pure ADB, full-frame 320k px, unconstrained JSON)
2. Phase 1: Systems & Transport (Scrcpy binary socket, fast clipboard, 60fps stream, adaptive settle, compact DSL)
3. Phase 2: Cheap Perception (Intent Teleportation, RapidOCR CPU grounding, centroid snapping)
4. Phase 3: Self-Evolving Memory (Procedural Memory replay, Sobel edge dHash, Foveated ROI)
5. Phase 4: Speculative & Closed-Loop (Streaming speculative dispatch, phase correlation scroll dead-reckoning)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fast_agent.benchmark import BenchmarkRunner, TaskSummary, format_markdown_table, get_standard_tasks
from fast_agent.config import AgentConfig
from fast_agent.device import AdbDevice, AndroidDevice, MockAndroidDevice
from fast_agent.vlm import VLMClient


@dataclass
class AblationConfig:
    name: str
    description: str
    config: AgentConfig


def get_ablation_presets() -> List[AblationConfig]:
    """Return incremental optimization layers for cumulative ablation."""
    return [
        AblationConfig(
            name="Phase 0: Baseline",
            description="Pure ADB, full-frame 320k px, static sleeps, verbose JSON",
            config=AgentConfig(),
        ),
        AblationConfig(
            name="Phase 1: Transport & DSL",
            description="+ Scrcpy binary socket, fast clipboard, 60 FPS stream, adaptive settle, compact DSL",
            config=AgentConfig(
                enable_os_tuning=True,
                enable_scrcpy_socket=True,
                enable_fast_clipboard=True,
                enable_stream_capture=True,
                enable_adaptive_settling=True,
                enable_prefix_caching=True,
                enable_compact_dsl=True,
            ),
        ),
        AblationConfig(
            name="Phase 2: Cheap Perception",
            description="+ Intent Teleportation, Xeon CPU RapidOCR, centroid snapping",
            config=AgentConfig(
                enable_os_tuning=True,
                enable_scrcpy_socket=True,
                enable_fast_clipboard=True,
                enable_stream_capture=True,
                enable_adaptive_settling=True,
                enable_prefix_caching=True,
                enable_compact_dsl=True,
                enable_intent_teleport=True,
                enable_cpu_ocr_verifier=True,
                enable_coordinate_snapping=True,
            ),
        ),
        AblationConfig(
            name="Phase 3: Procedural Memory & Foveated",
            description="+ SQLite macro cache, Sobel edge dHash, Foveated ROI compression",
            config=AgentConfig(
                enable_os_tuning=True,
                enable_scrcpy_socket=True,
                enable_fast_clipboard=True,
                enable_stream_capture=True,
                enable_adaptive_settling=True,
                enable_prefix_caching=True,
                enable_compact_dsl=True,
                enable_intent_teleport=True,
                enable_cpu_ocr_verifier=True,
                enable_coordinate_snapping=True,
                enable_procedural_memory=True,
                enable_foveated_roi=True,
            ),
        ),
        AblationConfig(
            name="Phase 4: Full Suite (Speculative + Closed-Loop)",
            description="+ Streaming speculative dispatch, Fourier scroll dead-reckoning, distillation",
            config=AgentConfig(
                enable_os_tuning=True,
                enable_scrcpy_socket=True,
                enable_fast_clipboard=True,
                enable_stream_capture=True,
                enable_adaptive_settling=True,
                enable_prefix_caching=True,
                enable_compact_dsl=True,
                enable_intent_teleport=True,
                enable_cpu_ocr_verifier=True,
                enable_coordinate_snapping=True,
                enable_procedural_memory=True,
                enable_foveated_roi=True,
                enable_scroll_dead_reckoning=True,
                enable_speculative_execution=True,
                enable_trajectory_distillation=True,
            ),
        ),
    ]


def run_ablation(
    task_id: str = "settings_battery",
    runs_per_config: int = 2,
    use_mock: bool = False,
) -> None:
    presets = get_ablation_presets()
    print("\n" + "=" * 70)
    print(f"FASTAGENT SYSTEMATIC ABLATION MATRIX: Task '{task_id}'")
    print(f"Configurations: {len(presets)} | Repetitions per config: {runs_per_config}")
    print(f"Execution Device: {'MockAndroidDevice' if use_mock else 'Physical Hardware'}")
    print("=" * 70 + "\n")

    summary_rows: List[Dict[str, Any]] = []

    for preset in presets:
        print(f"\n---> Running Ablation Layer: {preset.name} <---")
        print(f"Details: {preset.description}")

        dev: AndroidDevice = MockAndroidDevice() if use_mock else AdbDevice(preset.config)
        vlm = VLMClient(preset.config)
        runner = BenchmarkRunner(config=preset.config, device=dev, vlm_client=vlm)

        tasks = [t for t in get_standard_tasks() if t.task_id == task_id]
        if not tasks:
            print(f"Task {task_id} not found!")
            return

        summaries = runner.run_benchmark(tasks, runs_per_task=runs_per_config)
        if summaries:
            s = summaries[0]
            summary_rows.append(
                {
                    "layer": preset.name,
                    "success": s.success_rate,
                    "capture_ms": s.capture_ms.median,
                    "prefill_ms": s.vit_prefill_ms.median,
                    "decode_ms": s.decode_ms.median,
                    "inject_ms": s.input_inject_ms.median,
                    "step_total_ms": s.step_total_ms.median,
                    "tokens": f"{s.prompt_tokens_median:.0f} / {s.completion_tokens_median:.0f}",
                }
            )

    print("\n\n" + "=" * 70)
    print("SYSTEMATIC ABLATION TELEMETRY SUMMARY")
    print("=" * 70 + "\n")
    print("| Optimization Layer | Success | Capture | ViT Prefill | Decode | Inject | Step Total | Tokens |")
    print("|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        print(
            f"| **{r['layer']}** | {r['success']}% | {r['capture_ms']:.0f}ms | "
            f"{r['prefill_ms']:.0f}ms | {r['decode_ms']:.0f}ms | {r['inject_ms']:.0f}ms | "
            f"**{r['step_total_ms']:.0f}ms** | {r['tokens']} |"
        )
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAgent Ablation Suite")
    parser.add_argument("--task", type=str, default="settings_battery", help="Target benchmark task")
    parser.add_argument("--runs", type=int, default=2, help="Runs per configuration")
    parser.add_argument("--mock", action="store_true", help="Run on MockAndroidDevice")
    args = parser.parse_args()

    run_ablation(task_id=args.task, runs_per_config=args.runs, use_mock=args.mock)
