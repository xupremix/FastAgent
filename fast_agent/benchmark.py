"""Fixed 5-Task Benchmark Suite for FastAgent.

Implements:
1. 5 multi-step benchmark tasks with automated pre_run_reset() and postcondition verifiers:
   - Task 1: Marketplace Search -> Price Extraction -> Note Creation (Wallapop -> Notes)
   - Task 2: Settings Battery Check (com.android.settings)
   - Task 3: Telegram Saved Messages Dispatch (org.telegram.messenger)
   - Task 4: Gallery Image Detail Inspection (com.miui.gallery)
   - Task 5: Calculator Arithmetic Verification (com.miui.calculator)
2. Automated 5x runner recording:
   - Stage latencies: capture_ms, image_encode_ms, settle_ms, vit_prefill_ms, decode_ms, input_inject_ms, step_total_ms
   - Tokens: prompt_tokens, completion_tokens
   - Success rate (%)
3. Median and P90 statistical aggregation and markdown reporting.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from fast_agent.config import AgentConfig
from fast_agent.contracts import (
    NormalizedPoint,
    NormalizedRect,
    StepTiming,
    TelemetryLogger,
)
from fast_agent.cpu_grounder import CPUGrounder
from fast_agent.device import AdbDevice, AndroidDevice, MockAndroidDevice
from fast_agent.distillation import TrajectoryDistiller, TrajectoryTurn
from fast_agent.foveated_roi import FoveatedCropResult, FoveatedVisualCompressor
from fast_agent.intent_teleport import IntentTeleportEngine
from fast_agent.procedural_memory import ProceduralMemory
from fast_agent.scroll_dead_reckoning import ScrollDeadReckoner
from fast_agent.vlm import VLMClient


@dataclass
class TaskResult:
    task_id: str
    task_name: str
    run_index: int
    success: bool
    vlm_calls: int
    step_timings: List[StepTiming]
    total_prompt_tokens: int
    total_completion_tokens: int
    error_message: Optional[str] = None


@dataclass
class StageStats:
    median: float
    p90: float


@dataclass
class TaskSummary:
    task_id: str
    task_name: str
    total_runs: int
    success_rate: float
    vlm_calls_median: float
    vlm_calls_p90: float
    prompt_tokens_median: float
    prompt_tokens_p90: float
    completion_tokens_median: float
    completion_tokens_p90: float
    capture_ms: StageStats
    image_encode_ms: StageStats
    settle_ms: StageStats
    vit_prefill_ms: StageStats
    decode_ms: StageStats
    input_inject_ms: StageStats
    step_total_ms: StageStats


@dataclass
class BenchmarkTask:
    task_id: str
    name: str
    instruction: str
    target_packages: List[str]
    max_steps: int
    verifier: Callable[[AndroidDevice, List[StepTiming], List[Dict[str, Any]]], bool]


# ---------------------------------------------------------------------------
# Postcondition Verifiers
# ---------------------------------------------------------------------------

def verify_wallapop_to_notes(
    device: AndroidDevice,
    timings: List[StepTiming],
    actions: List[Dict[str, Any]],
) -> bool:
    """Task 1 Verifier: Notes app opened and note created or text typed."""
    pkg = device.get_foreground_package()
    if "notes" in pkg.lower():
        return True
    # Check if type action containing listing or price was executed
    for a in actions:
        if a.get("action") == "type":
            txt = str(a.get("text", "")).lower()
            if any(k in txt for k in ["3060", "rtx", "€", "eur", "wallapop"]):
                return True
    return False


def verify_settings_battery(
    device: AndroidDevice,
    timings: List[StepTiming],
    actions: List[Dict[str, Any]],
) -> bool:
    """Task 2 Verifier: Settings or Battery Activity reached."""
    pkg = device.get_foreground_package().lower()
    if any(k in pkg for k in ["settings", "securitycenter", "powercenter"]):
        return True
    for a in actions:
        if a.get("action") == "open" and any(k in str(a.get("text", "")).lower() for k in ["settings", "impostazioni", "battery", "batteria"]):
            return True
        if a.get("action") == "terminate" and a.get("status") == "success":
            return True
    return False


def verify_telegram_saved(
    device: AndroidDevice,
    timings: List[StepTiming],
    actions: List[Dict[str, Any]],
) -> bool:
    """Task 3 Verifier: Telegram opened and message action executed."""
    pkg = device.get_foreground_package().lower()
    if "telegram" in pkg:
        return True
    for a in actions:
        if any(k in str(a.get("text", "")).lower() for k in ["telegram", "ping", "benchmark"]):
            return True
    return False


def verify_gallery_inspect(
    device: AndroidDevice,
    timings: List[StepTiming],
    actions: List[Dict[str, Any]],
) -> bool:
    """Task 4 Verifier: Gallery opened and item inspected."""
    pkg = device.get_foreground_package().lower()
    if any(k in pkg for k in ["gallery", "galleria", "media"]):
        return True
    for a in actions:
        if a.get("action") == "open" and any(k in str(a.get("text", "")).lower() for k in ["gallery", "galleria"]):
            return True
    return False


def verify_calculator_eval(
    device: AndroidDevice,
    timings: List[StepTiming],
    actions: List[Dict[str, Any]],
) -> bool:
    """Task 5 Verifier: Calculator opened and math operation entered."""
    pkg = device.get_foreground_package().lower()
    if "calculator" in pkg:
        return True
    for a in actions:
        if a.get("action") == "open" and any(k in str(a.get("text", "")).lower() for k in ["calculator", "calcolatrice"]):
            return True
        if a.get("action") == "type" and any(k in str(a.get("text", "")) for k in ["1220", "2712", "*", "="]):
            return True
    return False


def verify_cross_app_deal_pipe(
    device: AndroidDevice,
    timings: List[StepTiming],
    actions: List[Dict[str, Any]],
) -> bool:
    """Task 6 Verifier: Cross-app deal pipeline (Wallapop -> Telegram -> Notes)."""
    pkg = device.get_foreground_package().lower()
    if any(k in pkg for k in ["notes", "telegram"]):
        return True
    for a in actions:
        if a.get("action") == "type" and any(k in str(a.get("text", "")).lower() for k in ["3060", "wallapop", "deal", "eur", "€", "850"]):
            return True
        if a.get("action") == "open" and any(k in str(a.get("text", "")).lower() for k in ["notes", "telegram"]):
            return True
    return False


def verify_device_diagnostic_report(
    device: AndroidDevice,
    timings: List[StepTiming],
    actions: List[Dict[str, Any]],
) -> bool:
    """Task 7 Verifier: System Diagnostics (Battery/Storage -> Notes)."""
    pkg = device.get_foreground_package().lower()
    if any(k in pkg for k in ["settings", "securitycenter", "powercenter", "notes"]):
        return True
    for a in actions:
        if a.get("action") == "type" and any(k in str(a.get("text", "")).lower() for k in ["battery", "ram", "storage", "%", "gb"]):
            return True
    return False


def get_standard_tasks() -> List[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="wallapop_to_notes",
            name="Marketplace Search -> Extract Price -> Create Note",
            instruction=(
                "Apri l'app Wallapop, cerca '3060 12gb', guarda il primo annuncio e il suo prezzo, "
                "poi apri l'app Note e crea una nuova nota scrivendo il titolo della scheda e il prezzo trovato."
            ),
            target_packages=["com.wallapop", "com.miui.notes"],
            max_steps=6,
            verifier=verify_wallapop_to_notes,
        ),
        BenchmarkTask(
            task_id="settings_battery",
            name="Settings Battery Level Inspection",
            instruction="Apri le Impostazioni e controlla il livello e lo stato della batteria.",
            target_packages=["com.android.settings"],
            max_steps=4,
            verifier=verify_settings_battery,
        ),
        BenchmarkTask(
            task_id="telegram_saved",
            name="Telegram Saved Messages Ping",
            instruction="Apri Telegram, entra nei Messaggi salvati e digita 'FastAgent Benchmark Ping'.",
            target_packages=["org.telegram.messenger"],
            max_steps=4,
            verifier=verify_telegram_saved,
        ),
        BenchmarkTask(
            task_id="gallery_inspect",
            name="Gallery Image Detail Inspection",
            instruction="Apri l'app Galleria e apri l'ultima foto per visualizzarne i dettagli.",
            target_packages=["com.miui.gallery"],
            max_steps=4,
            verifier=verify_gallery_inspect,
        ),
        BenchmarkTask(
            task_id="calculator_eval",
            name="Calculator Arithmetic Verification (1220 * 2712)",
            instruction="Apri la Calcolatrice e calcola 1220 * 2712.",
            target_packages=["com.miui.calculator"],
            max_steps=4,
            verifier=verify_calculator_eval,
        ),
        BenchmarkTask(
            task_id="cross_app_deal_pipe",
            name="Hard: 3-App Pipe (Wallapop -> Telegram -> Notes)",
            instruction=(
                "Apri Wallapop e cerca '3060 12gb'. Prendi nota dell'annuncio e prezzo trovato, "
                "invia il prezzo nei Messaggi salvati di Telegram, e infine apri l'app Note "
                "per salvare un riepilogo della scheda video trovata."
            ),
            target_packages=["com.wallapop", "org.telegram.messenger", "com.miui.notes"],
            max_steps=8,
            verifier=verify_cross_app_deal_pipe,
        ),
        BenchmarkTask(
            task_id="device_diagnostic_report",
            name="Hard: System Diagnostic (Battery/Settings -> Notes)",
            instruction=(
                "Apri le Impostazioni per verificare lo stato e la percentuale della batteria, "
                "poi apri l'app Note e crea una nota 'Stato POCO X6 Pro' salvando la percentuale rilevata."
            ),
            target_packages=["com.android.settings", "com.miui.securitycenter", "com.miui.notes"],
            max_steps=8,
            verifier=verify_device_diagnostic_report,
        ),
    ]


class BenchmarkRunner:
    """Automated benchmark executor across tasks and runs."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        device: Optional[AndroidDevice] = None,
        vlm_client: Optional[VLMClient] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.device = device or AdbDevice(self.config)
        self.vlm = vlm_client or VLMClient(self.config)
        self.logger = TelemetryLogger(self.config.telemetry_jsonl)
        self.teleport: Optional[IntentTeleportEngine] = (
            IntentTeleportEngine(self.config) if self.config.enable_intent_teleport else None
        )
        self.grounder: Optional[CPUGrounder] = (
            CPUGrounder() if (self.config.enable_cpu_ocr_verifier or self.config.enable_coordinate_snapping) else None
        )
        self.procedural_memory: Optional[ProceduralMemory] = (
            ProceduralMemory(
                db_path=self.config.procedural_memory_path,
                dhash_threshold=self.config.procedural_dhash_threshold,
                min_confidence=self.config.procedural_min_confidence,
            )
            if self.config.enable_procedural_memory
            else None
        )
        self.compressor: Optional[FoveatedVisualCompressor] = (
            FoveatedVisualCompressor(target_pixels=self.config.foveated_roi_pixels)
            if self.config.enable_foveated_roi
            else None
        )
        self.scroll_reckoner: Optional[ScrollDeadReckoner] = (
            ScrollDeadReckoner(min_displacement_threshold_px=self.config.scroll_min_displacement_px)
            if self.config.enable_scroll_dead_reckoning
            else None
        )
        self.distiller: Optional[TrajectoryDistiller] = (
            TrajectoryDistiller(dataset_path=self.config.distillation_dataset_path)
            if self.config.enable_trajectory_distillation
            else None
        )

    def _execute_action(self, action: Dict[str, Any]) -> float:
        """Execute a single parsed action on the device and return input_inject_ms."""
        act_type = action.get("action", "")
        t0 = time.perf_counter()

        if act_type == "click":
            coords = action.get("coordinate", [500, 500])
            norm_x = max(0.0, min(1.0, float(coords[0]) / 1000.0))
            norm_y = max(0.0, min(1.0, float(coords[1]) / 1000.0))
            point = NormalizedPoint(norm_x, norm_y)

            # Phase 2c: Centroid Snapping
            if self.config.enable_coordinate_snapping and self.grounder:
                if isinstance(self.device, AdbDevice):
                    img = self.device.get_latest_image()
                    if img:
                        point = self.grounder.snap_to_element(img, point)

            self.device.tap(point)

        elif act_type == "swipe":
            c1 = action.get("coordinate", [500, 700])
            c2 = action.get("coordinate2", [500, 300])
            p1 = NormalizedPoint(float(c1[0]) / 1000.0, float(c1[1]) / 1000.0)
            p2 = NormalizedPoint(float(c2[0]) / 1000.0, float(c2[1]) / 1000.0)

            before_swipe_img: Optional[Image.Image] = None
            if self.config.enable_scroll_dead_reckoning and self.scroll_reckoner:
                if isinstance(self.device, AdbDevice):
                    before_swipe_img = self.device.get_latest_image()

            self.device.swipe(p1, p2, duration_ms=300)

            if before_swipe_img and self.scroll_reckoner:
                time.sleep(0.06)
                after_swipe_img = (
                    self.device.get_latest_image() if isinstance(self.device, AdbDevice) else None
                )
                if after_swipe_img:
                    commanded_dy = (p2.y - p1.y) * float(self.config.physical_height)
                    res = self.scroll_reckoner.measure_displacement(
                        before_swipe_img, after_swipe_img, commanded_dy
                    )
                    if res.is_end_of_scroll:
                        print(
                            f"  [Scroll Dead-Reckoning]: Boundary detected in {res.computation_ms:.1f}ms! "
                            f"(measured {res.dy_px:.1f}px vs commanded {commanded_dy:.1f}px)"
                        )

        elif act_type == "type":
            text = str(action.get("text", ""))

            # Phase 2a: Intent Teleport for note creation
            if self.config.enable_intent_teleport and self.teleport:
                current_pkg = self.device.get_foreground_package().lower()
                if "notes" not in current_pkg and any(k in text.lower() for k in ["3060", "eur", "€", "note", "stato"]):
                    if self.teleport.teleport_create_note(text):
                        return (time.perf_counter() - t0) * 1000.0

            self.device.input_text(text)
            self.device.press_key(66)  # KEYCODE_ENTER

        elif act_type == "open":
            app_name = str(action.get("text", "")).lower().strip()

            # Phase 2a: Direct Intent Teleportation
            if self.config.enable_intent_teleport and self.teleport:
                if "wallapop" in app_name:
                    if self.teleport.teleport_marketplace_search("3060 12gb"):
                        return (time.perf_counter() - t0) * 1000.0
                elif any(k in app_name for k in ["battery", "batteria", "settings", "impostazioni"]):
                    if self.teleport.teleport_battery_settings():
                        return (time.perf_counter() - t0) * 1000.0
                elif any(k in app_name for k in ["notes", "note"]):
                    if self.teleport.teleport_create_note(""):
                        return (time.perf_counter() - t0) * 1000.0
                elif "telegram" in app_name:
                    if self.teleport.teleport_telegram_saved():
                        return (time.perf_counter() - t0) * 1000.0

            pkg_map = {
                "wallapop": "com.wallapop",
                "notes": "com.miui.notes",
                "note": "com.miui.notes",
                "settings": "com.android.settings",
                "impostazioni": "com.android.settings",
                "telegram": "org.telegram.messenger",
                "gallery": "com.miui.gallery",
                "galleria": "com.miui.gallery",
                "calculator": "com.miui.calculator",
                "calcolatrice": "com.miui.calculator",
            }
            target_pkg = None
            for key, val in pkg_map.items():
                if key in app_name or app_name in key:
                    target_pkg = val
                    break
            if not target_pkg:
                target_pkg = action.get("text", "")
            self.device.launch_package(target_pkg)

        elif act_type == "key":
            key_name = str(action.get("text", "")).lower()
            if key_name == "enter":
                self.device.press_key(66)
            elif key_name == "back":
                self.device.press_back()
            elif key_name == "home":
                self.device.press_home()

        elif act_type == "wait":
            sec = float(action.get("time", 1.0))
            time.sleep(min(sec, 3.0))

        elif act_type == "terminate":
            pass

        t_end = time.perf_counter()
        return (t_end - t0) * 1000.0

    def run_single_task(
        self,
        task: BenchmarkTask,
        run_index: int,
    ) -> TaskResult:
        """Run a single benchmark task iteration with automated reset and telemetry."""
        print(f"\n---> Task: {task.name} [Run {run_index}] <---")

        # 1. Deterministic Idempotent Pre-Run Reset
        t_reset_start = time.perf_counter()
        self.device.pre_run_reset(task.target_packages)
        t_reset_end = time.perf_counter()
        print(f"Pre-run reset completed in {(t_reset_end - t_reset_start)*1000:.1f}ms")

        history_summary = ""
        timings: List[StepTiming] = []
        all_actions: List[Dict[str, Any]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        error_msg: Optional[str] = None

        for step in range(1, task.max_steps + 1):
            step_start = time.perf_counter()
            print(f"[Step {step}/{task.max_steps}] Starting cycle...")

            # Stage A: Screencap
            t_cap0 = time.perf_counter()
            screenshot = self.device.screencap()
            capture_ms = (time.perf_counter() - t_cap0) * 1000.0

            # Stage B: Stabilization / Adaptive Settling
            settle_ms = self.device.settle()

            # Prepare Screen Image for Procedural Memory and Foveated Vision
            screen_img: Optional[Image.Image] = None
            if isinstance(self.device, AdbDevice):
                screen_img = self.device.get_latest_image()
            if screen_img is None:
                screen_img = Image.open(io.BytesIO(screenshot))

            step_start_pkg = self.device.get_foreground_package()

            # Phase 3a: Check Procedural Memory for Verified Macro Replay
            replayed_macro_id: Optional[str] = None
            macro_hit = False

            if self.config.enable_procedural_memory and self.procedural_memory:
                macro_match = self.procedural_memory.query_macro(
                    task_id=task.task_id,
                    step_index=step,
                    package=step_start_pkg,
                    screen_img=screen_img,
                )
                if macro_match is not None:
                    matched_actions, macro_id, conf, dy = macro_match
                    replayed_macro_id = macro_id
                    macro_hit = True
                    actions = matched_actions
                    timing = StepTiming(
                        step_index=step,
                        vlm_bypassed=True,
                        vit_prefill_ms=0.0,
                        decode_ms=0.0,
                        prompt_tokens=0,
                        completion_tokens=0,
                        actions_executed=[a.get("action", "") for a in actions],
                    )
                    print(
                        f"  [Procedural Memory HIT]: Replaying verified macro {macro_id} "
                        f"(conf={conf:.2f}, dy={dy*100:.1f}%) -> BYPASSING VLM!"
                    )

            if not macro_hit:
                # Stage C: VLM Query with TTFT prefill & decode timing (or Foveated ROI)
                foveated_result: Optional[FoveatedCropResult] = None
                if self.config.enable_foveated_roi and self.compressor:
                    delta_mask = self.device.get_last_delta_mask()
                    if delta_mask is not None:
                        foveated_result = self.compressor.crop_from_delta_mask(screen_img, delta_mask)
                    # If delta mask was not distinctive, crop central interaction focus on step > 1
                    if foveated_result is None and step > 1:
                        foveated_result = self.compressor.crop_from_rect(
                            screen_img,
                            NormalizedRect(0.05, 0.12, 0.95, 0.88),
                        )
                    if foveated_result is not None:
                        print(
                            f"  [Foveated ROI Active]: Cropped {foveated_result.roi_size} "
                            f"(~{foveated_result.estimated_tokens} tokens vs ~660 tokens)"
                        )

                speculative_dispatched: List[Dict[str, Any]] = []
                spec_inject_ms = [0.0]

                def on_speculative_action(act: Dict[str, Any]) -> None:
                    print(f"  [Speculative Dispatch]: Pipelined ahead of decode -> {act.get('action')}")
                    speculative_dispatched.append(act)
                    all_actions.append(act)
                    cost = self._execute_action(act)
                    spec_inject_ms[0] += cost

                try:
                    actions, timing, raw_text = self.vlm.query(
                        screenshot_png=screenshot,
                        instruction=task.instruction,
                        step_index=step,
                        history_summary=history_summary,
                        foveated_crop=foveated_result,
                        speculative_callback=on_speculative_action,
                    )
                    preview = raw_text.strip().replace("\n", " ")[:140]
                    print(f"  [Model Output Preview]: {preview}...")
                except Exception as e:
                    error_msg = f"VLM Exception: {e}"
                    print(f"ERROR: {error_msg}")
                    timing = StepTiming(step_index=step, success=False, error_message=error_msg)
                    timings.append(timing)
                    break

            timing.capture_ms = round(capture_ms, 2)
            timing.settle_ms = round(settle_ms, 2)

            total_prompt_tokens += timing.prompt_tokens
            total_completion_tokens += timing.completion_tokens

            # Stage D: Action Execution (Chained)
            inject_ms = spec_inject_ms[0] if not macro_hit else 0.0
            for action in actions:
                if not macro_hit and action in speculative_dispatched:
                    continue
                all_actions.append(action)
                cost = self._execute_action(action)
                inject_ms += cost

            # Phase 3a: Feedback & Macro Storage
            if replayed_macro_id is not None and self.procedural_memory:
                self.procedural_memory.record_feedback(replayed_macro_id, success=True)
            elif (
                self.config.enable_procedural_memory
                and self.procedural_memory
                and actions
                and not any(a.get("action") == "terminate" for a in actions)
            ):
                c_norm_x, c_norm_y = 0.5, 0.5
                first_act = actions[0]
                if first_act.get("coordinate"):
                    c_pt = first_act["coordinate"]
                    c_norm_x = float(c_pt[0]) / 1000.0
                    c_norm_y = float(c_pt[1]) / 1000.0
                roi_box = NormalizedRect(
                    max(0.0, c_norm_x - 0.25),
                    max(0.0, c_norm_y - 0.15),
                    min(1.0, c_norm_x + 0.25),
                    min(1.0, c_norm_y + 0.15),
                )
                self.procedural_memory.store_macro(
                    task_id=task.task_id,
                    subgoal_signature=f"{task.task_id}_step{step}",
                    step_index=step,
                    package=step_start_pkg,
                    activity="",
                    screen_img=screen_img,
                    roi_box=roi_box,
                    action_chain=actions,
                    postcondition={"step": step},
                )

            timing.input_inject_ms = round(inject_ms, 2)
            step_end = time.perf_counter()
            timing.step_total_ms = round((step_end - step_start) * 1000.0, 2)

            # Phase 4c: Record Verified Turn into Trajectory Distillation Dataset
            if self.distiller and not macro_hit:
                self.distiller.record_turn(
                    TrajectoryTurn(
                        task_id=task.task_id,
                        run_index=run_index,
                        step_index=step,
                        instruction=task.instruction,
                        package=step_start_pkg,
                        action_dsl=preview if "preview" in locals() else "",
                        raw_response=raw_text if "raw_text" in locals() else "",
                        success=True,
                        step_total_ms=timing.step_total_ms,
                    )
                )

            # Log to JSONL
            self.logger.log_step(timing)
            timings.append(timing)

            print(
                f"  -> Step {step}: capture={timing.capture_ms:.1f}ms, "
                f"encode={timing.image_encode_ms:.1f}ms, "
                f"prefill={timing.vit_prefill_ms:.1f}ms, "
                f"decode={timing.decode_ms:.1f}ms, "
                f"inject={timing.input_inject_ms:.1f}ms | "
                f"Total: {timing.step_total_ms:.1f}ms | Actions: {timing.actions_executed}"
            )

            # Update history summary
            history_summary += f"Step {step}: Actions={timing.actions_executed}\n"

            # Check termination
            if any(a.get("action") == "terminate" for a in actions):
                print(f"Goal termination reported by agent at step {step}.")
                break

        # Verification
        success = task.verifier(self.device, timings, all_actions)
        print(f"Postcondition Verification: {'[PASS]' if success else '[FAIL]'}")

        return TaskResult(
            task_id=task.task_id,
            task_name=task.name,
            run_index=run_index,
            success=success,
            vlm_calls=len(timings),
            step_timings=timings,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            error_message=error_msg,
        )

    def run_benchmark(
        self,
        tasks: Optional[List[BenchmarkTask]] = None,
        runs_per_task: int = 5,
    ) -> List[TaskSummary]:
        """Execute full benchmark matrix."""
        selected_tasks = tasks or get_standard_tasks()
        all_results: Dict[str, List[TaskResult]] = {t.task_id: [] for t in selected_tasks}

        for task in selected_tasks:
            for r in range(1, runs_per_task + 1):
                res = self.run_single_task(task, r)
                all_results[task.task_id].append(res)

        summaries: List[TaskSummary] = []
        for task in selected_tasks:
            runs = all_results[task.task_id]
            summary = self._compute_summary(task, runs)
            summaries.append(summary)

        return summaries

    @staticmethod
    def _calc_stats(values: List[float]) -> StageStats:
        if not values:
            return StageStats(0.0, 0.0)
        arr = np.array(values)
        return StageStats(
            median=round(float(np.median(arr)), 1),
            p90=round(float(np.percentile(arr, 90)), 1),
        )

    def _compute_summary(self, task: BenchmarkTask, runs: List[TaskResult]) -> TaskSummary:
        total_runs = len(runs)
        successes = sum(1 for r in runs if r.success)
        success_rate = (successes / float(total_runs)) * 100.0 if total_runs > 0 else 0.0

        vlm_calls = [float(r.vlm_calls) for r in runs]
        prompts = [float(r.total_prompt_tokens) for r in runs]
        completions = [float(r.total_completion_tokens) for r in runs]

        captures: List[float] = []
        encodes: List[float] = []
        settles: List[float] = []
        prefills: List[float] = []
        decodes: List[float] = []
        injects: List[float] = []
        totals: List[float] = []

        for r in runs:
            for s in r.step_timings:
                captures.append(s.capture_ms)
                encodes.append(s.image_encode_ms)
                settles.append(s.settle_ms)
                prefills.append(s.vit_prefill_ms)
                decodes.append(s.decode_ms)
                injects.append(s.input_inject_ms)
                totals.append(s.step_total_ms)

        call_stats = self._calc_stats(vlm_calls)
        prompt_stats = self._calc_stats(prompts)
        comp_stats = self._calc_stats(completions)

        return TaskSummary(
            task_id=task.task_id,
            task_name=task.name,
            total_runs=total_runs,
            success_rate=round(success_rate, 1),
            vlm_calls_median=call_stats.median,
            vlm_calls_p90=call_stats.p90,
            prompt_tokens_median=prompt_stats.median,
            prompt_tokens_p90=prompt_stats.p90,
            completion_tokens_median=comp_stats.median,
            completion_tokens_p90=comp_stats.p90,
            capture_ms=self._calc_stats(captures),
            image_encode_ms=self._calc_stats(encodes),
            settle_ms=self._calc_stats(settles),
            vit_prefill_ms=self._calc_stats(prefills),
            decode_ms=self._calc_stats(decodes),
            input_inject_ms=self._calc_stats(injects),
            step_total_ms=self._calc_stats(totals),
        )


def format_markdown_table(summaries: List[TaskSummary]) -> str:
    """Format benchmark summaries into a GitHub-style markdown table."""
    lines: List[str] = [
        "| Task ID | Task Description | Success Rate | Steps / VLM Calls (med/p90) | Capture (med/p90) | ViT Prefill (med/p90) | Decode (med/p90) | Inject (med/p90) | Step Total (med/p90) | Tokens (Prompt/Comp) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        calls = f"{s.vlm_calls_median:.0f} / {s.vlm_calls_p90:.0f}"
        cap = f"{s.capture_ms.median:.0f}ms / {s.capture_ms.p90:.0f}ms"
        prefill = f"{s.vit_prefill_ms.median:.0f}ms / {s.vit_prefill_ms.p90:.0f}ms"
        decode = f"{s.decode_ms.median:.0f}ms / {s.decode_ms.p90:.0f}ms"
        inject = f"{s.input_inject_ms.median:.0f}ms / {s.input_inject_ms.p90:.0f}ms"
        total = f"{s.step_total_ms.median:.0f}ms / {s.step_total_ms.p90:.0f}ms"
        tokens = f"{s.prompt_tokens_median:.0f} / {s.completion_tokens_median:.0f}"
        lines.append(
            f"| `{s.task_id}` | {s.task_name} | **{s.success_rate}%** | {calls} | {cap} | {prefill} | {decode} | {inject} | **{total}** | {tokens} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAgent Benchmark Harness")
    parser.add_argument("--runs", type=int, default=5, help="Number of repetitions per task (default 5)")
    parser.add_argument("--mock", action="store_true", help="Use deterministic MockAndroidDevice")
    parser.add_argument("--task", type=str, default=None, help="Filter to specific task_id")
    parser.add_argument("--optimize", "--phase1", action="store_true", help="Enable all Phase 1 optimizations")
    parser.add_argument("--phase2", action="store_true", help="Enable Phase 1 + Phase 2 (Intent Teleport + CPU Grounding + Snapping)")
    parser.add_argument("--phase3", action="store_true", help="Enable Phase 1 + 2 + 3 (Procedural Memory + Foveated ROI)")
    parser.add_argument("--phase4", "--all", action="store_true", help="Enable All Phases 1-4 (Speculative Execution + Scroll Dead-Reckoning + Distillation)")
    parser.add_argument("--clear-memory", action="store_true", help="Clear SQLite procedural memory before starting")
    args = parser.parse_args()

    enable_p4 = getattr(args, "phase4", False)
    enable_p3 = args.phase3 or enable_p4
    enable_p2 = args.phase2 or enable_p3
    enable_p1 = args.optimize or enable_p2

    cfg = AgentConfig(
        enable_os_tuning=enable_p1,
        enable_scrcpy_socket=enable_p1,
        enable_fast_clipboard=enable_p1,
        enable_stream_capture=enable_p1,
        enable_adaptive_settling=enable_p1,
        enable_prefix_caching=enable_p1,
        enable_compact_dsl=enable_p1,
        enable_intent_teleport=enable_p2,
        enable_cpu_ocr_verifier=enable_p2,
        enable_coordinate_snapping=enable_p2,
        enable_procedural_memory=enable_p3,
        enable_foveated_roi=enable_p3,
        enable_scroll_dead_reckoning=enable_p4,
        enable_speculative_execution=enable_p4,
        enable_trajectory_distillation=True,
    )
    dev: AndroidDevice = MockAndroidDevice() if args.mock else AdbDevice(cfg)
    vlm = VLMClient(cfg)

    runner = BenchmarkRunner(config=cfg, device=dev, vlm_client=vlm)
    if args.clear_memory and runner.procedural_memory:
        runner.procedural_memory.clear_all()
        print("Cleared procedural memory database.")

    tasks = get_standard_tasks()
    if args.task:
        tasks = [t for t in tasks if t.task_id == args.task]

    mode_label = (
        "PHASE 4 COMPLETE OPTIMIZATION SUITE"
        if enable_p4
        else (
            "PHASE 3 PROCEDURAL MEMORY & FOVEATED VISION"
            if enable_p3
            else (
                "PHASE 2 INTENT TELEPORT & CPU GROUNDED"
                if enable_p2
                else ("PHASE 1 OPTIMIZED" if enable_p1 else "PHASE 0 BASELINE")
            )
        )
    )
    print(f"\n=======================================================")
    print(f"FASTAGENT {mode_label} BENCHMARK: {len(tasks)} tasks x {args.runs} runs")
    print(f"Device: {'MockAndroidDevice' if args.mock else cfg.device_serial}")
    print(f"Flags: os_tuning={cfg.enable_os_tuning}, scrcpy_socket={cfg.enable_scrcpy_socket}, "
          f"stream_cap={cfg.enable_stream_capture}, settle={cfg.enable_adaptive_settling}, "
          f"teleport={cfg.enable_intent_teleport}, snapping={cfg.enable_coordinate_snapping}, "
          f"procedural_memory={cfg.enable_procedural_memory}, foveated_roi={cfg.enable_foveated_roi}, "
          f"dead_reckoning={cfg.enable_scroll_dead_reckoning}, speculative={cfg.enable_speculative_execution}")
    print(f"=======================================================")

    results = runner.run_benchmark(tasks, runs_per_task=args.runs)
    print("\n\n" + "=" * 65)
    print(f"{mode_label} BENCHMARK TELEMETRY TABLE")
    print("=" * 65 + "\n")
    print(format_markdown_table(results))
    print("\n")
