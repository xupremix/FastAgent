"""Phase 4c: Trajectory Distillation Dataset Exporter & N-Gram Draft Predictor.

Features:
- Records verified high-speed trajectories into ShareGPT / ChatML format for fine-tuning
  a 2B-3B student VLM (e.g. Qwen2-VL-2B or MobileVLM).
- Implements an N-Gram / State-Action prefix draft predictor that achieves speculative decoding
  on the Compact Action DSL.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TrajectoryTurn:
    """A single decision step in an agent trajectory."""
    task_id: str
    run_index: int
    step_index: int
    instruction: str
    package: str
    action_dsl: str
    raw_response: str
    success: bool
    step_total_ms: float
    timestamp: float = field(default_factory=time.time)

    def to_sharegpt(self) -> Dict[str, Any]:
        """Convert to ShareGPT format for LoRA/SFT training."""
        return {
            "id": f"{self.task_id}_r{self.run_index}_s{self.step_index}_{int(self.timestamp * 1000)}",
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\nGoal: {self.instruction}\nCurrent App: {self.package}\nStep: {self.step_index}",
                },
                {
                    "from": "gpt",
                    "value": self.raw_response.strip(),
                },
            ],
            "metadata": {
                "task_id": self.task_id,
                "package": self.package,
                "action_dsl": self.action_dsl,
                "latency_ms": self.step_total_ms,
                "success": self.success,
            },
        }


class TrajectoryDistiller:
    """Exports and persists successful agent trajectories to training datasets."""

    def __init__(self, dataset_path: str = "~/.fast_agent/trajectories_distill.jsonl") -> None:
        self.dataset_path = os.path.expanduser(dataset_path)
        os.makedirs(os.path.dirname(self.dataset_path), exist_ok=True)

    def record_turn(self, turn: TrajectoryTurn) -> None:
        """Append verified turn to JSONL dataset."""
        if not turn.success:
            return
        entry = turn.to_sharegpt()
        with open(self.dataset_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def count_samples(self) -> int:
        """Return total number of exported training samples."""
        if not os.path.exists(self.dataset_path):
            return 0
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


class NGramDraftPredictor:
    """Predicts common action sequences for speculative draft verification."""

    def __init__(self) -> None:
        # Key: (package, step_index) -> Counter of action strings
        self.transitions: Dict[Tuple[str, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def train_on_turn(self, package: str, step_index: int, action_dsl: str) -> None:
        """Record state-action occurrence."""
        self.transitions[(package, step_index)][action_dsl] += 1

    def predict_draft(self, package: str, step_index: int) -> Optional[str]:
        """Return most frequent action sequence for state or None."""
        cands = self.transitions.get((package, step_index))
        if not cands:
            return None
        return max(cands.items(), key=lambda kv: kv[1])[0]
