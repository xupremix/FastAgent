"""Persistent UI Knowledge Graph with Bayesian Coordinate Priors and Dead-End Traps.

Features:
- Cross-session SQLite knowledge graph store (~/.fast_agent/ui_knowledge_graph.db).
- UIStateNode: Keyed by (package, activity, skeleton_hash). Tracks screen summaries,
  visit counts, and Bayesian coordinate priors (mean_x, mean_y, std_x, std_y, edge_dhash, label)
  for statistically frequent controls.
- UITransitionEdge: Directed transitions tracking success/fail counts, mean latency,
  and dynamic edge cost weight: w = mean_latency_ms / max(0.05, ema_success_rate ** 2).
- DeadEndTrap: Records no-op clicks or external ad redirects to prevent repeated traps.
- UserPriors: Learned user defaults per app and category for zero-shot personalization.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from fast_agent.contracts import NormalizedPoint, NormalizedRect


@dataclass
class FrequentElement:
    """A statistically frequent control with Bayesian Gaussian coordinate priors."""
    element_id: str
    label: str
    mean_x: float
    mean_y: float
    std_x: float = 0.01
    std_y: float = 0.01
    edge_dhash: str = ""
    sample_count: int = 1

    def update_bayesian(self, new_x: float, new_y: float) -> None:
        """Update mean and running variance using Welford's algorithm."""
        self.sample_count += 1
        n = float(self.sample_count)
        delta_x = new_x - self.mean_x
        self.mean_x += delta_x / n
        delta_y = new_y - self.mean_y
        self.mean_y += delta_y / n
        # Relaxed variance update
        self.std_x = max(0.005, math.sqrt(((self.std_x**2) * (n - 1) + delta_x * (new_x - self.mean_x)) / n))
        self.std_y = max(0.005, math.sqrt(((self.std_y**2) * (n - 1) + delta_y * (new_y - self.mean_y)) / n))


@dataclass
class UIStateNode:
    """A unique structural and semantic UI state."""
    node_id: str
    package: str
    activity: str
    skeleton_hash: str
    summary: str
    visit_count: int = 1
    frequent_elements: List[FrequentElement] = field(default_factory=list)
    last_visited_ts: float = field(default_factory=time.time)


@dataclass
class UITransitionEdge:
    """Directed transition connecting two UI states via a parameterized action."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    action_type: str
    action_params: Dict[str, Any]
    success_count: int = 1
    fail_count: int = 0
    mean_latency_ms: float = 100.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5
        return self.success_count / float(total)

    @property
    def cost_weight(self) -> float:
        """Edge weight for shortest-path: w = mean_latency_ms / max(0.05, success_rate ** 2)."""
        sr = max(0.05, self.success_rate)
        return self.mean_latency_ms / (sr**2)


@dataclass
class DeadEndTrap:
    """Coordinate region on a node that produces no screen delta or triggers unwanted ads."""
    node_id: str
    norm_x: float
    norm_y: float
    radius: float = 0.05
    trap_type: str = "noop"  # 'noop' or 'ad_redirect'


class UIKnowledgeGraph:
    """Cross-session persistent knowledge graph engine."""

    def __init__(self, db_path: str = "~/.fast_agent/ui_knowledge_graph.db") -> None:
        self.db_path = os.path.expanduser(db_path)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    package TEXT NOT NULL,
                    activity TEXT NOT NULL,
                    skeleton_hash TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    visit_count INTEGER NOT NULL,
                    frequent_elements TEXT NOT NULL,
                    last_visited_ts REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS edges (
                    edge_id TEXT PRIMARY KEY,
                    source_node_id TEXT NOT NULL,
                    target_node_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_params TEXT NOT NULL,
                    success_count INTEGER NOT NULL,
                    fail_count INTEGER NOT NULL,
                    mean_latency_ms REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traps (
                    trap_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    norm_x REAL NOT NULL,
                    norm_y REAL NOT NULL,
                    radius REAL NOT NULL,
                    trap_type TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_priors (
                    prior_key TEXT PRIMARY KEY,
                    package TEXT NOT NULL,
                    category TEXT NOT NULL,
                    preference_val TEXT NOT NULL,
                    updated_ts REAL NOT NULL
                )
                """
            )
            conn.commit()

    def get_or_create_node(
        self,
        package: str,
        activity: str,
        skeleton_hash: str,
        summary: str = "",
    ) -> UIStateNode:
        """Find existing node or insert new node."""
        node_id = f"{package}::{activity}::{skeleton_hash}"
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM nodes WHERE node_id = ?", (node_id,))
            row = cur.fetchone()
            if row:
                elements_raw = json.loads(str(row["frequent_elements"]))
                elements = [FrequentElement(**e) for e in elements_raw]
                node = UIStateNode(
                    node_id=str(row["node_id"]),
                    package=str(row["package"]),
                    activity=str(row["activity"]),
                    skeleton_hash=str(row["skeleton_hash"]),
                    summary=str(row["summary"]),
                    visit_count=int(row["visit_count"]) + 1,
                    frequent_elements=elements,
                    last_visited_ts=time.time(),
                )
                conn.execute(
                    "UPDATE nodes SET visit_count = ?, last_visited_ts = ? WHERE node_id = ?",
                    (node.visit_count, node.last_visited_ts, node_id),
                )
                conn.commit()
                return node

            # Create new
            node = UIStateNode(
                node_id=node_id,
                package=package,
                activity=activity,
                skeleton_hash=skeleton_hash,
                summary=summary or f"Screen in {package}",
                visit_count=1,
                frequent_elements=[],
                last_visited_ts=time.time(),
            )
            conn.execute(
                """
                INSERT INTO nodes (node_id, package, activity, skeleton_hash, summary, visit_count, frequent_elements, last_visited_ts)
                VALUES (?, ?, ?, ?, ?, 1, '[]', ?)
                """,
                (node_id, package, activity, skeleton_hash, node.summary, node.last_visited_ts),
            )
            conn.commit()
            return node

    def record_element_interaction(
        self,
        node_id: str,
        label: str,
        norm_x: float,
        norm_y: float,
        edge_dhash: str = "",
    ) -> None:
        """Update Bayesian coordinate priors for a named UI element."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT frequent_elements FROM nodes WHERE node_id = ?", (node_id,))
            row = cur.fetchone()
            if not row:
                return
            elements_data = json.loads(str(row["frequent_elements"]))
            elements = [FrequentElement(**e) for e in elements_data]

            found = False
            for el in elements:
                if el.label == label:
                    el.update_bayesian(norm_x, norm_y)
                    if edge_dhash:
                        el.edge_dhash = edge_dhash
                    found = True
                    break
            if not found:
                elements.append(
                    FrequentElement(
                        element_id=f"el_{len(elements)+1}",
                        label=label,
                        mean_x=norm_x,
                        mean_y=norm_y,
                        edge_dhash=edge_dhash,
                        sample_count=1,
                    )
                )

            elements_json = json.dumps([asdict(e) for e in elements])
            conn.execute(
                "UPDATE nodes SET frequent_elements = ? WHERE node_id = ?",
                (elements_json, node_id),
            )
            conn.commit()

    def record_transition(
        self,
        source_node_id: str,
        target_node_id: str,
        action_type: str,
        action_params: Dict[str, Any],
        latency_ms: float,
        success: bool = True,
    ) -> str:
        """Record or update a directed transition edge between states."""
        edge_id = f"{source_node_id}->{target_node_id}::{action_type}"
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM edges WHERE edge_id = ?", (edge_id,))
            row = cur.fetchone()
            if row:
                s_count = int(row["success_count"]) + (1 if success else 0)
                f_count = int(row["fail_count"]) + (0 if success else 1)
                total = s_count + f_count
                old_mean = float(row["mean_latency_ms"])
                new_mean = old_mean + (latency_ms - old_mean) / float(total)

                conn.execute(
                    """
                    UPDATE edges 
                    SET success_count = ?, fail_count = ?, mean_latency_ms = ?
                    WHERE edge_id = ?
                    """,
                    (s_count, f_count, new_mean, edge_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO edges (edge_id, source_node_id, target_node_id, action_type, action_params, success_count, fail_count, mean_latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        edge_id,
                        source_node_id,
                        target_node_id,
                        action_type,
                        json.dumps(action_params),
                        1 if success else 0,
                        0 if success else 1,
                        latency_ms,
                    ),
                )
            conn.commit()
        return edge_id

    def record_trap(
        self,
        node_id: str,
        norm_x: float,
        norm_y: float,
        trap_type: str = "noop",
        radius: float = 0.05,
    ) -> None:
        """Blacklist coordinates that yield no screen change or navigate to ad traps."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO traps (node_id, norm_x, norm_y, radius, trap_type)
                VALUES (?, ?, ?, ?, ?)
                """,
                (node_id, norm_x, norm_y, radius, trap_type),
            )
            conn.commit()

    def is_trap(self, node_id: str, norm_x: float, norm_y: float) -> bool:
        """Check if target coordinates fall within a recorded dead-end trap."""
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT norm_x, norm_y, radius FROM traps WHERE node_id = ?", (node_id,)
            )
            for row in cur.fetchall():
                tx, ty, r = float(row["norm_x"]), float(row["norm_y"]), float(row["radius"])
                dist = math.sqrt((norm_x - tx) ** 2 + (norm_y - ty) ** 2)
                if dist <= r:
                    return True
        return False

    def get_outgoing_edges(self, source_node_id: str) -> List[UITransitionEdge]:
        """Fetch all outward transition edges from a node."""
        with self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM edges WHERE source_node_id = ?", (source_node_id,))
            edges: List[UITransitionEdge] = []
            for row in cur.fetchall():
                edges.append(
                    UITransitionEdge(
                        edge_id=str(row["edge_id"]),
                        source_node_id=str(row["source_node_id"]),
                        target_node_id=str(row["target_node_id"]),
                        action_type=str(row["action_type"]),
                        action_params=json.loads(str(row["action_params"])),
                        success_count=int(row["success_count"]),
                        fail_count=int(row["fail_count"]),
                        mean_latency_ms=float(row["mean_latency_ms"]),
                    )
                )
            return edges

    def set_user_prior(self, package: str, category: str, key: str, value: str) -> None:
        """Store user preference for an app category."""
        prior_key = f"{package}::{category}::{key}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_priors (prior_key, package, category, preference_val, updated_ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (prior_key, package, category, value, time.time()),
            )
            conn.commit()

    def get_user_prior(self, package: str, category: str, key: str) -> Optional[str]:
        """Fetch user preference."""
        prior_key = f"{package}::{category}::{key}"
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT preference_val FROM user_priors WHERE prior_key = ?", (prior_key,)
            )
            row = cur.fetchone()
            if row:
                return str(row["preference_val"])
            return None
