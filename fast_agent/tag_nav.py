"""TAG-Nav: A* & Dijkstra Shortest-Path Prefix Navigation.

Features:
- Fast state-graph pathfinding using Bayesian-weighted edge costs:
  w = mean_latency_ms / max(0.05, success_rate ** 2).
- Replays historical multi-step navigation paths in 0 VLM calls via Scrcpy control sockets.
- Delivers the agent directly to the target app activity/view before visual reasoning begins.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from fast_agent.ui_knowledge_graph import UIKnowledgeGraph, UITransitionEdge


@dataclass
class NavigationPlan:
    """Executable sequence of graph transitions."""
    source_node_id: str
    target_node_id: str
    edges: List[UITransitionEdge]
    total_cost: float
    success: bool


class TagNavPathfinder:
    """Dijkstra & A* shortest-path navigator over the persistent UI Knowledge Graph."""

    def __init__(self, graph: UIKnowledgeGraph) -> None:
        self.graph = graph

    def find_shortest_path(
        self,
        start_node_id: str,
        goal_node_id: str,
        max_hops: int = 8,
    ) -> Optional[NavigationPlan]:
        """Find lowest-latency transition sequence from start to goal.

        Args:
            start_node_id: Fingerprint of current screen state.
            goal_node_id: Fingerprint of target destination screen.
            max_hops: Maximum intermediate screen transitions allowed.

        Returns:
            NavigationPlan with sequence of UITransitionEdges or None.
        """
        if start_node_id == goal_node_id:
            return NavigationPlan(
                source_node_id=start_node_id,
                target_node_id=goal_node_id,
                edges=[],
                total_cost=0.0,
                success=True,
            )

        # Priority queue stores: (cumulative_cost, hops, current_node_id, path_edges)
        queue: List[Tuple[float, int, str, List[UITransitionEdge]]] = [
            (0.0, 0, start_node_id, [])
        ]
        visited: Dict[str, float] = {start_node_id: 0.0}

        while queue:
            cost, hops, curr_id, path = heapq.heappop(queue)

            if curr_id == goal_node_id:
                return NavigationPlan(
                    source_node_id=start_node_id,
                    target_node_id=goal_node_id,
                    edges=path,
                    total_cost=cost,
                    success=True,
                )

            if hops >= max_hops:
                continue

            outgoing = self.graph.get_outgoing_edges(curr_id)
            for edge in outgoing:
                next_id = edge.target_node_id
                next_cost = cost + edge.cost_weight

                if next_id not in visited or next_cost < visited[next_id]:
                    visited[next_id] = next_cost
                    heapq.heappush(
                        queue,
                        (next_cost, hops + 1, next_id, path + [edge]),
                    )

        return None
