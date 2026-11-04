"""Unit tests for Phase 5 Advanced Modules: UI Knowledge Graph, TAG-Nav, SoM Tagger, Crawler, and Reviewer."""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Dict, List

import pytest
from PIL import Image

from fast_agent.background_reviewer import BackgroundReviewer
from fast_agent.contracts import NormalizedPoint, StepTiming
from fast_agent.recon_crawler import DESTRUCTIVE_KEYWORDS, ReconCrawler
from fast_agent.som_tagger import SetOfMarkTagger
from fast_agent.tag_nav import TagNavPathfinder
from fast_agent.ui_knowledge_graph import FrequentElement, UIKnowledgeGraph


def test_ui_knowledge_graph_bayesian_and_traps() -> None:
    """Verify Bayesian coordinate priors and dead-end trap blacklisting."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_kg.db")
        kg = UIKnowledgeGraph(db_path=db_path)

        # 1. Create node
        node = kg.get_or_create_node("com.test.app", "MainActivity", "hash123", summary="Home Screen")
        assert node.visit_count == 1

        # 2. Update Bayesian prior on an element
        kg.record_element_interaction(node.node_id, label="Search Bar", norm_x=0.50, norm_y=0.20)
        kg.record_element_interaction(node.node_id, label="Search Bar", norm_x=0.52, norm_y=0.22)

        node_updated = kg.get_or_create_node("com.test.app", "MainActivity", "hash123")
        assert len(node_updated.frequent_elements) == 1
        el = node_updated.frequent_elements[0]
        assert pytest.approx(el.mean_x, abs=0.005) == 0.51
        assert pytest.approx(el.mean_y, abs=0.005) == 0.21
        assert el.sample_count == 2

        # 3. Dead-end traps
        kg.record_trap(node.node_id, norm_x=0.80, norm_y=0.80, trap_type="noop", radius=0.05)
        assert kg.is_trap(node.node_id, 0.81, 0.81)
        assert not kg.is_trap(node.node_id, 0.10, 0.10)

        # 4. User priors
        kg.set_user_prior("com.test.app", "marketplace", "preferred_gpu", "RTX 3060")
        assert kg.get_user_prior("com.test.app", "marketplace", "preferred_gpu") == "RTX 3060"


def test_tag_nav_shortest_path() -> None:
    """Verify Dijkstra pathfinding over transition edges."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_path = os.path.join(tmp_dir, "test_nav.db")
        kg = UIKnowledgeGraph(db_path=db_path)

        # Graph: A -> B (100ms, success=1) -> C (100ms, success=1)
        # Alternate: A -> C (500ms, success=1)
        kg.get_or_create_node("app", "A", "hA")
        kg.get_or_create_node("app", "B", "hB")
        kg.get_or_create_node("app", "C", "hC")

        node_a = "app::A::hA"
        node_b = "app::B::hB"
        node_c = "app::C::hC"

        kg.record_transition(node_a, node_b, "click", {"id": 1}, latency_ms=100.0, success=True)
        kg.record_transition(node_b, node_c, "click", {"id": 2}, latency_ms=100.0, success=True)
        kg.record_transition(node_a, node_c, "click", {"id": 3}, latency_ms=500.0, success=True)

        nav = TagNavPathfinder(kg)
        plan = nav.find_shortest_path(node_a, node_c)

        assert plan is not None
        assert plan.success
        assert len(plan.edges) == 2  # A -> B -> C has lower cost (200 vs 500)
        assert plan.edges[0].target_node_id == node_b
        assert plan.edges[1].target_node_id == node_c


def test_set_of_mark_tagger() -> None:
    """Verify OpenCV Set-of-Mark drawing and tag resolution in < 5ms."""
    tagger = SetOfMarkTagger()
    img = Image.new("RGB", (600, 1000), color=(240, 240, 240))

    elements = [
        FrequentElement(element_id="1", label="Search", mean_x=0.5, mean_y=0.15),
        FrequentElement(element_id="2", label="Filter", mean_x=0.8, mean_y=0.15),
    ]

    # Warmup OpenCV font engine
    tagger.tag_frame(img, elements)
    tagged_img, tag_map, elapsed_ms = tagger.tag_frame(img, elements)
    assert elapsed_ms < 10.0  # Must be fast (< 10ms on CPU)
    assert len(tag_map) == 2
    assert tag_map[1].label == "Search"

    pt1 = tagger.resolve_tag(1, tag_map)
    assert pt1 is not None
    assert pt1.x == 0.5
    assert pt1.y == 0.15


def test_recon_crawler_safety_filter() -> None:
    """Verify destructive keywords are strictly blacklisted."""
    for kw in ["Delete Account", "BUY NOW", "Pay €50", "Reset Settings", "Elimina"]:
        clean = kw.lower()
        is_safe = not any(d in clean for d in DESTRUCTIVE_KEYWORDS)
        assert not is_safe, f"Keyword '{kw}' should be blocked!"

    assert any("search" not in d for d in DESTRUCTIVE_KEYWORDS)


def test_background_reviewer_and_revision_loop() -> None:
    """Verify post-task revision loop audits data and promotes skills."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        skills_dir = os.path.join(tmp_dir, "skills")
        reviewer = BackgroundReviewer(skills_dir=skills_dir)

        actions = [
            {"action": "open", "text": "Wallapop"},
            {"action": "type", "text": "Trovata scheda video RTX 3060 a 250 € con batteria al 85%"},
        ]
        timings = [StepTiming(step_index=1), StepTiming(step_index=2)]

        report = reviewer.audit_task("deal_search", timings, actions, success=True)
        assert "€250" in report.revision_summary
        assert "85%" in report.revision_summary
        assert "RTX 3060" in report.revision_summary
        assert report.promoted_skill_path is not None
        assert os.path.exists(report.promoted_skill_path)

        # Thread spawn check
        thread = reviewer.spawn_background_review("deal_search", timings, actions, success=True)
        thread.join(timeout=2.0)
        assert not thread.is_alive()
