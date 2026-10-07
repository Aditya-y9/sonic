"""
Unit tests for the Multi-Agent Flow Visualizer.
Tests agent parsing, DAG generation, state transitions (amber, green, red), and event emission.
"""

import pytest
import asyncio
from pathlib import Path

from flow_visualizer.agent_parser import scan_agents_directory, parse_agent_file
from flow_visualizer.flow_engine import FlowEngine


def test_agent_parser_discovers_all_agents():
    agents_dir = Path.cwd() / ".github" / "agents"
    topology = scan_agents_directory(str(agents_dir))

    assert topology["agents_count"] == 7
    node_ids = {n["id"] for n in topology["nodes"]}
    expected_ids = {"orchestrator", "planner", "asr_engine", "diarization_expert", "quality_auditor", "exporter_agent", "reviewer_critic"}
    assert expected_ids.issubset(node_ids)

    # Validate orchestrator node
    orch = next(n for n in topology["nodes"] if n["id"] == "orchestrator")
    assert orch["is_orchestrator"] is True
    assert orch["model"] == "claude-3-7-sonnet"

    # Validate edges
    assert len(topology["edges"]) > 0


@pytest.mark.asyncio
async def test_flow_engine_state_transitions():
    engine = FlowEngine()
    engine.reset_state()

    events_captured = []
    engine.subscribe(lambda ev: events_captured.append(ev))

    # Test Amber processing
    await engine.set_agent_processing("orchestrator", "Initial parse", {"task": "test"})
    assert engine.nodes_state["orchestrator"]["status"] == "processing"
    assert engine.nodes_state["orchestrator"]["color_state"] == "amber"

    # Test Green passed
    await engine.set_agent_passed("orchestrator", {"status": "ok"}, tokens=100)
    assert engine.nodes_state["orchestrator"]["status"] == "passed"
    assert engine.nodes_state["orchestrator"]["color_state"] == "green"
    assert engine.nodes_state["orchestrator"]["tokens_used"] == 100

    # Test Red failed
    await engine.set_agent_failed("asr_engine", "Acoustic error", {"code": 500})
    assert engine.nodes_state["asr_engine"]["status"] == "failed"
    assert engine.nodes_state["asr_engine"]["color_state"] == "red"

    # Test data transfer event
    await engine.transfer_data("orchestrator", "planner", {"query": "hello"}, "Test Edge")

    assert len(events_captured) >= 4
    event_types = [ev["type"] for ev in events_captured]
    assert "agent_status_change" in event_types
    assert "data_transfer" in event_types
