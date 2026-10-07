"""
Agent Parser module for GitHub Copilot Custom Agents (*.agent.md).
Parses frontmatter metadata, markdown instructions, tool lists, schemas, and handoff relationships.
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


def parse_frontmatter(content: str) -> tuple[Dict[str, Any], str]:
    """Extract YAML frontmatter and markdown body from markdown content."""
    pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
    match = re.match(pattern, content, re.DOTALL)
    if match:
        fm_text = match.group(1)
        body = match.group(2)
        try:
            metadata = yaml.safe_load(fm_text) or {}
        except Exception as e:
            metadata = {"raw_frontmatter_error": str(e)}
        return metadata, body
    return {}, content


def parse_agent_file(file_path: Path) -> Dict[str, Any]:
    """Parse a single .agent.md file into a structured dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    metadata, body = parse_frontmatter(raw_text)
    file_id = metadata.get("id") or file_path.stem.replace(".agent", "")

    # Default attributes if not explicitly in frontmatter
    name = metadata.get("name", file_id.replace("_", " ").title())
    icon = metadata.get("icon", "bot")
    role = metadata.get("role", "Specialized Agent")
    model = metadata.get("model", "gpt-4o")
    tools = metadata.get("tools", [])
    invokable_by = metadata.get("invokable_by", [])
    subagents = metadata.get("subagents", [])
    input_schema = metadata.get("input_schema", {})
    output_schema = metadata.get("output_schema", {})

    is_orchestrator = "orchestrator" in file_id.lower() or len(subagents) > 0

    return {
        "id": file_id,
        "name": name,
        "filename": file_path.name,
        "file_path": str(file_path.resolve()),
        "icon": icon,
        "role": role,
        "model": model,
        "is_orchestrator": is_orchestrator,
        "tools": tools,
        "invokable_by": invokable_by,
        "subagents": subagents,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "body_markdown": body.strip(),
        "summary": body.strip().split("\n\n")[0] if body else "",
    }


def scan_agents_directory(agents_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Scan the .github/agents directory and build complete agent topology graph.
    Returns nodes and edges for the flow diagram.
    """
    if agents_dir is None:
        base_dir = Path.cwd()
        agents_dir_path = base_dir / ".github" / "agents"
    else:
        agents_dir_path = Path(agents_dir)

    agents: Dict[str, Dict[str, Any]] = {}
    
    if agents_dir_path.exists():
        for file in agents_dir_path.glob("*.agent.md"):
            agent_data = parse_agent_file(file)
            agents[agent_data["id"]] = agent_data

    # If no agents found, fallback check for any .md in directory
    if not agents and agents_dir_path.exists():
        for file in agents_dir_path.glob("*.md"):
            agent_data = parse_agent_file(file)
            agents[agent_data["id"]] = agent_data

    # Build Graph Nodes & Edges
    nodes = []
    edges = []

    # Identify orchestrator(s)
    orchestrator_ids = [aid for aid, a in agents.items() if a.get("is_orchestrator")]
    main_orchestrator_id = orchestrator_ids[0] if orchestrator_ids else (list(agents.keys())[0] if agents else "orchestrator")

    # Layout calculation helper (circular or tiered hierarchical coordinates)
    subagent_ids = [aid for aid in agents.keys() if aid != main_orchestrator_id]
    total_subagents = len(subagent_ids)

    # Position Orchestrator at Top/Center
    orch_data = agents.get(main_orchestrator_id, {
        "id": "orchestrator",
        "name": "Master Orchestrator",
        "role": "Central Controller",
        "icon": "cpu",
        "model": "claude-3-7-sonnet",
        "is_orchestrator": True,
        "tools": ["run_subagent", "delegate_task"],
        "subagents": subagent_ids
    })

    nodes.append({
        **orch_data,
        "x": 480,
        "y": 100,
        "status": "idle",
        "state_color": "slate"  # amber, green, red, slate
    })

    # Position subagents in a structured layout
    # Level 1: Planner (y=240, x=480)
    # Level 2: ASR Engine (x=260, y=380), Diarization Expert (x=700, y=380)
    # Level 3: Quality Auditor (x=480, y=520)
    # Level 4: Exporter Agent (x=300, y=660), Reviewer Critic (x=660, y=660)

    predefined_positions = {
        "planner": (480, 250),
        "asr_engine": (240, 400),
        "diarization_expert": (720, 400),
        "quality_auditor": (480, 550),
        "exporter_agent": (260, 700),
        "reviewer_critic": (700, 700)
    }

    import math
    for idx, aid in enumerate(subagent_ids):
        adata = agents[aid]
        if aid in predefined_positions:
            px, py = predefined_positions[aid]
        else:
            angle = (idx / max(total_subagents, 1)) * math.pi + (math.pi / 6)
            px = int(480 + 320 * math.cos(angle))
            py = int(350 + 260 * math.sin(angle))

        nodes.append({
            **adata,
            "x": px,
            "y": py,
            "status": "idle",
            "state_color": "slate"
        })

    # Build Edges based on orchestrator subagents list and invokable_by
    edge_set = set()

    for aid, adata in agents.items():
        # Orchestrator to subagents
        if adata.get("is_orchestrator"):
            for sub_id in adata.get("subagents", []):
                if sub_id in agents:
                    edge_key = (aid, sub_id)
                    if edge_key not in edge_set:
                        edge_set.add(edge_key)
                        edges.append({
                            "id": f"edge_{aid}_{sub_id}",
                            "source": aid,
                            "target": sub_id,
                            "label": "invoke / dispatch",
                            "type": "control_forward",
                            "animated": False
                        })
                        # Return edge for results
                        edges.append({
                            "id": f"edge_{sub_id}_{aid}",
                            "source": sub_id,
                            "target": aid,
                            "label": "return payload",
                            "type": "data_return",
                            "animated": False
                        })

        # Inter-agent transitions if specified
        for parent_id in adata.get("invokable_by", []):
            if parent_id in agents and (parent_id, aid) not in edge_set:
                edge_set.add((parent_id, aid))
                edges.append({
                    "id": f"edge_{parent_id}_{aid}",
                    "source": parent_id,
                    "target": aid,
                    "label": "invoke",
                    "type": "control_forward",
                    "animated": False
                })

    # Logical cross-agent data pipelines (e.g. planner -> ASR/Diarization -> Quality -> Exporter)
    cross_pipeline_edges = [
        ("planner", "asr_engine", "speech plan"),
        ("planner", "diarization_expert", "clustering plan"),
        ("asr_engine", "quality_auditor", "aligned transcript"),
        ("diarization_expert", "quality_auditor", "speaker turns"),
        ("quality_auditor", "exporter_agent", "verified segments"),
        ("quality_auditor", "reviewer_critic", "anomaly flags"),
        ("reviewer_critic", "asr_engine", "fallback parameters")
    ]

    for src, tgt, lbl in cross_pipeline_edges:
        if src in agents and tgt in agents:
            edges.append({
                "id": f"pipe_{src}_{tgt}",
                "source": src,
                "target": tgt,
                "label": lbl,
                "type": "pipeline_data",
                "animated": False
            })

    return {
        "agents_count": len(agents),
        "nodes": nodes,
        "edges": edges,
        "orchestrator_id": main_orchestrator_id,
        "raw_agents": agents
    }


if __name__ == "__main__":
    import json
    data = scan_agents_directory()
    print(json.dumps(data, indent=2))
