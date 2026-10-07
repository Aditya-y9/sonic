"""
Launch the Real-Time Multi-Agent Flow Visualizer Server and UI.
Usage:
    python run_visualizer.py [--port 8000] [--no-browser]
"""

import sys
import argparse
import webbrowser
import uvicorn
from pathlib import Path

from flow_visualizer.agent_parser import scan_agents_directory


def main():
    parser = argparse.ArgumentParser(description="Real-Time Flow Visualizer for GitHub Copilot Multi-Agent Orchestrator")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open default web browser")
    args = parser.parse_args()

    # Scan agents and report topology
    agents_dir = Path.cwd() / ".github" / "agents"
    topology = scan_agents_directory(str(agents_dir))

    print("=" * 70)
    print(" COPILOT MULTI-AGENT REAL-TIME FLOW VISUALIZER")
    print("=" * 70)
    print(f" Agent Directory : {agents_dir}")
    print(f" Discovered Agents ({len(topology['nodes'])}):")
    for node in topology["nodes"]:
        prefix = " [ORCHESTRATOR]" if node.get("is_orchestrator") else " [SUBAGENT]   "
        print(f"    {prefix}  @{node['id']:<24} {node['name']} ({node.get('model', 'gpt-4o')})")

    print(f"\n Flow Edges      : {len(topology['edges'])} channels")
    print(f" Web Dashboard   : http://{args.host}:{args.port}")
    print(f" WebSocket Stream: ws://{args.host}:{args.port}/ws")
    print(f" Standalone File : flow_visualizer/standalone_viewer.html")
    print("=" * 70)
    print(" Press CTRL+C to stop the server.\n")

    if not args.no_browser:
        try:
            webbrowser.open(f"http://{args.host}:{args.port}")
        except Exception:
            pass

    uvicorn.run("flow_visualizer.server:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
