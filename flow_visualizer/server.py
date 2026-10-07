"""
FastAPI Server for Real-Time Multi-Agent Flow Visualizer.
Provides REST and WebSocket endpoints for live topology, state tracking, and UI streaming.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from flow_visualizer.agent_parser import scan_agents_directory
from flow_visualizer.flow_engine import FlowEngine

app = FastAPI(title="Copilot Multi-Agent Real-Time Flow Visualizer")

# Root directory configuration
BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = Path(__file__).parent / "static"
AGENTS_DIR = BASE_DIR / ".github" / "agents"

# Initialize Flow Engine
engine = FlowEngine(str(AGENTS_DIR))

# Active WebSocket connections
active_connections: list[WebSocket] = []


class EventPayload(BaseModel):
    type: str
    data: Dict[str, Any]


class RunScenarioRequest(BaseModel):
    scenario_id: str
    speed: Optional[float] = 1.0


class SpeedRequest(BaseModel):
    speed: float


# Broadcast engine events to WebSockets
async def broadcast_event(event: Dict[str, Any]):
    message = json.dumps(event)
    for connection in list(active_connections):
        try:
            await connection.send_text(message)
        except Exception:
            if connection in active_connections:
                active_connections.remove(connection)

engine.subscribe(lambda ev: asyncio.create_task(broadcast_event(ev)))


@app.get("/api/agents")
async def get_agents():
    """Retrieve freshly parsed agent topology from .github/agents/*.agent.md."""
    topology = scan_agents_directory(str(AGENTS_DIR))
    engine.topology = topology
    return topology


@app.get("/api/state")
async def get_current_state():
    """Get current snapshot of all node states and history."""
    return {
        "nodes_state": engine.nodes_state,
        "is_running": engine.is_running,
        "is_paused": engine.is_paused,
        "speed": engine.speed_multiplier,
        "active_scenario": engine.active_scenario,
        "history_count": len(engine.history_events)
    }


@app.get("/api/scenarios")
async def get_scenarios():
    """Get list of playable simulation scenarios."""
    return engine.get_scenarios()


current_scenario_task: Optional[asyncio.Task] = None

@app.post("/api/scenarios/run")
async def run_scenario(req: RunScenarioRequest):
    """Trigger execution of a flow scenario."""
    global current_scenario_task
    if current_scenario_task and not current_scenario_task.done():
        current_scenario_task.cancel()
    
    engine.speed_multiplier = req.speed or 1.0
    current_scenario_task = asyncio.create_task(engine.run_scenario(req.scenario_id))
    return {"status": "started", "scenario_id": req.scenario_id}


@app.post("/api/scenarios/pause")
async def toggle_pause():
    """Pause or resume running scenario."""
    engine.is_paused = not engine.is_paused
    await engine.emit_event("execution_pause_toggled", {"is_paused": engine.is_paused})
    return {"is_paused": engine.is_paused}


@app.post("/api/scenarios/reset")
async def reset_flow():
    """Reset all nodes and clear active animations."""
    global current_scenario_task
    if current_scenario_task and not current_scenario_task.done():
        current_scenario_task.cancel()
    engine.reset_state()
    await engine.emit_event("flow_reset", {})
    return {"status": "reset"}


@app.post("/api/scenarios/speed")
async def set_speed(req: SpeedRequest):
    """Update execution speed multiplier."""
    engine.speed_multiplier = max(req.speed, 0.1)
    await engine.emit_event("speed_changed", {"speed": engine.speed_multiplier})
    return {"speed": engine.speed_multiplier}


@app.post("/api/events")
async def post_event(payload: EventPayload):
    """Ingest external live events from client scripts."""
    if payload.type == "agent_status_change":
        aid = payload.data.get("agent_id")
        status = payload.data.get("status")
        if status == "processing":
            await engine.set_agent_processing(aid, payload.data.get("active_task", "Processing"), payload.data.get("input_payload"))
        elif status == "passed":
            await engine.set_agent_passed(aid, payload.data.get("output_payload", {}), payload.data.get("tokens", 100))
        elif status == "failed":
            await engine.set_agent_failed(aid, payload.data.get("error", "Unknown error"), payload.data.get("context"))
    elif payload.type == "data_transfer":
        await engine.transfer_data(
            payload.data.get("source"),
            payload.data.get("target"),
            payload.data.get("payload", {}),
            payload.data.get("label", "Data Transfer"),
            payload.data.get("duration_ms", 800)
        )
    else:
        await engine.emit_event(payload.type, payload.data)

    return {"status": "received"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    # Send initial state and topology on connection
    init_data = {
        "type": "init",
        "data": {
            "topology": scan_agents_directory(str(AGENTS_DIR)),
            "nodes_state": engine.nodes_state,
            "scenarios": engine.get_scenarios(),
            "is_running": engine.is_running,
            "speed": engine.speed_multiplier
        }
    }
    await websocket.send_text(json.dumps(init_data))

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                msg = json.loads(raw_text)
                action = msg.get("action")
                if action == "run_scenario":
                    sc_id = msg.get("scenario_id", "full_transcription_pipeline")
                    sp = msg.get("speed", 1.0)
                    engine.speed_multiplier = sp
                    asyncio.create_task(engine.run_scenario(sc_id))
                elif action == "pause":
                    engine.is_paused = not engine.is_paused
                    await engine.emit_event("execution_pause_toggled", {"is_paused": engine.is_paused})
                elif action == "reset":
                    engine.reset_state()
                    await engine.emit_event("flow_reset", {})
                elif action == "set_speed":
                    engine.speed_multiplier = float(msg.get("speed", 1.0))
                    await engine.emit_event("speed_changed", {"speed": engine.speed_multiplier})
                elif action == "refresh_agents":
                    topo = scan_agents_directory(str(AGENTS_DIR))
                    engine.topology = topo
                    await engine.emit_event("topology_updated", {"topology": topo})
                elif action == "chaos_fail_node":
                    node_id = msg.get("node_id")
                    if node_id:
                        await engine.set_agent_failed(node_id, "Injected Chaos Fault: Synthetic process abort", {"reason": "Manual Chaos Injection"})
            except Exception as e:
                print(f"[WebSocket] Error processing client message: {e}")
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)


# Mount static web UI files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Flow Visualizer UI loading...</h1>")


def run():
    import uvicorn
    uvicorn.run("flow_visualizer.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
