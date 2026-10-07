"""
Flow Visualizer Client Library.
Enables any Python script or agent pipeline to emit live execution events
to the Flow Visualizer in real-time.
"""

import time
import requests
from typing import Dict, Any, Optional
from contextlib import contextmanager


class FlowTracker:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url.rstrip("/")

    def emit_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Post a generic event to the visualizer server."""
        try:
            resp = requests.post(f"{self.server_url}/api/events", json={
                "type": event_type,
                "data": data
            }, timeout=2.0)
            return resp.status_code == 200
        except Exception:
            return False

    def start_agent(self, agent_id: str, task: str, input_payload: Optional[Dict[str, Any]] = None):
        """Set an agent to AMBER (processing)."""
        return self.emit_event("agent_status_change", {
            "agent_id": agent_id,
            "status": "processing",
            "color_state": "amber",
            "active_task": task,
            "input_payload": input_payload
        })

    def pass_agent(self, agent_id: str, output_payload: Dict[str, Any], tokens: int = 150):
        """Set an agent to GREEN (passed)."""
        return self.emit_event("agent_status_change", {
            "agent_id": agent_id,
            "status": "passed",
            "color_state": "green",
            "output_payload": output_payload,
            "tokens": tokens
        })

    def fail_agent(self, agent_id: str, error_message: str, failure_context: Optional[Dict[str, Any]] = None):
        """Set an agent to RED (failed)."""
        return self.emit_event("agent_status_change", {
            "agent_id": agent_id,
            "status": "failed",
            "color_state": "red",
            "error": error_message,
            "context": failure_context
        })

    def transfer_data(self, source_id: str, target_id: str, payload: Dict[str, Any], label: str = "Data Transfer"):
        """Visualize data packet movement between two agents."""
        return self.emit_event("data_transfer", {
            "source": source_id,
            "target": target_id,
            "label": label,
            "payload": payload,
            "duration_ms": 750
        })

    @contextmanager
    def track_agent(self, agent_id: str, task: str, input_payload: Optional[Dict[str, Any]] = None):
        """
        Context manager for automatic Amber -> Green / Red lifecycle.
        Example:
            with tracker.track_agent("asr_engine", "Transcribing segment 1", {"audio": "file.wav"}) as node:
                result = run_model()
                node.set_output(result)
        """
        start = time.time()
        self.start_agent(agent_id, task, input_payload)
        holder = {"output": None, "tokens": 100}
        
        class NodeTracker:
            def set_output(self, output: Dict[str, Any], tokens: int = 100):
                holder["output"] = output
                holder["tokens"] = tokens

        tracker_obj = NodeTracker()
        try:
            yield tracker_obj
            duration_ms = int((time.time() - start) * 1000)
            self.pass_agent(agent_id, holder["output"] or {"status": "success", "duration_ms": duration_ms}, holder["tokens"])
        except Exception as err:
            self.fail_agent(agent_id, str(err), {"type": type(err).__name__})
            raise
