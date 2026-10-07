"""
Flow Engine module for simulating and managing real-time agent execution flows.
Handles state transitions:
  - 'processing' (Amber #f59e0b)
  - 'success' / 'passed' (Green #10b981)
  - 'failed' (Red #ef4444)
  - 'idle' (Slate #64748b)
Tracks data transfers between agents with packet payloads, metrics, and logs.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Callable
from flow_visualizer.agent_parser import scan_agents_directory


class FlowEngine:
    def __init__(self, agents_dir: Optional[str] = None):
        self.agents_dir = agents_dir
        self.topology = scan_agents_directory(agents_dir)
        self.active_scenario: Optional[str] = None
        self.is_running = False
        self.is_paused = False
        self.current_step_index = 0
        self.speed_multiplier = 1.0
        self.chaos_mode = False  # If True, injects simulated failure
        self.history_events: List[Dict[str, Any]] = []
        self.event_subscribers: List[Callable[[Dict[str, Any]], Any]] = []
        
        # Current snapshot of all nodes state
        self.nodes_state: Dict[str, Dict[str, Any]] = {}
        self.reset_state()

    def reset_state(self):
        """Reset all agent nodes to idle state and clear active packet transfers."""
        self.topology = scan_agents_directory(self.agents_dir)
        self.nodes_state = {}
        for node in self.topology["nodes"]:
            node_id = node["id"]
            self.nodes_state[node_id] = {
                "id": node_id,
                "status": "idle",       # idle | processing | passed | failed
                "color_state": "slate", # slate | amber | green | red
                "active_task": None,
                "last_input": None,
                "last_output": None,
                "error_details": None,
                "logs": [],
                "tokens_used": 0,
                "duration_ms": 0,
                "start_time": None
            }
        self.current_step_index = 0
        self.is_running = False
        self.is_paused = False

    def subscribe(self, callback: Callable[[Dict[str, Any]], Any]):
        """Subscribe to real-time execution events."""
        self.event_subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], Any]):
        if callback in self.event_subscribers:
            self.event_subscribers.remove(callback)

    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast event to all subscribers and append to event history."""
        event = {
            "event_id": str(uuid.uuid4())[:8],
            "timestamp": time.time(),
            "iso_time": time.strftime("%H:%M:%S", time.localtime()),
            "type": event_type,
            "data": data
        }
        self.history_events.append(event)
        if len(self.history_events) > 500:
            self.history_events.pop(0)

        for sub in list(self.event_subscribers):
            try:
                res = sub(event)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as e:
                print(f"[FlowEngine] Subscriber error: {e}")

    # --- Real-Time Execution State Helpers ---
    async def set_agent_processing(self, agent_id: str, task_desc: str, input_payload: Optional[Dict[str, Any]] = None):
        """Transition agent to AMBER processing state."""
        if agent_id in self.nodes_state:
            node = self.nodes_state[agent_id]
            node["status"] = "processing"
            node["color_state"] = "amber"
            node["active_task"] = task_desc
            node["start_time"] = time.time()
            if input_payload:
                node["last_input"] = input_payload
            log_msg = f"Started processing task: {task_desc}"
            node["logs"].append({"time": time.strftime("%H:%M:%S"), "text": log_msg, "level": "info"})
            
            await self.emit_event("agent_status_change", {
                "agent_id": agent_id,
                "status": "processing",
                "color_state": "amber",
                "active_task": task_desc,
                "input_payload": input_payload,
                "log": log_msg
            })

    async def set_agent_passed(self, agent_id: str, output_payload: Dict[str, Any], tokens: int = 150):
        """Transition agent to GREEN passed/success state."""
        if agent_id in self.nodes_state:
            node = self.nodes_state[agent_id]
            duration = int((time.time() - (node["start_time"] or time.time())) * 1000)
            node["status"] = "passed"
            node["color_state"] = "green"
            node["last_output"] = output_payload
            node["tokens_used"] += tokens
            node["duration_ms"] = duration
            log_msg = f"Completed successfully ({duration}ms, {tokens} tokens)"
            node["logs"].append({"time": time.strftime("%H:%M:%S"), "text": log_msg, "level": "success"})

            await self.emit_event("agent_status_change", {
                "agent_id": agent_id,
                "status": "passed",
                "color_state": "green",
                "output_payload": output_payload,
                "duration_ms": duration,
                "tokens": tokens,
                "log": log_msg
            })

    async def set_agent_failed(self, agent_id: str, error_message: str, failure_context: Optional[Dict[str, Any]] = None):
        """Transition agent to RED failed state."""
        if agent_id in self.nodes_state:
            node = self.nodes_state[agent_id]
            duration = int((time.time() - (node["start_time"] or time.time())) * 1000)
            node["status"] = "failed"
            node["color_state"] = "red"
            node["error_details"] = {
                "message": error_message,
                "context": failure_context,
                "timestamp": time.time()
            }
            node["duration_ms"] = duration
            log_msg = f"ERROR: {error_message}"
            node["logs"].append({"time": time.strftime("%H:%M:%S"), "text": log_msg, "level": "error"})

            await self.emit_event("agent_status_change", {
                "agent_id": agent_id,
                "status": "failed",
                "color_state": "red",
                "error": error_message,
                "context": failure_context,
                "duration_ms": duration,
                "log": log_msg
            })

    async def transfer_data(self, source_id: str, target_id: str, payload: Dict[str, Any], label: str = "Data Transfer", duration_ms: int = 800):
        """
        Visualize data passing between two agent nodes along the edge.
        Emits 'data_transfer' event with animated payload packet.
        """
        packet_id = f"pkt_{str(uuid.uuid4())[:6]}"
        await self.emit_event("data_transfer", {
            "packet_id": packet_id,
            "source": source_id,
            "target": target_id,
            "label": label,
            "payload": payload,
            "payload_summary": f"{len(str(payload))} bytes",
            "duration_ms": duration_ms
        })

    # --- Pre-Configured Simulation Scenarios ---
    def get_scenarios(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "full_transcription_pipeline",
                "name": "1. End-to-End Success Pipeline (Standard)",
                "description": "Orchestrator delegates to Planner -> Parallel ASR & Diarization -> Quality Audit -> Exporter -> Complete.",
                "complexity": "Standard Multi-Agent DAG"
            },
            {
                "id": "quality_failure_and_recovery",
                "name": "2. Quality Audit Failure & Critic Self-Healing",
                "description": "ASR produces low-confidence segments. Quality Auditor flags RED -> Orchestrator routes to Reviewer Critic -> Re-transcribe with prompt fallback -> Passes GREEN -> Exporter.",
                "complexity": "Error Recovery & Self-Healing Loop"
            },
            {
                "id": "acoustic_failure_immediate",
                "name": "3. Immediate ASR Acoustic Model Failure (Chaos Test)",
                "description": "Simulates corrupt audio input causing ASR Engine to throw RED failure, followed by Critic diagnostic intervention.",
                "complexity": "Failure Handling"
            }
        ]

    async def run_scenario(self, scenario_id: str):
        """Execute a full simulation scenario with realistic delays and payloads."""
        self.active_scenario = scenario_id
        self.is_running = True
        self.is_paused = False
        self.reset_state()

        await self.emit_event("scenario_started", {"scenario_id": scenario_id})

        try:
            if scenario_id == "full_transcription_pipeline":
                await self._scenario_full_pipeline()
            elif scenario_id == "quality_failure_and_recovery":
                await self._scenario_quality_recovery()
            elif scenario_id == "acoustic_failure_immediate":
                await self._scenario_acoustic_failure()
            else:
                await self._scenario_full_pipeline()
        except asyncio.CancelledError:
            print("[FlowEngine] Scenario cancelled.")
        finally:
            self.is_running = False
            await self.emit_event("scenario_completed", {"scenario_id": scenario_id})

    async def _step_delay(self, seconds: float = 1.2):
        """Sleep with pause/speed support."""
        elapsed = 0.0
        target = max(seconds / max(self.speed_multiplier, 0.1), 0.1)
        while elapsed < target:
            while self.is_paused:
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.05)
            elapsed += 0.05

    async def _scenario_full_pipeline(self):
        # 1. Orchestrator Receives Task
        job_payload = {
            "task": "Transcribe multi-speaker podcast recording with word timestamps & speaker tags",
            "audio_path": "C:\\audio\\tech_podcast_ep42.wav",
            "channels": 2,
            "sample_rate": 16000,
            "target_formats": ["json", "srt", "vtt"]
        }
        await self.set_agent_processing("orchestrator", "Analyzing user request & initializing pipeline context", job_payload)
        await self._step_delay(1.0)

        # 2. Orchestrator -> Planner
        await self.transfer_data("orchestrator", "planner", {"task_goal": job_payload["task"], "audio_metadata": {"duration_s": 142.5, "channels": 2}}, "Task Decomposition Request")
        await self._step_delay(0.6)

        await self.set_agent_processing("planner", "Decomposing task into DAG & checking hardware acceleration")
        await self._step_delay(1.2)

        plan_output = {
            "dag_steps": [
                {"step": 1, "target": "asr_engine", "mode": "parallel", "action": "acoustic_stt"},
                {"step": 1, "target": "diarization_expert", "mode": "parallel", "action": "speaker_clustering"},
                {"step": 2, "target": "quality_auditor", "mode": "sequential", "action": "verify_fidelity"},
                {"step": 3, "target": "exporter_agent", "mode": "sequential", "action": "render_artifacts"}
            ],
            "estimated_duration_s": 4.8
        }
        await self.set_agent_passed("planner", plan_output, tokens=210)
        await self.transfer_data("planner", "orchestrator", plan_output, "Execution DAG Plan")
        await self._step_delay(0.8)

        # 3. Orchestrator triggers ASR Engine and Diarization in parallel
        await self.transfer_data("orchestrator", "asr_engine", {"audio_path": job_payload["audio_path"], "model": "whisper-large-v3-turbo", "beam_size": 5}, "Dispatch ASR")
        await self.transfer_data("orchestrator", "diarization_expert", {"audio_path": job_payload["audio_path"], "vad_mode": "silero", "max_speakers": 3}, "Dispatch Diarization")
        await self._step_delay(0.4)

        # Both processing in AMBER
        await self.set_agent_processing("asr_engine", "Running Whisper acoustic inference & CTC alignment")
        await self.set_agent_processing("diarization_expert", "Extracting x-vector voiceprints & spectral clustering")
        await self._step_delay(1.6)

        # ASR completes
        asr_output = {
            "segments_count": 8,
            "detected_language": "en",
            "mean_confidence": 0.962,
            "sample_snippet": "Welcome back to the Deep Tech podcast. Today we are discussing multi-agent systems..."
        }
        await self.set_agent_passed("asr_engine", asr_output, tokens=480)
        await self.transfer_data("asr_engine", "orchestrator", asr_output, "ASR Transcript Segments")
        await self._step_delay(0.6)

        # Diarization completes
        diar_output = {
            "speaker_turns": 6,
            "speakers_detected": ["SPEAKER_00 (Host)", "SPEAKER_01 (Guest)"],
            "overlap_ratio": 0.041
        }
        await self.set_agent_passed("diarization_expert", diar_output, tokens=310)
        await self.transfer_data("diarization_expert", "orchestrator", diar_output, "Speaker Turn Matrix")
        await self._step_delay(0.8)

        # 4. Orchestrator -> Quality Auditor
        merged_data = {"segments": asr_output, "diarization": diar_output}
        await self.transfer_data("orchestrator", "quality_auditor", merged_data, "Merged Alignment Audit")
        await self._step_delay(0.5)

        await self.set_agent_processing("quality_auditor", "Auditing phonetic confidence, repetition loops & vocabulary")
        await self._step_delay(1.4)

        audit_output = {
            "passed": True,
            "composite_score": 0.978,
            "hallucination_rate": 0.0,
            "terminology_matches": ["multi-agent", "Copilot", "DAG", "CTC alignment"]
        }
        await self.set_agent_passed("quality_auditor", audit_output, tokens=190)
        await self.transfer_data("quality_auditor", "orchestrator", audit_output, "Audit Approval: PASS (Score: 0.98)")
        await self._step_delay(0.8)

        # 5. Orchestrator -> Exporter
        export_request = {"segments": asr_output, "formats": ["json", "srt", "vtt"], "output_dir": "output/transcripts"}
        await self.transfer_data("orchestrator", "exporter_agent", export_request, "Export Formatting Request")
        await self._step_delay(0.5)

        await self.set_agent_processing("exporter_agent", "Generating .SRT subtitle line-breaks & .JSON manifests")
        await self._step_delay(1.2)

        export_output = {
            "generated_files": [
                "output/transcripts/tech_podcast_ep42.srt",
                "output/transcripts/tech_podcast_ep42.vtt",
                "output/transcripts/tech_podcast_ep42.json"
            ],
            "total_bytes": 18450
        }
        await self.set_agent_passed("exporter_agent", export_output, tokens=120)
        await self.transfer_data("exporter_agent", "orchestrator", export_output, "Artifact Deliverables")
        await self._step_delay(0.8)

        # 6. Orchestrator Finish
        final_summary = {
            "status": "completed",
            "transcript_files": export_output["generated_files"],
            "total_speakers": 2,
            "total_duration_s": 142.5,
            "total_tokens_used": 1310
        }
        await self.set_agent_passed("orchestrator", final_summary, tokens=200)

    async def _scenario_quality_recovery(self):
        # 1. Orchestrator -> Planner
        await self.set_agent_processing("orchestrator", "Handling noisy audio file (SNR: 8dB)")
        await self._step_delay(0.8)
        await self.set_agent_processing("planner", "Formulating recovery-tolerant pipeline")
        await self._step_delay(0.8)
        await self.set_agent_passed("planner", {"strategy": "standard_with_critic_fallback"})
        await self.transfer_data("planner", "orchestrator", {"strategy": "standard_with_critic_fallback"})
        await self._step_delay(0.5)

        # 2. ASR Engine
        await self.transfer_data("orchestrator", "asr_engine", {"audio_path": "noisy_call.wav", "temperature": 0.0})
        await self.set_agent_processing("asr_engine", "Transcribing low SNR audio with default temperature")
        await self._step_delay(1.2)
        initial_asr = {"segments_count": 4, "mean_confidence": 0.54, "repetition_loop_detected": True}
        await self.set_agent_passed("asr_engine", initial_asr)
        await self.transfer_data("asr_engine", "quality_auditor", initial_asr, "Raw Segments")
        await self._step_delay(0.5)

        # 3. Quality Auditor FAILS -> RED STATE
        await self.set_agent_processing("quality_auditor", "Evaluating confidence & repetition thresholds")
        await self._step_delay(1.2)
        await self.set_agent_failed("quality_auditor", "Confidence score 0.54 below minimum threshold 0.85; repetitive loop detected in segment #3", {
            "confidence": 0.54,
            "threshold": 0.85,
            "loop_text": "and then and then and then..."
        })
        await self.transfer_data("quality_auditor", "orchestrator", {"status": "REJECTED", "error": "Confidence 0.54 < 0.85"}, "Audit Failure Notice")
        await self._step_delay(0.9)

        # 4. Orchestrator invokes Reviewer Critic
        await self.transfer_data("orchestrator", "reviewer_critic", {
            "failed_agent_id": "quality_auditor",
            "context": initial_asr,
            "problem": "Acoustic repetition loop due to background static"
        }, "Request Remediation Plan")
        await self._step_delay(0.5)

        await self.set_agent_processing("reviewer_critic", "Diagnosing acoustic anomaly & computing temperature fallback")
        await self._step_delay(1.5)

        remedy = {
            "diagnosis": "Hallucination triggered by low SNR; temperature 0.0 stuck in local minima",
            "recommended_action": "retry_with_temperature_fallback",
            "fallback_parameters": {"temperature": 0.2, "condition_on_previous_text": False, "compression_ratio_threshold": 2.4}
        }
        await self.set_agent_passed("reviewer_critic", remedy, tokens=290)
        await self.transfer_data("reviewer_critic", "orchestrator", remedy, "Self-Healing Fallback Config")
        await self._step_delay(0.8)

        # 5. Orchestrator re-dispatches ASR Engine with fallback
        await self.transfer_data("orchestrator", "asr_engine", remedy["fallback_parameters"], "Retry ASR (Fallback Mode)")
        await self.set_agent_processing("asr_engine", "Re-transcribing with temperature 0.2 & noise suppression")
        await self._step_delay(1.4)

        refined_asr = {"segments_count": 6, "mean_confidence": 0.942, "repetition_cleared": True}
        await self.set_agent_passed("asr_engine", refined_asr, tokens=380)
        await self.transfer_data("asr_engine", "quality_auditor", refined_asr, "Refined Transcript")
        await self._step_delay(0.5)

        # 6. Quality Auditor re-evaluates -> GREEN PASS
        await self.set_agent_processing("quality_auditor", "Re-auditing refined transcript")
        await self._step_delay(1.0)
        await self.set_agent_passed("quality_auditor", {"passed": True, "composite_score": 0.951}, tokens=150)
        await self.transfer_data("quality_auditor", "exporter_agent", refined_asr, "Verified Data")
        await self._step_delay(0.5)

        # 7. Exporter
        await self.set_agent_processing("exporter_agent", "Compiling final output files")
        await self._step_delay(0.9)
        await self.set_agent_passed("exporter_agent", {"files": ["output/noisy_call.srt", "output/noisy_call.json"]})
        await self.transfer_data("exporter_agent", "orchestrator", {"files_ready": 2})
        await self.set_agent_passed("orchestrator", {"status": "recovered_and_completed", "retries": 1})

    async def _scenario_acoustic_failure(self):
        await self.set_agent_processing("orchestrator", "Processing corrupt WAV header")
        await self._step_delay(0.8)
        await self.transfer_data("orchestrator", "asr_engine", {"audio_path": "corrupted_file.wav"})
        await self.set_agent_processing("asr_engine", "Attempting soundfile / ffmpeg decode")
        await self._step_delay(1.2)
        await self.set_agent_failed("asr_engine", "Acoustic Decode Error: Corrupted RIFF header at byte offset 0x0042", {"error_code": "AUDIO_DECODE_ERR_102"})
        await self.transfer_data("asr_engine", "orchestrator", {"status": "FATAL_ERROR", "code": 102}, "Failure Alert")
        await self._step_delay(0.8)
        await self.transfer_data("orchestrator", "reviewer_critic", {"error": "Corrupted audio stream"}, "Diagnostic Call")
        await self.set_agent_processing("reviewer_critic", "Inspecting binary header for byte repair options")
        await self._step_delay(1.2)
        await self.set_agent_passed("reviewer_critic", {"recommendation": "abort_with_report", "reason": "Unrecoverable audio header truncation"})
        await self.set_agent_failed("orchestrator", "Pipeline aborted: Input audio corrupted", {"aborted_at": "asr_engine"})
