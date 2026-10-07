"""
Demonstration script showing how real Python code or agent runs
can emit live real-time status, data packets, and logs to the Flow Visualizer.
Run this script while the visualizer server is running (python run_visualizer.py)
to watch the visualizer respond dynamically in real-time!
"""

import time
from flow_visualizer.client import FlowTracker


def run_live_pipeline_demo():
    client = FlowTracker(server_url="http://127.0.0.1:8000")
    print("[Live Pipeline] Connecting to Flow Visualizer...")

    # Step 1: Orchestrator initialization
    client.start_agent("orchestrator", "Parsing user query: 'Transcribe meeting with speaker labels'", {
        "user_query": "Transcribe meeting with speaker labels",
        "audio_file": "conference_call_16k.wav"
    })
    time.sleep(1.5)

    # Step 2: Orchestrator -> Planner
    client.transfer_data("orchestrator", "planner", {
        "objective": "Meeting transcription",
        "audio_specs": {"channels": 1, "duration_s": 94.2}
    }, "Task Specification")
    time.sleep(0.8)

    client.start_agent("planner", "Formulating optimal DAG execution graph")
    time.sleep(2.0)

    plan_data = {
        "parallel_tasks": ["asr_engine", "diarization_expert"],
        "post_eval": "quality_auditor",
        "sink": "exporter_agent"
    }
    client.pass_agent("planner", plan_data, tokens=190)
    client.transfer_data("planner", "orchestrator", plan_data, "Resolved DAG Plan")
    time.sleep(1.0)

    # Step 3: Parallel Dispatch to ASR Engine and Diarization Expert
    client.transfer_data("orchestrator", "asr_engine", {"model": "whisper-large-v3-turbo"}, "Dispatch ASR")
    client.transfer_data("orchestrator", "diarization_expert", {"min_speakers": 2}, "Dispatch Diarization")
    time.sleep(0.5)

    client.start_agent("asr_engine", "Running acoustic decoding & phoneme alignment")
    client.start_agent("diarization_expert", "Clustering speaker voiceprints")
    time.sleep(2.5)

    # ASR Finish
    asr_res = {"segments": 14, "confidence": 0.972, "language": "en"}
    client.pass_agent("asr_engine", asr_res, tokens=450)
    client.transfer_data("asr_engine", "quality_auditor", asr_res, "ASR Segments")
    time.sleep(0.8)

    # Diarization Finish
    diar_res = {"speakers": ["SPEAKER_00", "SPEAKER_01"], "turns": 9}
    client.pass_agent("diarization_expert", diar_res, tokens=280)
    client.transfer_data("diarization_expert", "quality_auditor", diar_res, "Speaker Turns")
    time.sleep(1.0)

    # Step 4: Quality Auditor
    client.start_agent("quality_auditor", "Validating acoustic grounding & WER metrics")
    time.sleep(1.8)
    audit_res = {"status": "PASSED", "wer_estimate": 0.024, "score": 0.98}
    client.pass_agent("quality_auditor", audit_res, tokens=140)
    client.transfer_data("quality_auditor", "orchestrator", audit_res, "Audit Approval")
    time.sleep(1.0)

    # Step 5: Exporter Agent
    client.transfer_data("orchestrator", "exporter_agent", {"formats": ["srt", "vtt", "json"]}, "Export Deliverables")
    client.start_agent("exporter_agent", "Generating subtitle cue intervals")
    time.sleep(1.5)
    export_res = {"files": ["output/meeting.srt", "output/meeting.vtt", "output/meeting.json"], "bytes": 24100}
    client.pass_agent("exporter_agent", export_res, tokens=120)
    client.transfer_data("exporter_agent", "orchestrator", export_res, "Files Ready")
    time.sleep(0.8)

    # Step 6: Orchestrator Final Pass
    client.pass_agent("orchestrator", {
        "status": "success",
        "output_files": export_res["files"],
        "total_tokens": 1180
    }, tokens=180)

    print("[Live Pipeline] Completed live pipeline run.")


if __name__ == "__main__":
    run_live_pipeline_demo()
