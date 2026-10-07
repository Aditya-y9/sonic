from pathlib import Path

from wav_transcriber.exporters import (
    format_timestamp_srt,
    format_timestamp_vtt,
    write_json,
    write_srt,
    write_txt,
    write_vtt,
)
from wav_transcriber.schemas import ProcessingInfo, Segment, SourceInfo, TranscriptDocument, Word


def _dummy_doc() -> TranscriptDocument:
    source = SourceInfo(
        path="test.wav",
        sha256="abcdef123456",
        duration_seconds=10.5,
        sample_rate_hz=16000,
        channels=1,
        sample_width_bytes=2,
        frame_count=168000,
    )
    proc = ProcessingInfo(
        asr_model="mock",
        language_hints=["en"],
        config_hash="cfg123",
        completed_at="2026-10-05T12:00:00Z",
    )
    seg1 = Segment(
        id="seg-000001",
        speaker="speaker_00",
        channel=0,
        start=2.184,
        end=5.931,
        text_original="Hello, how can I help?",
        languages=["en"],
        confidence=0.95,
        words=[
            Word(text="Hello,", start=2.184, end=2.560, confidence=0.98),
            Word(text="how", start=2.570, end=2.800, confidence=0.96),
        ],
    )
    return TranscriptDocument(
        schema_version="1.0",
        job_id="job-001",
        source=source,
        processing=proc,
        warnings=[],
        segments=[seg1],
        status="completed",
        review_reasons=[],
    )


def test_format_timestamp_srt() -> None:
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(2.184) == "00:00:02,184"
    assert format_timestamp_srt(3665.123) == "01:01:05,123"


def test_format_timestamp_vtt() -> None:
    assert format_timestamp_vtt(0.0) == "00:00:00.000"
    assert format_timestamp_vtt(2.184) == "00:00:02.184"


def test_write_json_and_txt(tmp_path: Path) -> None:
    doc = _dummy_doc()
    json_path = tmp_path / "out.json"
    txt_path = tmp_path / "out.txt"

    write_json(doc, json_path)
    write_txt(doc, txt_path)

    assert json_path.exists()
    assert txt_path.exists()

    txt_content = txt_path.read_text(encoding="utf-8")
    assert "# job_id: job-001" in txt_content
    assert "[speaker_00] Hello, how can I help?" in txt_content


def test_write_srt_and_vtt(tmp_path: Path) -> None:
    doc = _dummy_doc()
    srt_path = tmp_path / "out.srt"
    vtt_path = tmp_path / "out.vtt"

    write_srt(doc, srt_path)
    write_vtt(doc, vtt_path)

    assert srt_path.exists()
    assert vtt_path.exists()

    srt_content = srt_path.read_text(encoding="utf-8")
    assert "00:00:02,184 --> 00:00:05,931" in srt_content
    assert "[speaker_00] Hello, how can I help?" in srt_content

    vtt_content = vtt_path.read_text(encoding="utf-8")
    assert "WEBVTT" in vtt_content
    assert "00:00:02.184 --> 00:00:05.931" in vtt_content
