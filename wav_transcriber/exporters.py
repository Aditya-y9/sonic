from __future__ import annotations

import json
from pathlib import Path

from .schemas import TranscriptDocument


def format_timestamp_srt(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def write_json(document: TranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(document.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_txt(document: TranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# job_id: {document.job_id}")
    lines.append(f"# source: {document.source.path}")
    lines.append(f"# status: {document.status}")
    if document.warnings:
        lines.append("# warnings:")
        for warning in document.warnings:
            lines.append(f"# - {warning}")
    if document.review_reasons:
        lines.append("# review_reasons:")
        for reason in document.review_reasons:
            lines.append(f"# - {reason}")
    lines.append("")
    for segment in document.segments:
        speaker = segment.speaker or "unknown"
        lines.append(
            f"[{segment.start:08.3f} - {segment.end:08.3f}] [{speaker}] {segment.text_original}"
        )
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def write_srt(document: TranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for index, segment in enumerate(document.segments, start=1):
        start_str = format_timestamp_srt(segment.start)
        end_str = format_timestamp_srt(segment.end)
        speaker_prefix = f"[{segment.speaker}] " if segment.speaker else ""
        text = f"{speaker_prefix}{segment.text_original}".strip()
        entries.append(f"{index}\n{start_str} --> {end_str}\n{text}\n")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(entries).strip() + "\n")


def write_vtt(document: TranscriptDocument, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["WEBVTT", ""]
    for index, segment in enumerate(document.segments, start=1):
        start_str = format_timestamp_vtt(segment.start)
        end_str = format_timestamp_vtt(segment.end)
        speaker_prefix = f"<v {segment.speaker}>" if segment.speaker else ""
        text = f"{speaker_prefix}{segment.text_original}".strip()
        lines.append(f"{index}")
        lines.append(f"{start_str} --> {end_str}")
        lines.append(text)
        lines.append("")

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")

