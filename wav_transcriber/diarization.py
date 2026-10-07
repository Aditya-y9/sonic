from __future__ import annotations

from typing import Any

from .schemas import Segment


def diarize_segments(
    segments: list[Segment],
    enabled: bool,
    channel_map: dict[str | int, Any] | None = None,
) -> tuple[list[Segment], list[str]]:
    if not enabled:
        return segments, []

    warnings: list[str] = []
    cmap = channel_map or {}

    for segment in segments:
        # First priority: Channel map if channel is specified
        if segment.channel is not None:
            ch_key = str(segment.channel)
            ch_info = cmap.get(ch_key) or cmap.get(segment.channel)
            if isinstance(ch_info, dict) and "speaker" in ch_info:
                segment.speaker = ch_info["speaker"]
            elif segment.speaker is None:
                segment.speaker = f"speaker_{segment.channel:02d}"

        # Fallback if speaker is still None
        if segment.speaker is None:
            segment.speaker = "speaker_00"

    return segments, warnings

