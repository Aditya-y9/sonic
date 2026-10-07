from __future__ import annotations

from .schemas import Segment


def normalize_segment_ids(segments: list[Segment]) -> list[Segment]:
    """Sort segments deterministically and assign 1-based sequential IDs."""
    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.channel if s.channel is not None else -1, s.id))
    for idx, segment in enumerate(ordered, start=1):
        segment.id = f"seg-{idx:06d}"
    return ordered


def merge_channel_segments(channel_segments_list: list[list[Segment]]) -> list[Segment]:
    """Merge segments from independent channel transcriptions into a unified chronological stream."""
    all_segments: list[Segment] = []
    for seg_list in channel_segments_list:
        all_segments.extend(seg_list)
    return normalize_segment_ids(all_segments)

