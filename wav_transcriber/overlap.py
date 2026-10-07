from __future__ import annotations

from .schemas import Segment


def detect_overlap(
    segments: list[Segment],
    enabled: bool,
    threshold_seconds: float = 0.05,
) -> tuple[list[Segment], list[str]]:
    """Detect overlapping speech across concurrent speaker/channel segments without flattening dialogue."""
    if not enabled or len(segments) <= 1:
        return segments, []

    ordered = sorted(segments, key=lambda s: (s.start, s.end, s.id))
    warnings: list[str] = []

    # Reset flags
    for seg in ordered:
        seg.overlap = False

    overlap_count = 0
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            seg_a = ordered[i]
            seg_b = ordered[j]

            # Since sorted by start, if seg_b starts after seg_a ends, no further overlaps for seg_a
            if seg_b.start >= seg_a.end:
                break

            overlap_duration = min(seg_a.end, seg_b.end) - max(seg_a.start, seg_b.start)
            if overlap_duration > threshold_seconds:
                seg_a.overlap = True
                seg_b.overlap = True
                overlap_count += 1

    if overlap_count > 0:
        warnings.append(f"Detected {overlap_count} overlapping segment pair(s).")

    return ordered, warnings

