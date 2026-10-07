from __future__ import annotations

from .schemas import Segment


def align_segments(
    segments: list[Segment],
    enabled: bool,
    word_level: bool = True,
) -> tuple[list[Segment], list[str]]:
    if not enabled:
        return segments, []

    warnings: list[str] = []
    for segment in segments:
        if segment.end < segment.start:
            warnings.append(f"Segment {segment.id} had inverted timing [{segment.start} > {segment.end}]; clamped.")
            segment.end = segment.start

        if not word_level:
            continue

        prev_end = segment.start
        for word in segment.words:
            if word.start < segment.start:
                warnings.append(f"Word '{word.text}' start {word.start} clamped to segment start {segment.start}.")
                word.start = segment.start
            if word.end > segment.end:
                warnings.append(f"Word '{word.text}' end {word.end} clamped to segment end {segment.end}.")
                word.end = segment.end
            if word.end < word.start:
                warnings.append(f"Word '{word.text}' had inverted timing; corrected.")
                word.end = word.start
            if word.start < prev_end:
                # Slight overlap or monotonicity violation
                word.start = max(word.start, prev_end)
                word.end = max(word.end, word.start)
            prev_end = word.end

    return segments, warnings

