from __future__ import annotations

from dataclasses import dataclass, field

from .config import QualityConfig
from .schemas import Segment


@dataclass(slots=True)
class QualityResult:
    status: str  # "completed", "partial", "needs_review", "failed"
    review_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def run_quality_gates(
    segments: list[Segment],
    quality_config: QualityConfig,
    prior_warnings: list[str],
) -> QualityResult:
    warnings = list(prior_warnings)
    reasons: list[str] = []

    if not segments:
        message = "Transcript has no segments."
        if quality_config.fail_on_empty_transcript:
            reasons.append(message)
        else:
            warnings.append(message)

    for segment in segments:
        # Segment duration validation
        if segment.end < segment.start:
            reasons.append(f"Segment {segment.id} has negative duration ({segment.start} > {segment.end}).")

        # Speaker validation
        if not segment.speaker:
            reasons.append(f"Segment {segment.id} has missing speaker attribution.")

        # Channel validation
        if segment.channel is not None and segment.channel < 0:
            reasons.append(f"Segment {segment.id} has invalid negative channel ID {segment.channel}.")

        # Confidence validation
        if segment.confidence is not None and segment.confidence < quality_config.low_confidence_threshold:
            reasons.append(
                f"Segment {segment.id} below confidence threshold "
                f"({segment.confidence:.2f} < {quality_config.low_confidence_threshold:.2f})."
            )

        # Word duration and bounding validation
        for word in segment.words:
            if word.end < word.start:
                reasons.append(f"Word '{word.text}' in {segment.id} has negative duration.")
            if word.start < segment.start - 0.001 or word.end > segment.end + 0.001:
                warnings.append(
                    f"Word '{word.text}' [{word.start:.3f}-{word.end:.3f}] exceeds segment {segment.id} bounds [{segment.start:.3f}-{segment.end:.3f}]."
                )
            if word.confidence is not None and word.confidence < quality_config.low_confidence_threshold:
                warnings.append(
                    f"Word '{word.text}' in {segment.id} has low confidence ({word.confidence:.2f})."
                )

    if reasons:
        return QualityResult(status="needs_review", review_reasons=sorted(set(reasons)), warnings=warnings)
    if warnings:
        return QualityResult(status="partial", review_reasons=[], warnings=warnings)
    return QualityResult(status="completed", review_reasons=[], warnings=[])

