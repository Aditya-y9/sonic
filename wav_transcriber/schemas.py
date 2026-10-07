from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class LanguageSpan:
    start: float
    end: float
    language: str


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(slots=True)
class Segment:
    id: str
    speaker: str | None
    channel: int | None
    start: float
    end: float
    text_original: str
    text_cleaned: str | None = None
    language_spans: list[LanguageSpan] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    overlap: bool = False
    confidence: float | None = None
    words: list[Word] = field(default_factory=list)


@dataclass(slots=True)
class SourceInfo:
    path: str
    sha256: str
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    sample_width_bytes: int
    frame_count: int


@dataclass(slots=True)
class ProcessingInfo:
    asr_model: str
    alignment_model: str | None = None
    diarization_model: str | None = None
    language_hints: list[str] = field(default_factory=list)
    config_hash: str = ""
    completed_at: str = ""


@dataclass(slots=True)
class TranscriptDocument:
    job_id: str
    source: SourceInfo
    processing: ProcessingInfo
    warnings: list[str]
    segments: list[Segment]
    status: str
    schema_version: str = "1.0"
    review_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

