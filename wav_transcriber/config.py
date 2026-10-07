from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class JobConfig:
    output_dir: str = "output"
    resume: bool = True
    recursive: bool = False
    workers: int = 1


@dataclass(slots=True)
class AudioConfig:
    preserve_original: bool = True
    target_sample_rate_hz: int = 16000
    keep_channels: bool = True
    normalize_loudness: bool = False
    denoise: str = "optional"
    remove_long_silence: bool = False
    split_channels_for_stereo: bool = False
    channel_map: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AsrConfig:
    backend: str = "auto"
    model: str = "base"
    device: str = "auto"
    compute_type: str = "auto"
    task: str = "transcribe"
    language_hints: list[str] = field(default_factory=lambda: ["en", "hi"])
    temperature: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.2])
    beam_size: int = 5
    best_of: int = 5
    condition_on_previous_text: bool = False
    word_timestamps: bool = True
    vad_filter: bool = True
    vocabulary_files: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AlignmentConfig:
    enabled: bool = True
    word_level: bool = True
    model: str | None = None


@dataclass(slots=True)
class DiarizationConfig:
    enabled: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    detect_overlap: bool = True
    model: str | None = None


@dataclass(slots=True)
class LanguageConfig:
    detect_code_switch: bool = True


@dataclass(slots=True)
class QualityConfig:
    fail_on_empty_transcript: bool = False
    low_confidence_threshold: float = 0.6


@dataclass(slots=True)
class OutputConfig:
    write_txt: bool = True
    write_json: bool = True
    write_srt: bool = False
    write_vtt: bool = False
    include_words: bool = True
    include_confidence: bool = True
    include_provenance: bool = True


@dataclass(slots=True)
class AppConfig:
    job: JobConfig = field(default_factory=JobConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def config_hash(self) -> str:
        normalized = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _optional_yaml_loader() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "PyYAML is required to parse config files. Install dependencies with "
            "`python -m pip install -e .`"
        ) from exc
    return yaml


def load_config(config_path: Path | None) -> AppConfig:
    if config_path is None:
        cfg = AppConfig()
        cfg.raw = {
            "job": asdict(cfg.job),
            "audio": asdict(cfg.audio),
            "asr": asdict(cfg.asr),
            "alignment": asdict(cfg.alignment),
            "diarization": asdict(cfg.diarization),
            "language": asdict(cfg.language),
            "quality": asdict(cfg.quality),
            "output": asdict(cfg.output),
        }
        return cfg

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    yaml = _optional_yaml_loader()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = AppConfig(
        job=JobConfig(**raw.get("job", {})),
        audio=AudioConfig(**raw.get("audio", {})),
        asr=AsrConfig(**raw.get("asr", {})),
        alignment=AlignmentConfig(**raw.get("alignment", {})),
        diarization=DiarizationConfig(**raw.get("diarization", {})),
        language=LanguageConfig(**raw.get("language", {})),
        quality=QualityConfig(**raw.get("quality", {})),
        output=OutputConfig(**raw.get("output", {})),
        raw=raw,
    )
    return cfg

