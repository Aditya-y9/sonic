from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class PreprocessingConfig:
    """Fine-grained audio preprocessing controls.

    level:
      0 = none
      1 = sanitize only (mono, 16kHz resample)
      2 = sanitize + HP/LP filtering
      3 = sanitize + filtering + noise reduction (spectral gating)
      4 = sanitize + filtering + noise reduction + loudness normalization
    """
    level: int = 4
    noise_reduce_prop_decrease: float = 0.8
    noise_reduce_stationary: bool = False
    noise_reduce_time_constant_s: float = 2.0
    target_sample_rate_hz: int = 16000
    target_loudness_lufs: float = -20.0
    highpass_cutoff_hz: int = 80
    lowpass_cutoff_hz: int = 8000
    apply_pre_emphasis: bool = True
    pre_emphasis_coef: float = 0.97
    remove_long_silence: bool = False
    silence_threshold_db: float = -40.0
    min_silence_duration_ms: float = 1000.0
    enable_wpe: bool = False
    enable_deepfilter: bool = False
    wpe_taps: int = 10
    wpe_delay: int = 3


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
    normalize_loudness: bool = True
    target_loudness_lufs: float = -20.0
    denoise: str = "aggressive"
    denoise_strength: float = 0.5
    remove_long_silence: bool = True
    silence_threshold_db: float = -40.0
    min_silence_duration_ms: float = 1000.0
    split_channels_for_stereo: bool = False
    channel_map: dict[str, Any] = field(default_factory=dict)
    highpass_cutoff_hz: int = 80
    lowpass_cutoff_hz: int = 8000
    apply_pre_emphasis: bool = True
    pre_emphasis_coef: float = 0.97


@dataclass(slots=True)
class EnsembleConfig:
    """Multi-engine ensemble decoding configuration.

    When enabled, runs two ASR models in parallel and reconciles their
    outputs using ROVER (Recognizer Output Voting Error Reduction).
    """
    enabled: bool = True
    primary_model: str = "large-v3"
    secondary_model: str = "large-v3-turbo"
    primary_compute_type: str = "float16"
    secondary_compute_type: str = "float16"


@dataclass(slots=True)
class AsrConfig:
    backend: str = "auto"
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "float16"
    task: str = "transcribe"
    language_hints: list[str] = field(default_factory=lambda: ["en", "hi"])
    temperature: list[float] = field(default_factory=lambda: [0.0])
    beam_size: int = 10
    best_of: int = 10
    patience: float = 2.0
    length_penalty: float = 1.0
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 3
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    condition_on_previous_text: bool = True
    prompt_reset_on_temperature: float = 0.5
    initial_prompt: str | None = None
    prefix: str | None = None
    suppress_blank: bool = True
    suppress_tokens: list[int] = field(default_factory=lambda: [-1])
    word_timestamps: bool = True
    prepend_punctuations: str = "\"'¿([{-"
    append_punctuations: str = "\"'.。,，!！?？:：\")]}、"
    vad_filter: bool = True
    vad_parameters: dict[str, Any] = field(default_factory=dict)
    vocabulary_files: list[str] = field(default_factory=list)
    hallucination_silence_threshold: float = 2.0
    hotwords: str | None = None
    use_batched_pipeline: bool = False
    batch_size: int = 8
    temperature_fallback: list[float] = field(default_factory=lambda: [0.0, 0.2, 0.4, 0.6])
    reset_prompt_on_speaker_turn: bool = True


@dataclass(slots=True)
class AlignmentConfig:
    enabled: bool = True
    word_level: bool = True
    model: str | None = "WAV2VEC2_ASR_BASE_960H"
    return_char_alignments: bool = False
    interpolate_method: str = "nearest"
    refinement_steps: int = 5


@dataclass(slots=True)
class DiarizationConfig:
    enabled: bool = True
    min_speakers: int | None = None
    max_speakers: int | None = None
    detect_overlap: bool = True
    model: str | None = "pyannote/speaker-diarization-3.1"
    embedding_model: str | None = "pyannote/wespeaker-voxceleb-resnet34-LM"
    clustering_method: str = "centroid"
    min_segment_duration: float = 0.0
    min_speakers_per_cluster: int = 1
    auto_estimate_speakers: bool = True
    max_clusters: int = 6


@dataclass(slots=True)
class LanguageConfig:
    detect_code_switch: bool = True
    use_language_detection_model: bool = True
    min_language_confidence: float = 0.7
    context_window_size: int = 3


@dataclass(slots=True)
class QualityConfig:
    fail_on_empty_transcript: bool = False
    low_confidence_threshold: float = 0.75
    hallucination_detection: bool = True
    check_compression_ratio: bool = True
    max_compression_ratio: float = 2.4
    repetition_detection: bool = True
    max_repetition_score: float = 0.3
    cross_validation: bool = True
    ensemble_voting: bool = False


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
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
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
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        asr=AsrConfig(**raw.get("asr", {})),
        ensemble=EnsembleConfig(**raw.get("ensemble", {})),
        alignment=AlignmentConfig(**raw.get("alignment", {})),
        diarization=DiarizationConfig(**raw.get("diarization", {})),
        language=LanguageConfig(**raw.get("language", {})),
        quality=QualityConfig(**raw.get("quality", {})),
        output=OutputConfig(**raw.get("output", {})),
        raw=raw,
    )
    return cfg

