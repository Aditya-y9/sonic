from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .config import AsrConfig
from .schemas import LanguageSpan, Segment, Word


def load_vocabulary(vocabulary_files: list[str]) -> list[str]:
    """Load vocabulary hints from text files, returning list of terms."""
    terms: list[str] = []
    seen: set[str] = set()
    for file_entry in vocabulary_files:
        path = Path(file_entry)
        if not path.exists():
            continue
        try:
            content = path.read_text(encoding="utf-8")
            for line in content.splitlines():
                term = line.strip()
                if term and not term.startswith("#") and term not in seen:
                    terms.append(term)
                    seen.add(term)
        except OSError:
            continue
    return terms


@dataclass(slots=True)
class AsrResult:
    segments: list[Segment]
    warnings: list[str] = field(default_factory=list)
    engine_name: str = ""


class AsrBackend(Protocol):
    name: str

    def transcribe(self, audio_path: Path, duration_seconds: float) -> AsrResult:
        ...


class MockAsrBackend:
    name = "mock"

    def __init__(self, language_hints: list[str], vocabulary_files: list[str] | None = None) -> None:
        self._language_hints = language_hints
        self._vocabulary = load_vocabulary(vocabulary_files or [])

    def transcribe(self, audio_path: Path, duration_seconds: float) -> AsrResult:
        json_sidecar = audio_path.with_suffix(".json")
        if json_sidecar.exists():
            try:
                data = json.loads(json_sidecar.read_text(encoding="utf-8"))
                raw_segments = data.get("segments", [])
                segments: list[Segment] = []
                for idx, item in enumerate(raw_segments, start=1):
                    words = [
                        Word(
                            text=w.get("text", ""),
                            start=float(w.get("start", 0.0)),
                            end=float(w.get("end", 0.0)),
                            confidence=float(w.get("confidence", 0.9)) if w.get("confidence") is not None else None,
                        )
                        for w in item.get("words", [])
                    ]
                    spans = [
                        LanguageSpan(
                            start=float(s.get("start", 0.0)),
                            end=float(s.get("end", 0.0)),
                            language=str(s.get("language", "en")),
                        )
                        for s in item.get("language_spans", [])
                    ]
                    segments.append(
                        Segment(
                            id=item.get("id", f"seg-{idx:06d}"),
                            speaker=item.get("speaker"),
                            channel=item.get("channel"),
                            start=float(item.get("start", 0.0)),
                            end=float(item.get("end", duration_seconds)),
                            text_original=item.get("text_original", item.get("text", "")),
                            text_cleaned=item.get("text_cleaned"),
                            language_spans=spans,
                            languages=item.get("languages", self._language_hints[:1] if self._language_hints else []),
                            overlap=bool(item.get("overlap", False)),
                            confidence=float(item.get("confidence", 0.85)) if item.get("confidence") is not None else None,
                            words=words,
                        )
                    )
                return AsrResult(segments=segments, warnings=[], engine_name="mock")
            except Exception as exc:
                return AsrResult(
                    segments=[],
                    warnings=[f"Failed to parse JSON sidecar {json_sidecar}: {exc}"],
                    engine_name="mock",
                )

        txt_sidecar = audio_path.with_suffix(".txt")
        if not txt_sidecar.exists():
            return AsrResult(
                segments=[],
                warnings=[
                    "No ASR sidecar found. Expected a text or JSON file with the same basename for mock backend.",
                    f"Configured language hints: {', '.join(self._language_hints)}",
                ],
                engine_name="mock",
            )

        text = txt_sidecar.read_text(encoding="utf-8").strip()
        if not text:
            return AsrResult(segments=[], warnings=["ASR sidecar is empty."], engine_name="mock")

        words = text.split()
        word_duration = max(duration_seconds / max(len(words), 1), 0.01)
        word_items: list[Word] = []
        cursor = 0.0
        for token in words:
            start = cursor
            end = min(duration_seconds, cursor + word_duration)
            word_items.append(Word(text=token, start=round(start, 3), end=round(end, 3), confidence=0.85))
            cursor = end

        segment = Segment(
            id="seg-000001",
            speaker=None,
            channel=None,
            start=0.0,
            end=round(duration_seconds, 3),
            text_original=text,
            languages=self._language_hints[:1] if self._language_hints else [],
            overlap=False,
            confidence=0.85,
            words=word_items,
        )
        return AsrResult(segments=[segment], warnings=[], engine_name="mock")


class FasterWhisperBackend:
    name = "faster-whisper"

    def __init__(self, config: AsrConfig, engine_id: str | None = None) -> None:
        self._config = config
        self._engine_id = engine_id or "faster-whisper"
        self._vocabulary_list = load_vocabulary(config.vocabulary_files)
        self._model = None
        self._batched_model = None

    def _get_model(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed. Install with `pip install faster-whisper`"
                ) from exc

            device = self._config.device if self._config.device else "auto"
            compute_type = self._config.compute_type if self._config.compute_type else "auto"
            try:
                self._model = WhisperModel(
                    model_size_or_path=self._config.model,
                    device=device,
                    compute_type=compute_type,
                )
            except Exception:
                self._model = WhisperModel(
                    model_size_or_path=self._config.model,
                    device="cpu",
                    compute_type="int8",
                )
            # Build batched pipeline wrapper if requested
            if self._config.use_batched_pipeline:
                try:
                    from faster_whisper import BatchedInferencePipeline
                    self._batched_model = BatchedInferencePipeline(model=self._model)
                except ImportError:
                    pass
        return self._model

    def _build_hotwords(self) -> str | None:
        """Combine vocabulary terms and explicit hotwords into a single hotwords string."""
        parts: list[str] = []
        if self._vocabulary_list:
            parts.append(", ".join(self._vocabulary_list))
        if self._config.hotwords:
            parts.append(self._config.hotwords)
        return ", ".join(parts) if parts else None

    def transcribe(self, audio_path: Path, duration_seconds: float) -> AsrResult:
        model = self._get_model()
        primary_lang = self._config.language_hints[0] if self._config.language_hints else None
        vocab_hotwords = self._build_hotwords()
        warnings: list[str] = []
        temp_schedule = self._config.temperature_fallback if self._config.temperature_fallback else self._config.temperature
        transcribe_kwargs = {
            "language": primary_lang,
            "task": self._config.task,
            "beam_size": self._config.beam_size,
            "best_of": self._config.best_of,
            "temperature": temp_schedule,
            "condition_on_previous_text": self._config.condition_on_previous_text,
            "vad_filter": self._config.vad_filter,
            "word_timestamps": self._config.word_timestamps,
        }

        # Add optional transcription parameters
        if self._config.patience != 1.0:
            transcribe_kwargs["patience"] = self._config.patience
        if self._config.length_penalty != 1.0:
            transcribe_kwargs["length_penalty"] = self._config.length_penalty
        if self._config.repetition_penalty != 1.0:
            transcribe_kwargs["repetition_penalty"] = self._config.repetition_penalty
        if self._config.no_repeat_ngram_size > 0:
            transcribe_kwargs["no_repeat_ngram_size"] = self._config.no_repeat_ngram_size
        if self._config.compression_ratio_threshold is not None:
            transcribe_kwargs["compression_ratio_threshold"] = self._config.compression_ratio_threshold
        if self._config.log_prob_threshold is not None:
            transcribe_kwargs["log_prob_threshold"] = self._config.log_prob_threshold
        if self._config.no_speech_threshold is not None:
            transcribe_kwargs["no_speech_threshold"] = self._config.no_speech_threshold
        if self._config.hallucination_silence_threshold is not None:
            transcribe_kwargs["hallucination_silence_threshold"] = self._config.hallucination_silence_threshold
        if self._config.vad_parameters:
            transcribe_kwargs["vad_parameters"] = self._config.vad_parameters
        if vocab_hotwords:
            transcribe_kwargs["hotwords"] = vocab_hotwords
        if self._config.initial_prompt:
            transcribe_kwargs["initial_prompt"] = self._config.initial_prompt
        if self._config.suppress_blank is not None:
            transcribe_kwargs["suppress_blank"] = self._config.suppress_blank

        try:
            # Use batched pipeline if configured and available
            if self._batched_model is not None:
                batched_kwargs = {**transcribe_kwargs}
                batched_kwargs["batch_size"] = self._config.batch_size
                segments_iter, info = self._batched_model.transcribe(
                    str(audio_path), **batched_kwargs
                )
            else:
                segments_iter, info = model.transcribe(str(audio_path), **transcribe_kwargs)

            segments_list = list(segments_iter)
        except Exception as exc:
            if "cublas" in str(exc).lower() or "cuda" in str(exc).lower():
                warnings.append("CUDA libraries not found; fell back to CPU int8 inference.")
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    model_size_or_path=self._config.model,
                    device="cpu",
                    compute_type="int8",
                )
                model = self._model
                self._batched_model = None
                segments_iter, info = model.transcribe(str(audio_path), **transcribe_kwargs)
                segments_list = list(segments_iter)
            else:
                raise

        segments: list[Segment] = []
        for idx, fw_seg in enumerate(segments_list, start=1):
            words: list[Word] = []
            if fw_seg.words:
                for w in fw_seg.words:
                    words.append(
                        Word(
                            text=w.word.strip(),
                            start=round(w.start, 3),
                            end=round(w.end, 3),
                            confidence=round(w.probability, 3) if hasattr(w, "probability") else None,
                        )
                    )

            # Decode avg_logprob to confidence
            prob = None
            if hasattr(fw_seg, "avg_logprob") and fw_seg.avg_logprob is not None:
                import math
                prob = round(min(max(math.exp(fw_seg.avg_logprob), 0.0), 1.0), 3)

            detected_lang = [info.language] if info and info.language else (self._config.language_hints[:1] if self._config.language_hints else [])
            segments.append(
                Segment(
                    id=f"seg-{idx:06d}",
                    speaker=None,
                    channel=None,
                    start=round(fw_seg.start, 3),
                    end=round(fw_seg.end, 3),
                    text_original=fw_seg.text.strip(),
                    languages=detected_lang,
                    overlap=False,
                    confidence=prob,
                    words=words,
                )
            )

        if not segments:
            warnings.append("ASR returned 0 segments.")
        return AsrResult(segments=segments, warnings=warnings, engine_name=self._engine_id)


def build_asr_backend(config: AsrConfig, engine_id: str | None = None) -> AsrBackend:
    if config.backend == "mock":
        return MockAsrBackend(
            language_hints=config.language_hints,
            vocabulary_files=config.vocabulary_files,
        )
    if config.backend in ("faster-whisper", "faster_whisper"):
        return FasterWhisperBackend(config=config, engine_id=engine_id)
    if config.backend == "auto":
        try:
            import faster_whisper  # noqa: F401
            return FasterWhisperBackend(config=config, engine_id=engine_id)
        except Exception:
            return MockAsrBackend(
                language_hints=config.language_hints,
                vocabulary_files=config.vocabulary_files,
            )
    raise ValueError(
        f"Unsupported ASR backend '{config.backend}'. "
        "Supported backends: 'auto', 'faster-whisper', 'mock'."
    )