from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .alignment import align_segments
from .asr import AsrResult, build_asr_backend
from .audio import inspect_wav, preprocess_audio, split_channels
from .config import AppConfig
from .diarization import diarize_segments
from .exporters import write_json, write_srt, write_txt, write_vtt
from .language import annotate_languages
from .merge import merge_channel_segments, normalize_segment_ids
from .overlap import detect_overlap
from .phonetic import PhoneticRescorer
from .quality import apply_domain_corrections, run_quality_gates
from .rover import reconcile_ensemble
from .schemas import ProcessingInfo, TranscriptDocument
from .turn_detector import split_segments_into_turns


@dataclass(slots=True)
class FileOutcome:
    source_path: str
    status: str
    transcript_json: str | None
    transcript_txt: str | None
    checkpoint_path: str
    warnings: list[str]
    review_reasons: list[str]
    transcript_srt: str | None = None
    transcript_vtt: str | None = None


class TranscriptionPipeline:
    def __init__(self, config: AppConfig, output_dir: Path, resume: bool) -> None:
        self._config = config
        self._output_dir = output_dir
        self._resume = resume

        # Build primary ASR backend
        self._backend = build_asr_backend(config.asr)

        # Build secondary ASR backend for ensemble (if enabled and different model)
        self._secondary_backend = None
        if config.ensemble.enabled and config.ensemble.secondary_model != config.asr.model:
            from dataclasses import replace
            secondary_config = replace(
                config.asr,
                model=config.ensemble.secondary_model,
                compute_type=config.ensemble.secondary_compute_type,
            )
            self._secondary_backend = build_asr_backend(
                secondary_config, engine_id=config.ensemble.secondary_model
            )

        self._checkpoints_dir = output_dir / "checkpoints"
        self._transcripts_dir = output_dir / "transcripts"
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._transcripts_dir.mkdir(parents=True, exist_ok=True)

        from .asr import load_vocabulary
        vocab = load_vocabulary(config.asr.vocabulary_files)
        self._rescorer = PhoneticRescorer(vocabulary_terms=vocab)

    def process_file(self, audio_path: Path, job_id: str) -> FileOutcome:
        inspection = inspect_wav(audio_path)
        checkpoint_path = self._checkpoints_dir / f"{inspection.source.sha256}.json"
        if self._resume and checkpoint_path.exists():
            with checkpoint_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            cached_hash = payload.get("config_hash")
            cached_status = payload.get("status")
            if (cached_hash is None or cached_hash == self._config.config_hash) and cached_status in ("completed", "partial", "needs_review"):
                return FileOutcome(
                    source_path=str(audio_path),
                    status=payload["status"],
                    transcript_json=payload.get("transcript_json"),
                    transcript_txt=payload.get("transcript_txt"),
                    transcript_srt=payload.get("transcript_srt"),
                    transcript_vtt=payload.get("transcript_vtt"),
                    checkpoint_path=str(checkpoint_path),
                    warnings=payload.get("warnings", []),
                    review_reasons=payload.get("review_reasons", []),
                )

        warnings = list(inspection.warnings)
        preprocessed_path = audio_path

        # ---- Step 1: Audio Preprocessing ----
        if self._config.preprocessing.level > 0:
            try:
                preproc_result = preprocess_audio(
                    audio_path=audio_path,
                    config=self._config.preprocessing,
                    orig_sample_rate=inspection.source.sample_rate_hz,
                    orig_channels=inspection.source.channels,
                    orig_sample_width=inspection.source.sample_width_bytes,
                )
                warnings.extend(preproc_result.warnings)

                # Write preprocessed audio to temp file matching pipeline expectations
                preprocessed_path = self._checkpoints_dir / f"{audio_path.stem}_preprocessed.wav"
                # Use soundfile or scipy to write float32 WAV
                try:
                    import soundfile as sf
                    sf.write(str(preprocessed_path), preproc_result.samples, preproc_result.sample_rate)
                except ImportError:
                    from scipy.io import wavfile
                    # Convert float32 back to int16 for scipy
                    int_samples = (preproc_result.samples * 32767).astype("int16")
                    wavfile.write(str(preprocessed_path), preproc_result.sample_rate, int_samples)
            except Exception as exc:
                warnings.append(f"Audio preprocessing failed: {exc}. Using original audio.")
                preprocessed_path = audio_path

        # ---- Step 2: ASR (multi-engine ensemble or single) ----
        if self._secondary_backend and self._config.ensemble.enabled:
            # Run both engines
            primary_result: AsrResult = self._backend.transcribe(
                preprocessed_path, inspection.source.duration_seconds
            )
            secondary_result: AsrResult = self._secondary_backend.transcribe(
                preprocessed_path, inspection.source.duration_seconds
            )
            warnings.extend(primary_result.warnings)
            warnings.extend(secondary_result.warnings)

            # ROVER reconciliation
            rover_result = reconcile_ensemble(
                primary=primary_result.segments,
                secondary=secondary_result.segments,
                primary_engine=primary_result.engine_name or "large-v3",
                secondary_engine=secondary_result.engine_name or "turbo",
            )
            warnings.extend(rover_result.warnings)
            segments = rover_result.segments
            warnings.append(
                f"Ensemble: {primary_result.engine_name} ({len(primary_result.segments)} segs) + "
                f"{secondary_result.engine_name} ({len(secondary_result.segments)} segs) → "
                f"{len(segments)} reconciled segments via ROVER."
            )
        else:
            # Single engine
            if self._config.audio.split_channels_for_stereo and inspection.source.channels > 1:
                channel_wavs = split_channels(audio_path)
                channel_segments_list = []
                for ch_idx, ch_wav in enumerate(channel_wavs):
                    ch_asr = self._backend.transcribe(ch_wav, inspection.source.duration_seconds)
                    for seg in ch_asr.segments:
                        seg.channel = ch_idx
                    channel_segments_list.append(ch_asr.segments)
                    warnings.extend(ch_asr.warnings)
                segments = merge_channel_segments(channel_segments_list)
            else:
                asr_result = self._backend.transcribe(preprocessed_path, inspection.source.duration_seconds)
                warnings.extend(asr_result.warnings)
                segments = asr_result.segments

        # ---- Step 3: Forced Alignment ----
        segments, stage_warnings = align_segments(
            segments,
            enabled=self._config.alignment.enabled,
            word_level=self._config.alignment.word_level,
            model_name=self._config.alignment.model,
            audio_path=preprocessed_path if self._config.alignment.enabled and preprocessed_path != audio_path else audio_path,
        )
        warnings.extend(stage_warnings)

        segments = split_segments_into_turns(segments)

        # ---- Step 4: Speaker Diarization ----
        segments, stage_warnings = diarize_segments(
            segments,
            enabled=self._config.diarization.enabled,
            channel_map=self._config.audio.channel_map,
            model=self._config.diarization.model,
            audio_path=audio_path if self._config.diarization.enabled else None,
            min_speakers=self._config.diarization.min_speakers,
            max_speakers=self._config.diarization.max_speakers,
        )
        warnings.extend(stage_warnings)

        # ---- Step 5: Overlap Detection ----
        segments, stage_warnings = detect_overlap(
            segments,
            enabled=self._config.diarization.detect_overlap,
        )
        warnings.extend(stage_warnings)

        # ---- Step 6: Language Annotation ----
        segments, stage_warnings = annotate_languages(
            segments=segments,
            language_hints=self._config.asr.language_hints,
            detect_code_switch=self._config.language.detect_code_switch,
        )
        warnings.extend(stage_warnings)

        segments = normalize_segment_ids(segments)
        for seg in segments:
            seg.text_original = apply_domain_corrections(seg.text_original)
            seg.text_original = self._rescorer.rescore_text(seg.text_original)
            if seg.words:
                for w in seg.words:
                    w.text = apply_domain_corrections(w.text)
                    w.text = self._rescorer.rescore_token(w.text)

        # ---- Step 8: Quality Gates ----
        quality = run_quality_gates(segments, self._config.quality, warnings)

        document = TranscriptDocument(
            schema_version="1.0",
            job_id=job_id,
            source=inspection.source,
            processing=ProcessingInfo(
                asr_model=self._backend.name,
                alignment_model=self._config.alignment.model if self._config.alignment.enabled else None,
                diarization_model=self._config.diarization.model if self._config.diarization.enabled else None,
                language_hints=self._config.asr.language_hints,
                config_hash=self._config.config_hash,
                completed_at=datetime.now(timezone.utc).isoformat(),
            ),
            warnings=quality.warnings,
            segments=segments,
            status=quality.status,
            review_reasons=quality.review_reasons,
        )

        stem = audio_path.stem
        json_path = self._transcripts_dir / f"{stem}.json"
        txt_path = self._transcripts_dir / f"{stem}.txt"
        srt_path = self._transcripts_dir / f"{stem}.srt"
        vtt_path = self._transcripts_dir / f"{stem}.vtt"

        if self._config.output.write_json:
            write_json(document, json_path)
        if self._config.output.write_txt:
            write_txt(document, txt_path)
        if self._config.output.write_srt:
            write_srt(document, srt_path)
        if self._config.output.write_vtt:
            write_vtt(document, vtt_path)

        outcome = FileOutcome(
            source_path=str(audio_path),
            status=document.status,
            transcript_json=str(json_path) if self._config.output.write_json else None,
            transcript_txt=str(txt_path) if self._config.output.write_txt else None,
            transcript_srt=str(srt_path) if self._config.output.write_srt else None,
            transcript_vtt=str(vtt_path) if self._config.output.write_vtt else None,
            checkpoint_path=str(checkpoint_path),
            warnings=document.warnings,
            review_reasons=document.review_reasons,
        )
        self._write_checkpoint(outcome)
        return outcome

    def _write_checkpoint(self, outcome: FileOutcome) -> None:
        payload = {
            "source_path": outcome.source_path,
            "status": outcome.status,
            "config_hash": self._config.config_hash,
            "transcript_json": outcome.transcript_json,
            "transcript_txt": outcome.transcript_txt,
            "transcript_srt": outcome.transcript_srt,
            "transcript_vtt": outcome.transcript_vtt,
            "warnings": outcome.warnings,
            "review_reasons": outcome.review_reasons,
            "checkpoint_written_at": datetime.now(timezone.utc).isoformat(),
        }
        path = Path(outcome.checkpoint_path)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")