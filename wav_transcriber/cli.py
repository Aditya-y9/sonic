from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .pipeline import FileOutcome, TranscriptionPipeline


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local WAV transcription pipeline")
    parser.add_argument("input_path", help="Path to a .wav file or directory")
    parser.add_argument("--config", default=None, help="Path to YAML config")
    parser.add_argument("--output", default=None, help="Output directory override")
    parser.add_argument("--backend", default=None, help="ASR backend (auto, faster-whisper, mock)")
    parser.add_argument("--model", default=None, help="Model name or size (e.g. tiny, base, small, large-v3)")
    parser.add_argument("--device", default=None, help="Compute device (cpu, cuda, auto)")
    parser.add_argument("--language", nargs="+", default=None, help="Language hints (e.g. --language en hi)")
    parser.add_argument("--diarize", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable diarization")
    parser.add_argument("--align", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable forced alignment")
    parser.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable overlap detection")
    parser.add_argument("--include-words", action=argparse.BooleanOptionalAction, default=None, help="Include word-level timestamps")
    parser.add_argument("--write-srt", action="store_true", default=None, help="Export .srt subtitle file")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")
    parser.add_argument("--recursive", action="store_true", help="Recurse into input directory")
    parser.add_argument("--resume", action="store_true", help="Reuse existing checkpoints")
    parser.add_argument("--review", action="store_true", help="Print files needing review")
    parser.add_argument("--reprocess", action="store_true", help="Ignore checkpoints")
    return parser.parse_args()


def _collect_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*.wav" if recursive else "*.wav"
    files = sorted(input_path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No .wav files found under: {input_path}")
    return files


def _write_manifest(output_dir: Path, outcomes: list[FileOutcome], started_at: str) -> Path:
    status_counts: dict[str, int] = {}
    for outcome in outcomes:
        status_counts[outcome.status] = status_counts.get(outcome.status, 0) + 1

    payload = {
        "schema_version": "1.0",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "counts": status_counts,
        "files": [
            {
                "source_path": item.source_path,
                "status": item.status,
                "transcript_json": item.transcript_json,
                "transcript_txt": item.transcript_txt,
                "transcript_srt": item.transcript_srt,
                "transcript_vtt": item.transcript_vtt,
                "checkpoint_path": item.checkpoint_path,
                "warnings": item.warnings,
                "review_reasons": item.review_reasons,
            }
            for item in outcomes
        ],
    }
    manifest_path = output_dir / "job_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return manifest_path


def main() -> int:
    args = _parse_args()
    config = load_config(Path(args.config) if args.config else None)

    # Apply CLI overrides
    if args.backend:
        config.asr.backend = args.backend
    if args.model:
        config.asr.model = args.model
    if args.device:
        config.asr.device = args.device
    if args.language:
        config.asr.language_hints = args.language
    if args.diarize is not None:
        config.diarization.enabled = args.diarize
    if args.align is not None:
        config.alignment.enabled = args.align
    if args.overlap is not None:
        config.diarization.detect_overlap = args.overlap
    if args.include_words is not None:
        config.output.include_words = args.include_words
    if args.write_srt:
        config.output.write_srt = True
    if args.workers is not None:
        config.job.workers = args.workers

    input_path = Path(args.input_path).resolve()
    recursive = bool(args.recursive or config.job.recursive)
    files = _collect_files(input_path, recursive)

    output_dir = Path(args.output or config.job.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resume = bool(args.resume or config.job.resume)
    if args.reprocess:
        resume = False

    pipeline = TranscriptionPipeline(config=config, output_dir=output_dir, resume=resume)
    started_at = datetime.now(timezone.utc).isoformat()
    job_id = f"local-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    outcomes = [pipeline.process_file(path, job_id=job_id) for path in files]

    manifest_path = _write_manifest(output_dir=output_dir, outcomes=outcomes, started_at=started_at)
    if args.review:
        needs_review = [item for item in outcomes if item.status == "needs_review"]
        if needs_review:
            print("Files requiring review:")
            for item in needs_review:
                print(f"- {item.source_path}")
                for reason in item.review_reasons:
                    print(f"  - {reason}")
        else:
            print("No files currently marked as needs_review.")

    print(f"Processed {len(outcomes)} file(s).")
    print(f"Manifest: {manifest_path}")
    return 0

