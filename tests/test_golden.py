import json
from pathlib import Path

from wav_transcriber.config import load_config
from wav_transcriber.pipeline import TranscriptionPipeline


def test_golden_dataset(tmp_path: Path) -> None:
    golden_dir = Path("golden/audio")
    golden_expected = Path("golden/expected")
    if not golden_dir.exists():
        return

    out_dir = tmp_path / "golden_output"
    cfg = load_config(None)
    cfg.asr.backend = "mock"
    cfg.diarization.detect_overlap = True
    cfg.output.write_json = True
    cfg.output.write_srt = True

    pipeline = TranscriptionPipeline(config=cfg, output_dir=out_dir, resume=False)
    for wav_file in golden_dir.glob("*.wav"):
        outcome = pipeline.process_file(wav_file, job_id="test-golden")
        assert outcome.status in ("completed", "partial")
        assert outcome.transcript_json is not None

        generated_data = json.loads(Path(outcome.transcript_json).read_text(encoding="utf-8"))
        assert len(generated_data["segments"]) > 0

        expected_file = golden_expected / f"{wav_file.stem}.json"
        if expected_file.exists():
            expected_data = json.loads(expected_file.read_text(encoding="utf-8"))
            assert len(generated_data["segments"]) == len(expected_data["segments"])
