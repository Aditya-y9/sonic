import json
import wave
from pathlib import Path

from wav_transcriber.config import load_config
from wav_transcriber.pipeline import TranscriptionPipeline


def _create_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wavf:
        wavf.setnchannels(1)
        wavf.setsampwidth(2)
        wavf.setframerate(16000)
        wavf.writeframes(b"\x00\x00" * 3200)


def test_pipeline_writes_checkpoint_and_resumes(tmp_path: Path) -> None:
    wav_path = tmp_path / "a.wav"
    txt_sidecar = tmp_path / "a.txt"
    txt_sidecar.write_text("hello world", encoding="utf-8")
    _create_wav(wav_path)

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "job:",
                "  output_dir: output",
                "  resume: true",
                "asr:",
                "  backend: mock",
            ]
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_path)

    output_dir = tmp_path / "output"
    pipeline = TranscriptionPipeline(config=cfg, output_dir=output_dir, resume=True)
    first = pipeline.process_file(wav_path, job_id="job-1")
    second = pipeline.process_file(wav_path, job_id="job-1")

    assert first.checkpoint_path == second.checkpoint_path
    assert Path(first.checkpoint_path).exists()

    payload = json.loads(Path(first.checkpoint_path).read_text(encoding="utf-8"))
    assert payload["status"] in {"completed", "partial", "needs_review"}
