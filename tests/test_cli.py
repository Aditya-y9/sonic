import sys
import wave
from pathlib import Path
from unittest.mock import patch

from wav_transcriber.cli import main


def _create_test_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wavf:
        wavf.setnchannels(1)
        wavf.setsampwidth(2)
        wavf.setframerate(16000)
        wavf.writeframes(b"\x00\x00" * 3200)


def test_cli_end_to_end(tmp_path: Path) -> None:
    wav_path = tmp_path / "audio1.wav"
    txt_sidecar = tmp_path / "audio1.txt"
    _create_test_wav(wav_path)
    txt_sidecar.write_text("Testing the CLI pipeline", encoding="utf-8")

    out_dir = tmp_path / "cli_out"

    test_args = [
        "wav_transcriber",
        str(wav_path),
        "--output",
        str(out_dir),
        "--language",
        "en",
        "hi",
        "--diarize",
        "--align",
        "--overlap",
        "--include-words",
        "--write-srt",
        "--resume",
    ]

    with patch.object(sys, "argv", test_args):
        ret = main()
        assert ret == 0

    assert (out_dir / "job_manifest.json").exists()
    assert (out_dir / "transcripts" / "audio1.json").exists()
    assert (out_dir / "transcripts" / "audio1.txt").exists()
    assert (out_dir / "transcripts" / "audio1.srt").exists()
