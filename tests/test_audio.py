import wave
from pathlib import Path

import pytest
from wav_transcriber.audio import inspect_wav, split_channels


def test_inspect_wav_reads_metadata(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "wb") as wavf:
        wavf.setnchannels(1)
        wavf.setsampwidth(2)
        wavf.setframerate(16000)
        wavf.writeframes(b"\x00\x00" * 1600)

    inspected = inspect_wav(wav_path)
    assert inspected.source.channels == 1
    assert inspected.source.sample_rate_hz == 16000
    assert inspected.source.duration_seconds > 0
    assert inspected.source.sha256 != ""


def test_split_channels_stereo(tmp_path: Path) -> None:
    stereo_path = tmp_path / "stereo.wav"
    with wave.open(str(stereo_path), "wb") as wavf:
        wavf.setnchannels(2)
        wavf.setsampwidth(2)
        wavf.setframerate(16000)
        # 1600 frames of left=0x1000, right=0x2000
        frames = b"\x00\x10\x00\x20" * 1600
        wavf.writeframes(frames)

    channel_files = split_channels(stereo_path, output_dir=tmp_path / "split")
    assert len(channel_files) == 2
    assert channel_files[0].name == "stereo_left.wav"
    assert channel_files[1].name == "stereo_right.wav"

    with wave.open(str(channel_files[0]), "rb") as left_wav:
        assert left_wav.getnchannels() == 1
        assert left_wav.getframerate() == 16000
        assert left_wav.getnframes() == 1600


def test_inspect_wav_rejects_non_wav(tmp_path: Path) -> None:
    bad_file = tmp_path / "test.mp3"
    bad_file.write_text("not a wav")
    with pytest.raises(ValueError, match="Only .wav is supported"):
        inspect_wav(bad_file)


def test_inspect_wav_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inspect_wav(tmp_path / "non_existent.wav")


def test_wpe_dereverberation() -> None:
    import numpy as np
    from wav_transcriber.audio import _apply_wpe

    sine = np.sin(np.linspace(0, 100 * np.pi, 16000)).astype(np.float32)
    cleaned = _apply_wpe(sine, taps=4, delay=2)
    assert len(cleaned) == len(sine)
    assert cleaned.dtype == np.float32

