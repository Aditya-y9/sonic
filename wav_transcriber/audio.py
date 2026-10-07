from __future__ import annotations

import hashlib
import math
import wave
from dataclasses import dataclass
from pathlib import Path

from .schemas import SourceInfo


@dataclass(slots=True)
class AudioInspection:
    source: SourceInfo
    warnings: list[str]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inspect_wav(path: Path) -> AudioInspection:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() != ".wav":
        raise ValueError(f"Only .wav is supported in this implementation: {path}")

    warnings: list[str] = []
    try:
        with wave.open(str(path), "rb") as wavf:
            channels = wavf.getnchannels()
            sample_rate = wavf.getframerate()
            sample_width = wavf.getsampwidth()
            frame_count = wavf.getnframes()
            raw = wavf.readframes(frame_count)
    except wave.Error as exc:
        raise ValueError(f"Corrupt or unsupported WAV format ({path}): {exc}") from exc

    if frame_count <= 0:
        raise ValueError(f"Invalid WAV: zero frames ({path})")
    if sample_rate <= 0:
        raise ValueError(f"Invalid WAV: non-positive sample rate ({path})")
    if channels <= 0:
        raise ValueError(f"Invalid WAV: non-positive channel count ({path})")
    if sample_width not in (1, 2, 3, 4):
        warnings.append(f"Unexpected sample width {sample_width} bytes; continuing cautiously.")

    bytes_per_frame = channels * sample_width
    expected_bytes = frame_count * bytes_per_frame
    if len(raw) != expected_bytes:
        warnings.append(
            f"Frame-byte mismatch (expected {expected_bytes}, got {len(raw)}); file may be truncated."
        )

    duration = frame_count / sample_rate
    if duration < 0.2:
        warnings.append("Very short audio duration (<0.2s).")

    if _is_mostly_silence(raw, sample_width):
        warnings.append("Audio appears mostly silence.")
    if _is_heavily_clipped(raw, sample_width):
        warnings.append("Possible clipping detected.")

    source = SourceInfo(
        path=str(path),
        sha256=_sha256_file(path),
        duration_seconds=round(duration, 6),
        sample_rate_hz=sample_rate,
        channels=channels,
        sample_width_bytes=sample_width,
        frame_count=frame_count,
    )
    return AudioInspection(source=source, warnings=warnings)


def split_channels(path: Path, output_dir: Path | None = None) -> list[Path]:
    """Extract lossless mono WAV derivatives for each channel without mutating source."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    out_dir = output_dir or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with wave.open(str(path), "rb") as src_wav:
        channels = src_wav.getnchannels()
        sample_rate = src_wav.getframerate()
        sample_width = src_wav.getsampwidth()
        frame_count = src_wav.getnframes()
        raw_frames = src_wav.readframes(frame_count)

    if channels == 1:
        return [path]

    channel_paths: list[Path] = []
    side_names = ["left", "right"] if channels == 2 else [f"channel_{i}" for i in range(channels)]

    for ch_idx, side_name in enumerate(side_names):
        ch_path = out_dir / f"{path.stem}_{side_name}.wav"
        with wave.open(str(ch_path), "wb") as dst_wav:
            dst_wav.setnchannels(1)
            dst_wav.setsampwidth(sample_width)
            dst_wav.setframerate(sample_rate)

            # De-interleave channel frames
            ch_bytes = bytearray()
            bytes_per_frame = channels * sample_width
            for frame_start in range(0, len(raw_frames), bytes_per_frame):
                sample_start = frame_start + (ch_idx * sample_width)
                ch_bytes.extend(raw_frames[sample_start : sample_start + sample_width])

            dst_wav.writeframes(bytes(ch_bytes))
        channel_paths.append(ch_path)

    return channel_paths


def _is_mostly_silence(raw: bytes, sample_width: int) -> bool:
    if sample_width != 2 or not raw:
        return False
    silence_count = 0
    total = len(raw) // 2
    for i in range(0, len(raw), 2):
        sample = int.from_bytes(raw[i : i + 2], byteorder="little", signed=True)
        if abs(sample) < 64:
            silence_count += 1
    return (silence_count / max(total, 1)) > 0.95


def _is_heavily_clipped(raw: bytes, sample_width: int) -> bool:
    if sample_width != 2 or not raw:
        return False
    clipped = 0
    total = len(raw) // 2
    limit = math.floor(32767 * 0.98)
    for i in range(0, len(raw), 2):
        sample = int.from_bytes(raw[i : i + 2], byteorder="little", signed=True)
        if abs(sample) >= limit:
            clipped += 1
    return (clipped / max(total, 1)) > 0.02

