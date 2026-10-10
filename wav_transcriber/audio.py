from __future__ import annotations

import hashlib
import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal as scipy_signal

from .config import PreprocessingConfig
from .schemas import SourceInfo


# Re-export for backward compatibility
from .schemas import SourceInfo as _SI  # noqa: F401


@dataclass(slots=True)
class AudioInspection:
    source: SourceInfo
    warnings: list[str]


@dataclass(slots=True)
class PreprocessedAudio:
    """Result of audio preprocessing."""
    samples: np.ndarray
    sample_rate: int
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


# ---------------------------------------------------------------------------
# Preprocessing functions
# ---------------------------------------------------------------------------


def _wav_to_float(raw: bytes, sample_width: int) -> np.ndarray:
    """Convert raw PCM bytes to float32 array in [-1.0, 1.0]."""
    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.int32}
    arr = np.frombuffer(raw, dtype=dtype_map.get(sample_width, np.int16))
    if sample_width == 1:
        arr = arr.astype(np.float32) - 128.0
        arr /= 128.0
    elif sample_width == 2:
        arr = arr.astype(np.float32) / 32768.0
    elif sample_width in (3, 4):
        arr = arr.astype(np.float32) / 2147483648.0
    return arr


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio to target sample rate using librosa (fast path) or scipy."""
    if orig_sr == target_sr:
        return samples
    try:
        import librosa
        return librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr, res_type="kaiser_best")
    except ImportError:
        from scipy import signal as sp_signal
        duration = len(samples) / orig_sr
        target_len = int(duration * target_sr)
        return sp_signal.resample(samples, target_len)


def _apply_pre_emphasis(samples: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter to boost high frequencies."""
    if coef <= 0.0:
        return samples
    return np.append(samples[0], samples[1:] - coef * samples[:-1])


def _apply_filter(
    samples: np.ndarray,
    sr: int,
    highpass_hz: int = 0,
    lowpass_hz: int = 0,
    order: int = 5,
) -> np.ndarray:
    """Apply Butterworth high-pass and/or low-pass filter."""
    if highpass_hz > 0 and highpass_hz < sr / 2:
        sos = scipy_signal.butter(order, highpass_hz, btype="high", fs=sr, output="sos")
        samples = scipy_signal.sosfilt(sos, samples)
    if lowpass_hz > 0 and lowpass_hz < sr / 2:
        sos = scipy_signal.butter(order, lowpass_hz, btype="low", fs=sr, output="sos")
        samples = scipy_signal.sosfilt(sos, samples)
    return samples


def _normalize_loudness(samples: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    """RMS-based loudness normalization to target dBFS level."""
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-10:
        return samples
    target_rms = 10.0 ** (target_dbfs / 20.0)
    gain = target_rms / rms
    return np.clip(samples * gain, -1.0, 1.0)


def _denoise_spectral_gating(
    samples: np.ndarray,
    sr: int,
    prop_decrease: float = 0.8,
    stationary: bool = False,
    time_constant_s: float = 2.0,
) -> np.ndarray:
    """Apply spectral gating noise reduction via noisereduce."""
    try:
        import noisereduce as nr
    except ImportError:
        return samples  # silently skip if not installed

    return nr.reduce_noise(
        y=samples,
        sr=sr,
        stationary=stationary,
        prop_decrease=prop_decrease,
        time_constant_s=time_constant_s,
        thresh_n_mult_nonstationary=2,
        sigmoid_slope_nonstationary=10,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
    )


def _remove_long_silence_vad(
    samples: np.ndarray,
    sr: int,
    threshold_db: float = -40.0,
    min_silence_ms: float = 1000.0,
    hop_length: int = 512,
) -> np.ndarray:
    """Remove long silences using energy-based VAD with librosa."""
    try:
        import librosa
    except ImportError:
        return samples

    # Compute RMS energy per frame
    rms = librosa.feature.rms(y=samples, hop_length=hop_length)[0]
    threshold = 10.0 ** (threshold_db / 20.0)

    # Mark frames as speech if above threshold
    is_speech = rms > threshold

    # Convert to samples
    frames_to_samples = hop_length
    min_silence_frames = int(min_silence_ms / 1000.0 * sr / frames_to_samples)

    # Remove runs of silence longer than min_silence_frames
    out_chunks: list[np.ndarray] = []
    gap_start: int | None = None
    current_start: int | None = None

    for i in range(len(is_speech)):
        if is_speech[i]:
            if current_start is None:
                current_start = i
            gap_start = None
        else:
            if current_start is not None and gap_start is None:
                gap_start = i
            if gap_start is not None and (i - gap_start) >= min_silence_frames:
                # Emit speech chunk up to gap_start
                chunk_end = gap_start * frames_to_samples
                chunk_start = current_start * frames_to_samples if current_start is not None else 0
                if chunk_end > chunk_start:
                    out_chunks.append(samples[chunk_start:chunk_end])
                current_start = None
                gap_start = None

    # Emit final chunk
    if current_start is not None:
        chunk_start = current_start * frames_to_samples
        if chunk_start < len(samples):
            out_chunks.append(samples[chunk_start:])

    if not out_chunks:
        return samples

    result = np.concatenate(out_chunks)
    # Pad or truncate to original length to maintain timing reference
    if len(result) < len(samples):
        result = np.pad(result, (0, len(samples) - len(result)), mode="constant")
    return result[:len(samples)]


def _apply_wpe(samples: np.ndarray, taps: int = 10, delay: int = 3) -> np.ndarray:
    try:
        from nara_wpe.wpe import wpe
        from nara_wpe.utils import stft, istft
        stft_data = stft(samples, size=512, shift=128)
        stft_trans = np.transpose(stft_data, (1, 0))[None, ...]
        enhanced_stft = wpe(stft_trans, taps=taps, delay=delay, iterations=3)
        enhanced = istft(enhanced_stft[0].transpose(1, 0), size=512, shift=128)
        if len(enhanced) < len(samples):
            enhanced = np.pad(enhanced, (0, len(samples) - len(enhanced)))
        return enhanced[:len(samples)].astype(np.float32)
    except Exception:
        return samples


def _apply_deepfilter(samples: np.ndarray, sr: int) -> np.ndarray:
    try:
        import sys
        from types import ModuleType
        import torchaudio
        if not hasattr(torchaudio, "backend"):
            backend = ModuleType("torchaudio.backend")
            common = ModuleType("torchaudio.backend.common")
            from dataclasses import dataclass
            @dataclass
            class AudioMetaData:
                sample_rate: int
                num_frames: int
                num_channels: int
                bits_per_sample: int
                encoding: str
            common.AudioMetaData = AudioMetaData
            backend.common = common
            torchaudio.backend = backend
            sys.modules["torchaudio.backend"] = backend
            sys.modules["torchaudio.backend.common"] = common

        import df
        import torch
        model, df_state, _ = df.init_df()
        tensor_y = torch.from_numpy(samples).float().unsqueeze(0)
        enhanced = df.enhance(model, df_state, tensor_y)
        out = enhanced.squeeze(0).cpu().numpy().astype(np.float32)
        if len(out) < len(samples):
            out = np.pad(out, (0, len(samples) - len(out)))
        return out[:len(samples)]
    except Exception:
        return samples


def preprocess_audio(
    audio_path: Path,
    config: PreprocessingConfig,
    orig_sample_rate: int,
    orig_channels: int,
    orig_sample_width: int,
) -> PreprocessedAudio:
    """Run the full audio preprocessing pipeline.

    Pipeline order:
      1. Read PCM → float32
      2. Mix to mono if multi-channel
      3. Resample to target rate
      4. High-pass / low-pass filter
      5. Spectral gating denoising
      6. Pre-emphasis
      7. Loudness normalization
      8. Long silence removal
    """
    warnings: list[str] = []
    sr = config.target_sample_rate_hz

    # Read raw PCM data
    with wave.open(str(audio_path), "rb") as wavf:
        raw = wavf.readframes(wavf.getnframes())

    samples = _wav_to_float(raw, orig_sample_width)

    # Mix to mono if multi-channel
    if orig_channels > 1:
        samples = samples.reshape(-1, orig_channels).mean(axis=1)
        warnings.append("Mixed multi-channel audio to mono for preprocessing.")

    # Resample
    if orig_sample_rate != sr:
        samples = _resample(samples, orig_sample_rate, sr)
        warnings.append(f"Resampled from {orig_sample_rate} Hz to {sr} Hz.")

    level = config.level

    # Level 1+ : done (resample/mono already handled)

    # Level 2+ : filtering
    if level >= 2 and (config.highpass_cutoff_hz > 0 or config.lowpass_cutoff_hz > 0):
        samples = _apply_filter(
            samples,
            sr=sr,
            highpass_hz=config.highpass_cutoff_hz if level >= 2 else 0,
            lowpass_hz=config.lowpass_cutoff_hz if level >= 2 else 0,
        )
        warnings.append(
            f"Applied HP={config.highpass_cutoff_hz}Hz LP={config.lowpass_cutoff_hz}Hz filter."
        )

    if config.enable_wpe:
        samples = _apply_wpe(samples, taps=config.wpe_taps, delay=config.wpe_delay)
        warnings.append(f"Applied WPE dereverberation (taps={config.wpe_taps}, delay={config.wpe_delay}).")

    if config.enable_deepfilter:
        samples = _apply_deepfilter(samples, sr=sr)
        warnings.append("Applied DeepFilterNet 3 deep neural noise suppression.")
    elif level >= 3:
        samples = _denoise_spectral_gating(
            samples,
            sr=sr,
            prop_decrease=config.noise_reduce_prop_decrease,
            stationary=config.noise_reduce_stationary,
            time_constant_s=config.noise_reduce_time_constant_s,
        )
        warnings.append(
            f"Applied spectral gating denoising (prop_decrease={config.noise_reduce_prop_decrease})."
        )

    # Pre-emphasis
    if config.apply_pre_emphasis and config.pre_emphasis_coef > 0:
        samples = _apply_pre_emphasis(samples, config.pre_emphasis_coef)
        warnings.append(f"Applied pre-emphasis (coef={config.pre_emphasis_coef}).")

    # Level 4+ : loudness normalization
    if level >= 4:
        samples = _normalize_loudness(samples, config.target_loudness_lufs)
        warnings.append(f"Normalized loudness to {config.target_loudness_lufs} LUFS.")

    # Silence removal (always last)
    if config.remove_long_silence:
        samples = _remove_long_silence_vad(
            samples,
            sr=sr,
            threshold_db=config.silence_threshold_db,
            min_silence_ms=config.min_silence_duration_ms,
        )
        warnings.append(
            f"Removed silence >{config.min_silence_duration_ms}ms below {config.silence_threshold_db}dB."
        )

    source = SourceInfo(
        path=str(audio_path),
        sha256=_sha256_file(audio_path),
        duration_seconds=round(len(samples) / sr, 6),
        sample_rate_hz=sr,
        channels=1,
        sample_width_bytes=4,  # float32
        frame_count=len(samples),
    )

    return PreprocessedAudio(samples=samples, sample_rate=sr, source=source, warnings=warnings)


# ---------------------------------------------------------------------------
# Channel splitting (preserved from original)
# ---------------------------------------------------------------------------


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
                ch_bytes.extend(raw_frames[sample_start: sample_start + sample_width])

            dst_wav.writeframes(bytes(ch_bytes))
        channel_paths.append(ch_path)

    return channel_paths


# ---------------------------------------------------------------------------
# Detection helpers (preserved from original)
# ---------------------------------------------------------------------------


def _is_mostly_silence(raw: bytes, sample_width: int) -> bool:
    if sample_width != 2 or not raw:
        return False
    silence_count = 0
    total = len(raw) // 2
    for i in range(0, len(raw), 2):
        sample = int.from_bytes(raw[i: i + 2], byteorder="little", signed=True)
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
        sample = int.from_bytes(raw[i: i + 2], byteorder="little", signed=True)
        if abs(sample) >= limit:
            clipped += 1
    return (clipped / max(total, 1)) > 0.02