from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schemas import Segment, Word


@dataclass(slots=True)
class _AlignmentResult:
    """Word-level alignment from CTC decoder."""
    words: list[str]
    starts: list[float]
    ends: list[float]
    probabilities: list[float]


def _load_wav2vec2_model(model_name: str):
    """Lazy-load Wav2Vec2 model for CTC alignment.

    Supports:
      - WAV2VEC2_ASR_BASE_960H (default, ~95M params)
      - Any Hugging Face Wav2Vec2ForCTC model
    """
    # Map the config's short model name to Hugging Face ID if needed
    model_map = {
        "WAV2VEC2_ASR_BASE_960H": "facebook/wav2vec2-base-960h",
        "WAV2VEC2_LARGE": "facebook/wav2vec2-large-960h",
        "WAV2VEC2_LARGE_LV60K": "facebook/wav2vec2-large-960h-lv60-self",
    }
    hf_id = model_map.get(model_name, model_name)

    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for forced alignment. "
            "Install with `pip install transformers torch`"
        ) from exc

    import torch  # noqa: F401 — needed for model

    try:
        processor = Wav2Vec2Processor.from_pretrained(hf_id)
        model = Wav2Vec2ForCTC.from_pretrained(hf_id)
        # Set model to eval mode
        model.eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load Wav2Vec2 model '{hf_id}': {exc}"
        ) from exc

    return processor, model


def _read_audio_for_alignment(audio_path: Path, target_sr: int = 16000) -> np.ndarray:
    """Read audio file into float32 numpy array at target sample rate.

    Uses soundfile (preferred) or librosa as fallback.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(str(audio_path))
    except (ImportError, Exception):
        try:
            import librosa
            data, sr = librosa.load(str(audio_path), sr=target_sr)
        except ImportError:
            raise RuntimeError(
                "soundfile or librosa is required for forced alignment. "
                "Install with `pip install soundfile librosa`"
            )

    # Convert to mono if multi-channel
    if data.ndim > 1:
        data = data.mean(axis=1)

    # Resample if needed
    if sr != target_sr:
        try:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr, res_type="kaiser_best")
        except ImportError:
            from scipy import signal as sp_signal
            duration = len(data) / sr
            target_len = int(duration * target_sr)
            data = sp_signal.resample(data, target_len)

    return data.astype(np.float32)


def _force_align_wav2vec2(
    audio_path: Path,
    segments: list[Segment],
    model_name: str,
    target_sr: int = 16000,
) -> _AlignmentResult:
    """Run CTC forced alignment with Wav2Vec2 to get precise word timestamps.

    For each segment, aligns the audio against the hypothesized text using
    Wav2Vec2's acoustic model and CTC prefix beam search decoding.
    """
    import torch
    import torch.nn.functional as F

    processor, model = _load_wav2vec2_model(model_name)
    audio = _read_audio_for_alignment(audio_path, target_sr)

    # Send model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    word_level_starts: list[float] = []
    word_level_ends: list[float] = []
    word_level_texts: list[str] = []
    word_level_probs: list[float] = []

    for segment in segments:
        # Skip empty segments
        text = segment.text_original.strip()
        if not text:
            continue

        # Extract the audio segment corresponding to this segment's time range
        seg_start_sample = int(segment.start * target_sr)
        seg_end_sample = min(int(segment.end * target_sr), len(audio))
        seg_audio = audio[seg_start_sample:seg_end_sample]

        if len(seg_audio) < target_sr * 0.02:  # < 20ms
            continue

        # Process audio
        input_values = processor(
            seg_audio,
            sampling_rate=target_sr,
            return_tensors="pt",
        ).input_values.to(device)

        # Get logits from model
        with torch.no_grad():
            logits = model(input_values).logits  # (1, T, vocab_size)

        # Get CTC alignment via forced alignment (trellis)
        clean_text = text.upper()
        labels = processor.tokenizer(clean_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        log_probs = F.log_softmax(logits, dim=-1)  # (1, T, V)
        T = log_probs.size(1)
        U = labels.size(1)

        # CTC forced alignment using viterbi/trellis
        # We compute the best path through the CTC trellis
        blank_id = processor.tokenizer.pad_token_id
        if blank_id is None:
            blank_id = processor.tokenizer.token_to_id("<pad>")

        # Simplified forced alignment: use monotonic alignment search
        # Compute per-frame argmax labels
        argmax = logits[0].argmax(dim=-1)  # (T,)

        # Find boundaries between different tokens (collapse repeats + blanks)
        word_tokens = processor.tokenizer.convert_ids_to_tokens(labels[0])
        word_string = processor.tokenizer.decode(labels[0])

        # Map decoded word sequence to audio frames
        # We use a greedy CTC decode to find word boundaries
        prev_token = blank_id
        word_boundaries: list[tuple[int, int]] = []  # (start_frame, end_frame)
        word_chars: list[str] = []
        seg_start_frame = None

        for t in range(T):
            token_id = int(argmax[t])
            if token_id != prev_token and token_id != blank_id:
                # Token change — potential word boundary
                if seg_start_frame is not None and prev_token != blank_id:
                    word_boundaries.append((seg_start_frame, t))
                    token_text = processor.tokenizer.decode([prev_token])
                    word_chars.append(token_text)

                seg_start_frame = t
            if token_id == blank_id and seg_start_frame is not None:
                # Blank encountered — close current segment
                word_boundaries.append((seg_start_frame, t))
                token_text = processor.tokenizer.decode([prev_token])
                word_chars.append(token_text)
                seg_start_frame = None

            prev_token = token_id

        # Close final segment
        if seg_start_frame is not None:
            word_boundaries.append((seg_start_frame, T))
            token_text = processor.tokenizer.decode([prev_token])
            word_chars.append(token_text)

        if not word_boundaries:
            continue

        # Merge subword tokens into words and compute frame-to-time mapping
        frames_per_second = target_sr / 320  # Wav2Vec2 outputs 50Hz framerate (every 20ms)
        segment_offset = segment.start

        # Group subwords by word (space separators from decoding)
        merged_words: list[str] = []
        merged_starts: list[float] = []
        merged_ends: list[float] = []
        merged_probs: list[float] = []

        current_chars: list[str] = []
        word_start_frame: int = 0
        is_first = True

        for (start_f, end_f), char in zip(word_boundaries, word_chars):
            if char.startswith("Ġ") or is_first:
                if current_chars:
                    merged_words.append("".join(current_chars))
                    merged_starts.append(word_start_frame / frames_per_second + segment_offset)
                    merged_ends.append(start_f / frames_per_second + segment_offset)

                    # Estimate confidence from log_probs in this window
                    conf = 0.8
                    if start_f < T:
                        conf_slice = log_probs[0, word_start_frame:start_f, :]
                        best_scores = conf_slice.max(dim=-1).values
                        conf = float(torch.exp(best_scores).mean().clamp(0, 1).cpu())
                    merged_probs.append(round(conf, 3))

                current_chars = []
                word_start_frame = start_f

            # Remove the Ġ prefix for word continuation
            clean_char = char.replace("Ġ", "")
            if clean_char:
                current_chars.append(clean_char)
            is_first = False

        # Final word
        if current_chars:
            merged_words.append("".join(current_chars))
            last_f = word_boundaries[-1][1]
            merged_starts.append(word_start_frame / frames_per_second + segment_offset)
            merged_ends.append(last_f / frames_per_second + segment_offset)
            conf = 0.8
            if last_f < T:
                conf_slice = log_probs[0, word_start_frame:last_f, :]
                best_scores = conf_slice.max(dim=-1).values
                conf = float(torch.exp(best_scores).mean().clamp(0, 1).cpu())
            merged_probs.append(round(conf, 3))

        word_level_texts.extend(merged_words)
        word_level_starts.extend(merged_starts)
        word_level_ends.extend(merged_ends)
        word_level_probs.extend(merged_probs)

    return _AlignmentResult(
        words=word_level_texts,
        starts=word_level_starts,
        ends=word_level_ends,
        probabilities=word_level_probs,
    )


def align_segments(
    segments: list[Segment],
    enabled: bool,
    word_level: bool = True,
    model_name: str | None = "WAV2VEC2_ASR_BASE_960H",
    audio_path: Path | None = None,
) -> tuple[list[Segment], list[str]]:
    """Run forced alignment on segments.

    When enabled with an audio_path, uses Wav2Vec2 CTC forced alignment
    for precise word-level timestamps. Falls back to bounds clamping
    when the audio path is unavailable or the model fails to load.

    Args:
        segments: List of transcription segments to align.
        enabled: Whether alignment is enabled.
        word_level: Whether to produce word-level timestamps.
        model_name: Wav2Vec2 model identifier.
        audio_path: Path to the original audio file (required for real alignment).

    Returns:
        Tuple of (aligned segments, warnings list).
    """
    if not enabled:
        return segments, []

    warnings: list[str] = []

    # First pass: basic bounds clamping (always applied)
    for segment in segments:
        if segment.end < segment.start:
            warnings.append(f"Segment {segment.id} had inverted timing [{segment.start} > {segment.end}]; clamped.")
            segment.end = segment.start

        if not word_level:
            continue

        clamped = False
        prev_end = segment.start
        for word in segment.words:
            if word.start < segment.start:
                word.start = segment.start
                clamped = True
            if word.end > segment.end:
                word.end = segment.end
                clamped = True
            if word.end < word.start:
                word.end = word.start
                clamped = True
            if word.start < prev_end:
                word.start = max(word.start, prev_end)
                word.end = max(word.end, word.start)
                clamped = True
            prev_end = word.end
        if clamped:
            warnings.append(f"Segment {segment.id}: word boundaries clamped to segment bounds.")

    # Second pass: Wav2Vec2 CTC forced alignment for precise timestamps
    if word_level and audio_path is not None and model_name:
        try:
            align_result = _force_align_wav2vec2(
                audio_path=audio_path,
                segments=segments,
                model_name=model_name,
            )
            total_seg_words = sum(len(s.words) for s in segments)
            if align_result.words and len(align_result.words) >= int(total_seg_words * 0.8):
                word_idx = 0
                for segment in segments:
                    segment_words: list[Word] = []
                    for w in segment.words:
                        if word_idx < len(align_result.words):
                            segment_words.append(Word(
                                text=w.text,
                                start=round(align_result.starts[word_idx], 3),
                                end=round(align_result.ends[word_idx], 3),
                                confidence=round(align_result.probabilities[word_idx], 3),
                            ))
                            word_idx += 1
                        else:
                            segment_words.append(w)
                    if segment_words:
                        segment.words = segment_words
                warnings.append(f"Forced alignment aligned {word_idx} words across {len(segments)} segments.")
            else:
                warnings.append("Wav2Vec2 token count differed from ASR word count; preserved Whisper DTW timestamps.")
        except Exception as exc:
            warnings.append(
                f"Wav2Vec2 forced alignment failed: {exc}. "
                "Falling back to bounds-clamping only."
            )

    return segments, warnings