from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import Segment


def _init_pyannote_pipeline(
    model_name: str,
) -> Any:
    """Lazy-load and return a pyannote speaker diarization pipeline.

    Supports:
      - pyannote/speaker-diarization-3.1 (legacy, ~2.2M params)
      - pyannote/speaker-diarization-community-1 (newer open-source, ~150MB)
      - pyannote/speaker-diarization-precision-2 (premium API)
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "pyannote.audio is required for speaker diarization. "
            "Install with `pip install pyannote.audio torch`"
        ) from exc

    try:
        import torch

        pipeline = Pipeline.from_pretrained(model_name)
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load pyannote pipeline '{model_name}': {exc}. "
            "Ensure you have accepted the user conditions on Hugging Face "
            "and set the HF_TOKEN environment variable."
        ) from exc

    return pipeline


def _run_pyannote_diarization(
    pipeline: Any,
    audio_path: Path,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict]:
    """Run pyannote diarization pipeline and return structured results.

    Returns list of dicts with keys: speaker, start, end.
    """
    kwargs: dict[str, Any] = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    diarization = pipeline(str(audio_path), **kwargs)

    turns: list[dict] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append({
            "speaker": str(speaker),
            "start": turn.start,
            "end": turn.end,
        })

    return turns


def _assign_speakers_to_segments(
    segments: list[Segment],
    diarization_turns: list[dict],
) -> list[Segment]:
    """Assign speaker labels to segments based on diarization turn overlap.

    For each segment, finds the diarization turn with the most temporal
    overlap and assigns that speaker label.
    """
    if not diarization_turns:
        return segments

    for segment in segments:
        max_overlap = 0.0
        best_speaker: str | None = None

        seg_duration = segment.end - segment.start
        if seg_duration <= 0:
            continue

        for turn in diarization_turns:
            overlap_start = max(segment.start, turn["start"])
            overlap_end = min(segment.end, turn["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = turn["speaker"]

        if best_speaker and max_overlap / seg_duration > 0.1:
            segment.speaker = best_speaker

    return segments


def _diarize_simple(
    segments: list[Segment],
    channel_map: dict[str | int, Any] | None = None,
) -> tuple[list[Segment], list[str]]:
    """Simple channel-map based diarization (original behavior, no pyannote)."""
    warnings: list[str] = []
    cmap = channel_map or {}

    for segment in segments:
        if segment.channel is not None:
            ch_key = str(segment.channel)
            ch_info = cmap.get(ch_key) or cmap.get(segment.channel)
            if isinstance(ch_info, dict) and "speaker" in ch_info:
                segment.speaker = ch_info["speaker"]
            elif segment.speaker is None:
                segment.speaker = f"speaker_{segment.channel:02d}"

        if segment.speaker is None:
            segment.speaker = "speaker_00"

    return segments, warnings


def _run_wavlm_diarization(
    audio_path: Path,
    segments: list[Segment],
    n_speakers: int = 2,
) -> bool:
    try:
        import numpy as np
        import soundfile as sf
        import torch
        from sklearn.cluster import AgglomerativeClustering
        from transformers import AutoFeatureExtractor, AutoModelForAudioXVector
    except ImportError:
        return False

    if len(segments) < 2:
        for s in segments:
            s.speaker = "speaker_00"
        return True

    try:
        model_id = "microsoft/wavlm-base-plus-sv"
        extractor = AutoFeatureExtractor.from_pretrained(model_id, local_files_only=True)
        model = AutoModelForAudioXVector.from_pretrained(model_id, local_files_only=True)
    except Exception:
        try:
            model_id = "microsoft/wavlm-base-plus-sv"
            extractor = AutoFeatureExtractor.from_pretrained(model_id)
            model = AutoModelForAudioXVector.from_pretrained(model_id)
        except Exception:
            return False

    model.eval()
    try:
        y, sr = sf.read(str(audio_path))
    except Exception:
        return False

    if y.ndim > 1:
        y = y.mean(axis=1)

    embeddings = []
    with torch.no_grad():
        for seg in segments:
            start_s = max(0, int(seg.start * sr))
            end_s = min(len(y), int(seg.end * sr))
            clip = y[start_s:end_s]
            if len(clip) < 160:
                clip = np.pad(clip, (0, 160 - len(clip)))
            inputs = extractor(clip, sampling_rate=16000, return_tensors="pt", padding=True)
            out = model(**inputs)
            emb = torch.nn.functional.normalize(out.embeddings, dim=-1).squeeze().cpu().numpy()
            embeddings.append(emb)

    embeddings_arr = np.array(embeddings)
    from sklearn.metrics import silhouette_score

    dist_matrix = np.clip(1.0 - (embeddings_arr @ embeddings_arr.T), 0.0, 2.0)
    k = n_speakers
    if k is None or k <= 0:
        max_k = min(6, len(segments) - 1)
        if max_k >= 2:
            best_score = -1.0
            best_k = 2
            for cand_k in range(2, max_k + 1):
                c = AgglomerativeClustering(n_clusters=cand_k, metric="precomputed", linkage="average")
                labels = c.fit_predict(dist_matrix)
                score = silhouette_score(dist_matrix, labels, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_k = cand_k
            k = best_k
        else:
            k = 1

    durations = np.array([max(0.01, s.end - s.start) for s in segments])
    long_mask = durations >= 1.0
    if np.sum(long_mask) >= k and k > 1:
        long_indices = np.where(long_mask)[0]
        long_dist = dist_matrix[np.ix_(long_indices, long_indices)]
        long_clust = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        long_labels = long_clust.fit_predict(long_dist)

        centroids = []
        for cluster_id in range(k):
            members = embeddings_arr[long_indices[long_labels == cluster_id]]
            cent = members.mean(axis=0)
            cent = cent / (np.linalg.norm(cent) + 1e-8)
            centroids.append(cent)
        centroids = np.array(centroids)

        preds = []
        for emb in embeddings_arr:
            sims = centroids @ emb
            preds.append(int(np.argmax(sims)))
    else:
        k = min(k, len(segments))
        clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        preds = clustering.fit_predict(embeddings_arr)

    for seg, pred in zip(segments, preds):
        seg.speaker = f"speaker_{pred:02d}"

    return True


def _run_acoustic_diarization(
    audio_path: Path,
    segments: list[Segment],
    n_speakers: int = 2,
) -> bool:
    try:
        import librosa
        import numpy as np
        import soundfile as sf
        from sklearn.cluster import AgglomerativeClustering
    except ImportError:
        return False

    if len(segments) < 2:
        for s in segments:
            s.speaker = "speaker_00"
        return True

    try:
        y, sr = sf.read(str(audio_path))
    except Exception:
        return False

    if y.ndim > 1:
        y = y.mean(axis=1)

    feats = []
    for seg in segments:
        start_s = max(0, int(seg.start * sr))
        end_s = min(len(y), int(seg.end * sr))
        clip = y[start_s:end_s]
        if len(clip) < 512:
            clip = np.pad(clip, (0, 512 - len(clip)))
        mfcc = librosa.feature.mfcc(y=clip, sr=sr, n_mfcc=13)
        mfcc_mean = mfcc.mean(axis=1)
        cent = librosa.feature.spectral_centroid(y=clip, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(clip).mean()
        vec = np.concatenate([mfcc_mean, [cent, zcr]])
        feats.append(vec)

    feats_arr = np.array(feats)
    std = feats_arr.std(axis=0) + 1e-8
    feats_norm = (feats_arr - feats_arr.mean(axis=0)) / std

    k = min(n_speakers, len(segments))
    clustering = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    preds = clustering.fit_predict(feats_norm)

    for seg, pred in zip(segments, preds):
        seg.speaker = f"speaker_{pred:02d}"

    return True


def diarize_segments(
    segments: list[Segment],
    enabled: bool,
    channel_map: dict[str | int, Any] | None = None,
    model: str | None = None,
    audio_path: Path | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> tuple[list[Segment], list[str]]:
    """Assign speaker labels to segments using speaker diarization.

    When enabled and audio_path is provided, runs the pyannote/speaker-diarization
    pipeline for neural speaker diarization. Falls back to simple channel-map
    based assignment when audio_path is unavailable or pyannote is not installed.

    Args:
        segments: Segments to diarize.
        enabled: Whether diarization is enabled.
        channel_map: Channel-to-speaker mapping for simple diarization.
        model: Pyannote model name (e.g. "pyannote/speaker-diarization-3.1").
        audio_path: Path to the audio file (required for neural diarization).
        min_speakers: Minimum number of speakers.
        max_speakers: Maximum number of speakers.

    Returns:
        Tuple of (diarized segments, warnings list).
    """
    if not enabled:
        return segments, []

    warnings: list[str] = []
    num_speakers = max_speakers or min_speakers or 2

    if audio_path is not None and model and "pyannote" in model:
        try:
            pipeline = _init_pyannote_pipeline(model)
            turns = _run_pyannote_diarization(
                pipeline,
                audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
            if turns:
                segments = _assign_speakers_to_segments(segments, turns)
                return segments, [f"Pyannote diarization assigned {len(set(t['speaker'] for t in turns))} speakers across {len(turns)} turns."]
        except Exception as exc:
            warnings.append(f"Pyannote diarization unavailable ({exc}); trying token-free neural diarization.")

    if audio_path is not None:
        if _run_wavlm_diarization(audio_path, segments, n_speakers=num_speakers):
            speakers = set(s.speaker for s in segments if s.speaker)
            warnings.append(f"WavLM speaker diarization assigned {len(speakers)} speakers across {len(segments)} segments.")
            return segments, warnings

        if _run_acoustic_diarization(audio_path, segments, n_speakers=num_speakers):
            speakers = set(s.speaker for s in segments if s.speaker)
            warnings.append(f"Acoustic feature clustering assigned {len(speakers)} speakers across {len(segments)} segments.")
            return segments, warnings

    segments_out, simple_warnings = _diarize_simple(segments, channel_map)
    warnings.extend(simple_warnings)
    return segments_out, warnings