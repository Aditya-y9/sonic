from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import Segment, Word


@dataclass(slots=True)
class ROVERWord:
    """A word hypothesis with provenance."""
    text: str
    confidence: float | None
    start: float
    end: float
    source_engine: str


@dataclass(slots=True)
class ROVERResult:
    segments: list[Segment]
    warnings: list[str] = field(default_factory=list)


def _levenshtein_alignment(
    words_a: list[str],
    words_b: list[str],
) -> list[tuple[int, int, str]]:
    """Compute edit script between two word sequences.

    Returns list of (op, pos_a, pos_b) where op is 0=match, 1=sub, 2=ins, 3=del.
    """
    n, m = len(words_a), len(words_b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if words_a[i - 1] == words_b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # match/substitution
            )

    # Backtrace
    alignment: list[tuple[int, int, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if words_a[i - 1] == words_b[j - 1] else 1):
            if words_a[i - 1] == words_b[j - 1]:
                alignment.append((0, i - 1, words_a[i - 1]))  # match
            else:
                alignment.append((1, i - 1, words_a[i - 1]))  # substitution (use a's word as ref)
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            alignment.append((3, i - 1, words_a[i - 1]))  # deletion in b
            i -= 1
        else:
            alignment.append((2, j - 1, words_b[j - 1]))  # insertion in b
            j -= 1

    alignment.reverse()
    return alignment


def _build_word_map(segments: list[Segment]) -> tuple[list[str], list[dict]]:
    """Flatten segments into word list with metadata lookup."""
    words: list[str] = []
    meta: list[dict] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                words.append(w.text)
                meta.append({
                    "segment_id": seg.id,
                    "speaker": seg.speaker,
                    "channel": seg.channel,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                    "text": w.text,
                })
        else:
            # Fallback: split segment text into words
            for tok in seg.text_original.split():
                words.append(tok)
                meta.append({
                    "segment_id": seg.id,
                    "speaker": seg.speaker,
                    "channel": seg.channel,
                    "start": seg.start,
                    "end": seg.end,
                    "confidence": seg.confidence,
                    "text": tok,
                })
    return words, meta


def _reconcile_aligned_words(
    primary_words: list[str],
    primary_meta: list[dict],
    secondary_words: list[str],
    secondary_meta: list[dict],
) -> list[dict]:
    """Reconcile two word sequences into a single list via ROVER voting.

    Uses confidence-weighted voting when words differ at aligned positions.
    """
    alignment = _levenshtein_alignment(primary_words, secondary_words)
    result: list[dict] = []

    sec_idx = 0
    for op, pri_idx, word in alignment:
        if op == 0:  # match — take primary
            result.append({**primary_meta[pri_idx]})
        elif op == 1:  # substitution — confidence-weighted vote
            pri = primary_meta[pri_idx]
            sec = secondary_meta[sec_idx]
            pri_conf = pri.get("confidence") or 0.0
            sec_conf = sec.get("confidence") or 0.0
            if sec_conf > pri_conf:
                result.append({**sec, "source": "rover_secondary"})
            else:
                result.append({**pri, "source": "rover_primary"})
            sec_idx += 1
        elif op == 3:  # deletion (only in primary)
            result.append({**primary_meta[pri_idx], "source": "rover_primary_only"})
        elif op == 2:  # insertion (only in secondary) — skip, or include if confident
            sec = secondary_meta[sec_idx]
            sec_conf = sec.get("confidence") or 0.0
            if sec_conf > 0.5:
                result.append({**sec, "source": "rover_secondary_only"})
            sec_idx += 1

    return result


def _reconstruct_segments(aligned_words: list[dict]) -> list[Segment]:
    """Reconstruct segments from aligned word list, merging consecutive words with same speaker."""
    if not aligned_words:
        return []

    segments: list[Segment] = []
    current_words: list[Word] = []
    current_seg_id: str | None = None
    current_speaker: str | None = None
    current_channel: int | None = None
    current_start: float = 0.0
    segment_counter = 0

    for i, w in enumerate(aligned_words):
        # Determine segment boundary: new segment at speaker/channel changes or gap > 0.5s
        needs_new = False
        if current_words and current_speaker != w.get("speaker"):
            needs_new = True
        if current_words and current_channel != w.get("channel"):
            needs_new = True
        if current_words and (w["start"] - current_words[-1].end) > 0.5:
            needs_new = True

        if needs_new and current_words:
            segment_counter += 1
            end = current_words[-1].end
            text = " ".join(t.text for t in current_words)
            segments.append(Segment(
                id=f"seg-{segment_counter:06d}",
                speaker=current_speaker,
                channel=current_channel,
                start=current_start,
                end=end,
                text_original=text,
                overlap=False,
                confidence=sum((t.confidence or 0) for t in current_words) / max(len(current_words), 1),
                words=list(current_words),
            ))
            current_words = []

        # Determine word confidence (prefer explicit, then segment-level)
        conf = w.get("confidence")
        if conf is None:
            conf = 0.85

        current_words.append(Word(
            text=w["text"],
            start=w["start"],
            end=w["end"],
            confidence=conf,
        ))
        if len(current_words) == 1:
            current_start = w["start"]
        current_speaker = w.get("speaker")
        current_channel = w.get("channel")

    # Final segment
    if current_words:
        segment_counter += 1
        end = current_words[-1].end
        text = " ".join(t.text for t in current_words)
        segments.append(Segment(
            id=f"seg-{segment_counter:06d}",
            speaker=current_speaker,
            channel=current_channel,
            start=current_start,
            end=end,
            text_original=text,
            overlap=False,
            confidence=sum((t.confidence or 0) for t in current_words) / max(len(current_words), 1),
            words=list(current_words),
        ))

    return segments


def reconcile_ensemble(
    primary: list[Segment],
    secondary: list[Segment],
    primary_engine: str = "primary",
    secondary_engine: str = "secondary",
) -> ROVERResult:
    """Reconcile two ASR engine outputs via ROVER confidence-weighted voting.

    The primary result takes precedence; the secondary result fills gaps and
    overrides when its confidence is significantly higher.
    """
    if not primary:
        return ROVERResult(segments=secondary, warnings=["Primary engine returned 0 segments; using secondary."])
    if not secondary:
        return ROVERResult(segments=primary, warnings=["Secondary engine returned 0 segments; using primary."])

    warnings: list[str] = []

    # Flatten both engine outputs into word lists
    primary_words, primary_meta = _build_word_map(primary)
    secondary_words, secondary_meta = _build_word_map(secondary)

    # Align and reconcile
    aligned = _reconcile_aligned_words(primary_words, primary_meta, secondary_words, secondary_meta)

    # Count voting decisions
    primary_count = sum(1 for a in aligned if a.get("source", "").startswith("rover_primary"))
    secondary_count = sum(1 for a in aligned if a.get("source", "").startswith("rover_secondary"))
    total = len(aligned)

    if primary_count > 0:
        warnings.append(
            f"ROVER: primary={primary_engine} won {primary_count}/{total} words, "
            f"secondary={secondary_engine} won {secondary_count}/{total} words."
        )

    # Reconstruct segments
    segments = _reconstruct_segments(aligned)

    return ROVERResult(segments=segments, warnings=warnings)