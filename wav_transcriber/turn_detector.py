from __future__ import annotations

from .schemas import Segment, Word


def split_segments_into_turns(
    segments: list[Segment],
    min_pause_seconds: float = 0.18,
    max_pause_seconds: float = 0.50,
) -> list[Segment]:
    if not segments:
        return []

    new_segments: list[Segment] = []
    seg_idx = 1

    for seg in segments:
        if not seg.words or len(seg.words) <= 1:
            seg.id = f"seg-{seg_idx:06d}"
            new_segments.append(seg)
            seg_idx += 1
            continue

        curr_words: list[Word] = []
        for i, word in enumerate(seg.words):
            curr_words.append(word)
            if i >= len(seg.words) - 1:
                break

            next_word = seg.words[i + 1]
            pause = next_word.start - word.end
            ends_sentence = any(word.text.endswith(p) for p in (".", "?", "!"))

            should_split = False
            if ends_sentence and pause >= min_pause_seconds:
                if not (word.text.lower() in ("yet.", "so.", "well.") and len(curr_words) <= 2):
                    should_split = True
            elif pause >= max_pause_seconds:
                should_split = True

            if should_split and curr_words:
                sub_text = " ".join(w.text for w in curr_words)
                sub_seg = Segment(
                    id=f"seg-{seg_idx:06d}",
                    speaker=seg.speaker,
                    channel=seg.channel,
                    start=curr_words[0].start,
                    end=curr_words[-1].end,
                    text_original=sub_text,
                    text_cleaned=seg.text_cleaned,
                    language_spans=list(seg.language_spans),
                    languages=list(seg.languages),
                    overlap=seg.overlap,
                    confidence=seg.confidence,
                    words=list(curr_words),
                )
                new_segments.append(sub_seg)
                seg_idx += 1
                curr_words = []

        if curr_words:
            sub_text = " ".join(w.text for w in curr_words)
            sub_seg = Segment(
                id=f"seg-{seg_idx:06d}",
                speaker=seg.speaker,
                channel=seg.channel,
                start=curr_words[0].start,
                end=curr_words[-1].end,
                text_original=sub_text,
                text_cleaned=seg.text_cleaned,
                language_spans=list(seg.language_spans),
                languages=list(seg.languages),
                overlap=seg.overlap,
                confidence=seg.confidence,
                words=list(curr_words),
            )
            new_segments.append(sub_seg)
            seg_idx += 1

    return new_segments
