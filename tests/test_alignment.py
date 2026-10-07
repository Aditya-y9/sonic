from wav_transcriber.alignment import align_segments
from wav_transcriber.schemas import Segment, Word


def test_align_segments_clamps_bounds() -> None:
    words = [
        Word(text="first", start=0.5, end=1.5),
        Word(text="second", start=2.0, end=4.0),
    ]
    seg = Segment(
        id="seg-1",
        speaker="speaker_00",
        channel=0,
        start=1.0,
        end=3.0,
        text_original="first second",
        words=words,
    )
    aligned, warnings = align_segments([seg], enabled=True, word_level=True)
    assert len(aligned) == 1
    # first word was clamped to seg.start (1.0)
    assert aligned[0].words[0].start == 1.0
    # second word end was clamped to seg.end (3.0)
    assert aligned[0].words[1].end == 3.0
    assert len(warnings) > 0


def test_align_segments_inverts_correction() -> None:
    seg = Segment(
        id="seg-1",
        speaker="speaker_00",
        channel=0,
        start=5.0,
        end=3.0,
        text_original="test",
        words=[],
    )
    aligned, warnings = align_segments([seg], enabled=True)
    assert aligned[0].end == 5.0
    assert any("inverted timing" in w for w in warnings)
