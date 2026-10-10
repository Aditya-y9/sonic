from wav_transcriber.diarization import diarize_segments
from wav_transcriber.overlap import detect_overlap
from wav_transcriber.schemas import Segment, Word
from wav_transcriber.turn_detector import split_segments_into_turns


def test_split_segments_into_turns() -> None:
    words = [
        Word(text="Hello.", start=0.0, end=0.5, confidence=0.9),
        Word(text="How", start=0.8, end=1.0, confidence=0.9),
        Word(text="are", start=1.0, end=1.2, confidence=0.9),
        Word(text="you?", start=1.2, end=1.5, confidence=0.9),
    ]
    seg = Segment(
        id="s1",
        speaker=None,
        channel=0,
        start=0.0,
        end=1.5,
        text_original="Hello. How are you?",
        words=words,
    )
    turns = split_segments_into_turns([seg], min_pause_seconds=0.2)
    assert len(turns) == 2
    assert turns[0].text_original == "Hello."
    assert turns[0].start == 0.0
    assert turns[0].end == 0.5
    assert turns[1].text_original == "How are you?"
    assert turns[1].start == 0.8
    assert turns[1].end == 1.5


def test_diarize_with_channel_map() -> None:
    cmap = {
        "0": {"side": "left", "role": "agent", "speaker": "agent_01"},
        "1": {"side": "right", "role": "customer", "speaker": "cust_02"},
    }
    seg0 = Segment(
        id="s1",
        speaker=None,
        channel=0,
        start=0.0,
        end=2.0,
        text_original="Agent speaking",
    )
    seg1 = Segment(
        id="s2",
        speaker=None,
        channel=1,
        start=1.0,
        end=3.0,
        text_original="Customer speaking",
    )
    diarized, warnings = diarize_segments([seg0, seg1], enabled=True, channel_map=cmap)
    assert diarized[0].speaker == "agent_01"
    assert diarized[1].speaker == "cust_02"


def test_detect_overlap_concurrent_speakers() -> None:
    seg0 = Segment(
        id="s1",
        speaker="speaker_00",
        channel=0,
        start=10.0,
        end=15.0,
        text_original="First person talking",
    )
    seg1 = Segment(
        id="s2",
        speaker="speaker_01",
        channel=1,
        start=12.0,
        end=16.0,
        text_original="Second person interrupting",
    )
    seg2 = Segment(
        id="s3",
        speaker="speaker_00",
        channel=0,
        start=20.0,
        end=25.0,
        text_original="No overlap here",
    )

    detected, warnings = detect_overlap([seg0, seg1, seg2], enabled=True)
    assert detected[0].overlap is True
    assert detected[1].overlap is True
    assert detected[2].overlap is False
