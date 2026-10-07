from wav_transcriber.diarization import diarize_segments
from wav_transcriber.overlap import detect_overlap
from wav_transcriber.schemas import Segment


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
