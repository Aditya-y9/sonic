from wav_transcriber.config import QualityConfig
from wav_transcriber.quality import run_quality_gates
from wav_transcriber.schemas import Segment


def test_quality_marks_needs_review_for_low_confidence() -> None:
    seg = Segment(
        id="seg-1",
        speaker="speaker_00",
        channel=0,
        start=0.0,
        end=1.0,
        text_original="hello",
        confidence=0.2,
    )
    result = run_quality_gates([seg], QualityConfig(low_confidence_threshold=0.5), [])
    assert result.status == "needs_review"
    assert any("below confidence threshold" in reason for reason in result.review_reasons)
