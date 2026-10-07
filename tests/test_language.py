from wav_transcriber.language import annotate_languages
from wav_transcriber.schemas import Segment, Word


def test_annotate_languages_codeswitching() -> None:
    words = [
        Word(text="Kal", start=0.0, end=0.5),
        Word(text="meeting", start=0.5, end=1.0),
        Word(text="reschedule", start=1.0, end=1.8),
        Word(text="karte", start=1.8, end=2.2),
        Word(text="hain", start=2.2, end=2.6),
    ]
    seg = Segment(
        id="seg-1",
        speaker="speaker_00",
        channel=0,
        start=0.0,
        end=2.6,
        text_original="Kal meeting reschedule karte hain",
        words=words,
    )
    annotated, warnings = annotate_languages([seg], language_hints=["hi", "en"], detect_code_switch=True)
    assert len(annotated[0].language_spans) > 0
    assert "en" in annotated[0].languages or "hi" in annotated[0].languages


def test_annotate_languages_devanagari() -> None:
    words = [
        Word(text="कल", start=0.0, end=0.5),
        Word(text="meeting", start=0.5, end=1.0),
        Word(text="होगी", start=1.0, end=1.5),
    ]
    seg = Segment(
        id="seg-1",
        speaker="speaker_00",
        channel=0,
        start=0.0,
        end=1.5,
        text_original="कल meeting होगी",
        words=words,
    )
    annotated, warnings = annotate_languages([seg], language_hints=["hi", "en"], detect_code_switch=True)
    spans = annotated[0].language_spans
    assert len(spans) == 3
    assert spans[0].language == "hi"
    assert spans[1].language == "en"
    assert spans[2].language == "hi"
    assert "hi" in annotated[0].languages
    assert "en" in annotated[0].languages
