from wav_transcriber.phonetic import PhoneticRescorer


def test_phonetic_rescorer_fixes_misspellings() -> None:
    rescorer = PhoneticRescorer(vocabulary_terms=["diarization", "real-time"])
    assert rescorer.rescore_token("diorization") == "diarization"
    assert rescorer.rescore_token("diarizzation") == "diarization"
    assert rescorer.rescore_text("the diorization feature works in real-time.") == "the diarization feature works in real-time."


def test_phonetic_rescorer_preserves_case() -> None:
    rescorer = PhoneticRescorer(vocabulary_terms=["diarization"])
    assert rescorer.rescore_token("Diorization") == "Diarization"
    assert rescorer.rescore_token("DIORIZATION") == "DIARIZATION"
