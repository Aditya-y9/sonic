import json
from pathlib import Path

from wav_transcriber.asr import MockAsrBackend, load_vocabulary


def test_load_vocabulary(tmp_path: Path) -> None:
    f1 = tmp_path / "vocab1.txt"
    f2 = tmp_path / "vocab2.txt"

    f1.write_text("API\n# Comment\nSDK\n", encoding="utf-8")
    f2.write_text("SDK\nEndpoint\n\n", encoding="utf-8")

    vocab = load_vocabulary([str(f1), str(f2)])
    assert vocab == ["API", "SDK", "Endpoint"]


def test_mock_asr_json_sidecar(tmp_path: Path) -> None:
    wav_path = tmp_path / "sample.wav"
    wav_path.write_bytes(b"dummy")
    json_sidecar = tmp_path / "sample.json"
    json_sidecar.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "text": "Hello world",
                        "speaker": "speaker_01",
                        "words": [
                            {"text": "Hello", "start": 0.0, "end": 0.8},
                            {"text": "world", "start": 0.9, "end": 1.8},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    backend = MockAsrBackend(language_hints=["en"])
    res = backend.transcribe(wav_path, duration_seconds=2.0)
    assert len(res.segments) == 1
    assert res.segments[0].text_original == "Hello world"
    assert res.segments[0].speaker == "speaker_01"
    assert len(res.segments[0].words) == 2
