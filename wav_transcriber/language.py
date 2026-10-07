from __future__ import annotations

import re

from .schemas import LanguageSpan, Segment

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def _classify_token_script(token: str, default_lang: str) -> str:
    if _DEVANAGARI_RE.search(token):
        return "hi"
    if _LATIN_RE.search(token):
        return "en"
    return default_lang


def annotate_languages(
    segments: list[Segment],
    language_hints: list[str],
    detect_code_switch: bool = True,
) -> tuple[list[Segment], list[str]]:
    warnings: list[str] = []
    primary_lang = language_hints[0] if language_hints else "en"

    if not language_hints:
        warnings.append("No language hints provided; defaulting to 'en'.")

    for segment in segments:
        # If segment already has explicit spans, preserve and derive languages
        if segment.language_spans:
            langs = []
            for s in segment.language_spans:
                if s.language not in langs:
                    langs.append(s.language)
            segment.languages = langs
            continue

        if not detect_code_switch or not segment.words:
            # Single span across entire segment
            lang = segment.languages[0] if segment.languages else primary_lang
            segment.languages = [lang]
            segment.language_spans = [
                LanguageSpan(start=segment.start, end=segment.end, language=lang)
            ]
            continue

        # Detect word-by-word code-switching
        spans: list[LanguageSpan] = []
        current_lang: str | None = None
        span_start: float = segment.start
        span_end: float = segment.start

        for word in segment.words:
            w_lang = _classify_token_script(word.text, primary_lang)
            if current_lang is None:
                current_lang = w_lang
                span_start = word.start
                span_end = word.end
            elif w_lang == current_lang:
                span_end = word.end
            else:
                spans.append(LanguageSpan(start=span_start, end=span_end, language=current_lang))
                current_lang = w_lang
                span_start = word.start
                span_end = word.end

        if current_lang is not None:
            spans.append(LanguageSpan(start=span_start, end=span_end, language=current_lang))
        else:
            spans.append(LanguageSpan(start=segment.start, end=segment.end, language=primary_lang))

        segment.language_spans = spans
        distinct_langs: list[str] = []
        for s in spans:
            if s.language not in distinct_langs:
                distinct_langs.append(s.language)
        segment.languages = distinct_langs

    return segments, warnings

