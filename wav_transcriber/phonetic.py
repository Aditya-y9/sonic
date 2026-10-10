from __future__ import annotations

import re
from symspellpy import SymSpell, Verbosity


class PhoneticRescorer:
    def __init__(self, vocabulary_terms: list[str] | None = None) -> None:
        self._symspell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
        self._canonical: dict[str, str] = {}

        default_terms = [
            "diarization",
            "diarize",
            "real-time",
            "transcription",
            "Microsoft Speech Service",
            "batch transcription",
        ]
        all_terms = list(default_terms)
        if vocabulary_terms:
            all_terms.extend(vocabulary_terms)

        for term in all_terms:
            clean = term.strip()
            if not clean:
                continue
            lower_clean = clean.lower()
            self._canonical[lower_clean] = clean
            for token in lower_clean.split():
                if len(token) >= 4:
                    self._symspell.create_dictionary_entry(token, 10000)

    def rescore_token(self, token: str) -> str:
        clean = re.sub(r"[^\w-]", "", token)
        if not clean or len(clean) < 4:
            return token

        lower = clean.lower()
        if lower in self._canonical:
            return self._preserve_case(token, self._canonical[lower])

        suggestions = self._symspell.lookup(lower, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions and suggestions[0].distance <= 2:
            best = suggestions[0].term
            if best in self._canonical:
                return self._preserve_case(token, self._canonical[best])
            return self._preserve_case(token, best)

        return token

    def rescore_text(self, text: str) -> str:
        words = text.split()
        rescored = [self.rescore_token(w) for w in words]
        return " ".join(rescored)

    def _preserve_case(self, original: str, replacement: str) -> str:
        prefix = ""
        suffix = ""
        m = re.match(r"^([^\w]*)(.*?)([^\w]*)$", original)
        if m:
            prefix, core, suffix = m.groups()
        else:
            core = original

        if core.isupper():
            rep = replacement.upper()
        elif core and core[0].isupper():
            rep = replacement.capitalize()
        else:
            rep = replacement

        return f"{prefix}{rep}{suffix}"
