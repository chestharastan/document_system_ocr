"""Stage 4 (post-correction) - only the no-op ships today.

To add one: subclass PostCorrector, register it in POSTCORRECTORS under a name and start the backend with
POSTCORRECT=<name>. It runs after recognition on all lines of a page (see PostCorrector.correct for the contract).
"""

from __future__ import annotations

from app.services.stages.base import LineResult, PostCorrector


class IdentityPostCorrector(PostCorrector):
    name = "none"

    def correct(self, lines: list[LineResult], context: dict) -> list[LineResult]:
        return lines


POSTCORRECTORS: dict[str, type[PostCorrector]] = {"none": IdentityPostCorrector}


def build_postcorrector(name: str) -> PostCorrector:
    key = (name or "none").strip().lower()
    if key not in POSTCORRECTORS:
        raise ValueError(f"Unknown POSTCORRECT={name!r}; available: {', '.join(POSTCORRECTORS)}")
    return POSTCORRECTORS[key]()
