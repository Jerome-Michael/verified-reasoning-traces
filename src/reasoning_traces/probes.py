"""Question variants used to test whether the model *knows* an answer or just landed on it.

Every variant records how a letter chosen on it maps back to an option text of the original
question, so answers from all variants can be compared on the same scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import QAItem

YES_NO_FLIP = {"Yes": "No", "No": "Yes"}
# Rephrasings whose correct answer is the opposite of the original question's answer.
NEGATIONS = [("be sensitive to", "be resistant to")]


@dataclass(frozen=True)
class Variant:
    kind: str  # "original" | "swapped" | "negated"
    question: str
    options: dict[str, str]
    letter_to_original: dict[str, str]  # letter on this variant -> equivalent original option text

    def original_text_for(self, letter: str | None) -> str | None:
        return self.letter_to_original.get(letter) if letter else None


def original(item: QAItem) -> Variant:
    return Variant("original", item.question, dict(item.options), dict(item.options))


def swapped(item: QAItem) -> Variant | None:
    """Same question, option texts in reverse order. A model that knows the answer follows the content."""
    if len(item.options) < 2:
        return None
    keys = list(item.options)
    opts = dict(zip(keys, reversed(list(item.options.values())), strict=False))
    return Variant("swapped", item.question, opts, dict(opts))


def negated(item: QAItem) -> Variant | None:
    """'Would X be resistant to Y?' A model that says Yes to both phrasings is agreeing, not reasoning."""
    if set(item.options.values()) != {"Yes", "No"}:
        return None
    for phrase, opposite in NEGATIONS:
        if phrase in item.question:
            question = item.question.replace(phrase, opposite, 1)
            mapping = {k: YES_NO_FLIP[v] for k, v in item.options.items()}
            return Variant("negated", question, dict(item.options), mapping)
    return None


BUILDERS = {"original": original, "swapped": swapped, "negated": negated}
