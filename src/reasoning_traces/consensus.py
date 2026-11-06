"""Stage 3: does the model actually know the answer to this question?

On a two-option question a model with no knowledge is right half the time, so a correct final
answer alone is weak evidence. We ask each question several ways (original, options swapped,
question negated) and only trust questions where every view lands on the ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from .checks import ScoredCompletion
from .config import FilterConfig
from .data import QAItem

VERDICTS = {
    "robust": "right answer under every phrasing and option order",
    "unstable": "different answers to the identical question across samples (a coin flip)",
    "framing_bias": "same Yes/No for 'sensitive?' and 'resistant?' (agrees instead of reasoning)",
    "position_bias": "picks the same letter when the option order is swapped",
    "confidently_wrong": "consistently contradicts the ground truth (knowledge gap or label noise)",
    "insufficient": "too few parseable answers to judge",
}


@dataclass
class QuestionVerdict:
    question_id: int
    verdict: str
    n_valid: int
    n_agree: int

    @property
    def agreement(self) -> float:
        return self.n_agree / self.n_valid if self.n_valid else 0.0


def judge_question(item: QAItem, completions: list[ScoredCompletion], cfg: FilterConfig) -> QuestionVerdict:
    # A probe counts if we could read an answer from it (truncated or malformed outputs do not vote).
    valid = [c for c in completions if c.original_text is not None and "complete" not in c.hard_failures]
    agree = [c for c in valid if c.original_text == item.answer_text]

    def verdict(name: str) -> QuestionVerdict:
        return QuestionVerdict(item.question_id, name, len(valid), len(agree))

    if len(valid) < cfg.min_valid_probes:
        return verdict("insufficient")
    if len(agree) / len(valid) >= cfg.min_probe_agreement:
        return verdict("robust")
    if not agree:
        return verdict("confidently_wrong")

    by_kind: dict[str, list[ScoredCompletion]] = {}
    for c in valid:
        by_kind.setdefault(c.variant.kind, []).append(c)
    originals = by_kind.get("original", [])
    # Different answers to the identical prompt: the model is sampling, not knowing.
    if len({o.chosen_text for o in originals}) > 1:
        return verdict("unstable")
    # Consistent on the original, but said "Yes" (or "No") both to "sensitive?" and to "resistant?"
    if any(o.chosen_text == n.chosen_text for o in originals for n in by_kind.get("negated", [])):
        return verdict("framing_bias")
    # Consistent on the original, but picked the same letter although the option texts were swapped
    if any(o.parse.answer == s.parse.answer for o in originals for s in by_kind.get("swapped", [])):
        return verdict("position_bias")
    return verdict("unstable")
