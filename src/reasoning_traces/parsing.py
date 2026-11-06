"""Tiered parser for model outputs: strict -> recovered -> failed.

Every repair is recorded as an issue, so nothing is fixed silently and the report can count them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple

THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
ANY_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)

# "Answer: B", "**Answer:** B", "Final answer: (B)", "The answer is option B", "Answer: B. Yes",
# and the non-standard "the decision is: A. Yes" / "my choice is B" seen with thinking mode off.
LETTER_ANSWER = re.compile(
    r"(?i:(?:final\s+)?(answer|decision|choice))[*_\s]*(?:(?i:is)\s*)?[:：]?[*_\s]*(?:(?i:option)\s*)?[\(\[]?\s*"
    r"([A-Z])(?![A-Za-z0-9])([^\n]*)"
)
# A last line that is just an option, e.g. "B. Yes". Accepted only if letter and text agree.
OPTION_LINE = re.compile(r"^[*_\s]*[\(\[]?([A-Z])[\)\]]?\s*[.:)\-]\s*(.+?)[\s.*_]*$")
BOXED = re.compile(r"\\boxed\{\s*([A-Z])\s*\}")
BARE_LETTER = re.compile(r"^[*_\s]*[\(\[]?([A-Z])[\)\]]?\.?[*_\s]*$")

# Issues that make the output unusable. Everything else is a recoverable formatting problem.
FATAL = {
    "empty_output", "unclosed_think", "empty_reasoning", "no_answer",
    "multiple_answers", "answer_not_in_options", "answer_text_conflict",
}


class _Answer(NamedTuple):
    pos: int
    letter: str
    how: str  # "letter" (Answer: X) | "boxed" | "keyword" (decision: X) | "text" (Answer: Yes) | "option_line"
    trailing: str


@dataclass
class ParseResult:
    status: str  # "strict" | "recovered" | "failed"
    reasoning: str | None
    answer: str | None
    issues: list[str] = field(default_factory=list)


def _find_answers(segment: str, options: dict[str, str]) -> list[_Answer]:
    found = [_Answer(m.start(), m.group(2), "letter" if m.group(1).lower() == "answer" else "keyword", m.group(3))
             for m in LETTER_ANSWER.finditer(segment)]
    found += [_Answer(m.start(), m.group(1), "boxed", "") for m in BOXED.finditer(segment)]
    # "Answer: Yes" -> map the option text back to its letter
    texts = {v.lower(): k for k, v in options.items()}
    if texts:
        alternatives = "|".join(re.escape(t) for t in sorted(texts, key=len, reverse=True))
        text_answer = re.compile(
            rf"(?i:(?:final\s+)?(?:answer|decision|choice))[*_\s]*(?:(?i:is)\s*)?[:：]?[*_\s\"']*({alternatives})\b",
            re.IGNORECASE
        )
        found += [_Answer(m.start(), texts[m.group(1).lower()], "text", "") for m in text_answer.finditer(segment)]
    lines = [m for m in re.finditer(r"[^\n]+", segment) if m.group(0).strip()]
    if lines:
        last = OPTION_LINE.match(lines[-1].group(0))
        if last and options.get(last.group(1), "").lower() == last.group(2).strip().lower():
            found.append(_Answer(lines[-1].start(), last.group(1), "option_line", ""))
    return sorted(found)


def _final_answer(tail: str, options: dict[str, str]) -> tuple[str | None, list[str]]:
    answers = _find_answers(tail, options)
    if not answers:
        bare = BARE_LETTER.match(tail.strip())
        if bare:
            answers = [_Answer(0, bare.group(1), "letter", "")]
    if not answers:
        return None, ["no_answer"]
    if len({a.letter for a in answers}) > 1:
        return None, ["multiple_answers"]
    letter = answers[-1].letter
    if letter not in options:
        return None, ["answer_not_in_options"]
    issues = []
    if all(a.how == "text" for a in answers):
        issues.append("answer_from_text")
    elif not any(a.how in {"letter", "boxed"} for a in answers):
        issues.append("nonstandard_answer_line")
    for a in answers:  # "Answer: B. No" when B is "Yes"
        named = {k for k, v in options.items() if re.search(rf"\b{re.escape(v)}\b", a.trailing, re.IGNORECASE)}
        if named and letter not in named:
            return None, ["answer_text_conflict"]
    return letter, issues


def parse_completion(text: str | None, options: dict[str, str], min_outside_words: int = 20) -> ParseResult:
    if not text or not text.strip():
        return ParseResult("failed", None, None, ["empty_output"])

    issues: list[str] = []
    opens = list(THINK_OPEN.finditer(text))
    closes = list(THINK_CLOSE.finditer(text))

    if closes:
        close = closes[-1]
        head, tail = text[: close.start()], text[close.end():]
        if opens and opens[0].start() < close.start():
            reasoning = head[opens[0].end():]
            if text[: opens[0].start()].strip():
                issues.append("text_before_think")
        else:
            issues.append("missing_think_open")  # e.g. the server pre-filled <think>
            reasoning = head
        if len(opens) > 1 or len(closes) > 1:
            issues.append("multiple_think_blocks")
        reasoning = ANY_THINK_TAG.sub(" ", reasoning).strip()
        if not reasoning:
            # Qwen3 with thinking disabled emits an empty <think></think> and reasons after it.
            answers = _find_answers(tail, options)
            cut = answers[-1].pos if answers else len(tail)
            outside = tail[:cut].strip()
            if len(outside.split()) >= min_outside_words:
                issues.append("reasoning_outside_tags")
                reasoning, tail = outside, tail[cut:]
            else:
                issues.append("empty_reasoning")
    elif opens:
        # <think> never closed: almost always the token limit cut the reasoning off.
        return ParseResult("failed", text[opens[0].end():].strip() or None, None, ["unclosed_think"])
    else:
        issues.append("no_think_tags")
        answers = _find_answers(text, options)
        if not answers:
            return ParseResult("failed", text.strip(), None, issues + ["no_answer"])
        cut = answers[-1].pos
        reasoning, tail = text[:cut].strip(), text[cut:]
        if len(reasoning.split()) < min_outside_words:
            issues.append("empty_reasoning")

    letter, answer_issues = _final_answer(tail, options)
    issues += answer_issues
    status = "failed" if FATAL.intersection(issues) else ("recovered" if issues else "strict")
    return ParseResult(status, reasoning or None, letter if status != "failed" else None, issues)
