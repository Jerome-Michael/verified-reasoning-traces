"""Stage 2: checks on a single completion.

Hard checks reject the completion. Soft checks only lower its quality score, which ranks the
survivors of a question. Each check returns a named result so the report can count every reason.
"""

from __future__ import annotations

import re
import zlib
from collections import Counter
from dataclasses import dataclass, field

from .config import FilterConfig
from .data import QAItem
from .generate import Generation
from .parsing import ParseResult, parse_completion
from .probes import Variant
from .reference import MoaReference, drug_components, norm_name


@dataclass
class Check:
    name: str
    passed: bool
    hard: bool
    detail: str = ""


@dataclass
class ScoredCompletion:
    gen: Generation
    variant: Variant
    parse: ParseResult
    chosen_text: str | None    # option text picked, as worded on this variant ("Yes"/"No")
    original_text: str | None  # the same answer expressed on the original question
    checks: list[Check] = field(default_factory=list)

    @property
    def hard_failures(self) -> list[str]:
        return [c.name for c in self.checks if c.hard and not c.passed]

    @property
    def soft_failures(self) -> list[str]:
        return [c.name for c in self.checks if not c.hard and not c.passed]

    @property
    def passed(self) -> bool:
        return not self.hard_failures

    @property
    def quality(self) -> float:
        penalty = 0.1 * len(self.soft_failures) + (0.05 if self.parse.status == "recovered" else 0.0)
        return round(1.0 - penalty, 3)


# ---------------------------------------------------------------- patterns

CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
FAKE_REFS = [
    re.compile(r"\bet al\b|\bPMID\b|\bdoi\s*[:/]|https?://|www\.", re.IGNORECASE),
    re.compile(r"\([A-Z][A-Za-z\-]+(?: et al\.?)?,? (?:19|20)\d{2}\)"),  # (Smith, 2019)
]
# A conclusion that admits it is a coin flip. Checked only in the last few sentences.
# ("educated guess" is deliberately *not* here: in practice it introduces reasoning from the drug
# class, which is what we want; the question-level probes already catch answers that are lucky.)
GUESS = re.compile(
    r"\b(?:if I (?:have|had|must|need) to (?:choose|guess|pick|decide)|I(?:'ll| will| would) (?:have to )?guess"
    r"|(?:best|random) guess|cannot (?:be )?determined?|can't (?:be )?determined?"
    r"|(?:not|im)possible to (?:determine|say|know)|insufficient (?:information|data|evidence)"
    r"|not enough (?:information|data|evidence)|coin[ -]?flip|toss[ -]?up|50/50"
    r"|does not support a (?:clear )?conclusion)",
    re.IGNORECASE,
)
WEAK_HEDGE = re.compile(
    r"\b(?:not (?:entirely |completely |very |100% )?sure|uncertain|I don't (?:know|recall|remember)"
    r"|can(?:'t|not) be (?:sure|certain)|not certain|unclear|educated guess)\b",
    re.IGNORECASE,
)
DRUG_UNKNOWN = re.compile(
    r"\b(?:not familiar with|unfamiliar with|don't recognize|do not recognize|not (?:a )?well[- ]known"
    r"|not well[- ]characteri[sz]ed|I'?m not sure what|don't know what|no information (?:on|about))",
    re.IGNORECASE,
)
META_TALK = re.compile(
    r"\bthe user\b|\bas an AI\b|\bI (?:can't|cannot|can not|don't have(?: the ability to)?) "
    r"(?:access|look (?:it |this )?up|browse|search|check (?:external|online|databases))"
    r"|\bexternal (?:resources|databases)\b",
    re.IGNORECASE,
)
CONCENTRATION = re.compile(r"(\d+(?:\.\d+)?)\s*(pM|nM|µM|μM|uM|mM)\b")
TO_NM = {"pm": 1e-3, "nm": 1.0, "µm": 1e3, "μm": 1e3, "um": 1e3, "mm": 1e6}
THRESHOLDS_NM = {100.0, 1000.0}  # the two thresholds given in the prompt

# Conclusion stance. Explicit statements win over sensitivity wording.
LETTER_STANCE = re.compile(
    r"(?i:answer|option|choice|choose|select|go with|pick)[*_\s]*(?:(?i:is|would be|should be|=|:)\s*)?"
    r"[*_\s]*(?:(?i:option)\s*)?[\(\[]?([A-Z])(?![A-Za-z0-9])"
)
NOT_SENSITIVE = re.compile(
    r"\b(?:not (?:be |likely be |likely to be )?(?:very |highly )?sensitive|insensitive|resistant|resistance"
    r"|not respond|unlikely to (?:be sensitive|respond)|lacks? sensitivity)\b",
    re.IGNORECASE,
)
SENSITIVE = re.compile(
    r"\b(?:(?:is|be|are|likely|highly|very|quite|would be|should be) sensitive|responsive|would respond"
    r"|likely to respond)\b",
    re.IGNORECASE,
)
DEFINITION_CONTEXT = re.compile(r"^\s*(?:if\b|\(|:|means|is defined|requires|threshold|=|<|>)", re.IGNORECASE)


# ---------------------------------------------------------------- helpers

def conclusion_window(text: str, fraction: float = 0.35, min_chars: int = 800) -> str:
    return text[-max(int(len(text) * fraction), min_chars):]


def _sensitivity_stance(window: str) -> list[tuple[int, bool]]:
    """(position, is_sensitive) for each stance phrase that is not a restatement of the definition."""
    found, negative_spans = [], []
    for m in NOT_SENSITIVE.finditer(window):
        if not DEFINITION_CONTEXT.match(window[m.end(): m.end() + 15]):
            found.append((m.start(), False))
        negative_spans.append((m.start(), m.end()))
    for m in SENSITIVE.finditer(window):
        inside_negative = any(s <= m.start() < e for s, e in negative_spans)
        if not inside_negative and not DEFINITION_CONTEXT.match(window[m.end(): m.end() + 15]):
            found.append((m.start(), True))
    return found


def reasoning_stance(reasoning: str, variant: Variant) -> str | None:
    """Which option the reasoning itself concludes, as an option text of the variant, or None."""
    window = conclusion_window(reasoning)
    explicit: list[tuple[int, str]] = []
    for m in LETTER_STANCE.finditer(window):
        if m.group(1) in variant.options:
            explicit.append((m.start(), variant.options[m.group(1)]))
    texts = "|".join(re.escape(t) for t in variant.options.values())
    for m in re.finditer(rf"(?i:answer|conclusion)\s*(?:is|would be|should be)?\s*[:：]?[*_\s\"']*({texts})\b", window, re.I):
        explicit.append((m.start(), next(t for t in variant.options.values() if t.lower() == m.group(1).lower())))
    if explicit:
        return max(explicit)[1]
    if set(variant.options.values()) == {"Yes", "No"}:
        stances = _sensitivity_stance(window)
        if stances:
            is_sensitive = max(stances)[1]
            asks_resistance = variant.kind == "negated"
            return "Yes" if is_sensitive != asks_resistance else "No"
    return None


def compression_ratio(text: str) -> float:
    raw = text.encode()
    return len(raw) / max(len(zlib.compress(raw)), 1)


def max_shingle_repeats(text: str, n: int = 8) -> int:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return 0
    counts = Counter(tuple(words[i: i + n]) for i in range(len(words) - n + 1))
    return max(counts.values())


def _mentions(text: str, name: str) -> bool:
    return norm_name(name) in re.sub(r"[^a-z0-9]", "", text.lower())


# ---------------------------------------------------------------- main entry

def score_completion(
    gen: Generation, item: QAItem, variant: Variant, cfg: FilterConfig, moa: MoaReference | None
) -> ScoredCompletion:
    parse = parse_completion(gen.text, variant.options)
    chosen = variant.options.get(parse.answer) if parse.answer else None
    sc = ScoredCompletion(gen, variant, parse, chosen, variant.original_text_for(parse.answer))
    add = sc.checks.append

    format_ok = parse.status == "strict" or (parse.status == "recovered" and cfg.allow_recovered_format)
    add(Check("format", format_ok, True, ",".join(parse.issues) or parse.status))
    add(Check("complete", gen.finish_reason != "length", True, f"finish_reason={gen.finish_reason}"))
    add(Check("correct", sc.original_text == item.answer_text, True,
              f"model={sc.original_text} truth={item.answer_text}"))

    reasoning = parse.reasoning
    if not reasoning:
        return sc

    stance = reasoning_stance(reasoning, variant)
    if stance is None:
        add(Check("explicit_conclusion", False, False, "reasoning never states which option it favours"))
    else:
        add(Check("faithful", chosen is None or stance == chosen, True, f"reasoning concludes {stance}, answer {chosen}"))

    tail = reasoning[-600:]
    guess = GUESS.search(tail)
    add(Check("decisive", guess is None, True, f"guess marker: {guess.group(0)!r}" if guess else ""))

    n_words = len(reasoning.split())
    add(Check("length", cfg.min_reasoning_words <= n_words <= cfg.max_reasoning_words, True, f"{n_words} words"))

    cjk = CJK.findall(reasoning)
    add(Check("language", not cjk, True, f"{len(cjk)} CJK characters" if cjk else ""))

    ratio, repeats = compression_ratio(reasoning), max_shingle_repeats(reasoning)
    add(Check("no_loops", ratio <= cfg.max_compression_ratio and repeats <= cfg.max_shingle_repeats, True,
              f"compression={ratio:.2f} max_8gram_repeats={repeats}"))
    add(Check("not_repetitive", repeats < cfg.soft_shingle_repeats, False, f"max_8gram_repeats={repeats}"))

    refs = [m.group(0) for p in FAKE_REFS for m in p.finditer(reasoning)]
    add(Check("no_citations", not refs, True, f"{refs[:3]}" if refs else ""))

    drug, cell_line = item.metadata.get("drug"), item.metadata.get("cell_line")
    if drug:
        components = drug_components(drug) or [drug]  # combinations: every component must be handled
        missing = [c for c in components if not _mentions(reasoning, c)]
        add(Check("mentions_drug", not missing, True, f"not mentioned: {missing}" if missing else drug))
        unknown = [m.group(0) for m in DRUG_UNKNOWN.finditer(reasoning)
                   if any(_mentions(reasoning[max(0, m.start() - 80): m.end() + 80], c) for c in components)
                   or re.search(r"drug|compound", reasoning[m.end(): m.end() + 40], re.I)]
        add(Check("knows_drug", not unknown, True, f"{unknown[:2]}" if unknown else ""))
        if moa is not None:
            entries = [(c, moa.lookup(c)) for c in components]
            # Drop the drug names first, so "I-BRD9" cannot vouch for a BRD9 mechanism by itself.
            claims = reasoning
            for c in components:
                claims = re.sub(re.escape(c), " ", claims, flags=re.IGNORECASE)
            checked = [(c, e, MoaReference.mentions(claims, e.keywords())) for c, e in entries if e is not None]
            unverifiable = [c for c, e in entries if e is None]
            # Hand audit of the first full run: 3 of the 4 kept traces about drugs missing from the
            # reference stated a wrong mechanism, so an unverifiable mechanism is not accepted by default.
            add(Check("mechanism_verifiable", not unverifiable, cfg.require_verifiable_mechanism,
                      f"not in reference: {unverifiable}" if unverifiable else ""))
            if checked:
                ungrounded = [f"{c} (reference: {', '.join(e.moa)})" for c, e, hits in checked if not hits]
                matched = sorted({h for _, _, hits in checked for h in hits})
                add(Check("mechanism_grounded", not ungrounded, True,
                          f"no mechanism match for {ungrounded}" if ungrounded else f"matched {matched[:3]}"))
    if cell_line:
        add(Check("mentions_cell_line", _mentions(reasoning, cell_line), False, cell_line))

    values = [float(v) * TO_NM[u.lower()] for v, u in CONCENTRATION.findall(reasoning)]
    invented = [v for v in values if round(v, 6) not in THRESHOLDS_NM]
    add(Check("few_specific_values", len(invented) < 2, False, f"{len(invented)} non-threshold concentrations"))

    hedges = WEAK_HEDGE.findall(reasoning)
    add(Check("low_hedging", len(hedges) < 3, False, f"{len(hedges)} hedges"))
    meta = META_TALK.findall(reasoning)
    add(Check("no_meta_talk", not meta, False, f"{meta[:2]}" if meta else ""))
    return sc
