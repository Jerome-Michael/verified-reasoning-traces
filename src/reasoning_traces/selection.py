"""Stage 4: choose which surviving traces go into the dataset."""

from __future__ import annotations

import re

from .checks import ScoredCompletion
from .config import FilterConfig
from .consensus import QuestionVerdict


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i: i + n]) for i in range(max(len(words) - n + 1, 1))}


def jaccard(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    return len(sa & sb) / max(len(sa | sb), 1)


def select_for_question(
    completions: list[ScoredCompletion], verdict: QuestionVerdict, cfg: FilterConfig
) -> list[ScoredCompletion]:
    """Keep up to max_per_question passing original-order traces of a robust question, best first, no near-duplicates."""
    if verdict.verdict != "robust":
        return []
    candidates = [c for c in completions if c.variant.kind == "original" and c.passed and c.quality >= cfg.min_quality]
    candidates.sort(key=lambda c: (-c.quality, c.gen.sample_idx))
    kept: list[ScoredCompletion] = []
    for c in candidates:
        if len(kept) >= cfg.max_per_question:
            break
        if all(jaccard(c.parse.reasoning or "", k.parse.reasoning or "") < cfg.dedup_jaccard for k in kept):
            kept.append(c)
    return kept


def balance_by_label(
    selected: dict[int, list[ScoredCompletion]], truth: dict[int, str]
) -> dict[int, list[ScoredCompletion]]:
    """Same number of rows for every ground-truth label.

    The probes catch bias in *how* a question is asked, not a content prior: a model convinced that
    "most drugs fail at 100 nM" answers No consistently and passes the probes on every No question.
    Without balancing, the dataset would teach that prior. Within a label, each question's best trace
    goes first, so balancing removes second traces before it removes questions.
    """
    labels = set(truth.values())
    pools: dict[str, list[tuple[int, float, int, ScoredCompletion]]] = {label: [] for label in labels}
    for qid, comps in selected.items():
        for rank, c in enumerate(comps):
            pools[truth[qid]].append((rank, -c.quality, qid, c))
    k = min(len(pool) for pool in pools.values()) if pools else 0
    keep = {id(c) for pool in pools.values() for *_, c in sorted(pool, key=lambda t: t[:3])[:k]}
    return {qid: [c for c in comps if id(c) in keep] for qid, comps in selected.items()}
