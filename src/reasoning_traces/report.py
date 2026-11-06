"""Run metrics and a human-readable report.md: what the model did and what the filter did about it."""

from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .checks import ScoredCompletion
from .config import Config
from .consensus import VERDICTS
from .pipeline import RunResult


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion. With n=50, expect roughly +/-14 points."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(centre - half, 3), round(centre + half, 3))


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n)}


def coin_flip_pass_probability(n_probes: int, threshold: float) -> float:
    """Chance that a model guessing 50/50 on every probe reaches the agreement threshold."""
    need = math.ceil(threshold * n_probes)
    return sum(math.comb(n_probes, k) for k in range(need, n_probes + 1)) / 2**n_probes


def threshold_sweep(result: RunResult, cfg: Config) -> list[dict[str, Any]]:
    """How yield trades against protection from lucky guesses as the agreement threshold moves."""
    n_probes = cfg.sampling.n_original + cfg.sampling.n_swapped + cfg.sampling.n_negated
    rows = []
    for t in (0.5, 0.75, 1.0):
        kept_q = kept_rows = 0
        for qid, v in result.verdicts.items():
            if v.n_valid >= cfg.filter.min_valid_probes and v.agreement >= t:
                passing = [c for c in result.scored[qid] if c.variant.kind == "original" and c.passed]
                kept_q += bool(passing)
                kept_rows += min(len(passing), cfg.filter.max_per_question)
        rows.append({"min_agreement": t, "questions_with_rows": kept_q, "rows_before_dedup": kept_rows,
                     "coin_flipper_pass_probability": round(coin_flip_pass_probability(n_probes, t), 3)})
    return rows


def build_metrics(result: RunResult, cfg: Config) -> dict[str, Any]:
    all_c = [c for comps in result.scored.values() for c in comps]
    orig = [c for c in all_c if c.variant.kind == "original"]
    by_id = {it.question_id: it for it in result.items}
    selected = [c for comps in result.selected.values() for c in comps]

    answered = [c for c in orig if c.original_text is not None]
    correct = [c for c in answered if c.original_text == by_id[c.gen.question_id].answer_text]
    said_yes = [c for c in answered if c.original_text == "Yes"]
    # Accuracy per ground-truth label: overall accuracy hides a model that just leans one way.
    per_label = {}
    for label in sorted({it.answer_text for it in result.items}):
        pool = [c for c in answered if by_id[c.gen.question_id].answer_text == label]
        per_label[label] = _rate(sum(c.original_text == label for c in pool), len(pool))
    rates = [r["rate"] for r in per_label.values() if r["rate"] is not None]
    balanced_accuracy = round(sum(rates) / len(rates), 3) if rates else None
    truth_yes = sum(1 for it in result.items if it.answer_text == "Yes")

    # Probe consistency, one pair per question (sample 0 of each kind), only when both are readable.
    def pair_consistency(kind: str) -> dict[str, Any]:
        agree = total = 0
        for comps in result.scored.values():
            o = next((c for c in comps if c.variant.kind == "original" and c.gen.sample_idx == 0), None)
            p = next((c for c in comps if c.variant.kind == kind and c.gen.sample_idx == 0), None)
            if o and p and o.original_text and p.original_text:
                total += 1
                agree += o.original_text == p.original_text
        return _rate(agree, total)

    # Funnel over original-order completions (the only dataset candidates).
    fmt_ok = [c for c in orig if "format" not in c.hard_failures]
    complete = [c for c in fmt_ok if "complete" not in c.hard_failures]
    naive = [c for c in complete if "correct" not in c.hard_failures]  # what "answer matches truth" would keep
    clean = [c for c in naive if c.passed]
    robust = [c for c in clean if result.verdicts[c.gen.question_id].verdict == "robust"]

    # Why did the naive filter's picks not make it?
    selected_keys = {c.gen.key for c in selected}
    before_balance_keys = {c.gen.key for comps in result.selected_before_balance.values() for c in comps}
    rejected_from_naive: Counter[str] = Counter()
    for c in naive:
        if c.gen.key in selected_keys:
            continue
        if not c.passed:
            rejected_from_naive.update(f"trace: {f}" for f in c.hard_failures)
        elif result.verdicts[c.gen.question_id].verdict != "robust":
            rejected_from_naive[f"question: {result.verdicts[c.gen.question_id].verdict}"] += 1
        elif c.quality < cfg.filter.min_quality:
            rejected_from_naive["trace: too many soft failures"] += 1
        elif c.gen.key in before_balance_keys:
            rejected_from_naive["dataset: Yes/No balance"] += 1
        else:
            rejected_from_naive["dedup / per-question cap"] += 1

    hard = Counter(f for c in orig for f in c.hard_failures)
    soft = Counter(f for c in orig for f in c.soft_failures)
    words = [len((c.parse.reasoning or "").split()) for c in selected]
    tokens = [c.gen.completion_tokens for c in all_c if c.gen.completion_tokens]
    latency = [c.gen.latency_s for c in all_c if c.gen.latency_s]
    grounded = [c for c in selected if any(k.name == "mechanism_grounded" for k in c.checks)]

    return {
        "run": {
            "model": cfg.llm.model, "reasoning_mode": cfg.llm.reasoning_mode, "prompt_version": cfg.llm.prompt_version,
            "n_questions": len(result.items), "sampling": cfg.data.sampling, "seed": cfg.data.seed,
            "samples_per_question": {"original": cfg.sampling.n_original, "swapped": cfg.sampling.n_swapped,
                                     "negated": cfg.sampling.n_negated},
            "mechanism_reference": result.moa_available,
        },
        "generator": {
            "completions": len(all_c),
            "errors": sum(1 for c in all_c if c.gen.error),
            "parse_status": dict(Counter(c.parse.status for c in all_c)),
            "parse_issues": dict(Counter(i for c in all_c for i in c.parse.issues)),
            "truncated": sum(1 for c in all_c if c.gen.finish_reason == "length"),
            "accuracy_original": _rate(len(correct), len(answered)),
            "accuracy_by_truth": per_label,
            "balanced_accuracy": balanced_accuracy,
            "yes_rate_original": _rate(len(said_yes), len(answered)),
            "yes_rate_ground_truth": _rate(truth_yes, len(result.items)),
            "swap_consistency": pair_consistency("swapped"),
            "negation_consistency": pair_consistency("negated"),
            "mean_completion_tokens": round(statistics.mean(tokens)) if tokens else None,
            "mean_latency_s": round(statistics.mean(latency), 1) if latency else None,
        },
        "question_verdicts": dict(Counter(v.verdict for v in result.verdicts.values())),
        "funnel_original_completions": {
            "generated": len(orig), "format_ok": len(fmt_ok), "complete": len(complete),
            "correct (naive filter)": len(naive), "passed trace checks": len(clean),
            "robust question": len(robust), "best per question (≤2, de-duplicated)": len(before_balance_keys),
            "after Yes/No balance (final)": len(selected),
        },
        "hard_failures_original": dict(hard.most_common()),
        "soft_failures_original": dict(soft.most_common()),
        "naive_accepted_but_rejected": dict(rejected_from_naive.most_common()),
        "agreement_threshold_sweep": threshold_sweep(result, cfg),
        "dataset": {
            "rows": len(selected),
            "questions_covered": sum(1 for v in result.selected.values() if v),
            "label_balance": dict(Counter(by_id[c.gen.question_id].answer_text for c in selected)),
            "letter_balance": dict(Counter(c.parse.answer for c in selected)),
            "reasoning_words": {"min": min(words), "median": statistics.median(words), "max": max(words)} if words else None,
            "mechanism_checked_rows": len(grounded),
        },
    }


def _table(rows: list[tuple], header: tuple) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(r: dict[str, Any]) -> str:
    if r["rate"] is None:
        return "n/a"
    lo, hi = r["ci95"]
    return f"{r['rate']:.0%} ({r['k']}/{r['n']}, 95% CI {lo:.0%}–{hi:.0%})"


def _example(c: ScoredCompletion, limit: int = 900) -> str:
    text = (c.parse.reasoning or c.gen.text or "").strip()
    text = text if len(text) <= limit else text[:limit] + " […]"
    return "\n".join("> " + line for line in text.splitlines() if line.strip())


def write_report(result: RunResult, cfg: Config, m: dict[str, Any], path: Path) -> None:
    g, run = m["generator"], m["run"]
    lines = [
        "# Reasoning-trace generation report", "",
        f"Model `{run['model']}` · mode `{run['reasoning_mode']}` · prompt `{run['prompt_version']}` · "
        f"{run['n_questions']} questions ({run['sampling']}, seed {run['seed']}) · samples per question "
        f"{run['samples_per_question']} · mechanism reference {'on' if run['mechanism_reference'] else 'OFF'}", "",
        "## Generator", "",
        _table([
            ("Completions (all variants)", g["completions"]),
            ("Request errors", g["errors"]),
            ("Parse status", ", ".join(f"{k}: {v}" for k, v in g["parse_status"].items())),
            ("Truncated at max_tokens", g["truncated"]),
            ("Accuracy on original questions", _pct(g["accuracy_original"])),
            *[(f"Accuracy when the truth is {k}", _pct(v)) for k, v in g["accuracy_by_truth"].items()],
            ("Balanced accuracy (mean of the above)", f"{g['balanced_accuracy']:.0%}" if g["balanced_accuracy"] is not None else "n/a"),
            ("Model says Yes", _pct(g["yes_rate_original"])),
            ("Ground truth is Yes", _pct(g["yes_rate_ground_truth"])),
            ("Same answer when options swapped", _pct(g["swap_consistency"])),
            ("Answer flips when question negated", _pct(g["negation_consistency"])),
            ("Mean completion tokens / latency", f"{g['mean_completion_tokens']} tok / {g['mean_latency_s']} s"),
        ], ("Metric", "Value")), "",
        "Parse issues (all completions): " + (", ".join(f"`{k}` {v}" for k, v in g["parse_issues"].items()) or "none"), "",
        "## Question verdicts", "",
        _table([(f"`{k}`", m["question_verdicts"].get(k, 0), desc) for k, desc in VERDICTS.items()],
               ("Verdict", "Questions", "Meaning")), "",
        "## Funnel (original-order completions)", "",
        _table(list(m["funnel_original_completions"].items()), ("Stage", "Completions")), "",
        "## Why the naive filter's picks were rejected", "",
        "A naive filter keeps every well-formed completion whose answer matches the ground truth. "
        "These are the reasons ours dropped some of them:", "",
        _table(list(m["naive_accepted_but_rejected"].items()) or [("none", 0)], ("Reason", "Completions")), "",
        "## Agreement threshold: yield vs protection from lucky guesses", "",
        "A question is trusted only if this share of its probes lands on the ground truth. The last column is the "
        "chance that a model guessing 50/50 would still pass.", "",
        _table([(r["min_agreement"], r["questions_with_rows"], r["rows_before_dedup"],
                 f"{r['coin_flipper_pass_probability']:.0%}") for r in m["agreement_threshold_sweep"]],
               ("Min agreement", "Questions with rows", "Rows (before de-dup)", "Coin-flipper passes")), "",
        "## Check failures on original-order completions", "",
        _table([(f"`{k}`", "hard", v) for k, v in m["hard_failures_original"].items()]
               + [(f"`{k}`", "soft", v) for k, v in m["soft_failures_original"].items()],
               ("Check", "Type", "Completions failing")), "",
        "## Final dataset", "",
        _table(list(m["dataset"].items()), ("Property", "Value")), "",
    ]

    selected = [c for comps in result.selected.values() for c in comps]
    if selected:
        best = max(selected, key=lambda c: c.quality)
        lines += ["## Example: accepted trace", "", f"Question {best.gen.question_id}: {best.variant.question}", "",
                  _example(best), "", f"Answer: {best.parse.answer} ({best.chosen_text})", ""]
    shown: set[str] = set()
    rejected_examples = []
    for comps in result.scored.values():
        for c in comps:
            if c.variant.kind == "original" and c.hard_failures and c.parse.reasoning:
                reason = next(f for f in c.hard_failures if f != "correct") if set(c.hard_failures) - {"correct"} else "correct"
                if reason not in shown and len(rejected_examples) < 4:
                    shown.add(reason)
                    rejected_examples.append((reason, c))
    if rejected_examples:
        lines += ["## Examples: rejected traces", ""]
        for reason, c in rejected_examples:
            detail = next((k.detail for k in c.checks if k.name == reason), "")
            lines += [f"**`{reason}`** ({detail}) — question {c.gen.question_id}: {c.variant.question}", "",
                      _example(c, 600), ""]
    path.write_text("\n".join(lines))
