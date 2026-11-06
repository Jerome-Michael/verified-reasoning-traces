"""Evaluating the filter itself.

1. Fault injection ("stress"): take traces the filter accepted, break them in one known way each,
   and measure how often the intended check catches it. This gives recall per check on real text.
2. Blind audit: export a sample of traces the naive filter would keep (half accepted by our filter
   where possible, half rejected) without showing the decision, have a human or an LLM judge label
   them (judge_prompt.md is the rubric), then score the filter against the labels.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .checks import ScoredCompletion, score_completion
from .config import Config
from .data import QAItem
from .parsing import THINK_CLOSE
from .pipeline import RunResult, evaluate
from .reference import MoaReference, load_reference
from .report import wilson

Mutation = Callable[[str, QAItem, ScoredCompletion, MoaReference | None], tuple[str, str] | None]


def _before_close(text: str, insert: str) -> str:
    m = list(THINK_CLOSE.finditer(text))[-1]
    return text[: m.start()] + insert + text[m.start():]


def _flip_answer(text, item, sc, moa):
    other = next(k for k in sc.variant.options if k != sc.parse.answer)
    head, sep, _ = text.rpartition("</think>")
    return f"{head}{sep}\n\nAnswer: {other}", "stop"


def _truncate(text, item, sc, moa):
    cut = int(text.find("</think>") * 0.6)
    return text[:cut], "length"


def _loop(text, item, sc, moa):
    sentences = re.split(r"(?<=[.!?])\s+", (sc.parse.reasoning or "").strip())
    chunk = " ".join(sentences[-2:])
    return _before_close(text, (" " + chunk) * 15 + "\n"), "stop"  # real loops run until the token limit


def _cjk(text, item, sc, moa):
    return _before_close(text, " 因此，该细胞系可能对该药物敏感。\n"), "stop"


def _citation(text, item, sc, moa):
    reasoning = sc.parse.reasoning or ""
    first = re.search(r"[.!?]\s", reasoning)
    if not first:
        return None
    return text.replace(reasoning[: first.start() + 1], reasoning[: first.start()] + " (Smith et al., 2019).", 1), "stop"


def _guess(text, item, sc, moa):
    return _before_close(text, " If I had to guess, I would still pick this option.\n"), "stop"


def _unknown_drug(text, item, sc, moa):
    drug = item.metadata.get("drug")
    if not drug:
        return None
    return text.replace("<think>", f"<think>\nI'm not familiar with {drug}, so I will reason generally.", 1), "stop"


def _wrong_mechanism(text, item, sc, moa):
    entry = moa.lookup(item.metadata.get("drug")) if moa else None
    if entry is None:
        return None
    kws = sorted(entry.keywords(), key=len, reverse=True)
    if any(k in {"egfr", "erbb1"} for k in kws):
        return None
    mutated = text
    for kw in kws:  # same matching rules as MoaReference.mentions
        pattern = rf"(?<![a-z0-9]){re.escape(kw)}s?(?![a-z0-9])" if len(kw) <= 4 else re.escape(kw)
        mutated = re.sub(pattern, "EGFR", mutated, flags=re.IGNORECASE)
    return mutated, "stop"


def _strip_tags(text, item, sc, moa):
    return re.sub(r"</?think>", "", text), "stop"


# name -> (mutation, check that should catch it). "parse_recovers" means it should NOT be rejected.
MUTATIONS: dict[str, tuple[Mutation, str]] = {
    "final answer flipped": (_flip_answer, "faithful"),
    "cut off mid-reasoning": (_truncate, "complete"),
    "repetition loop appended": (_loop, "no_loops"),
    "language switch (Chinese)": (_cjk, "language"),
    "fake citation inserted": (_citation, "no_citations"),
    "guess added to conclusion": (_guess, "decisive"),
    "admits not knowing the drug": (_unknown_drug, "knows_drug"),
    "mechanism replaced by a wrong one": (_wrong_mechanism, "mechanism_grounded"),
    "<think> tags removed": (_strip_tags, "parse_recovers"),
}


def stress_test(result: RunResult, moa: MoaReference | None, cfg: Config) -> dict:
    by_id = {it.question_id: it for it in result.items}
    # Base: every real trace that passed all hard checks (any question, any probe variant).
    accepted = [c for comps in result.scored.values() for c in comps if c.passed]
    rows = {}
    for name, (mutate, target) in MUTATIONS.items():
        n = caught = rejected = 0
        for sc in accepted:
            item = by_id[sc.gen.question_id]
            out = mutate(sc.gen.text, item, sc, moa)
            if out is None or out[0] == sc.gen.text:
                continue
            text, finish = out
            new = score_completion(replace(sc.gen, text=text, finish_reason=finish), item, sc.variant, cfg.filter, moa)
            n += 1
            if target == "parse_recovers":
                caught += new.parse.status == "recovered" and new.parse.answer == sc.parse.answer
            else:
                caught += target in new.hard_failures
                rejected += not new.passed
        rows[name] = {"target_check": target, "n": n, "caught_by_target": caught, "rejected_overall": rejected}
    return {"accepted_traces": len(accepted), "mutations": rows}


def cmd_stress(cfg: Config) -> int:
    from .data import load_private_dataset

    d = cfg.data
    items = load_private_dataset(d.n_questions, d.split, d.sampling, d.seed, d.repo)
    result = evaluate(items, cfg, offline=True)
    moa = load_reference(cfg.reference_dir, cfg.filter.moa_reference_url) if cfg.filter.use_moa_reference else None
    res = stress_test(result, moa, cfg)
    out = Path(cfg.out_dir)
    (out / "stress_test.json").write_text(json.dumps(res, indent=2))
    lines = ["# Filter stress test (fault injection)", "",
             f"Each of the {res['accepted_traces']} real traces that passed every check is broken in one known way and re-checked.", "",
             "| Fault injected | Check expected to catch it | Caught by that check | Rejected by any check |",
             "|---|---|---|---|"]
    for name, r in res["mutations"].items():
        if r["target_check"] == "parse_recovers":
            lines.append(f"| {name} | parser should *recover* it | {r['caught_by_target']}/{r['n']} recovered | n/a |")
        else:
            lines.append(f"| {name} | `{r['target_check']}` | {r['caught_by_target']}/{r['n']} | {r['rejected_overall']}/{r['n']} |")
    (out / "stress_test.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# ------------------------------------------------------------------ blind audit (human or LLM judge)

AUDIT_GUIDE = (
    "Label 1 if this is a trace you would want a model trained on: correct drug mechanism, no invented facts, "
    "the reasoning actually supports the final answer, and it commits to an answer. Otherwise 0."
)


JUDGE_PROMPT = Path(__file__).with_name("judge_prompt.md")


def _load_audit(out_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (out_dir / "audit.jsonl").read_text().splitlines() if line.strip()]


def cmd_audit_sheet(cfg: Config, size: int) -> int:
    out = Path(cfg.out_dir)
    rows = _load_audit(out)
    naive = [r for r in rows if r["variant"] == "original" and r["answer_on_original"] == r["ground_truth"]
             and "format" not in r["hard_failures"] and "complete" not in r["hard_failures"]]
    accepted = [r for r in naive if r["selected"]]
    rejected = [r for r in naive if not r["selected"]]
    rng = random.Random(cfg.data.seed)
    sample = rng.sample(accepted, min(size // 2, len(accepted))) + rng.sample(rejected, min(size - size // 2, len(rejected)))
    rng.shuffle(sample)
    audit_dir = out / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    with (audit_dir / "audit_sheet.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "question", "options", "ground_truth", "model_answer", "reasoning", "human_label", "notes"])
        for i, r in enumerate(sample):
            w.writerow([i, r["question"], json.dumps(r["options"]), r["ground_truth"], r["answer_text"], r["reasoning"], "", ""])
    with (audit_dir / "audit_key.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "key", "pipeline_decision", "hard_failures", "question_verdict"])
        for i, r in enumerate(sample):
            w.writerow([i, r["key"], "accept" if r["selected"] else "reject", "|".join(r["hard_failures"]), r["question_verdict"]])
    (audit_dir / "population.json").write_text(json.dumps({"accepted": len(accepted), "rejected": len(rejected)}))
    # Same sample as a readable page; labels go into the CSV's human_label column (1 = good, 0 = bad).
    md = [f"# Blind audit ({len(sample)} traces)", "", AUDIT_GUIDE, ""]
    for i, r in enumerate(sample):
        options = ", ".join(f"{k}: {v}" for k, v in r["options"].items())
        md += [f"## Trace {i}", "", f"**Question:** {r['question']}  ", f"**Options:** {options}  ",
               f"**Ground truth:** {r['ground_truth']} · **Model answered:** {r['answer_text']}", "",
               (r["reasoning"] or "").strip(), "", "**Label (1 good / 0 bad):** ___", ""]
    (audit_dir / "audit_sheet.md").write_text("\n".join(md))
    # Same sample again, behind the LLM-judge rubric: upload this one file to a judge model.
    judge = [JUDGE_PROMPT.read_text().rstrip(), ""]
    for i, r in enumerate(sample):
        options = ", ".join(f"{k}: {v}" for k, v in r["options"].items())
        judge += [f"## Trace {i}", "", f"**Question:** {r['question']}  ", f"**Options:** {options}  ",
                  f"**Correct answer:** {r['ground_truth']} · **Model's answer:** {r['answer_text']}", "",
                  "<reasoning>", (r["reasoning"] or "").strip(), "</reasoning>", ""]
    judge += ["---", "", "Reminder: write short notes for each trace, then end with the single CSV block of "
              f"{len(sample)} rows described at the top."]
    (audit_dir / "judge_input.md").write_text("\n".join(judge) + "\n")
    print(f"Wrote {len(sample)} traces to {audit_dir}/audit_sheet.csv (decisions hidden in audit_key.csv).")
    print(f"For an LLM judge: upload {audit_dir}/judge_input.md, save its CSV, then run "
          "traces audit-score --labels <csv> --labeller '<model>'.")
    print(f"Guide: {AUDIT_GUIDE}")
    return 0


def cmd_audit_score(cfg: Config, labels_path: str, labeller: str = "human") -> int:
    """Compare blind labels (1 good / 0 bad) with the filter's hidden decisions.

    The labels file needs `audit_id` and either `human_label` (the filled-in audit sheet) or `label`
    (an LLM judge's CSV). An optional `reason` or `notes` column is quoted for disagreements.
    """
    audit_dir = Path(cfg.out_dir) / "audit"
    key = {r["audit_id"]: r for r in csv.DictReader((audit_dir / "audit_key.csv").open())}
    pop = json.loads((audit_dir / "population.json").read_text())
    rows = list(csv.DictReader(open(labels_path)))
    col = "human_label" if rows and "human_label" in rows[0] else "label"
    labelled = sorted((r for r in rows if (r.get(col) or "").strip() in {"0", "1"} and r["audit_id"] in key),
                      key=lambda r: int(r["audit_id"]))
    if not labelled:
        print(f"No 0/1 labels found in column {col!r} of {labels_path}.")
        return 1

    good = {"accept": [0, 0], "reject": [0, 0]}  # decision -> [n labelled good, n]
    disagreements = []
    for r in labelled:
        k, label = key[r["audit_id"]], int(r[col])
        decision = k["pipeline_decision"]
        good[decision][0] += label
        good[decision][1] += 1
        if (decision == "accept") != bool(label):
            reasons = [f for f in k["hard_failures"].split("|") if f]
            if k["question_verdict"] != "robust":
                reasons.append(f"question {k['question_verdict']}")
            why = "passed every check" if decision == "accept" else ", ".join(reasons) or "quality or balance"
            note = (r.get("reason") or r.get("notes") or "").replace("|", "/").strip()
            confidence = f" ({r['confidence']} confidence)" if r.get("confidence") else ""
            disagreements.append(
                f"| {r['audit_id']} | {decision}ed, labelled {'good' if label else 'bad'}{confidence} | {why} | {note} |")
    (acc_k, acc_n), (rej_k, rej_n) = good["accept"], good["reject"]
    p_acc = acc_k / acc_n if acc_n else 0.0
    p_rej = rej_k / rej_n if rej_n else 0.0
    # The sample over-represents accepted traces; re-weight by the real population to estimate
    # the share of good traces a naive "correct answer" filter would keep.
    total = pop["accepted"] + pop["rejected"]
    naive_precision = (pop["accepted"] * p_acc + pop["rejected"] * p_rej) / total if total else 0.0
    agree = len(labelled) - len(disagreements)

    def ci(k: int, n: int) -> str:
        lo, hi = wilson(k, n)
        return f"{k / n:.0%} (95% CI {lo:.0%}–{hi:.0%})" if n else "n/a"

    lines = [
        "# Blind audit of the filter", "",
        f"Labelled by: {labeller}. {len(labelled)} traces, all with the correct final answer (so all kept by a naive "
        "filter). The labeller did not see the filter's decisions.", "",
        "| | Labelled good | Labelled bad | Share good |", "|---|---|---|---|",
        f"| Accepted by our filter | {acc_k} | {acc_n - acc_k} | {ci(acc_k, acc_n)} |",
        f"| Rejected by our filter | {rej_k} | {rej_n - rej_k} | {ci(rej_k, rej_n)} |",
        "",
        f"Agreement with the filter: {agree}/{len(labelled)} ({agree / len(labelled):.0%}).",
        f"Share of good traces, naive filter (population-weighted, {pop['accepted']} accepted / {pop['rejected']} "
        f"rejected): {naive_precision:.0%}.",
        f"Share of good traces, this filter: {p_acc:.0%}.",
    ]
    if disagreements:
        lines += ["", "| id | disagreement | filter's reason | labeller's reason |", "|---|---|---|---|", *disagreements]
    (audit_dir / "audit_results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0
