"""End-to-end pipeline: load -> generate (cached) -> check -> judge questions -> select -> export + report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .checks import ScoredCompletion, score_completion
from .config import Config
from .consensus import QuestionVerdict, judge_question
from .data import QAItem, load_private_dataset
from .generate import generate, plan_tasks
from .prompts import TRAINING_SYSTEM_PROMPT, build_messages, render_full_prompt, user_message
from .reference import load_reference
from .schema import SFTRecord
from .selection import balance_by_label, select_for_question


@dataclass
class RunResult:
    items: list[QAItem]
    scored: dict[int, list[ScoredCompletion]]
    verdicts: dict[int, QuestionVerdict]
    selected: dict[int, list[ScoredCompletion]]
    records: list[dict[str, Any]] = field(default_factory=list)
    moa_available: bool = False
    selected_before_balance: dict[int, list[ScoredCompletion]] = field(default_factory=dict)


def evaluate(items: list[QAItem], cfg: Config, offline: bool = False) -> RunResult:
    """Everything after loading the data. offline=True re-scores cached generations without the model."""
    tasks = plan_tasks(items, cfg)
    gens = generate(tasks, cfg, offline=offline)
    moa = load_reference(cfg.reference_dir, cfg.filter.moa_reference_url) if cfg.filter.use_moa_reference else None

    by_id = {it.question_id: it for it in items}
    scored: dict[int, list[ScoredCompletion]] = {it.question_id: [] for it in items}
    for task, gen in zip(tasks, gens, strict=False):
        scored[task.question_id].append(score_completion(gen, by_id[task.question_id], task.variant, cfg.filter, moa))

    verdicts, before_balance, records = {}, {}, []
    for item in items:
        verdicts[item.question_id] = judge_question(item, scored[item.question_id], cfg.filter)
        before_balance[item.question_id] = select_for_question(scored[item.question_id], verdicts[item.question_id], cfg.filter)
    selected = (balance_by_label(before_balance, {it.question_id: it.answer_text for it in items})
                if cfg.filter.balance_labels else before_balance)
    for item in items:
        for c in selected[item.question_id]:
            record = SFTRecord.model_validate({
                "question_id": item.question_id,
                "completion_id": c.gen.sample_idx,  # index of the original-order sample, traceable in audit.jsonl
                "hf_sample": item.hf_sample(),
                "completion": {"reasoning": c.parse.reasoning, "answer": c.parse.answer},
                "prompting": {"full_prompt": render_full_prompt(build_messages(c.variant, cfg.llm.reasoning_mode, cfg.llm.prompt_version))},
            })
            records.append(record.model_dump())
    return RunResult(items, scored, verdicts, selected, records, moa_available=moa is not None,
                     selected_before_balance=before_balance)


def _audit_rows(result: RunResult) -> list[dict[str, Any]]:
    by_id = {it.question_id: it for it in result.items}
    rows = []
    for qid, comps in result.scored.items():
        chosen_keys = {c.gen.key for c in result.selected[qid]}
        for c in comps:
            rows.append({
                "key": c.gen.key, "question_id": qid, "variant": c.variant.kind, "sample_idx": c.gen.sample_idx,
                "question": c.variant.question, "options": c.variant.options,
                "ground_truth": by_id[qid].answer_text, "answer": c.parse.answer, "answer_text": c.chosen_text,
                "answer_on_original": c.original_text, "parse_status": c.parse.status, "parse_issues": c.parse.issues,
                "passed_checks": c.passed, "hard_failures": c.hard_failures, "soft_failures": c.soft_failures,
                "quality": c.quality, "question_verdict": result.verdicts[qid].verdict,
                "selected": c.gen.key in chosen_keys,
                "checks": {k.name: {"passed": k.passed, "hard": k.hard, "detail": k.detail} for k in c.checks},
                "finish_reason": c.gen.finish_reason, "completion_tokens": c.gen.completion_tokens,
                "latency_s": c.gen.latency_s, "error": c.gen.error, "reasoning": c.parse.reasoning, "raw_text": c.gen.text,
            })
    return rows


def _chat_rows(result: RunResult) -> list[dict[str, Any]]:
    rows = []
    for qid, comps in result.selected.items():
        for c in comps:
            rows.append({
                "question_id": qid, "completion_id": c.gen.sample_idx,
                "messages": [
                    {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message(c.variant)},
                    {"role": "assistant", "content": f"<think>\n{c.parse.reasoning}\n</think>\n\nAnswer: {c.parse.answer}"},
                ],
            })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export(result: RunResult, cfg: Config) -> Path:
    from .report import build_metrics, write_report

    out = Path(cfg.out_dir)
    write_jsonl(out / "sft_dataset.jsonl", result.records)
    write_jsonl(out / "sft_chat.jsonl", _chat_rows(result))
    write_jsonl(out / "audit.jsonl", _audit_rows(result))
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    metrics = build_metrics(result, cfg)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_report(result, cfg, metrics, out / "report.md")
    return out


def run(cfg: Config, items: list[QAItem] | None = None, offline: bool = False) -> RunResult:
    if items is None:
        d = cfg.data
        items = load_private_dataset(d.n_questions, d.split, d.sampling, d.seed, d.repo)
    result = evaluate(items, cfg, offline=offline)
    export(result, cfg)
    return result


def generate_reasoning_traces(qa_samples: list[Any], cfg: Config | None = None) -> list[dict[str, Any]]:
    """Entry point matching the starter stub: Q&A samples in, validated SFT rows out (also written to disk)."""
    cfg = cfg or Config.load()
    items = [
        s if isinstance(s, QAItem) else QAItem.from_row(int(s.get("question_id", i)), s)
        for i, s in enumerate(qa_samples)
    ]
    return run(cfg, items).records
