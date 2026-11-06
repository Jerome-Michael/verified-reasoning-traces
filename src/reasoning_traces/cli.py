"""Command line: `traces check | run | filter | stress | audit-sheet | audit-score`."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import Config


def _config(args: argparse.Namespace) -> Config:
    cfg = Config.load(args.config)
    if args.smoke:
        cfg.data.n_questions, cfg.out_dir = 6, "outputs/smoke"
    if args.n is not None:
        cfg.data.n_questions = args.n
    if args.sampling:
        cfg.data.sampling = args.sampling
    if args.split:
        cfg.data.split = args.split
    if args.prompt:
        cfg.llm.prompt_version = args.prompt
    if args.mode:
        cfg.llm.reasoning_mode = args.mode
    if args.model:
        cfg.llm.model = args.model
    if args.concurrency:
        cfg.llm.concurrency = args.concurrency
    if args.out_dir:
        cfg.out_dir = args.out_dir
    return cfg


def cmd_check(cfg: Config) -> int:
    from .data import load_all
    from .llm import LLMClient, LLMError

    async def server() -> list[str]:
        async with LLMClient(cfg.llm) as client:
            return await client.health_check()

    ok = True
    try:
        print(f"[ok] model server {cfg.llm.base_url}: {asyncio.run(server())}")
    except LLMError as e:
        ok = False
        print(f"[fail] {e}")
    try:
        items = load_all(cfg.data.repo, cfg.data.split)
        print(f"[ok] dataset {cfg.data.repo}/{cfg.data.split}: {len(items)} rows")
    except Exception as e:  # noqa: BLE001 - surface any auth / network problem as-is
        ok = False
        print(f"[fail] dataset: {e}")
    return 0 if ok else 1


def cmd_run(cfg: Config, offline: bool) -> int:
    from .pipeline import run

    print(f"Fetching private Q&A dataset ({cfg.data.n_questions} questions, {cfg.data.sampling} sampling)...")
    result = run(cfg, offline=offline)
    metrics = json.loads((Path(cfg.out_dir) / "metrics.json").read_text())
    g = metrics["generator"]
    print(f"Accuracy on original questions: {g['accuracy_original']['rate']}  "
          f"verdicts: {metrics['question_verdicts']}")
    funnel = metrics["funnel_original_completions"]
    print("Funnel (original-order answers): " + " -> ".join(f"{v} {k}" for k, v in funnel.items()))
    reasons = list(metrics["naive_accepted_but_rejected"].items())[:5]
    if reasons:
        print("Why correct answers were dropped: " + ", ".join(f"{k} ({v})" for k, v in reasons))
    print(f"Generated {len(result.records)} high-quality training pairs "
          f"from {metrics['dataset']['questions_covered']} questions -> {cfg.out_dir}/sft_dataset.jsonl")
    if not result.records:
        print("No trace met every condition. This is common on small samples; the report explains each rejection.")
    print(f"Report: {cfg.out_dir}/report.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="traces", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_ in [
        ("check", "check the model server, model and dataset access"),
        ("run", "generate (cached) + filter + export + report"),
        ("filter", "re-run filtering and the report from cached generations only (no model calls)"),
        ("stress", "fault-injection test of the filter on accepted traces"),
        ("audit-sheet", "export a blind sample of traces for a human or LLM judge"),
        ("audit-score", "score the filter against blind labels (human or LLM judge)"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--config", help="YAML file overriding defaults (see configs/default.yaml)")
        p.add_argument("--smoke", action="store_true", help="6 questions, writes to outputs/smoke")
        p.add_argument("--n", type=int, help="number of questions")
        p.add_argument("--sampling", choices=["stratified", "head"])
        p.add_argument("--split")
        p.add_argument("--mode", choices=["native", "prompted"])
        p.add_argument("--prompt", help="prompt version, e.g. v1 or v2")
        p.add_argument("--model")
        p.add_argument("--concurrency", type=int)
        p.add_argument("--out-dir")
        if name == "audit-sheet":
            p.add_argument("--size", type=int, default=32, help="traces to sample (half accepted where possible)")
        if name == "audit-score":
            p.add_argument("--labels", required=True,
                           help="CSV with audit_id and human_label (filled-in audit_sheet.csv) or label (LLM judge)")
            p.add_argument("--labeller", default="human", help="who labelled, for the report (e.g. 'LLM judge: <model>')")
    args = parser.parse_args(argv)
    cfg = _config(args)

    if args.command == "check":
        return cmd_check(cfg)
    if args.command == "run":
        return cmd_run(cfg, offline=False)
    if args.command == "filter":
        return cmd_run(cfg, offline=True)

    from . import evaluation

    if args.command == "stress":
        return evaluation.cmd_stress(cfg)
    if args.command == "audit-sheet":
        return evaluation.cmd_audit_sheet(cfg, args.size)
    if args.command == "audit-score":
        return evaluation.cmd_audit_score(cfg, args.labels, args.labeller)
    return 1


if __name__ == "__main__":
    sys.exit(main())
