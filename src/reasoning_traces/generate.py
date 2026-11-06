"""Stage 1: plan and run generations. Slow, so every result is cached and runs are resumable."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .data import QAItem
from .llm import LLMClient, LLMError
from .probes import BUILDERS, Variant
from .prompts import build_messages


@dataclass
class Task:
    key: str
    question_id: int
    variant: Variant
    sample_idx: int
    messages: list[dict[str, str]]
    seed: int


@dataclass
class Generation:
    key: str
    question_id: int
    variant_kind: str
    sample_idx: int
    seed: int
    model: str
    reasoning_mode: str
    prompt_version: str
    text: str | None
    finish_reason: str | None
    completion_tokens: int | None
    latency_s: float | None
    error: str | None = None


def _task_key(cfg: Config, messages: list[dict[str, str]], sample_idx: int) -> str:
    """Same inputs -> same key -> cache hit. Any change to model, prompt or sampling -> new generation."""
    c = cfg.llm
    payload = {
        "model": c.model, "mode": c.reasoning_mode, "prompt_version": c.prompt_version, "messages": messages,
        "temperature": c.temperature, "top_p": c.top_p, "top_k": c.top_k, "max_tokens": c.max_tokens,
        "sample_idx": sample_idx,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def plan_tasks(items: list[QAItem], cfg: Config) -> list[Task]:
    counts = {"original": cfg.sampling.n_original, "swapped": cfg.sampling.n_swapped, "negated": cfg.sampling.n_negated}
    tasks: list[Task] = []
    for item in items:  # question by question, so a partial run still yields complete questions
        for kind, n in counts.items():
            variant = BUILDERS[kind](item)
            if variant is None:
                continue
            messages = build_messages(variant, cfg.llm.reasoning_mode, cfg.llm.prompt_version)
            for sample_idx in range(n):
                key = _task_key(cfg, messages, sample_idx)
                tasks.append(Task(key, item.question_id, variant, sample_idx, messages, seed=int(key[:8], 16)))
    return tasks


class GenerationCache:
    """Append-only JSONL. Only successful generations are stored, so failures are retried next run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: dict[str, Generation] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = Generation(**json.loads(line))
                    self._records[rec.key] = rec

    def __contains__(self, key: str) -> bool:
        return key in self._records

    def get(self, key: str) -> Generation | None:
        return self._records.get(key)

    def add(self, rec: Generation) -> None:
        self._records[rec.key] = rec
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")


def _record(task: Task, cfg: Config, **kw) -> Generation:
    return Generation(
        key=task.key, question_id=task.question_id, variant_kind=task.variant.kind, sample_idx=task.sample_idx,
        seed=task.seed, model=cfg.llm.model, reasoning_mode=cfg.llm.reasoning_mode, prompt_version=cfg.llm.prompt_version, **kw,
    )


async def _run(tasks: list[Task], cfg: Config, cache: GenerationCache) -> dict[str, Generation]:
    pending = list({t.key: t for t in tasks if t.key not in cache}.values())
    failures: dict[str, Generation] = {}
    if not pending:
        return failures
    sem = asyncio.Semaphore(cfg.llm.concurrency)
    async with LLMClient(cfg.llm) as client:
        await client.health_check()
        bar = tqdm(total=len(pending), desc="generating", unit="gen")

        async def one(task: Task) -> None:
            async with sem:
                try:
                    r = await client.chat(task.messages, task.seed)
                    cache.add(_record(task, cfg, text=r.text, finish_reason=r.finish_reason,
                                      completion_tokens=r.completion_tokens, latency_s=round(r.latency_s, 2)))
                except LLMError as e:
                    failures[task.key] = _record(task, cfg, text=None, finish_reason=None,
                                                 completion_tokens=None, latency_s=None, error=str(e))
                bar.update(1)

        await asyncio.gather(*(one(t) for t in pending))
        bar.close()
    return failures


def generate(tasks: list[Task], cfg: Config, cache: GenerationCache | None = None, offline: bool = False) -> list[Generation]:
    """Returns one Generation per task, in task order. offline=True never calls the model (cache only)."""
    cache = cache or GenerationCache(cfg.cache_path)
    failures = {} if offline else asyncio.run(_run(tasks, cfg, cache))
    out = []
    for t in tasks:
        rec = cache.get(t.key) or failures.get(t.key)
        if rec is None:  # offline and never generated
            rec = _record(t, cfg, text=None, finish_reason=None, completion_tokens=None, latency_s=None,
                          error="not generated (offline mode)")
        out.append(rec)
    return out
