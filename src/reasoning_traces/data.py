"""Loading the private Hugging Face dataset and choosing the question subset."""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

PRIVATE_REPO = "owkin/technical_test"


@dataclass(frozen=True)
class QAItem:
    question_id: int  # row index in the HF split, so every output row traces back to its source
    question: str
    options: dict[str, str]
    answer: str  # ground-truth option key, e.g. "B"
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)  # never shown to the model

    @property
    def answer_text(self) -> str:
        return self.options[self.answer]

    def hf_sample(self) -> dict[str, Any]:
        return {"question": self.question, "answer": self.answer, "options": dict(self.options)}

    @classmethod
    def from_row(cls, question_id: int, row: dict[str, Any]) -> "QAItem":
        options = parse_options(row["options"])
        answer = str(row["answer"]).strip()
        if answer not in options:
            raise ValueError(f"Row {question_id}: answer {answer!r} is not one of {list(options)}")
        return cls(
            question_id=question_id,
            question=str(row["question"]).strip(),
            options=options,
            answer=answer,
            metadata=dict(row.get("metadata") or {}),
        )


def parse_options(raw: Any) -> dict[str, str]:
    """Options arrive as a JSON string in the HF dataset ('{"A": "Yes", "B": "No"}')."""
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Unexpected options format: {raw!r}")
    return {str(k).strip(): str(v).strip() for k, v in raw.items()}


def get_hf_token() -> str:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Please set the HF_TOKEN environment variable (or put it in a .env file).")
    return token


def load_all(repo: str = PRIVATE_REPO, split: str = "train") -> list[QAItem]:
    from datasets import load_dataset  # heavy import, only when needed

    dataset = load_dataset(repo, token=get_hf_token(), split=split)
    return [QAItem.from_row(i, row) for i, row in enumerate(dataset)]


def select_subset(items: list[QAItem], n: int, sampling: str = "stratified", seed: int = 42) -> list[QAItem]:
    """Pick n questions.

    The stub's `select(range(50))` is a trap on this dataset: rows are sorted by label, so the first
    50 are all "Yes". A dataset built from them would teach a model to always answer "Yes".
    Default is a seeded sample with equal Yes/No counts.
    """
    if n >= len(items):
        return list(items)
    if sampling == "head":
        return list(items[:n])
    if sampling != "stratified":
        raise ValueError(f"Unknown sampling strategy: {sampling!r}")

    rng = random.Random(seed)
    by_label: dict[str, list[QAItem]] = defaultdict(list)
    for item in items:
        by_label[item.answer_text].append(item)
    for bucket in by_label.values():
        rng.shuffle(bucket)
    labels = sorted(by_label)
    chosen: list[QAItem] = []
    # round-robin over labels keeps the classes balanced for any n
    while len(chosen) < n:
        for label in labels:
            if by_label[label] and len(chosen) < n:
                chosen.append(by_label[label].pop())
    return sorted(chosen, key=lambda it: it.question_id)


def load_private_dataset(
    n: int = 50, split: str = "train", sampling: str = "stratified", seed: int = 42, repo: str = PRIVATE_REPO
) -> list[QAItem]:
    """Loads the pre-built Q&A dataset from Hugging Face (same entry point as the starter stub)."""
    return select_subset(load_all(repo, split), n, sampling, seed)
