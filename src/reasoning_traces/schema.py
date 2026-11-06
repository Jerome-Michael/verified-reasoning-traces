"""The output contract from the brief, enforced on every row before it is written."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HFSample(_Strict):
    question: str
    answer: str
    options: dict[str, str]


class CompletionOut(_Strict):
    reasoning: str
    answer: str


class Prompting(_Strict):
    full_prompt: str


class SFTRecord(_Strict):
    question_id: int
    completion_id: int
    hf_sample: HFSample
    completion: CompletionOut
    prompting: Prompting

    @model_validator(mode="after")
    def _consistent(self) -> "SFTRecord":
        if self.completion.answer not in self.hf_sample.options:
            raise ValueError(f"answer {self.completion.answer!r} is not an option key")
        if self.completion.answer != self.hf_sample.answer:
            raise ValueError("completion answer contradicts the ground truth")
        reasoning = self.completion.reasoning
        if not reasoning.strip():
            raise ValueError("empty reasoning")
        if "<think>" in reasoning.lower() or "</think>" in reasoning.lower():
            raise ValueError("reasoning must be stored without <think> tags")
        return self
