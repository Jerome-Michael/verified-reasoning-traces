"""All tunable settings in one place. Defaults are the values used for the reported run."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass
class DataConfig:
    repo: str = "owkin/technical_test"
    split: str = "train"
    n_questions: int = 50
    # "stratified": seeded sample balanced on the Yes/No label.
    # "head": the first N rows, like the starter stub. On this dataset the first 50 rows are all "Yes".
    sampling: str = "stratified"
    seed: int = 42


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:12434/engines/v1"  # Docker Model Runner (host TCP)
    model: str = "hf.co/qwen/qwen3-4b-gguf"
    # "native": the model's own thinking mode emits <think>...</think>.
    # "prompted": thinking mode off; the prompt alone asks for <think> tags (for non-reasoning models).
    reasoning_mode: str = "native"
    prompt_version: str = "v1"
    temperature: float = 0.6  # Qwen3 thinking-mode recommendation; greedy decoding causes loops
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 2048
    concurrency: int = 2
    timeout_s: float = 600.0
    max_retries: int = 3


@dataclass
class SamplingConfig:
    n_original: int = 2  # completions on the question as given: the only candidates for the dataset
    n_swapped: int = 1   # probe: A/B option order reversed (detects letter-position bias)
    n_negated: int = 1   # probe: "sensitive" -> "resistant", answer must flip (detects yes-bias)


@dataclass
class FilterConfig:
    allow_recovered_format: bool = True  # keep traces whose tags were repaired by the lenient parser
    min_reasoning_words: int = 60
    max_reasoning_words: int = 1400
    # Calibrated on 190 Qwen3-4B traces: median 8-gram repeat count 2, p95 5, the one real loop 19.
    max_compression_ratio: float = 5.0   # zlib ratio; degenerate loops compress unusually well (p99 = 4.3)
    max_shingle_repeats: int = 10        # hard: an 8-word sequence repeated more than this = degenerate loop
    soft_shingle_repeats: int = 5        # soft: circular reasoning that keeps restating the same condition
    min_valid_probes: int = 3            # question needs this many parseable answers to be judged
    min_probe_agreement: float = 1.0     # fraction of valid probes that must match the ground truth
    max_per_question: int = 2
    min_quality: float = 0.75            # each soft failure costs 0.1: at most 2 soft failures (hand review: 3+ = flip-flopping)
    balance_labels: bool = True          # equal Yes/No rows in the final dataset (see selection.balance_by_label)
    dedup_jaccard: float = 0.8           # near-duplicate threshold between two kept traces
    use_moa_reference: bool = True
    require_verifiable_mechanism: bool = True  # drug must be in the reference (hard); False = soft penalty only
    moa_reference_url: str = (
        "https://s3.amazonaws.com/data.clue.io/repurposing/downloads/repurposing_drugs_20200324.txt"
    )


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    out_dir: str = "outputs/main"
    cache_path: str = "outputs/cache/generations.jsonl"
    reference_dir: str = "data/reference"

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        """Defaults <- YAML file (optional) <- environment (LLM_BASE_URL, LLM_MODEL)."""
        load_dotenv()
        cfg = cls()
        if path:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            for section, values in raw.items():
                target = getattr(cfg, section, None)
                if target is None:
                    raise ValueError(f"Unknown config section: {section!r}")
                if not isinstance(values, dict):
                    setattr(cfg, section, values)
                    continue
                known = {f.name for f in fields(target)}
                for key, value in values.items():
                    if key not in known:
                        raise ValueError(f"Unknown config key: {section}.{key}")
                    setattr(target, key, value)
        cfg.llm.base_url = os.getenv("LLM_BASE_URL", cfg.llm.base_url)
        cfg.llm.model = os.getenv("LLM_MODEL", cfg.llm.model)
        return cfg
