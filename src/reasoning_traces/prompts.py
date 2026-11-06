"""Prompt templates, versioned. The version is part of the generation cache key, so editing a prompt
means adding a new version, never changing an old one in place."""

from __future__ import annotations

from .probes import Variant

_HEADER = """You are an expert in cancer pharmacology and pharmacogenomics. You answer questions about whether a cancer cell line responds to a drug.

Definitions used for every question:
- Sensitive: the drug's IC50 in the cell line is below 0.1 µM (100 nM).
- Not sensitive (resistant): the IC50 is above 1 µM.

Reason through the question before answering, covering in order:
1. Drug: its class, primary molecular target(s) and mechanism of action.
2. Cell line: tissue of origin and any driver alterations or dependencies that matter for this drug.
"""

_FOOTER = """4. Decision: weigh the evidence and commit to one option.

Only state facts you are confident about. If you do not know something, say so briefly and reason from what you do know. Never invent mutations, IC50 values, studies or citations. Write as a scientist analysing the problem; do not refer to "the user". Keep the reasoning focused: about 150 to 400 words."""

# The IC50 definition is a dataset-wide labelling rule (Yes < 0.1 µM, No > 1 µM, inferred from the
# metadata), not a per-question hint. Without it, "sensitive" is ambiguous and the task is ill-posed.
SYSTEM_PROMPTS = {
    "v1": _HEADER
    + "3. Potency: how potent this kind of drug usually is in cancer cell lines, compared with the 0.1 µM and 1 µM thresholds.\n"
    + _FOOTER,
    # v2 changes step 3 only. Error analysis on v1: the model argued from clinical use ("taxanes are
    # less effective in bladder cancer") although the labels come from in vitro viability assays.
    "v2": _HEADER
    + "3. Potency: these IC50s come from in vitro cell viability assays. Judge how potent this kind of drug typically "
    "is against cancer cells in culture, compared with the 0.1 µM and 1 µM thresholds. Whether the drug is used "
    "clinically in this cancer type is not evidence either way.\n"
    + _FOOTER,
}

# Native mode: the model's chat template already wraps its thinking in <think> tags.
NATIVE_FORMAT = """When you are done reasoning, reply with one line in exactly this format:
Answer: <letter>"""

# Prompted mode: thinking mode is off, so the prompt has to impose the structure.
PROMPTED_FORMAT = """Format your whole reply exactly like this:
<think>
your step-by-step reasoning
</think>
Answer: <letter>

All reasoning goes inside the <think> tags. After </think>, write only the answer line."""


# For the optional chat-format export: the task definition stays, the step-by-step scaffold goes.
# The student model should learn to reason this way without being told how (context distillation).
TRAINING_SYSTEM_PROMPT = """You are an expert in cancer pharmacology and pharmacogenomics.
A cell line is sensitive to a drug if its IC50 is below 0.1 µM, and not sensitive (resistant) if its IC50 is above 1 µM.
Think inside <think> tags, then reply with one line: Answer: <letter>"""


def user_message(variant: Variant) -> str:
    options = "\n".join(f"{key}. {text}" for key, text in variant.options.items())
    return f"Question: {variant.question}\n\nOptions:\n{options}"


def build_messages(variant: Variant, reasoning_mode: str, prompt_version: str = "v1") -> list[dict[str, str]]:
    if prompt_version not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown prompt version {prompt_version!r}; available: {list(SYSTEM_PROMPTS)}")
    fmt = NATIVE_FORMAT if reasoning_mode == "native" else PROMPTED_FORMAT
    return [
        {"role": "system", "content": f"{SYSTEM_PROMPTS[prompt_version]}\n\n{fmt}"},
        {"role": "user", "content": user_message(variant)},
    ]


def render_full_prompt(messages: list[dict[str, str]]) -> str:
    """The exact text sent, one block per chat message. The server then applies the model's chat template."""
    return "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
