import pytest
from pydantic import ValidationError

from conftest import GOOD_REASONING, make_gen
from reasoning_traces.checks import ScoredCompletion
from reasoning_traces.consensus import judge_question
from reasoning_traces.parsing import ParseResult
from reasoning_traces.probes import BUILDERS
from reasoning_traces.report import wilson
from reasoning_traces.schema import SFTRecord


def fake(item, kind, letter, idx=0):
    v = BUILDERS[kind](item)
    parse = ParseResult("strict", GOOD_REASONING, letter, [])
    return ScoredCompletion(make_gen("x", kind, idx), v, parse, v.options[letter], v.original_text_for(letter))


# item: options {"A": "No", "B": "Yes"}, truth "Yes" (B)

def test_robust_when_every_view_is_right(item, fcfg):
    comps = [fake(item, "original", "B", 0), fake(item, "original", "B", 1),
             fake(item, "swapped", "A"), fake(item, "negated", "A")]  # swapped A = Yes; on "resistant?" A = No
    assert judge_question(item, comps, fcfg).verdict == "robust"


def test_yes_sayer_is_caught_by_negation(item, fcfg):
    # Says "Yes" to everything: right on the original and swapped questions, wrong on "resistant?"
    comps = [fake(item, "original", "B", 0), fake(item, "original", "B", 1),
             fake(item, "swapped", "A"), fake(item, "negated", "B")]
    assert judge_question(item, comps, fcfg).verdict == "framing_bias"


def test_letter_bias_is_caught_by_swap(item, fcfg):
    comps = [fake(item, "original", "B", 0), fake(item, "original", "B", 1), fake(item, "swapped", "B")]
    assert judge_question(item, comps, fcfg).verdict == "position_bias"


def test_confidently_wrong(item, fcfg):
    comps = [fake(item, "original", "A", 0), fake(item, "original", "A", 1),
             fake(item, "swapped", "B"), fake(item, "negated", "B")]
    assert judge_question(item, comps, fcfg).verdict == "confidently_wrong"


def test_insufficient_when_too_few_readable_answers(item, fcfg):
    comps = [fake(item, "original", "B", 0), fake(item, "swapped", "A")]
    assert judge_question(item, comps, fcfg).verdict == "insufficient"


def _record(**overrides):
    rec = {
        "question_id": 0, "completion_id": 0,
        "hf_sample": {"question": "q?", "answer": "B", "options": {"A": "Hydrocortisone", "B": "CCT 018159"}},
        "completion": {"reasoning": "First, compare mechanism...", "answer": "B"},
        "prompting": {"full_prompt": "[system]\n...\n\n[user]\nq?"},
    }
    rec.update(overrides)
    return rec


def test_schema_accepts_the_brief_example():
    assert SFTRecord.model_validate(_record()).model_dump() == _record()


@pytest.mark.parametrize("bad", [
    {"extra_field": 1},
    {"completion": {"reasoning": "x", "answer": "A"}},              # contradicts ground truth
    {"completion": {"reasoning": "<think>x</think>", "answer": "B"}},  # tags must be stripped
    {"completion": {"reasoning": "  ", "answer": "B"}},
])
def test_schema_rejects_invalid_rows(bad):
    with pytest.raises(ValidationError):
        SFTRecord.model_validate(_record(**bad))


def test_wilson_interval():
    lo, hi = wilson(25, 50)
    assert 0.36 < lo < 0.37 and 0.63 < hi < 0.64


def test_unstable_when_identical_prompts_disagree(item, fcfg):
    comps = [fake(item, "original", "B", 0), fake(item, "original", "A", 1),
             fake(item, "swapped", "A"), fake(item, "negated", "B")]
    assert judge_question(item, comps, fcfg).verdict == "unstable"


def test_label_balance_keeps_equal_yes_and_no(item, fcfg):
    from reasoning_traces.selection import balance_by_label
    yes = [fake(item, "original", "B", 0), fake(item, "original", "B", 1)]
    no_q1, no_q2 = [fake(item, "original", "A", 0)], [fake(item, "original", "A", 0)]
    out = balance_by_label({1: yes, 2: no_q1, 3: no_q2}, {1: "Yes", 2: "No", 3: "No"})
    assert sum(map(len, out.values())) == 4
    # A model that only ever produces "No" rows yields an empty, not a biased, dataset.
    assert sum(map(len, balance_by_label({2: no_q1}, {1: "Yes", 2: "No"}).values())) == 0


def test_selection_skips_traces_below_min_quality(item, fcfg):
    from reasoning_traces.checks import Check
    from reasoning_traces.consensus import QuestionVerdict
    from reasoning_traces.selection import select_for_question
    good, sloppy = fake(item, "original", "B", 0), fake(item, "original", "B", 1)
    sloppy.checks += [Check(n, False, False) for n in ("low_hedging", "not_repetitive", "no_meta_talk")]
    kept = select_for_question([good, sloppy], QuestionVerdict(item.question_id, "robust", 2, 2), fcfg)
    assert kept == [good]
