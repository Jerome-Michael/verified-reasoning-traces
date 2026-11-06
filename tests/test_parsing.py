"""The parser against the ways small models actually break the format."""

import pytest

from reasoning_traces.parsing import parse_completion

YN = {"A": "Yes", "B": "No"}
R = "Panobinostat is a pan-HDAC inhibitor with low nanomolar potency across most cancer cell lines. " * 3


@pytest.mark.parametrize(
    "text, status, answer, issue",
    [
        # well formed, in the many ways models write the answer line
        (f"<think>\n{R}\n</think>\n\nAnswer: A", "strict", "A", None),
        (f"<think>{R}</think>\n**Answer:** A", "strict", "A", None),
        (f"<think>{R}</think>\nFinal answer: (B)", "strict", "B", None),
        (f"<think>{R}</think>\nThe answer is option B.", "strict", "B", None),
        (f"<think>{R}</think>\nAnswer: A. Yes", "strict", "A", None),
        (f"<think>{R}</think>\n\\boxed{{A}}", "strict", "A", None),
        (f"<think>{R}</think>\nB", "strict", "B", None),
        # recoverable: the content is fine, the wrapper is not
        (f"{R}\n</think>\nAnswer: A", "recovered", "A", "missing_think_open"),
        (f"<think>\n\n</think>\n\n{R}\nAnswer: B", "recovered", "B", "reasoning_outside_tags"),
        (f"{R}\nAnswer: A", "recovered", "A", "no_think_tags"),
        (f"<think>{R}</think>\nAnswer: Yes", "recovered", "A", "answer_from_text"),
        (f"<think>{R}</think><think>{R}</think>\nAnswer: A", "recovered", "A", "multiple_think_blocks"),
        # seen in the prompted-mode run: reasoning after an empty <think></think>, no "Answer:" keyword
        (f"<think>\n\n</think>\n\n{R}\n\nTherefore, the decision is: A. Yes.", "recovered", "A", "nonstandard_answer_line"),
        (f"<think>{R}</think>\n\nB. No", "recovered", "B", "nonstandard_answer_line"),
        # unusable
        (f"<think>{R} and then the token budget ran out", "failed", None, "unclosed_think"),
        ("<think>\n\n</think>\nAnswer: A", "failed", None, "empty_reasoning"),
        (f"<think>{R}</think>\nI think it is sensitive.", "failed", None, "no_answer"),
        (f"<think>{R}</think>\nAnswer: A\nAnswer: B", "failed", None, "multiple_answers"),
        (f"<think>{R}</think>\nAnswer: C", "failed", None, "answer_not_in_options"),
        (f"<think>{R}</think>\nAnswer: A. No", "failed", None, "answer_text_conflict"),
        (f"<think>{R}</think>\n\nB. Yes", "failed", None, "no_answer"),  # letter and text disagree: ambiguous
        (f"<think>{R}</think>\nAnswer: A\n\nB. No", "failed", None, "multiple_answers"),
        ("", "failed", None, "empty_output"),
        (None, "failed", None, "empty_output"),
    ],
)
def test_parse(text, status, answer, issue):
    result = parse_completion(text, YN)
    assert result.status == status, result.issues
    assert result.answer == answer
    if issue:
        assert issue in result.issues


def test_reasoning_is_stored_without_tags():
    result = parse_completion(f"<think>\n{R}\n</think>\n\nAnswer: A", YN)
    assert result.reasoning == R.strip()
    assert "<think>" not in result.reasoning


def test_answer_after_think_wins_over_letters_inside_reasoning():
    text = f"<think>{R} At first option A looks right, but the answer is B.</think>\nAnswer: B"
    assert parse_completion(text, YN).answer == "B"
