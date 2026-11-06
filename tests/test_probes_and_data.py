from reasoning_traces.data import QAItem, parse_options, select_subset
from reasoning_traces.probes import negated, original, swapped
from reasoning_traces.prompts import build_messages, render_full_prompt


def _correct_letter(variant, truth_text):
    return next(k for k, v in variant.letter_to_original.items() if v == truth_text)


def test_every_variant_maps_its_correct_letter_back_to_the_ground_truth(item):
    for build in (original, swapped, negated):
        v = build(item)
        letter = _correct_letter(v, item.answer_text)
        assert v.original_text_for(letter) == item.answer_text


def test_swapped_reverses_option_texts(item):
    v = swapped(item)
    assert v.options == {"A": "Yes", "B": "No"}
    assert v.question == item.question


def test_negated_flips_question_and_answer(item):
    v = negated(item)
    assert "be resistant to" in v.question
    # On "resistant?" the right answer is "No" (option B), which means "Yes" on the original question.
    assert v.options["A"] == "No" and v.letter_to_original["A"] == "Yes"


def test_negation_skipped_when_not_applicable(item):
    other = QAItem(1, "Which drug is more potent?", {"A": "X", "B": "Y"}, "A")
    assert negated(other) is None
    assert original(other).options == {"A": "X", "B": "Y"}


def test_parse_options_from_json_string():
    assert parse_options('{"A": "No", "B": "Yes"}') == {"A": "No", "B": "Yes"}


def test_stratified_sampling_is_label_balanced_even_when_data_is_sorted():
    # Mimics the real dataset: sorted by label, so the first rows are all "Yes".
    items = [QAItem(i, f"q{i}", {"A": "Yes", "B": "No"}, "A" if i < 300 else "B") for i in range(600)]
    head = select_subset(items, 50, "head")
    assert all(it.answer_text == "Yes" for it in head)
    strat = select_subset(items, 50, "stratified", seed=1)
    assert sum(it.answer_text == "Yes" for it in strat) == 25
    assert strat == select_subset(items, 50, "stratified", seed=1)  # reproducible


def test_prompt_never_leaks_metadata(item):
    for build in (original, swapped, negated):
        prompt = render_full_prompt(build_messages(build(item), "native"))
        assert str(item.metadata["ic50_recomputed"])[:6] not in prompt
        assert item.metadata["study"] not in prompt
        assert "metadata" not in prompt.lower()
