"""Each trace check fires on the failure it targets, and stays quiet on a good trace."""

from conftest import GOOD_REASONING, make_gen
from reasoning_traces.checks import reasoning_stance, score_completion
from reasoning_traces.probes import negated, original


def _score(item, fcfg, moa, reasoning=GOOD_REASONING, answer="B", finish="stop", variant=None):
    variant = variant or original(item)
    text = f"<think>\n{reasoning}\n</think>\n\nAnswer: {answer}"
    return score_completion(make_gen(text, finish=finish), item, variant, fcfg, moa)


def _failed(sc):
    return set(sc.hard_failures)


def test_good_trace_passes_everything(item, fcfg, moa):
    sc = _score(item, fcfg, moa)
    assert sc.passed, sc.hard_failures
    assert sc.quality >= 0.9


def test_wrong_answer(item, fcfg, moa):
    sc = _score(item, fcfg, moa, reasoning=GOOD_REASONING.replace("The answer is B.", "So the answer is A."), answer="A")
    assert "correct" in _failed(sc)


def test_reasoning_contradicts_final_answer(item, fcfg, moa):
    # Reasoning argues for B (Yes) but the final line says A (No).
    sc = _score(item, fcfg, moa, answer="A")
    assert "faithful" in _failed(sc)


def test_stance_from_sensitivity_wording_ignores_the_definition(item):
    text = "A line is sensitive if its IC50 is below 0.1 µM. " + "Filler about the drug. " * 30 + \
           "Overall I expect HT-1197 to be resistant to this compound."
    assert reasoning_stance(text, original(item)) == "No"


def test_stance_on_negated_question(item):
    text = "Filler about the drug. " * 30 + "Therefore the cell line is likely resistant."
    # Asked "resistant?" -> concluding resistant means answering "Yes".
    assert reasoning_stance(text, negated(item)) == "Yes"


def test_truncated_output(item, fcfg, moa):
    sc = _score(item, fcfg, moa, finish="length")
    assert "complete" in _failed(sc)


def test_guess_in_conclusion(item, fcfg, moa):
    sc = _score(item, fcfg, moa, reasoning=GOOD_REASONING + " Honestly, if I had to guess, I'd say sensitive.")
    assert "decisive" in _failed(sc)


def test_language_switch(item, fcfg, moa):
    sc = _score(item, fcfg, moa, reasoning=GOOD_REASONING + " 因此该细胞系对该药物敏感。")
    assert "language" in _failed(sc)


def test_repetition_loop(item, fcfg, moa):
    loop = " But wait, I need to check the IC50 of panobinostat in this line again." * 15
    sc = _score(item, fcfg, moa, reasoning=GOOD_REASONING + loop)
    assert "no_loops" in _failed(sc)


def test_mild_repetition_is_only_a_soft_penalty(item, fcfg, moa):
    circular = " If the cell line is not dependent on this pathway it may be resistant." * 6
    sc = _score(item, fcfg, moa, reasoning=GOOD_REASONING.replace("The answer is B.", circular + " The answer is B."))
    assert "no_loops" not in _failed(sc) and "not_repetitive" in sc.soft_failures


def test_fabricated_citation(item, fcfg, moa):
    sc = _score(item, fcfg, moa, reasoning=GOOD_REASONING.replace("histone deacetylases,", "histone deacetylases (Smith et al., 2019),"))
    assert "no_citations" in _failed(sc)


def test_model_admits_not_knowing_the_drug(item, fcfg, moa):
    sc = _score(item, fcfg, moa, reasoning="I'm not familiar with Panobinostat. " + GOOD_REASONING)
    assert "knows_drug" in _failed(sc)


def test_invented_mechanism_is_caught(item, fcfg, moa):
    wrong = ("Panobinostat is a selective EGFR tyrosine kinase inhibitor that blocks growth factor signalling. "
             "HT-1197 has no EGFR amplification, but EGFR inhibitors are often potent in urothelial lines, so it is "
             "likely sensitive. " * 3) + "The answer is B."
    sc = _score(item, fcfg, moa, reasoning=wrong)
    assert "mechanism_grounded" in _failed(sc)


def test_mechanism_synonym_is_accepted(item, fcfg, moa):
    text = GOOD_REASONING.replace("pan-HDAC inhibitor", "histone deacetylase inhibitor").replace("HDAC", "deacetylase")
    sc = _score(item, fcfg, moa, reasoning=text)
    assert "mechanism_grounded" not in _failed(sc)


def test_too_short(item, fcfg, moa):
    sc = _score(item, fcfg, moa, reasoning="Panobinostat is potent. Sensitive. The answer is B.")
    assert "length" in _failed(sc)


def test_combination_requires_every_component(fcfg, moa):
    from reasoning_traces.data import QAItem
    combo = QAItem(3, "Would the Bladder/Urinary Tract cancer cell line UM-UC-1 be sensitive to treatment with "
                      "docetaxel:tanespimycin (2:1 mol/mol)?", {"A": "Yes", "B": "No"}, "A",
                   {"drug": "docetaxel:tanespimycin (2:1 mol/mol)", "cell_line": "UM-UC-1"})
    # The model misspells tanespimycin, as Qwen3-4B did in the smoke run.
    text = GOOD_REASONING.replace("Panobinostat", "Docetaxel plus talespimycin").replace("The answer is B.", "The answer is A.")
    sc = score_completion(make_gen(f"<think>{text}</think>\nAnswer: A"), combo, original(combo), fcfg, moa)
    assert "mentions_drug" in _failed(sc)
    assert "tanespimycin" in next(c.detail for c in sc.checks if c.name == "mentions_drug")


def test_educated_guess_is_only_a_soft_penalty(item, fcfg, moa):
    hedged = GOOD_REASONING.replace("The answer is B.", "This is an educated guess from the drug class. The answer is B.")
    sc = _score(item, fcfg, moa, reasoning=hedged)
    assert sc.passed


def test_mechanism_match_accepts_plurals(item, fcfg, moa):
    text = GOOD_REASONING.replace("pan-HDAC inhibitor that blocks class I and II histone deacetylases", "drug that inhibits HDACs")
    sc = _score(item, fcfg, moa, reasoning=text)
    assert "mechanism_grounded" not in _failed(sc)


def test_drug_name_cannot_vouch_for_its_own_mechanism(fcfg):
    from reasoning_traces.data import QAItem
    from reasoning_traces.reference import MoaEntry, MoaReference
    ref = MoaReference({"ibrd9": MoaEntry("I-BRD9", ("bromodomain inhibitor",), ("BRD9",))})
    q = QAItem(5, "Would the Bladder/Urinary Tract cancer cell line X be sensitive to treatment with I-BRD9?",
               {"A": "Yes", "B": "No"}, "B", {"drug": "I-BRD9", "cell_line": "X"})
    text = ("I-BRD9 is an androgen receptor antagonist used in prostate cancer. " * 6) + "So the cell line is resistant. The answer is B."
    sc = score_completion(make_gen(f"<think>{text}</think>\nAnswer: B"), q, original(q), fcfg, ref)
    assert "mechanism_grounded" in _failed(sc)


def test_drug_missing_from_reference_is_unverifiable(item, fcfg):
    from reasoning_traces.reference import MoaReference
    sc = _score(item, fcfg, MoaReference({}))
    assert "mechanism_verifiable" in _failed(sc)
