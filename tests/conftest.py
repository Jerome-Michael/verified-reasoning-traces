from __future__ import annotations

import pytest

from reasoning_traces.config import FilterConfig
from reasoning_traces.data import QAItem
from reasoning_traces.generate import Generation
from reasoning_traces.reference import MoaEntry, MoaReference

GOOD_REASONING = (
    "Panobinostat is a pan-HDAC inhibitor that blocks class I and II histone deacetylases, causing histone "
    "hyperacetylation, cell-cycle arrest and apoptosis. HT-1197 is a urothelial carcinoma cell line; I am not aware "
    "of an alteration that would make it resistant to HDAC inhibition. Pan-HDAC inhibitors such as panobinostat are "
    "among the most potent cytotoxic agents in cell line screens, typically reaching IC50 values in the low "
    "nanomolar range across most tumour types, well below the 0.1 µM threshold. Weighing the broad, mechanism-driven "
    "potency against the absence of a known resistance mechanism, HT-1197 is likely sensitive. The answer is B."
)


@pytest.fixture
def item() -> QAItem:
    return QAItem(
        question_id=7,
        question="Would the Bladder/Urinary Tract cancer cell line HT-1197 be sensitive to treatment with Panobinostat?",
        options={"A": "No", "B": "Yes"},
        answer="B",
        metadata={"drug": "Panobinostat", "cell_line": "HT-1197", "ic50_recomputed": 0.0699084807277752,
                  "label": "Yes", "study": "CTRPv2_2015"},
    )


@pytest.fixture
def fcfg() -> FilterConfig:
    return FilterConfig()


@pytest.fixture
def moa() -> MoaReference:
    entry = MoaEntry("Panobinostat", ("HDAC inhibitor",), ("HDAC1", "HDAC2", "HDAC3"))
    return MoaReference({"panobinostat": entry})


def make_gen(text: str | None, kind: str = "original", sample_idx: int = 0, finish: str = "stop", qid: int = 7) -> Generation:
    return Generation(key=f"{kind}-{sample_idx}-{hash(text) % 10**6}", question_id=qid, variant_kind=kind,
                      sample_idx=sample_idx, seed=1, model="m", reasoning_mode="native", prompt_version="v1",
                      text=text, finish_reason=finish, completion_tokens=100, latency_s=1.0)
