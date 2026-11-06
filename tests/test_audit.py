import csv
import json

from reasoning_traces.config import Config
from reasoning_traces.evaluation import cmd_audit_score


def _write_csv(path, header, rows):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _audit_dir(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    _write_csv(audit / "audit_key.csv", ["audit_id", "key", "pipeline_decision", "hard_failures", "question_verdict"], [
        ["0", "k0", "accept", "", "robust"],
        ["1", "k1", "accept", "", "robust"],
        ["2", "k2", "reject", "mechanism_grounded", "robust"],
        ["3", "k3", "reject", "", "unstable"],
    ])
    (audit / "population.json").write_text(json.dumps({"accepted": 2, "rejected": 8}))
    return audit


def test_audit_score_reads_llm_labels_and_lists_disagreements(tmp_path):
    audit = _audit_dir(tmp_path)
    labels = tmp_path / "judge.csv"
    _write_csv(labels, ["audit_id", "label", "reason"], [
        ["0", "1", "fine"], ["1", "0", "wrong indication"], ["2", "0", "wrong target"], ["3", "1", "looks fine"],
    ])

    cfg = Config(out_dir=str(tmp_path))
    assert cmd_audit_score(cfg, str(labels), "LLM judge") == 0
    out = (audit / "audit_results.md").read_text()
    assert "Labelled by: LLM judge" in out
    assert "Agreement with the filter: 2/4 (50%)" in out
    # naive share = (2 * 0.5 + 8 * 0.5) / 10
    assert "naive filter (population-weighted, 2 accepted / 8 rejected): 50%" in out
    assert "| 1 | accepted, labelled bad | passed every check | wrong indication |" in out
    assert "| 3 | rejected, labelled good | question unstable | looks fine |" in out

