"""Side-by-side table of several runs (models, prompt modes): uv run python scripts/compare_runs.py outputs/a outputs/b"""

import json
import sys
from pathlib import Path


def pct(r):
    return "n/a" if r["rate"] is None else f"{r['rate']:.0%} [{r['ci95'][0]:.0%}–{r['ci95'][1]:.0%}]"


def main(dirs):
    rows = []
    for d in dirs:
        m = json.loads((Path(d) / "metrics.json").read_text())
        g, run, ds, f = m["generator"], m["run"], m["dataset"], m["funnel_original_completions"]
        v = m["question_verdicts"]
        rows.append([
            Path(d).name, run["model"].split("/")[-1], run["reasoning_mode"], run["n_questions"],
            f"{g['parse_status'].get('strict', 0)}+{g['parse_status'].get('recovered', 0)}/{g['completions']}",
            pct(g["accuracy_original"]), pct(g["accuracy_by_truth"]["Yes"]), pct(g["accuracy_by_truth"]["No"]),
            pct(g["yes_rate_original"]), pct(g["negation_consistency"]), pct(g["swap_consistency"]),
            g["mean_completion_tokens"], g["mean_latency_s"], v.get("robust", 0), f["correct (naive filter)"],
            ds["rows"], ds["questions_covered"],
        ])
    header = ["run", "model", "mode", "questions", "parsed (strict+recovered)", "accuracy", "acc. on Yes",
              "acc. on No", "says Yes", "negation-consistent", "swap-consistent", "tokens", "latency s",
              "robust questions", "naive keeps", "final rows", "questions covered"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


if __name__ == "__main__":
    main(sys.argv[1:])
