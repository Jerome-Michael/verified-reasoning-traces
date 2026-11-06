# Reasoning-trace dataset generation (Owkin take-home)

A local pipeline that turns Owkin's biology Q&A pairs into an SFT dataset of R1-style reasoning traces
(`<think>…</think>` then an answer), with an automated gate that decides which traces are good enough to train on.
Runs end to end on a 16 GB Apple Silicon Mac with Qwen3-4B served by Docker Model Runner.

## Results at a glance

Main run: 50 questions × 4 prompts = 200 generations with Qwen3-4B (thinking mode), about an hour on an M5 / 16 GB.

| | |
|---|---|
| Well-formed outputs | 199 / 200 |
| Accuracy on the questions as asked | 54.5% (95% CI 45–64%): right 45% of the time when the truth is Yes, 64% when it is No |
| Traces a naive "answer matches the ground truth" filter would keep | 54 |
| **Traces kept by this pipeline** | **6, from 5 questions, 3 Yes / 3 No** |

The gap between 54 and 6 is the point of the exercise. Three measurements explain it:

1. **The model is close to guessing.** On a Yes/No question a coin is right half the time. Asking each question four
   ways shows that the model's answer survives rephrasing for only 9 of 50 questions (§3).
2. **Correct answers often rest on invented facts.** Where the drug's mechanism can be checked against a curated
   reference, 49% of traces state a wrong one, and so do **16 of the 36 traces with the correct final answer** (§4).
   Example: Osi-027 "is a specific PARP inhibitor" (it is an mTOR inhibitor).
3. **The model has a prior, not just noise.** It answers "No" 60% of the time on a 50/50 dataset. A dataset built
   from its confident answers would teach that prior, so the final set is balanced on the label (§3).

The filter itself is evaluated five ways (§7): a funnel with every rejection reason, a sweep of its strictness
against the chance that a pure guesser passes, fault injection on 36 real traces (every injected fault caught), a
hand review of what survives, and a blind audit by an LLM judge (ChatGPT). The judge agreed with the filter on 17
of 22 traces. Traces kept by a naive "answer matches" filter have good reasoning about 24% of the time; traces kept
by this filter, 67% (4 of 6; the hand review says 3 of 6). What still gets through are fine-grained factual errors
(a wrong indication, claims attributed to unnamed studies) that no check here verifies. The checks enforce
necessary conditions, not sufficient ones, and a 4B teacher is the bottleneck (§6, §8).

Model and prompt were chosen by measurement, not assumption: five configurations on the same 50 questions (§5, §6).
The 8B model was *less* useful than the 4B: more self-consistent and fewer invented mechanisms, but right on only 16%
of "Yes" questions.

## Quickstart

Prerequisites: an Apple Silicon Mac, Docker Desktop ≥ 4.62, and [uv](https://docs.astral.sh/uv/)
(`curl -LsSf https://astral.sh/uv/install.sh | sh`).

```bash
git clone https://github.com/Jerome-Michael/verified-reasoning-traces.git
cd verified-reasoning-traces
cp .env.example .env          # then paste your HF_TOKEN into .env
make setup                    # Python 3.12 + locked dependencies (uv.lock)
make model                    # Docker Model Runner on port 12434, Qwen3-4B (~2.5 GB), 8k context (see below)
make check                    # model server, model and dataset access, with fix-it hints on failure
make smoke                    # 6 questions end to end (~8 min): a plumbing check, often 0–2 rows survive
make run                      # 50 questions (~1 h on an M5 / 16 GB) -> outputs/main/
```

`uv run python main.py` runs the same pipeline through the starter stub's interface
(`load_private_dataset()` → `generate_reasoning_traces()`).

| Command | What it does |
|---|---|
| `make filter` | re-runs every check and the report from cached generations, without calling the model |
| `make stress` | fault-injection test of the filter (§7) |
| `make audit-sheet` | blind sample of traces for a human or LLM judge, decisions hidden (§7) |
| `make audit-score` | compares the judge's labels with the filter's hidden decisions (§7) |
| `make test` | 68 unit tests: parser edge cases, every check, probes, verdicts, schema, no metadata leakage |
| `make lint` | static checks (ruff) |
| `uv run traces run --model hf.co/qwen/qwen3-8b-gguf --out-dir outputs/qwen3-8b` | another model |
| `uv run traces run --mode prompted` | thinking mode off, `<think>` structure imposed by the prompt only |
| `scripts/run_experiments.sh` | the comparison grid of §5–6 (`MODELS=1` adds the model comparison) |

Any OpenAI-compatible server works (Ollama, LM Studio, llama-server): set `LLM_BASE_URL` and `LLM_MODEL` in `.env`.

**Two portability notes from running this on a 16 GB laptop:**
- Docker Model Runner loads Qwen3 with its full 40,960-token context: 8.4 GB resident and heavy swapping (throughput
  fell to 2 generations/min). `make model` creates a same-name variant with an 8,192-token context (4.2 GB), enough
  for 3 parallel requests of prompt + 2,048 output. `docker model configure` is not enough: it is forgotten whenever
  the model unloads, including the automatic unload after 5 idle minutes.
- In an iCloud-synced folder (Desktop, Documents), macOS hides uv's `.pth` file and Python 3.12 then ignores it. The
  Makefile sets `PYTHONPATH=src` so this cannot break a run; keeping the virtualenv outside iCloud also avoids
  syncing it (`export UV_PROJECT_ENVIRONMENT=~/.venvs/owkin-traces`).

### Outputs (`outputs/<run>/`)

| File | Content |
|---|---|
| `sft_dataset.jsonl` | the deliverable: exactly the schema of the brief, validated row by row |
| `sft_chat.jsonl` | same rows as chat `messages` for SFT trainers, without the prompt scaffold (§5) |
| `audit.jsonl` | every generation, kept or not, with its parse status, every check result and the question verdict |
| `report.md`, `metrics.json` | funnel, rejection reasons, model behaviour, threshold sweep, dataset statistics, examples |
| `stress_test.md`, `audit/` | filter evaluation (§7) |
| `config.json` | the exact configuration of the run |

`question_id` is the row index in the HF split and `completion_id` the index of the sample, so every row traces
back to its source and to its line in `audit.jsonl`. Generations are cached in `outputs/cache/generations.jsonl`,
keyed on model, prompt version, sampling settings and sample index: an interrupted run resumes where it stopped,
and changing a filter never requires generating again.

## 1. What the data actually is

`owkin/technical_test` has 604 train and 460 test rows, all from one template:

> *Would the Bladder/Urinary Tract cancer cell line **HT-1197** be sensitive to treatment with **Panobinostat**?*
> A. No · B. Yes

plus metadata: drug, cell line, source study (CTRPv2, GDSC, CCLE; PRISM for test) and the measured IC50.
Profiling it before writing any pipeline code changed several decisions:

| Finding | Consequence |
|---|---|
| **The stub's `select(range(50))` returns 50 "Yes" questions**: rows are sorted by label. | A dataset built from them teaches "always answer Yes". Default sampling is seeded and label-balanced; `--sampling head` reproduces the stub. |
| **Labels follow a fixed rule:** Yes ⇔ IC50 < 0.1 µM, No ⇔ IC50 > 1 µM, the band in between is excluded. | Without it "sensitive" is undefined, so the definition is in the prompt. It is a dataset-wide rule, not a per-question hint. |
| **The metadata contains the answer** (`ic50_recomputed`, `label`). | Only `question` and `options` reach the model; a test asserts no metadata appears in any prompt. |
| Two options, A/B order already shuffled per row. | A model with no knowledge is right half the time (§3). |
| For 74% of drugs with ≥3 rows the label is the same in every cell line. | The drug's mechanism and typical potency carry most of the signal, so the mechanism is the claim most worth verifying (§4). |
| ~10% of drugs are combinations (`JQ-1:carboplatin (1:1 mol/mol)`). | Checks handle each component. |
| The test split (PRISM, "seen/unseen cell line/drug" scenarios) is a held-out benchmark. | Not used: the pipeline reads the train split only. |

## 2. Pipeline

```
HF dataset ─► label-balanced sample ─► 4 prompts per question ─► local LLM ──► cache (resumable)
                                        2 × original                 │
                                        1 × options swapped          ▼
                                        1 × "resistant?" (negated)   parse: strict → recovered → failed
                                                                      │
                                                trace checks: hard reject / soft penalty
                                                                      │
                                                question verdict from all 4 answers
                                                                      │
                                    quality bar · ≤2 per question · de-dup · Yes/No balance ─► sft_dataset.jsonl
                                                                                              audit.jsonl, report.md
```

Generation is the slow stage; everything after it is deterministic and re-runs from the cache in seconds
(`make filter`). That separation is what made it cheap to calibrate the checks on real outputs (§4).

## 3. The core problem: a correct answer is weak evidence

With two options, keep "any correct sample out of 4" and a model with *zero* knowledge passes 94% of questions
(1 − 0.5⁴). The obvious filter, *final answer matches the ground truth*, mostly certifies lucky guesses, and training
on them teaches a model to rationalise. So each question is asked four ways, and only the two original-order samples
can enter the dataset (so `hf_sample` always matches `full_prompt`); the other two are probes.

| Verdict | Pattern | Main run | Action |
|---|---|---|---|
| `robust` | every probe lands on the ground truth | 9 | keep |
| `unstable` | the two identical prompts get different answers | 15 | drop: sampling, not knowing |
| `framing_bias` | same Yes/No to "sensitive?" and "resistant?" | 11 | drop |
| `position_bias` | same letter after the option texts are swapped | 7 | drop |
| `confidently_wrong` | every probe contradicts the ground truth | 8 | drop, listed in the report |

The 8 `confidently_wrong` questions are worth reading: 6 are drugs that are extremely potent in vitro (bortezomib,
paclitaxel, leptomycin B, brefeldin A twice, parbendazole) where the model insists the line is resistant. That is a
knowledge gap of the model, not label noise; the pipeline lists these rather than silently dropping them because in
other datasets the same verdict is how label errors surface.

**What the probes cannot catch: a prior.** A model convinced that "most drugs miss 100 nM" answers No consistently
under every phrasing and passes the probes on every No question. Qwen3-8B does exactly this (right on 82% of No
questions, 16% of Yes questions), and all of its robust questions are No. The final step therefore keeps equal
numbers of Yes and No rows. For the 8B model that leaves zero rows, which is the honest answer: its data would teach
the prior.

**How strict to be** is a trade-off between yield and protection from lucky guesses (`filter.min_probe_agreement`):

| Share of probes that must be right | Questions with rows | Rows (before de-dup/balance) | Chance a 50/50 guesser passes |
|---|---|---|---|
| 0.5 | 9 | 16 | 69% |
| 0.75 | 7 | 14 | 31% |
| **1.0 (default)** | 6 | 12 | **6%** |

## 4. Trace checks

Each completion is parsed by a tiered parser and then checked. Hard checks reject; soft checks lower the score that
ranks the survivors of a question.

| Check | Catches | Type |
|---|---|---|
| `format` | unclosed `<think>` (truncation), empty reasoning, no or several answers, a letter that is not an option, "Answer: B. No" when B is Yes. Repairable cases are *recovered and flagged*, never silently fixed: missing opening tag, reasoning written after an empty `<think></think>`, no tags, "Answer: Yes", "the decision is: A. Yes", a bare final "B. No" (only if letter and text agree). | hard |
| `complete` | output cut off by the token limit | hard |
| `correct` | final answer ≠ ground truth | hard |
| `faithful` | the reasoning concludes one option and the final line gives the other | hard |
| `mechanism_grounded` | the drug's mechanism disagrees with the Broad Drug Repurposing Hub (synonyms handled; the drug's own name is removed first so "I-BRD9" cannot vouch for "BRD9") | hard |
| `mechanism_verifiable` | the drug is not in the reference, so the key claim cannot be checked | hard (configurable) |
| `knows_drug` | "I'm not familiar with X", followed by an answer anyway | hard |
| `mentions_drug` | the reasoning never names the drug (or one component of a combination) | hard |
| `decisive` | the conclusion is an explicit coin flip ("if I have to choose", "cannot determine") | hard |
| `no_citations` | invented references: et al., PMID, DOI, URLs | hard |
| `language` | switching to Chinese mid-trace (a known Qwen failure; DeepSeek-R1 added a reward against it) | hard |
| `no_loops` | degenerate repetition (an 8-word sequence > 10 times, or zlib compression ratio > 5) | hard |
| `length` | under 60 or over 1,400 words | hard |
| `low_hedging`, `not_repetitive`, `few_specific_values`, `no_meta_talk`, `mentions_cell_line`, `explicit_conclusion` | heavy hedging, circular restating, invented IC50 values, "the user is asking…", no explicit conclusion. Each costs 0.1 of the quality score; below 0.75 (3+ soft failures) a trace is not kept, because the one that did reach the dataset before this bar flip-flopped between Yes and No for 800 words. | soft |

**Every threshold was calibrated on real traces, and three changed because of what the traces showed:**
- *Repetition.* The first threshold rejected 9 traces; only 1 was a real loop (one sentence 19 times). Over 190
  traces the median 8-gram repeat count is 2 and the 95th percentile 5, so the hard limit became 10 and 5+ became a
  soft penalty for circular reasoning.
- *"Educated guess".* First a hard reject. In the traces it usually introduces reasoning from the drug class, which
  is what we want, and the probes already catch lucky answers. Now a soft penalty.
- *Unverifiable mechanisms.* Reading the 12 traces that survived an earlier version found 4 about drugs absent from
  the reference, and 3 of them stated a wrong mechanism: PF-00299804 (dacomitinib, an EGFR inhibitor) treated as a
  PARP inhibitor; BRD-K03536150 called a "BRD inhibitor" because of its name (BRD-K is the Broad Institute's ID
  prefix). Drugs that cannot be verified are now rejected by default (`filter.require_verifiable_mechanism`).

The mechanism check is the most valuable one and was audited by hand: of the 16 distinct drugs it rejected in an
earlier pass, 13 were confident inventions (Osi-027 and KX2-391 as PARP inhibitors, SNX-2112 as "a derivative of
bortezomib", JQ-1 as an HDAC inhibitor, tanespimycin as an HDAC inhibitor or a taxane), 2 traces never stated a
mechanism, and 1 was a false alarm (gemcitabine correctly described as a nucleoside analogue), fixed with a synonym.

Traces are never edited beyond stripping the `<think>` tags, as the brief requires.

## 5. Prompts

The system prompt (`prompts.py`, versioned; the version is part of the cache key):
- states the **label definition** (IC50 < 0.1 µM sensitive, > 1 µM not sensitive);
- asks for a **domain scaffold**: drug class/target/mechanism → cell line biology → typical potency against the
  thresholds → decision. Loose enough that traces do not all read the same;
- asks for **epistemic honesty**: say when you do not know, never invent mutations, IC50 values or citations;
- ends with a strict answer format: `Answer: <letter>`.

The model is **never shown the correct answer**. Hinting it (STaR-style rationalisation) would raise yield, but the
traces would justify a known answer instead of reasoning towards it.

Sampling follows Qwen's recommendation for thinking mode (temperature 0.6, top-p 0.95, top-k 20); Qwen documents
that greedy decoding causes repetition loops in thinking mode.

`sft_chat.jsonl` keeps the label definition but drops the scaffold: the student should learn to reason this way
without being told how (context distillation). `prompting.full_prompt` records the exact prompt, as required.

**Three prompt variants, same model, same 50 questions:**

| Variant | Accuracy | Right on Yes / No | Robust questions | Final rows |
|---|---|---|---|---|
| **v1, thinking mode** (default) | 55% | 45% / 64% | 9 | 6 |
| v2: "these IC50s come from in vitro assays; clinical use is not evidence" | 52% | 40% / 65% | 12 | 4 |
| v1, thinking off, `<think>` requested by the prompt only | 52% | 38% / 66% | 4 | 0 |

- **v2 tested a hypothesis from error analysis**: v1 traces argued from clinical use ("taxanes are less effective in
  bladder cancer"). It did not help. Reading the misses showed the cause is knowledge, not framing: the model believes
  proteasome inhibitors and taxanes sit around 0.1–1 µM in cell lines, where they are typically low-nanomolar. A prompt
  cannot supply missing knowledge, so v1 stays.
- **Prompt-only structure fails on a hybrid thinking model.** With thinking disabled, Qwen3's chat template inserts an
  empty `<think></think>`; in 200/200 outputs the model then reasoned *outside* the tags, and in 145 it ended with
  "Therefore, the decision is: A. Yes" instead of the requested line. The parser recovers these (flagged), but the
  traces are shorter (323 vs 770 tokens), hallucinate mechanisms more often (53% vs 49%), and survive the probes less
  often (4 robust questions vs 9). It is 3× faster; for a model without native thinking (e.g. Qwen2.5-3B-Instruct)
  it is the only option and the pipeline supports it (`--mode prompted`).

## 6. Model choice and size trade-offs

Same prompt (v1), same 50 questions, only the model changes:

| | Qwen3-4B (default) | Qwen3-8B | Qwen3.5-4B (community GGUF) |
|---|---|---|---|
| Generation speed, 1 request | 40 tok/s | 22 tok/s | — |
| Server memory (8k context) | 4.2 GB | 6.1 GB | GPU out of memory at 3 parallel requests |
| Accuracy (right on Yes / No) | 55% (45% / 64%) | 49% (16% / 82%) | — |
| Self-consistency: swap / negation | 61% / 67% | 90% / 76% | — |
| Wrong mechanism, where checkable | 49% | 37% | — |
| Robust questions | 9 | 14 (all "No") | — |
| Final balanced rows | **6** | 0 | — |

**Why 4B.** On a 16 GB laptop the trade-off is concrete:
- **8B pros**: knows more (12 points fewer invented mechanisms) and is far more self-consistent.
- **8B cons**: 1.8× slower (its 50-question run took 1 h 50 min here), 1.5× the memory (on a 16 GB Mac shared with the OS
  and Docker, this is where swapping starts), and, decisive here, a strong "resistant" prior: it is right on 16% of
  Yes questions. Consistency built on a prior passes the probes, so its robust set is all-No and teaches the prior.
- **4B pros**: fast enough to sample each question 4 times in about an hour, fits comfortably, and its errors are
  closer to random than systematic, which the probes are designed to catch.
- **4B cons**: barely above chance on this task and invents mechanisms in half of checkable traces, so yield after
  filtering is low (6 rows from 50 questions).
- **Qwen3.5-4B** (released Feb 2026) failed under the same budget: its thinking is much longer (9/9 completed outputs
  hit the 2,048-token limit before answering) and three parallel requests ran the Metal GPU out of memory. It would
  need one request at a time and a ~3× token budget, roughly a 6-hour run.

In short: none of the laptop-sized models knows this domain well. A 3–8B model is a weak *teacher*, so the value of
this pipeline is the verifier around it; the generator is one config line (`llm.model`) and can be swapped for a
stronger teacher without touching anything else.

## 7. Evaluation of the filtering logic

**a. Funnel (main run, original-order completions).** Every stage and every rejection reason is in `report.md`.

| Stage | Completions |
|---|---|
| generated | 100 |
| well-formed and complete | 99 |
| correct final answer (= what a naive filter keeps) | 54 |
| passed every trace check | 16 |
| on a robust question | 12 |
| quality bar, best ≤2 per question, de-duplicated | 11 |
| after Yes/No balance (final) | 6 |

Why the naive filter's 54 were not all kept (a trace can have several reasons): unverifiable mechanism 19, wrong
mechanism 16, reasoning contradicts the answer 6, Yes/No balance 5, question not robust 4, explicit coin flip 3,
other 4.

**b. Strictness sweep** (table in §3): going from a 69% to a 6% chance that a guesser passes costs 4 rows.

**c. Fault injection** (`make stress`). Each of the 36 real traces that passed every check is broken in one known way
and re-checked:

| Fault injected | Check expected to catch it | Caught |
|---|---|---|
| final answer flipped | `faithful` | 36/36 |
| cut off mid-reasoning | `complete` | 36/36 |
| repetition loop appended | `no_loops` | 36/36 |
| language switch (Chinese) | `language` | 36/36 |
| fake citation inserted | `no_citations` | 36/36 |
| guess added to the conclusion | `decisive` | 36/36 |
| "I'm not familiar with this drug" added | `knows_drug` | 36/36 |
| mechanism replaced by a wrong one | `mechanism_grounded` | 32/32 |
| `<think>` tags removed | parser should *recover* it | 36/36 recovered |

This measures that each check works on real text. It does not measure recall on the faults models actually make,
which are subtler; that is what (d) and (e) are for.

**d. Hand review.** Every rejection reason above was spot-checked on real traces, and four rules changed as a
result (§4 and the quality bar). The 6 final traces were then read in full:

| Trace | Verdict |
|---|---|
| Vincristine → Yes | good: microtubule inhibitor, potent in vitro, no resistance mechanism known |
| Trifluoperazine → No | good: antipsychotic (dopamine antagonist), no reason for sub-micromolar cytotoxicity |
| Sorafenib → No | good: multi-kinase inhibitor without a matching dependency |
| Gemcitabine → Yes (×2) | weak: right mechanism, but both estimate the IC50 at ~0.5 µM (which by the definition means *not* sensitive) and then answer Yes because gemcitabine is a bladder-cancer drug |
| Alpelisib → No | mixed: right target (PI3Kα) and logic (needs a PIK3CA mutation), wrong indication ("HER2-positive breast cancer"; it is HR+/HER2−) |

So 3 of 6 are traces I would train on as they are. One more trace about I-BRD9 (claimed to target the androgen
receptor and to be a BET inhibitor; bladder cancer's main driver called FGFR2) passed every check, because the
mechanism check verifies the class (bromodomain), not the specific target, and left the dataset only through the
Yes/No balance. These residual errors are fine-grained facts: target, indication, driver genes, and internal
consistency of a potency estimate with the definition. Catching them needs richer references or a stronger judge (§8).

**e. Blind audit by an LLM judge** (`make audit-sheet`, then `make audit-score`). The sample is 22 traces whose
final answer is correct, so all would pass the naive filter: all 6 that this filter accepted and 16 of the 48 it
rejected, shuffled, with the decisions hidden. The judge (ChatGPT) saw only the traces and a rubric
(`src/reasoning_traces/judge_prompt.md`; `make audit-sheet` writes it together with the traces to
`judge_input.md`). It scored four criteria (mechanism correct, no invented facts, reasoning supports the answer,
commits to an answer), labelled a trace good only if all four passed, and gave its confidence.

| | Judge: good | Judge: bad | Share good |
|---|---|---|---|
| Accepted by this filter | 4 | 2 | 67% (95% CI 30–90%) |
| Rejected by this filter | 3 | 13 | 19% (95% CI 7–43%) |

- **Agreement with the filter: 17 of 22 (77%).** The judge marked 4 of the 5 disagreements as low-confidence.
- **Precision.** Re-weighted to the real population (6 accepted, 48 rejected), a naive "answer matches" filter keeps
  traces that are good about 24% of the time; this filter, 67%. The intervals are wide at this sample size.
- **The mechanism checks hold up.** Of the 12 traces rejected for a wrong or unverifiable mechanism, the judge rated
  11 bad and independently found the mechanism itself wrong in 10 (Methylstat called a folate antagonist, HBX-41108
  and CR-1-31B called PARP inhibitors, JQ-1 called an HDAC inhibitor, SNX-2112 called a proteasome inhibitor, …).
  The "unverifiable ⇒ reject" rule blocked 6 bad traces at the cost of 1 good one.

The five disagreements:

| Trace | Filter | Judge | Reading |
|---|---|---|---|
| Gemcitabine → Yes (a final trace) | kept | bad (low confidence): invented DSH1 study results and a bladder-line IC50 | judge is right; the hand review (d) flagged the same ~0.5 µM estimate |
| Alpelisib → No (a final trace) | kept | bad (high): calls it a HER2-positive drug | judge is right; (d) also flagged it |
| Momelotinib → No | rejected: drug not in the reference; the answer also flipped when the question was negated | good (low) | the text is fine; a reference gap, but the probes drop the question anyway |
| Paclitaxel → Yes | rejected: two identical asks gave opposite answers | good (low) | by design, see below |
| Methotrexate → Yes | rejected: the answer flipped when the options were swapped | good (low; "argument weak") | by design, see below |

Both errors the filter let through are the kind (d) already found: indications and invented study claims, which no
check here verifies. The three good traces it rejected were lost to rules about the *model*, not the text: the drug
is outside the reference, or the model answers differently when asked again or with the options swapped. A judge
reading one trace cannot see that. Such traces read well but rationalise one side of a coin flip, and this filter
rejects them on purpose. Caveats: one judge, one run, 22 traces, so the intervals are wide; a domain expert's labels
would make this stronger. The judge's labels and reasons are in `outputs/main/audit/llm_judge.csv`, the scored
result in `audit_results.md`.

**Tests**: 68 unit tests cover ~25 malformed-output patterns, every check (firing and staying quiet), the probe
answer mapping, verdict logic, label balancing, the output schema, audit scoring, and that no metadata reaches a
prompt.

## 8. Limitations and next steps

- **Small numbers.** 50 questions give ±14-point confidence intervals on accuracy; the comparisons in §5–6 are
  directional. The pipeline runs on all 604 train rows (`--n 604`); at the observed yield that is ~70 rows and
  ~12 hours on this laptop.
- **Better teacher, same verifier.** A larger model on a GPU server, or an API model, behind the same
  OpenAI-compatible client. The probes, checks and audit stay as they are.
- **Finer fact checking.** The mechanism check verifies the drug class; checking the specific target, the
  indication and cell-line driver claims (DepMap mutation calls) would catch the errors found in the hand review.
- **Wider fact checking.** The reference covers 54% of train drugs. CTRP's own compound annotations or ChEMBL would
  cover the tool compounds (BRD-K…) and newer drugs (Momelotinib, rejected in the audit for this reason, is missing).
- **An LLM judge as a filter stage.** The blind audit shows a strong judge catches what these checks miss (wrong
  indications, invented study claims). Run it on every trace that passes the cheap checks (one call per survivor),
  calibrated against a small set of expert labels.

## Repository layout

```
src/reasoning_traces/
  data.py         load_private_dataset(): HF download, option parsing, label-balanced sampling
  probes.py       question variants: original, options swapped, question negated
  prompts.py      versioned prompts
  llm.py          async OpenAI-compatible client: retries, health check, <think> handling
  generate.py     sampling plan + resumable cache
  parsing.py      tiered parser: strict -> recovered -> failed
  reference.py    drug mechanism reference (Broad Drug Repurposing Hub, downloaded on first run)
  checks.py       per-trace checks (hard reject / soft penalty)
  consensus.py    per-question verdict from the probes
  selection.py    ranking, de-duplication, per-question cap, Yes/No balance
  schema.py       output contract (pydantic)
  pipeline.py     orchestration + exports; generate_reasoning_traces()
  report.py       metrics + report.md
  evaluation.py   fault injection + blind audit (human or LLM judge)
  judge_prompt.md rubric for the LLM judge
  cli.py          `traces` command
configs/          default.yaml: every setting with its default (override with --config)
scripts/          experiment grid, run comparison, model benchmark
tests/            unit tests
main.py           starter-stub entry point
```

The private dataset and everything derived from it (`data/`, `outputs/`) are git-ignored and never committed.
