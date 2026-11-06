# PYTHONPATH makes the package importable even when the checkout sits in an iCloud-synced
# folder (Desktop/Documents), where macOS hides uv's .pth file from Python.
export PYTHONPATH := src
UV ?= uv
MODEL ?= hf.co/Qwen/Qwen3-4B-GGUF

.PHONY: setup model check smoke run filter test lint stress audit-sheet audit-score

setup:  ## Python 3.12 + locked dependencies
	$(UV) sync

model:  ## enable Docker Model Runner (host port 12434), download the model (~2.5 GB), cap its context
	docker desktop enable model-runner --tcp=12434
	-docker model install-runner --backend llama.cpp
	docker model pull $(MODEL)
	@# The runner loads the model's full context (40,960 tokens for Qwen3): 8.4 GB resident and heavy swapping
	@# on a 16 GB Mac. A same-name variant with an 8,192-token context (3 parallel requests of prompt + 2,048
	@# output) uses 4.3 GB. `docker model configure` is not enough: it is forgotten when the model unloads.
	-docker model unload --all
	docker model package --from $(MODEL) --context-size 8192 $(MODEL)

check:  ## model server, model and dataset access
	$(UV) run traces check

smoke:  ## 6 questions end to end, a few minutes
	$(UV) run traces run --smoke

run:  ## the full run (50 questions)
	$(UV) run traces run

filter:  ## re-run filtering + report from cached generations (no model calls)
	$(UV) run traces filter

stress:  ## fault-injection evaluation of the filter
	$(UV) run traces stress

audit-sheet:  ## blind sample of traces for a human or LLM judge (writes audit_sheet.csv and judge_input.md)
	$(UV) run traces audit-sheet

LABELS ?= outputs/main/audit/llm_judge.csv
LABELLER ?= LLM judge (ChatGPT), blind
audit-score:  ## compare blind labels with the filter's hidden decisions
	$(UV) run traces audit-score --labels $(LABELS) --labeller "$(LABELLER)"

test:  ## unit tests (no model or network needed)
	$(UV) run pytest -q

lint:  ## static checks (ruff)
	$(UV) run ruff check src tests scripts main.py
