#!/usr/bin/env bash
# Runs the comparison grid reported in the README (same 50 questions for every run).
# Generations are cached, so re-running this script only generates what is missing.
set -euo pipefail
export PYTHONPATH=src
UV=${UV:-uv}

run() { echo "=== $*"; $UV run traces run "$@" 2>&1 | grep -v "gen/s\]$" | tail -4; }
# Free memory before switching models: two resident models make a 16 GB Mac swap.
switch_model() { docker model unload --all >/dev/null 2>&1 || true; }

run --prompt v1 --out-dir outputs/qwen3-4b_native_v1 --concurrency 3
run --prompt v2 --out-dir outputs/qwen3-4b_native_v2 --concurrency 3
run --prompt v1 --mode prompted --out-dir outputs/qwen3-4b_prompted_v1 --concurrency 3

# Model comparison: prompt fixed at v1, only the model changes.
if [[ "${MODELS:-0}" == "1" ]]; then
  switch_model
  run --prompt v1 --model hf.co/qwen/qwen3-8b-gguf --out-dir outputs/qwen3-8b_native_v1 --concurrency 3
  switch_model
  run --prompt v1 --model hf.co/unsloth/qwen3.5-4b-gguf:Q4_K_M --out-dir outputs/qwen3.5-4b_native_v1 --concurrency 3
fi
