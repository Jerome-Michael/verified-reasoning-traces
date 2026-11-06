"""Same flow as the starter stub in the brief: `uv run python main.py`.

`make run` / `uv run traces run` does the same with more options (see README).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from reasoning_traces.config import Config  # noqa: E402
from reasoning_traces.data import load_private_dataset  # noqa: E402
from reasoning_traces.pipeline import generate_reasoning_traces  # noqa: E402

if __name__ == "__main__":
    cfg = Config.load()
    print("Fetching private Q&A dataset...")
    raw_qa = load_private_dataset(cfg.data.n_questions, cfg.data.split, cfg.data.sampling, cfg.data.seed)
    print(f"Loaded {len(raw_qa)} pairs. Running reasoning generation...")
    sft_dataset = generate_reasoning_traces(raw_qa, cfg)
    print(f"Generated {len(sft_dataset)} high-quality training pairs. Output and report: {cfg.out_dir}/")
