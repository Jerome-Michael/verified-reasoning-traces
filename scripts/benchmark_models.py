"""Single-request speed and memory of each local model (for the model-size section of the README).

uv run python scripts/benchmark_models.py hf.co/qwen/qwen3-4b-gguf hf.co/qwen/qwen3-8b-gguf
"""

import subprocess
import sys
import time

import httpx

URL = "http://localhost:12434/engines/v1/chat/completions"
PROMPT = "Explain in about 200 words how a proteasome inhibitor kills cancer cells."


def server_memory_gb() -> float:
    """Resident memory of the llama.cpp server process, from `ps` (macOS)."""
    out = subprocess.run(["ps", "-axo", "rss,command"], capture_output=True, text=True).stdout
    kb = sum(int(line.split()[0]) for line in out.splitlines() if "llama-server" in line and "grep" not in line)
    return kb / 1024 / 1024


def main(models: list[str]) -> None:
    print("| model | generation tok/s (1 request) | prompt tok/s | server memory (GB) |\n|---|---|---|---|")
    for model in models:
        subprocess.run(["docker", "model", "unload", "--all"], capture_output=True)
        body = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 400,
                "temperature": 0.6, "chat_template_kwargs": {"enable_thinking": False}}
        httpx.post(URL, json=body, timeout=600)  # warm-up (loads the model)
        speeds = []
        for _ in range(2):
            t = httpx.post(URL, json=body, timeout=600).json().get("timings", {})
            speeds.append((t.get("predicted_per_second", 0), t.get("prompt_per_second", 0)))
            time.sleep(1)
        gen = sum(s[0] for s in speeds) / len(speeds)
        pp = sum(s[1] for s in speeds) / len(speeds)
        print(f"| {model} | {gen:.1f} | {pp:.0f} | {server_memory_gb():.1f} |")
    subprocess.run(["docker", "model", "unload", "--all"], capture_output=True)


if __name__ == "__main__":
    main(sys.argv[1:])
