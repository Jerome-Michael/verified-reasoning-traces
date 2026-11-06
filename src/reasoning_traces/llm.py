"""Minimal async client for any OpenAI-compatible local server (Docker Model Runner, llama-server, Ollama, LM Studio)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import httpx

from .config import LLMConfig


class LLMError(RuntimeError):
    pass


@dataclass
class ChatResult:
    text: str  # full raw output; includes the <think> block when the model produced one
    finish_reason: str | None
    completion_tokens: int | None
    latency_s: float


def _norm_model_id(model_id: str) -> str:
    m = model_id.lower().removeprefix("huggingface.co/").removeprefix("hf.co/")
    return m.removesuffix(":latest")


SETUP_HINT = """Start the local model server (Docker Desktop >= 4.62):
  docker desktop enable model-runner --tcp=12434
  docker model install-runner --backend llama.cpp
  docker model pull {model}
Or point LLM_BASE_URL at another OpenAI-compatible server (e.g. Ollama: http://localhost:11434/v1)."""


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = httpx.AsyncClient(base_url=cfg.base_url.rstrip("/"), timeout=cfg.timeout_s)

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def health_check(self) -> list[str]:
        """Fails fast with an actionable message instead of 200 timeouts later."""
        try:
            resp = await self._client.get("/models", timeout=10)
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(
                f"Cannot reach a model server at {self.cfg.base_url} ({e.__class__.__name__}).\n"
                + SETUP_HINT.format(model=self.cfg.model)
            ) from e
        available = [m.get("id", "") for m in resp.json().get("data", [])]
        if available and _norm_model_id(self.cfg.model) not in {_norm_model_id(a) for a in available}:
            raise LLMError(
                f"Model {self.cfg.model!r} is not available on the server. Available: {available}\n"
                + SETUP_HINT.format(model=self.cfg.model)
            )
        return available

    def _body(self, messages: list[dict[str, str]], seed: int) -> dict:
        c = self.cfg
        body = {
            "model": c.model,
            "messages": messages,
            "temperature": c.temperature,
            "top_p": c.top_p,
            "top_k": c.top_k,
            "max_tokens": c.max_tokens,
            "seed": seed,
            # llama.cpp: return the raw text with <think> tags so *our* parser sees exactly what the
            # model wrote. Servers that ignore this and split reasoning out are handled in chat().
            "reasoning_format": "none",
        }
        if c.reasoning_mode == "prompted":
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    async def chat(self, messages: list[dict[str, str]], seed: int) -> ChatResult:
        body = self._body(messages, seed)
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            start = time.perf_counter()
            try:
                resp = await self._client.post("/chat/completions", json=body)
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(f"server error {resp.status_code}", request=resp.request, response=resp)
                if resp.status_code >= 400:  # bad request: retrying will not help
                    raise LLMError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                data = resp.json()
                choice = data["choices"][0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                reasoning = message.get("reasoning_content") or message.get("reasoning")
                text = f"<think>\n{reasoning}\n</think>\n\n{content}" if reasoning else content
                return ChatResult(
                    text=text,
                    finish_reason=choice.get("finish_reason"),
                    completion_tokens=(data.get("usage") or {}).get("completion_tokens"),
                    latency_s=time.perf_counter() - start,
                )
            except (httpx.TransportError, httpx.HTTPStatusError, KeyError, ValueError) as e:
                last_error = e
                if attempt < self.cfg.max_retries:
                    await asyncio.sleep(2**attempt)
        raise LLMError(f"Generation failed after {self.cfg.max_retries + 1} attempts: {last_error!r}")
