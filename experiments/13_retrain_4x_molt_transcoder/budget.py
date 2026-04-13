"""Budget-tracking wrapper around delphi's OpenRouter client.

Intercepts each httpx response to capture `usage.prompt_tokens` /
`usage.completion_tokens`, accumulates spend, and raises BudgetExceeded
before the next call if the cap is hit.

Pricing is hard-coded for Llama-3.1-70B-Instruct on OpenRouter:
    $0.40 per M input tokens, $0.40 per M output tokens.
Update PRICE_* constants if the provider or model changes.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

import httpx

# delphi.clients.__init__ imports from .offline which imports vllm.
# We don't use Offline (OpenRouter only), so stub vllm to satisfy that import
# without forcing a vllm install that would break our torch + CUDA setup.
if "vllm" not in sys.modules:
    def _mk(name, attrs=None):
        m = types.ModuleType(name)
        m.__path__ = []
        for k, v in (attrs or {}).items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    _mk("vllm", {"LLM": None, "SamplingParams": None})
    _mk("vllm.distributed")
    _mk("vllm.distributed.parallel_state", {
        "destroy_model_parallel": lambda *a, **k: None,
        "destroy_distributed_environment": lambda *a, **k: None,
        "get_tensor_model_parallel_rank": lambda *a, **k: 0,
        "get_pipeline_model_parallel_rank": lambda *a, **k: 0,
    })
    _mk("vllm.inputs", {"TokensPrompt": dict})

from delphi.clients.openrouter import OpenRouter
from delphi.clients.client import Response

PRICE_IN_PER_M = 0.40
PRICE_OUT_PER_M = 0.40


class BudgetExceeded(RuntimeError):
    """Raised when cumulative spend would exceed the configured cap."""


class CostTracker:
    """Thread-safe running cost accumulator with a hard cap."""

    def __init__(self, cap_usd: float, log_path: Path | None = None):
        self.cap = cap_usd
        self.spent = 0.0
        self.n_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()
        self._t0 = time.time()
        self.log_path = log_path
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, prompt_tokens: int, completion_tokens: int) -> None:
        delta = (
            prompt_tokens * PRICE_IN_PER_M + completion_tokens * PRICE_OUT_PER_M
        ) / 1e6
        with self._lock:
            self.spent += delta
            self.n_calls += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            should_raise = self.spent > self.cap
            snapshot = self.snapshot()
        if self.log_path is not None:
            with self.log_path.open("a") as f:
                f.write(json.dumps(snapshot) + "\n")
        if should_raise:
            raise BudgetExceeded(
                f"Spend ${snapshot['spent']:.4f} > cap ${self.cap:.2f} "
                f"after {snapshot['n_calls']} calls"
            )

    def snapshot(self) -> dict:
        return {
            "spent": round(self.spent, 6),
            "cap": self.cap,
            "n_calls": self.n_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "elapsed_s": round(time.time() - self._t0, 2),
        }

    def remaining(self) -> float:
        return self.cap - self.spent


class BudgetedOpenRouter(OpenRouter):
    """OpenRouter subclass that records per-response token usage to a
    CostTracker and retries rate-limit / transient failures with exponential
    backoff instead of delphi's single-attempt default.
    """

    def __init__(
        self,
        model: str,
        tracker: CostTracker,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        max_tokens: int = 500,
        temperature: float = 0.0,
        max_retries: int = 6,
        base_backoff_s: float = 2.0,
    ):
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self.tracker = tracker
        self._max_retries = max_retries
        self._base_backoff = base_backoff_s

    def postprocess(self, response: httpx.Response) -> Response:
        rj = response.json()
        usage = rj.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        if pt or ct:
            self.tracker.record(pt, ct)
        msg = rj["choices"][0]["message"]["content"]
        return Response(text=msg)

    async def generate(self, prompt, max_retries=None, **kwargs):
        import asyncio

        kwargs.pop("schema", None)
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        temperature = kwargs.pop("temperature", self.temperature)
        data = {
            "model": self.model,
            "messages": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        retries = max_retries if max_retries is not None else self._max_retries

        last_err = None
        for attempt in range(retries):
            try:
                response = await self.client.post(
                    url=self.url, json=data, headers=self.headers, timeout=120,
                )
                status = response.status_code
                if status == 429 or 500 <= status < 600:
                    # Respect Retry-After if present
                    ra = response.headers.get("retry-after")
                    wait = float(ra) if ra and ra.replace(".", "", 1).isdigit() else \
                        min(self._base_backoff * (2 ** attempt), 60.0)
                    last_err = RuntimeError(f"HTTP {status}: {response.text[:200]}")
                    await asyncio.sleep(wait)
                    continue
                # Non-429 non-5xx — try to postprocess, bubble real errors
                return self.postprocess(response)
            except Exception as e:
                last_err = e
                wait = min(self._base_backoff * (2 ** attempt), 60.0)
                await asyncio.sleep(wait)

        raise RuntimeError(
            f"OpenRouter generate failed after {retries} attempts; last={last_err}"
        )


def check_openrouter_credits(api_key: str) -> dict:
    """Query OpenRouter's credits endpoint; return raw JSON or raise."""
    r = httpx.get(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15.0,
    )
    r.raise_for_status()
    return r.json()
