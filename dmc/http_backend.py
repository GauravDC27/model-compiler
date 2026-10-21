"""Backend for an OpenAI-compatible completions endpoint (vLLM, SGLang), for GPU Phase 0.

``score`` sends ``prompt + continuation`` with ``echo=true`` and sums the log-probs of the
tokens whose text offset lies at or beyond the end of the prompt. ``generate`` uses greedy
decoding (``temperature=0``) or the server's seeded sampler. The rendered chat prompt is built
locally from the tokenizer's chat template (with an explicit system turn, R13a), so the bytes
sent are under our control and hashable.

Only the standard library is used for HTTP so the backend runs anywhere.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Sequence


class OpenAICompatBackend:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        render: Callable[[str, str, str | None], str] | None = None,
        timeout: float = 300.0,
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"{model}@{self.base_url}"
        self.timeout = timeout
        self.api_key = api_key
        self._render = render

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {})},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def render_chat(self, system: str, user: str, now: str | None = None) -> str:
        if self._render is None:
            raise ValueError("pass render= (e.g. a tokenizer chat-template function) to render chats")
        return self._render(system, user, now)

    def score(self, prompt: str, continuations: Sequence[str]) -> list[float]:
        out = []
        for cont in continuations:
            body = self._post("/v1/completions", {
                "model": self.model, "prompt": prompt + cont, "max_tokens": 1,
                "temperature": 0, "echo": True, "logprobs": 1,
            })
            lp = body["choices"][0]["logprobs"]
            total = 0.0
            n_prompt_chars = len(prompt)
            n_echo = len(lp["tokens"]) - 1  # the last token is the generated one
            for i in range(n_echo):
                if lp["text_offset"][i] >= n_prompt_chars:
                    total += lp["token_logprobs"][i]
            out.append(total)
        return out

    def generate(self, prompt: str, max_new_tokens: int, seed: int | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "max_tokens": max_new_tokens}
        if seed is None:
            payload["temperature"] = 0
        else:
            payload.update({"temperature": 1.0, "seed": seed})
        return self._post("/v1/completions", payload)["choices"][0]["text"]

    def lock_fields(self) -> dict[str, object]:
        info: dict[str, object] = {"engine": "openai-compat", "endpoint": self.base_url,
                                   "model": self.model}
        try:
            req = urllib.request.Request(self.base_url + "/version")
            with urllib.request.urlopen(req, timeout=10) as resp:
                info["engine_version"] = json.loads(resp.read().decode("utf-8")).get("version")
        except Exception:  # noqa: BLE001 - version endpoint is optional
            info["engine_version"] = None
        return info
