from __future__ import annotations
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

@dataclass
class LLMCall:
    model: str
    input_tokens: int
    output_tokens: int
    latency: float
    cost: float = 0.0
    step: str = ""
    executed: bool = True

class BaseLLMClient(ABC):
    def __init__(self):
        self.calls: List[LLMCall] = []

    @abstractmethod
    def generate(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 1024, step: str = "generation") -> str:
        ...
    def generate_json(self, system: str, prompt: str, max_tokens: int = 512, step: str = "json_generation") -> Dict[str, Any]:
        text = self.generate(system, [{"role": "user", "content": prompt}], max_tokens=max_tokens, step=step)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            text = text.rsplit("```", 1)[0]
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                return json.loads(text[start:end + 1])
            raise

    def reset_calls(self) -> None:
        self.calls.clear()

    @property
    def total_cost(self) -> float:
        return sum(c.cost for c in self.calls)

_GEMINI_PRICING_PER_1M = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "default": {"input": 0.30, "output": 2.50},}

def _gemini_price(model_name: str) -> Dict[str, float]:
    name = (model_name or "").lower()
    for key, pricing in _GEMINI_PRICING_PER_1M.items():
        if key != "default" and key in name:
            return pricing
    return _GEMINI_PRICING_PER_1M["default"]

class GeminiClient(BaseLLMClient):

    def __init__(self, model: str = "gemini-3.5-flash-lite", api_key: Optional[str] = None, temperature: float = 0.0):
        super().__init__()
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError( "Install the Gemini SDK with: pip install -U google-genai" ) from exc
        self.model = model
        self.temperature = temperature
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.types = types

    def _build_thinking_config(self):
        name = (self.model or "").lower()
        if "gemini-3" in name:
            return self.types.ThinkingConfig(thinking_level="minimal")
        if "pro" in name:
            return None
        return self.types.ThinkingConfig(thinking_budget=0)

    def generate(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 1024, step: str = "generation") -> str:
        contents = []
        for m in messages:
            contents.append(f"{m['role'].upper()}:\n{m['content']}")
        prompt = f"{system}\n\n" + "\n\n".join(contents)
        started = time.perf_counter()
        config_kwargs = dict(temperature=self.temperature, max_output_tokens=max_tokens)
        thinking_config = self._build_thinking_config()
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        response = self.client.models.generate_content( model=self.model, contents=prompt, config=self.types.GenerateContentConfig(**config_kwargs), )
        latency = time.perf_counter() - started
        usage = getattr(response, "usage_metadata", None)
        input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
        if not input_tokens:
            input_tokens = max(1, len(prompt) // 4)
        if not output_tokens:
            output_tokens = max(1, len(response.text or "") // 4)

        pricing = _gemini_price(self.model)
        cost = (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
        self.calls.append(LLMCall( model=self.model, input_tokens=input_tokens, output_tokens=output_tokens, latency=latency, cost=cost, step=step, ))

        text = response.text or ""
        if not text.strip():
            candidate = (getattr(response, "candidates", None) or [None])[0]
            finish_reason = getattr(candidate, "finish_reason", None)
            raise RuntimeError(
                f"GeminiClient: model '{self.model}' returned an empty response "
                f"for step '{step}' (finish_reason={finish_reason!r}, "
                f"input_tokens={input_tokens}, output_tokens={output_tokens}, "
                f"max_tokens={max_tokens}). If finish_reason is MAX_TOKENS, "
                f"raise max_tokens for this call." )
        return text

class MockLLMClient(BaseLLMClient):
    def generate(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 1024, step: str = "generation") -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        context = self._extract(last, "context")
        question = self._extract(last, "question")
        text = "I couldn't find that in the provided documents."
        if context and not context.startswith("(No relevant context"):
            text = f"Based on the retrieved context, the relevant evidence is: {context[:500]}"
        self.calls.append(LLMCall("mock", len(last)//4, len(text)//4, 0.0, 0.0, step))
        return text + " [MockLLMClient]"

    @staticmethod
    def _extract(text: str, tag: str) -> str:
        a, b = f"<{tag}>", f"</{tag}>"
        i, j = text.find(a), text.find(b)
        return text[i + len(a):j].strip() if i >= 0 and j > i else ""

def create_llm_client(backend: str = "gemini", **kwargs) -> BaseLLMClient:
    backend = backend.lower()
    if backend == "mock":
        return MockLLMClient()
    if backend in {"gemini", "auto"}:
        return GeminiClient(**kwargs)
    raise ValueError(f"Unsupported LLM backend: {backend}")