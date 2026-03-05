from __future__ import annotations

import importlib
from abc import ABC, abstractmethod


class TokenCounter(ABC):
    @abstractmethod
    def count(self, text: str) -> int:
        ...


class GPTTokenCounter(TokenCounter):
    def count(self, text: str) -> int:
        return max(1, len(text) // 4)


class TiktokenCounter(TokenCounter):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._fallback = GPTTokenCounter()
        self._encoding = None

        try:
            tiktoken = importlib.import_module("tiktoken")

            try:
                self._encoding = tiktoken.encoding_for_model(model)
            except Exception:
                self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None

    def count(self, text: str) -> int:
        if self._encoding is None:
            return self._fallback.count(text)
        return max(1, len(self._encoding.encode(text)))


class ClaudeTokenCounter(TokenCounter):
    def count(self, text: str) -> int:
        return max(1, int(len(text) // 3.5))


def get_token_counter(model: str) -> TokenCounter:
    model_name = (model or "").lower()
    if "claude" in model_name:
        return ClaudeTokenCounter()
    if model_name.startswith("gpt") or "o1" in model_name or "o3" in model_name:
        return GPTTokenCounter()
    return GPTTokenCounter()
