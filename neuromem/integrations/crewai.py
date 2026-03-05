from __future__ import annotations

from typing import Any

from ..context_manager import ContextManager


def _text_from_item(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return str(item.get("content", item.get("text", "")))
    content = getattr(item, "content", None)
    if content is not None:
        return str(content)
    return str(item)


class NeuromemCrewMemory:
    def __init__(
        self,
        token_budget: int = 4096,
        *,
        context_manager: ContextManager | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._context = context_manager or ContextManager(token_budget=token_budget)
        if system_prompt:
            self._context.add_system(system_prompt)

    def save(self, value: Any, metadata: dict[str, Any] | None = None) -> None:
        text = _text_from_item(value)
        role = "user"
        if isinstance(value, dict):
            role = str(value.get("role", "user"))
        else:
            role = str(getattr(value, "role", "user"))
        self._context.add(role, text, metadata=metadata)

    def search(self, query: str, limit: int = 5) -> list[dict[str, str]]:
        lowered = query.lower()
        matches = [
            message
            for message in reversed(self._context.get_raw_history())
            if lowered in message.get("content", "").lower()
        ]
        return list(matches[:limit])

    def reset(self, keep_system: bool = True) -> None:
        self._context.clear(keep_system=keep_system)
