from __future__ import annotations

from typing import Any, List

from ..context_manager import ContextManager

try:
    from llama_index.core.memory import BaseChatMemoryBuffer  # pyright: ignore[reportMissingImports]
    from llama_index.core.base.llms.types import ChatMessage, MessageRole  # pyright: ignore[reportMissingImports]

    _llamaindex_available = True
except ImportError:
    _llamaindex_available = False

    class BaseChatMemoryBuffer:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    ChatMessage = None  # type: ignore
    MessageRole = None  # type: ignore


_BaseChatMemoryBuffer = BaseChatMemoryBuffer


def _role_to_neuromem(value: Any) -> str:
    role_value = str(value).lower() if value is not None else "user"
    if "system" in role_value:
        return "system"
    if "assistant" in role_value:
        return "assistant"
    return "user"


def _role_to_llamaindex(role: str) -> Any:
    if MessageRole is None:
        return role
    if role == "system":
        return getattr(MessageRole, "SYSTEM", "system")
    if role == "assistant":
        return getattr(MessageRole, "ASSISTANT", "assistant")
    return getattr(MessageRole, "USER", "user")


def _to_chat_message(message: dict[str, str]) -> Any:
    if ChatMessage is None:
        return {
            "role": message.get("role", "user"),
            "content": message.get("content", ""),
        }
    return ChatMessage(
        role=_role_to_llamaindex(message.get("role", "user")),
        content=message.get("content", ""),
    )


class NeuromemChatMemoryBuffer(_BaseChatMemoryBuffer):  # pyright: ignore[reportGeneralTypeIssues]

    def __init__(
        self,
        token_limit: int = 4096,
        *,
        token_budget: int | None = None,
        context_manager: ContextManager | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        budget = token_budget if token_budget is not None else token_limit
        self.token_limit = budget
        self._context = context_manager or ContextManager(token_budget=budget)
        if system_prompt:
            self._context.add_system(system_prompt)

    def put(self, message: Any) -> None:
        role = _role_to_neuromem(getattr(message, "role", "user"))
        content = getattr(message, "content", "")
        self._context.add(role, str(content))

    def get(self, *args: Any, **kwargs: Any) -> List[Any]:
        return self.get_all()

    def get_all(self) -> List[Any]:
        return [_to_chat_message(m) for m in self._context.get_messages()]

    def reset(self) -> None:
        self._context.clear(keep_system=False)
