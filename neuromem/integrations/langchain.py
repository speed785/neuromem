"""
neuromem.integrations.langchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LangChain-compatible memory class that uses neuromem for intelligent
context compression.

Usage (LangChain ≥ 0.1)::

    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI
    from neuromem.integrations.langchain import NeuromemMemory

    memory = NeuromemMemory(token_budget=6000)

    chain = ConversationChain(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        memory=memory,
        verbose=True,
    )

    chain.predict(input="Hello, who are you?")
    chain.predict(input="Tell me about black holes.")
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..context_manager import ContextManager
from ..scorer import MessageScorer
from ..summarizer import Summarizer
from ..pruner import Pruner

try:
    from langchain.memory.chat_memory import BaseChatMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.schema.messages import get_buffer_string
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    # Provide a stub so the module imports without LangChain installed
    class BaseChatMemory:  # type: ignore
        pass


def _require_langchain():
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is required for this integration. "
            "Install it with: pip install langchain langchain-openai"
        )


# ---------------------------------------------------------------------------

class NeuromemMemory(BaseChatMemory):
    """
    LangChain memory backed by neuromem's ContextManager.

    Attributes
    ----------
    token_budget : int
    memory_key : str
        The key injected into the chain's input dict (default: "history").
    return_messages : bool
        If True, return a list of BaseMessage objects; else a formatted string.
    human_prefix / ai_prefix : str
    system_prompt : str, optional
    """

    memory_key: str = "history"
    return_messages: bool = True
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    token_budget: int = 4096
    system_prompt: str = ""

    # neuromem internals (excluded from pydantic serialisation)
    _context: ContextManager = None  # type: ignore

    def __init__(self, **kwargs):
        _require_langchain()
        super().__init__(**kwargs)
        self._context = ContextManager(token_budget=self.token_budget)
        if self.system_prompt:
            self._context.add_system(self.system_prompt)

    # ------------------------------------------------------------------
    # LangChain BaseChatMemory interface
    # ------------------------------------------------------------------

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return managed messages for injection into the prompt."""
        _require_langchain()
        raw_msgs = self._context.get_messages()
        lc_msgs = _to_langchain_messages(raw_msgs)

        if self.return_messages:
            return {self.memory_key: lc_msgs}
        return {self.memory_key: get_buffer_string(
            lc_msgs,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Called by LangChain after each chain step."""
        human_text = inputs.get("input") or inputs.get("human_input") or ""
        ai_text = outputs.get("response") or outputs.get("output") or ""
        self._context.add_user(str(human_text))
        self._context.add_assistant(str(ai_text))

    def clear(self) -> None:
        """Reset conversation (keeps system prompt if set)."""
        self._context.clear(keep_system=bool(self.system_prompt))

    # ------------------------------------------------------------------
    # Extra helpers
    # ------------------------------------------------------------------

    @property
    def context_stats(self) -> dict:
        return self._context.stats()

    @property
    def neuromem_context(self) -> ContextManager:
        """Direct access to the underlying ContextManager."""
        return self._context


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _to_langchain_messages(messages: List[dict]):
    """Convert neuromem dicts → LangChain BaseMessage objects."""
    _require_langchain()
    result = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            result.append(SystemMessage(content=content))
        elif role == "user":
            result.append(HumanMessage(content=content))
        else:
            result.append(AIMessage(content=content))
    return result


def _from_langchain_messages(messages) -> List[dict]:
    """Convert LangChain BaseMessage objects → neuromem dicts."""
    _require_langchain()
    result = []
    for m in messages:
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        result.append({"role": role, "content": m.content})
    return result
