"""
neuromem.integrations.openai
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Drop-in context-aware wrapper around the OpenAI Chat Completions API.

Usage::

    import openai
    from neuromem.integrations.openai import ContextAwareOpenAI

    client = ContextAwareOpenAI(
        openai_client=openai.OpenAI(),
        token_budget=8000,
        model="gpt-4o",
    )

    # Exactly like the standard API — but your context is auto-managed
    reply = client.chat("Tell me about relativity.")
    reply = client.chat("Can you give me an example?")   # history preserved

    print(client.context.stats())
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from ..context_manager import ContextManager
from ..scorer import MessageScorer
from ..summarizer import Summarizer
from ..pruner import Pruner


class ContextAwareOpenAI:
    """
    Wraps an ``openai.OpenAI`` client and manages the context window automatically.

    Parameters
    ----------
    openai_client :
        A pre-configured ``openai.OpenAI()`` instance.
    model : str
        Default model to use for chat completions.
    token_budget : int
        Maximum tokens for the context window.
    system_prompt : str, optional
        Initial system message inserted on construction.
    summarize_mode : "extractive" | "abstractive"
        How to summarize old context.  Use "abstractive" for better quality
        (requires the LLM to be called for summarization).
    context_manager : ContextManager, optional
        Supply your own pre-configured ContextManager to override all defaults.
    """

    def __init__(
        self,
        openai_client,
        model: str = "gpt-4o-mini",
        token_budget: int = 4096,
        system_prompt: Optional[str] = None,
        summarize_mode: str = "extractive",
        context_manager: Optional[ContextManager] = None,
    ) -> None:
        self._client = openai_client
        self.model = model

        if context_manager is not None:
            self.context = context_manager
        else:
            summarizer = Summarizer(
                mode=summarize_mode,
                client=openai_client if summarize_mode == "abstractive" else None,
                model=model,
            )
            self.context = ContextManager(
                token_budget=token_budget,
                summarizer=summarizer,
            )

        if system_prompt:
            self.context.add_system(system_prompt)

    # ------------------------------------------------------------------

    def chat(
        self,
        user_message: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        extra_params: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Send *user_message*, manage context automatically, return the reply.

        Parameters
        ----------
        user_message : str
        model : str, optional  — overrides instance default
        temperature : float
        max_tokens : int
        stream : bool          — if True, streams and returns concatenated text
        extra_params : dict    — additional kwargs forwarded to the API

        Returns
        -------
        str — the assistant's reply content
        """
        self.context.add_user(user_message)
        messages = self.context.get_messages()

        kwargs: Dict[str, Any] = dict(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **(extra_params or {}),
        )

        if stream:
            reply = self._stream(kwargs)
        else:
            response = self._client.chat.completions.create(**kwargs)
            reply = response.choices[0].message.content or ""

        self.context.add_assistant(reply)
        return reply

    def _stream(self, kwargs: dict) -> str:
        kwargs["stream"] = True
        chunks: List[str] = []
        for chunk in self._client.chat.completions.create(**kwargs):
            delta = chunk.choices[0].delta.content or ""
            chunks.append(delta)
        return "".join(chunks)

    # ------------------------------------------------------------------

    def set_system(self, prompt: str) -> None:
        """Replace or set the system prompt."""
        self.context.replace_system(prompt)

    def reset(self, keep_system: bool = True) -> None:
        """Clear conversation history."""
        self.context.clear(keep_system=keep_system)

    def stats(self) -> dict:
        """Return context statistics."""
        return self.context.stats()

    # ------------------------------------------------------------------
    # Raw API pass-through (for advanced usage)
    # ------------------------------------------------------------------

    def raw_create(self, **kwargs) -> Any:
        """
        Direct pass-through to ``openai_client.chat.completions.create``.
        The caller is responsible for message management.
        """
        return self._client.chat.completions.create(**kwargs)
