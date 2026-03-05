from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..context_manager import ContextManager
from ..summarizer import Summarizer
from ..token_counter import get_token_counter


class ContextAwareAnthropic:
    def __init__(
        self,
        anthropic_client,
        model: str = "claude-3-5-sonnet-latest",
        token_budget: int = 4096,
        system_prompt: Optional[str] = None,
        summarize_mode: str = "extractive",
        context_manager: Optional[ContextManager] = None,
    ) -> None:
        self._client = anthropic_client
        self.model = model

        if context_manager is not None:
            self.context = context_manager
        else:
            mode = "extractive" if summarize_mode == "abstractive" else summarize_mode
            summarizer = Summarizer(mode=mode)
            self.context = ContextManager(
                token_budget=token_budget,
                summarizer=summarizer,
                token_counter=get_token_counter(model),
            )

        if system_prompt:
            self.context.add_system(system_prompt)

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
        self.context.add_user(user_message)
        context_messages = self.context.get_messages()
        payload_messages, system_text = self._convert_messages(context_messages)

        kwargs: Dict[str, Any] = dict(
            model=model or self.model,
            messages=payload_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **(extra_params or {}),
        )
        if system_text:
            kwargs["system"] = system_text

        if stream:
            reply = self._stream(kwargs)
        else:
            response = self._client.messages.create(**kwargs)
            reply = self._extract_text(response)

        self.context.add_assistant(reply)
        return reply

    def _stream(self, kwargs: dict[str, Any]) -> str:
        chunks: List[str] = []
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    chunks.append(text)
        return "".join(chunks)

    def _convert_messages(self, messages: List[dict[str, str]]) -> tuple[List[dict[str, str]], str]:
        system_parts: List[str] = []
        converted: List[dict[str, str]] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                if content:
                    system_parts.append(content)
                continue
            if role not in ("user", "assistant"):
                role = "user"
            converted.append({"role": role, "content": content})

        if not converted:
            converted.append({"role": "user", "content": ""})

        return converted, "\n\n".join(system_parts)

    def _extract_text(self, response: Any) -> str:
        parts = getattr(response, "content", []) or []
        text_parts: List[str] = []
        for part in parts:
            part_type = getattr(part, "type", None)
            if part_type == "text":
                text_parts.append(getattr(part, "text", "") or "")
        if text_parts:
            return "".join(text_parts)
        return ""

    def set_system(self, prompt: str) -> None:
        self.context.replace_system(prompt)

    def reset(self, keep_system: bool = True) -> None:
        self.context.clear(keep_system=keep_system)

    def stats(self) -> dict[str, float | int]:
        return self.context.stats()

    def raw_create(self, **kwargs) -> Any:
        return self._client.messages.create(**kwargs)
