"""
neuromem.summarizer
~~~~~~~~~~~~~~~~~~~
Summarizes older conversation chunks to free context space.

Two modes:
  1. **Extractive** (default, no API key needed): picks the highest-scored
     sentences from each message to form a compressed summary.
  2. **Abstractive**: calls the configured LLM to write a concise summary
     (requires an OpenAI-compatible client to be passed in).
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .scorer import MessageScorer, _estimate_tokens


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SummaryResult:
    """Returned by :meth:`Summarizer.summarize`."""
    summary_text: str
    original_message_count: int
    original_token_count: int
    summary_token_count: int
    compression_ratio: float          # summary_tokens / original_tokens

    def as_message(self) -> dict[str, str]:
        """Return the summary formatted as a system message."""
        return {
            "role": "system",
            "content": (
                f"[Context Summary — {self.original_message_count} messages compressed]\n"
                f"{self.summary_text}"
            ),
        }


# ---------------------------------------------------------------------------
# Extractive helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Very lightweight sentence splitter."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _score_sentence(sentence: str, keyword_set: set[str]) -> float:
    words = re.findall(r"[a-z]+", sentence.lower())
    if not words:
        return 0.0
    kw_hits = sum(1 for w in words if w in keyword_set)
    length_bonus = min(1.0, len(words) / 20)
    return kw_hits * 0.15 + length_bonus * 0.5


_CRITICAL_KW: set[str] = {
    "goal", "objective", "task", "requirement", "decided", "decision",
    "conclusion", "result", "answer", "solution", "error", "bug", "fix",
    "must", "never", "always", "important", "critical", "instruction",
    "rule", "policy",
}


def _extractive_summarize(
    messages: Sequence[dict[str, str]],
    target_ratio: float = 0.35,
    max_sentences_per_msg: int = 3,
) -> str:
    """
    Build a concise extractive summary from *messages*.

    For each message we pick the top N sentences by a heuristic score and
    render them as a bulleted digest.
    """
    lines: List[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "").strip()
        if not content:
            continue

        sentences = _split_sentences(content)
        if not sentences:
            continue

        scored = sorted(
            sentences,
            key=lambda s: _score_sentence(s, _CRITICAL_KW),
            reverse=True,
        )

        # How many sentences to keep?
        keep = max(1, round(len(sentences) * target_ratio))
        keep = min(keep, max_sentences_per_msg)
        picked = scored[:keep]

        # Re-order to original sequence
        order = {s: i for i, s in enumerate(sentences)}
        picked_ordered = sorted(picked, key=lambda s: order.get(s, 0))

        snippet = " ".join(picked_ordered)
        # Trim to ~120 chars if still too long
        if len(snippet) > 240:
            snippet = textwrap.shorten(snippet, width=240, placeholder="…")

        lines.append(f"[{role.upper()}] {snippet}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summarizer
# ---------------------------------------------------------------------------

class Summarizer:
    """
    Compresses a slice of conversation history into a short summary message.

    Parameters
    ----------
    mode : "extractive" | "abstractive"
        Extraction mode.  Use "abstractive" only when you supply a *client*.
    client : optional
        An OpenAI-compatible client (``openai.OpenAI`` instance).  Only used
        in abstractive mode.
    model : str
        Model to use for abstractive summarization.
    target_ratio : float
        For extractive mode: fraction of sentences to retain per message.
    scorer : MessageScorer, optional
        Custom scorer instance.
    """

    ABSTRACTIVE_PROMPT = (
        "You are a concise conversation summarizer.  "
        "Summarize the following conversation segment in no more than "
        "{max_words} words, preserving all decisions, goals, constraints, "
        "and key facts.  Do not add commentary.\n\n"
        "Conversation:\n{conversation}"
    )

    def __init__(
        self,
        mode: str = "extractive",
        client=None,
        model: str = "gpt-4o-mini",
        target_ratio: float = 0.35,
        max_abstractive_words: int = 200,
        scorer: Optional[MessageScorer] = None,
    ) -> None:
        if mode not in ("extractive", "abstractive"):
            raise ValueError("mode must be 'extractive' or 'abstractive'")
        self.mode = mode
        self.client = client
        self.model = model
        self.target_ratio = target_ratio
        self.max_abstractive_words = max_abstractive_words
        self.scorer = scorer or MessageScorer()

    # ------------------------------------------------------------------

    def summarize(self, messages: Sequence[dict[str, str]]) -> SummaryResult:
        """
        Summarize *messages* and return a :class:`SummaryResult`.

        Parameters
        ----------
        messages : list of {"role": str, "content": str}
            The messages to compress.  Should NOT include system messages
            you want to keep verbatim (those should be excluded before calling).
        """
        if not messages:
            return SummaryResult(
                summary_text="",
                original_message_count=0,
                original_token_count=0,
                summary_token_count=0,
                compression_ratio=1.0,
            )

        original_tokens = sum(_estimate_tokens(m.get("content", "")) for m in messages)

        if self.mode == "abstractive" and self.client is not None:
            text = self._abstractive(messages)
        else:
            text = _extractive_summarize(messages, target_ratio=self.target_ratio)

        summary_tokens = _estimate_tokens(text)
        ratio = summary_tokens / original_tokens if original_tokens else 1.0

        return SummaryResult(
            summary_text=text,
            original_message_count=len(messages),
            original_token_count=original_tokens,
            summary_token_count=summary_tokens,
            compression_ratio=round(ratio, 4),
        )

    # ------------------------------------------------------------------

    def _abstractive(self, messages: Sequence[dict[str, str]]) -> str:
        """Call the LLM for an abstractive summary."""
        client = self.client
        if client is None:
            raise RuntimeError("Abstractive summarization requires a client")
        conversation_text = "\n".join(
            f"{m.get('role','user').upper()}: {m.get('content','')}"
            for m in messages
        )
        prompt = self.ABSTRACTIVE_PROMPT.format(
            max_words=self.max_abstractive_words,
            conversation=conversation_text,
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_abstractive_words * 2,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    # ------------------------------------------------------------------

    def should_summarize(
        self,
        messages: Sequence[dict[str, str]],
        token_budget: int,
        summarize_threshold: float = 0.8,
    ) -> bool:
        """
        Return True when the context has consumed *summarize_threshold*
        fraction of *token_budget* and summarization would help.
        """
        used = sum(_estimate_tokens(m.get("content", "")) for m in messages)
        return used >= token_budget * summarize_threshold
