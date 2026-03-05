"""
neuromem.context_manager
~~~~~~~~~~~~~~~~~~~~~~~~
Core class that orchestrates message history, scoring, summarization,
and pruning within a configurable token budget.

Quick start::

    from neuromem import ContextManager

    cm = ContextManager(token_budget=8000)
    cm.add_system("You are a helpful assistant.")
    cm.add_user("Tell me about quantum computing.")
    cm.add_assistant("Quantum computing uses qubits…")

    # Get a pruned, budget-aware message list ready for the API
    messages = cm.get_messages()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .observability import MemoryLogger
from .pruner import Pruner, PruneResult
from .scorer import MessageScorer
from .summarizer import Summarizer
from .token_counter import GPTTokenCounter, TokenCounter


# ---------------------------------------------------------------------------
# Internal message record
# ---------------------------------------------------------------------------

@dataclass
class _MessageRecord:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# ContextManager
# ---------------------------------------------------------------------------

class ContextManager:
    """
    Manages an LLM conversation history with automatic token-budget enforcement.

    The manager keeps an internal log of every message ever added.  When the
    context grows beyond *token_budget* tokens it calls the :class:`~neuromem.pruner.Pruner`
    which may summarize and/or drop low-importance messages.

    Parameters
    ----------
    token_budget : int
        Maximum total tokens for the context window (default: 4096).
    auto_prune : bool
        Automatically prune when adding a message would exceed the budget.
    prune_threshold : float
        Fraction of budget at which auto-pruning triggers (default: 0.9).
    scorer : MessageScorer, optional
    summarizer : Summarizer, optional
    pruner : Pruner, optional
        Supply your own fully-configured instances to override defaults.
    always_keep_last_n : int
        Forwarded to the default Pruner if you don't supply one.
    """

    def __init__(
        self,
        token_budget: int = 4096,
        auto_prune: bool = True,
        prune_threshold: float = 0.9,
        scorer: Optional[MessageScorer] = None,
        summarizer: Optional[Summarizer] = None,
        pruner: Optional[Pruner] = None,
        always_keep_last_n: int = 4,
        token_counter: Optional[TokenCounter] = None,
        memory_logger: Optional[MemoryLogger] = None,
        debug: bool = False,
    ) -> None:
        self.token_budget = token_budget
        self.auto_prune = auto_prune
        self.prune_threshold = prune_threshold
        self._token_counter = token_counter or GPTTokenCounter()
        self._memory_logger = memory_logger or MemoryLogger()
        self.debug = debug

        self._scorer = scorer or MessageScorer()
        self._summarizer = summarizer or Summarizer()
        self._pruner = pruner or Pruner(
            token_budget=token_budget,
            scorer=self._scorer,
            summarizer=self._summarizer,
            always_keep_last_n=always_keep_last_n,
            token_counter=self._token_counter,
            memory_logger=self._memory_logger,
        )

        self._history: List[_MessageRecord] = []
        self._prune_history: List[PruneResult] = []

    # ------------------------------------------------------------------
    # Adding messages
    # ------------------------------------------------------------------

    def add(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add a message with an arbitrary *role*."""
        record = _MessageRecord(role=role, content=content, metadata=metadata or {})
        self._history.append(record)
        current_total = self.token_count
        self._memory_logger.log_event(
            "message_added",
            token_count=current_total,
            budget=self.token_budget,
            message_count=self.message_count,
            compression_ratio=0.0,
            role=role,
            added_token_count=self._token_counter.count(content),
            current_total=current_total,
        )
        utilization = current_total / self.token_budget if self.token_budget else 0.0
        if utilization >= 0.9:
            self._memory_logger.log_event(
                "budget_enforced",
                token_count=current_total,
                budget=self.token_budget,
                message_count=self.message_count,
                compression_ratio=0.0,
                utilization=round(utilization, 4),
            )
        if self.auto_prune:
            self._maybe_prune()

    def add_system(self, content: str, **meta) -> None:
        """Add a system message (always preserved by default)."""
        self.add("system", content, meta)

    def add_user(self, content: str, **meta) -> None:
        """Add a user message."""
        self.add("user", content, meta)

    def add_assistant(self, content: str, **meta) -> None:
        """Add an assistant message."""
        self.add("assistant", content, meta)

    def add_messages(self, messages: Sequence[dict[str, str]]) -> None:
        """Bulk-add a list of ``{"role": …, "content": …}`` dicts."""
        for m in messages:
            self.add(m.get("role", "user"), m.get("content", ""))

    # ------------------------------------------------------------------
    # Retrieving messages
    # ------------------------------------------------------------------

    def get_messages(self, *, force_prune: bool = False) -> List[dict[str, str]]:
        """
        Return the current message list, optionally forcing a prune pass.

        This is the main method you pass to your LLM call::

            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=cm.get_messages(),
            )
        """
        raw = [r.to_dict() for r in self._history]
        if force_prune or self._pruner.needs_pruning(raw):
            if self.debug:
                self._print_scoring_breakdown(raw)
            result = self._pruner.prune(raw)
            self._prune_history.append(result)
            self._memory_logger.log_event(
                "budget_enforced",
                token_count=result.total_tokens,
                budget=self.token_budget,
                message_count=len(result.messages),
                compression_ratio=(result.summary_result.compression_ratio if result.summary_result else 0.0),
                force_prune=force_prune,
            )
            return result.messages
        return raw

    def get_raw_history(self) -> List[dict[str, str]]:
        """Return every message ever added (no pruning)."""
        return [r.to_dict() for r in self._history]

    # ------------------------------------------------------------------
    # Stats & inspection
    # ------------------------------------------------------------------

    @property
    def token_count(self) -> int:
        """Estimated token count of the current (unpruned) history."""
        return sum(self._token_counter.count(r.content) for r in self._history)

    @property
    def message_count(self) -> int:
        return len(self._history)

    def stats(self) -> dict[str, float | int]:
        """Return a summary dict useful for logging / debugging."""
        return {
            "message_count": self.message_count,
            "token_count": self.token_count,
            "token_budget": self.token_budget,
            "utilization": round(self.token_count / self.token_budget, 3),
            "prune_events": len(self._prune_history),
            "total_removed": sum(p.removed_count for p in self._prune_history),
            "total_summaries": sum(1 for p in self._prune_history if p.summary_inserted),
        }

    def score_current(self):
        """Score the current history and return scored messages."""
        raw = [r.to_dict() for r in self._history]
        return self._scorer.score_messages(raw)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def clear(self, keep_system: bool = True) -> None:
        """
        Clear conversation history.

        Parameters
        ----------
        keep_system : bool
            If True, retain system messages.
        """
        if keep_system:
            self._history = [r for r in self._history if r.role == "system"]
        else:
            self._history = []

    def pop(self) -> Optional[dict[str, str]]:
        """Remove and return the last message."""
        if self._history:
            return self._history.pop().to_dict()
        return None

    def replace_system(self, content: str) -> None:
        """Replace the first system message (or add one if absent)."""
        for r in self._history:
            if r.role == "system":
                r.content = content
                return
        self._history.insert(0, _MessageRecord(role="system", content=content))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _maybe_prune(self) -> None:
        raw = [r.to_dict() for r in self._history]
        used = sum(self._token_counter.count(m.get("content", "")) for m in raw)
        if used >= self.token_budget * self.prune_threshold:
            if self.debug:
                self._print_scoring_breakdown(raw)
            result = self._pruner.prune(raw)
            self._prune_history.append(result)
            self._memory_logger.log_event(
                "budget_enforced",
                token_count=result.total_tokens,
                budget=self.token_budget,
                message_count=len(result.messages),
                compression_ratio=(result.summary_result.compression_ratio if result.summary_result else 0.0),
                threshold=self.prune_threshold,
            )
            # Reconstruct history from pruned messages
            self._history = [
                _MessageRecord(role=m["role"], content=m["content"])
                for m in result.messages
            ]

    def _print_scoring_breakdown(self, messages: Sequence[dict[str, str]]) -> None:
        scored = self._scorer.score_messages(messages)
        print("[neuromem debug] scoring breakdown before prune")
        for item in scored:
            reasons = ", ".join(item.reasons)
            print(
                f"  idx={item.index} role={item.role} tokens={item.token_count} "
                f"score={item.score:.4f} reasons={reasons}"
            )

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "ContextManager":
        return self

    def __exit__(self, *_) -> None:
        pass

    def __len__(self) -> int:
        return self.message_count

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ContextManager(messages={self.message_count}, "
            f"tokens={self.token_count}/{self.token_budget})"
        )
