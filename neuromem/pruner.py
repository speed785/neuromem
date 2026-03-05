"""
neuromem.pruner
~~~~~~~~~~~~~~~
Safely prunes low-importance messages while respecting token budgets
and preserving critical content.

The pruner works in several passes:
  1. Always keep: system messages, the most recent N turns.
  2. Score remaining messages via :class:`~neuromem.scorer.MessageScorer`.
  3. Drop lowest-scored messages until the token budget is met.
  4. Optionally trigger summarization before hard pruning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .scorer import MessageScorer, ScoredMessage, _estimate_tokens
from .summarizer import Summarizer, SummaryResult


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class PruneResult:
    """Outcome of a :meth:`Pruner.prune` operation."""
    messages: List[dict]              # final message list (ready to send)
    removed_count: int
    removed_tokens: int
    summary_inserted: bool
    summary_result: Optional[SummaryResult] = None
    scores: List[ScoredMessage] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(_estimate_tokens(m.get("content", "")) for m in self.messages)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PruneResult(kept={len(self.messages)}, removed={self.removed_count}, "
            f"tokens={self.total_tokens}, summary={self.summary_inserted})"
        )


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------

class Pruner:
    """
    Intelligently prunes a message list to fit within a token budget.

    Parameters
    ----------
    token_budget : int
        Hard upper-bound on total context tokens (default: 4096).
    min_score_threshold : float
        Messages below this score are candidates for pruning (0–1).
    always_keep_last_n : int
        Always preserve the most recent *n* turns regardless of score.
    summarize_before_prune : bool
        If True, try to summarize pruneable messages before hard-dropping them.
    scorer : MessageScorer, optional
        Custom scorer instance.
    summarizer : Summarizer, optional
        Custom summarizer instance.
    preserve_roles : list of str
        Roles that must never be pruned (default: ["system"]).
    """

    def __init__(
        self,
        token_budget: int = 4096,
        min_score_threshold: float = 0.3,
        always_keep_last_n: int = 4,
        summarize_before_prune: bool = True,
        scorer: Optional[MessageScorer] = None,
        summarizer: Optional[Summarizer] = None,
        preserve_roles: Optional[List[str]] = None,
    ) -> None:
        self.token_budget = token_budget
        self.min_score_threshold = min_score_threshold
        self.always_keep_last_n = always_keep_last_n
        self.summarize_before_prune = summarize_before_prune
        self.scorer = scorer or MessageScorer()
        self.summarizer = summarizer or Summarizer()
        self.preserve_roles = set(preserve_roles or ["system"])

    # ------------------------------------------------------------------

    def prune(
        self,
        messages: Sequence[dict],
        *,
        force: bool = False,
    ) -> PruneResult:
        """
        Prune *messages* to fit within :attr:`token_budget`.

        Parameters
        ----------
        messages : sequence of {"role": str, "content": str}
        force : bool
            If True, prune even if already under budget.

        Returns
        -------
        PruneResult
        """
        msgs = list(messages)

        current_tokens = sum(_estimate_tokens(m.get("content", "")) for m in msgs)
        if not force and current_tokens <= self.token_budget:
            return PruneResult(
                messages=msgs,
                removed_count=0,
                removed_tokens=0,
                summary_inserted=False,
            )

        # --- partition messages ---
        protected_indices, candidate_indices = self._partition(msgs)

        # Score candidate messages
        scored = self.scorer.score_messages(msgs)
        score_by_idx = {s.index: s for s in scored}

        candidate_scored = sorted(
            [score_by_idx[i] for i in candidate_indices if i in score_by_idx],
            key=lambda s: s.score,
        )

        removed_count = 0
        removed_tokens = 0
        summary_inserted = False
        summary_result: Optional[SummaryResult] = None

        # --- try summarization first ---
        if self.summarize_before_prune and candidate_scored:
            low_scored = [s for s in candidate_scored if s.score < self.min_score_threshold]
            if low_scored:
                chunk = [msgs[s.index] for s in low_scored]
                summary_result = self.summarizer.summarize(chunk)

                # Only insert summary if it actually saves tokens
                saved = summary_result.original_token_count - summary_result.summary_token_count
                if saved > 0:
                    summary_msg = summary_result.as_message()
                    # Remove summarized messages from msgs
                    remove_set = {s.index for s in low_scored}
                    new_msgs: List[dict] = []
                    inserted = False
                    for i, m in enumerate(msgs):
                        if i in remove_set:
                            removed_count += 1
                            removed_tokens += _estimate_tokens(m.get("content", ""))
                        else:
                            if not inserted and i not in protected_indices and i == min(
                                j for j in range(len(msgs)) if j not in remove_set and j not in protected_indices
                            ):
                                new_msgs.append(summary_msg)
                                inserted = True
                            new_msgs.append(m)
                    if not inserted:
                        # insert right after last system message
                        insert_at = 0
                        for i, m in enumerate(new_msgs):
                            if m.get("role") == "system":
                                insert_at = i + 1
                        new_msgs.insert(insert_at, summary_msg)

                    msgs = new_msgs
                    summary_inserted = True
                    current_tokens = sum(_estimate_tokens(m.get("content", "")) for m in msgs)

        # --- hard prune if still over budget ---
        if current_tokens > self.token_budget:
            # Re-score after potential summary insertion
            scored2 = self.scorer.score_messages(msgs)
            score_by_idx2 = {s.index: s for s in scored2}
            protected2, candidates2 = self._partition(msgs)
            cands_sorted = sorted(
                [score_by_idx2[i] for i in candidates2 if i in score_by_idx2],
                key=lambda s: s.score,
            )

            remove_set2: set[int] = set()
            for sm in cands_sorted:
                if current_tokens <= self.token_budget:
                    break
                current_tokens -= sm.token_count
                remove_set2.add(sm.index)
                removed_count += 1
                removed_tokens += sm.token_count

            msgs = [m for i, m in enumerate(msgs) if i not in remove_set2]

        return PruneResult(
            messages=msgs,
            removed_count=removed_count,
            removed_tokens=removed_tokens,
            summary_inserted=summary_inserted,
            summary_result=summary_result,
            scores=scored,
        )

    # ------------------------------------------------------------------

    def _partition(self, msgs: List[dict]) -> tuple[set[int], set[int]]:
        """Split indices into (protected, candidates)."""
        n = len(msgs)
        protected: set[int] = set()

        for i, m in enumerate(msgs):
            if m.get("role") in self.preserve_roles:
                protected.add(i)

        # Always-keep last N non-system messages
        non_system = [i for i in range(n) if i not in protected]
        for idx in non_system[-self.always_keep_last_n:]:
            protected.add(idx)

        candidates = set(range(n)) - protected
        return protected, candidates

    # ------------------------------------------------------------------

    def needs_pruning(self, messages: Sequence[dict]) -> bool:
        """Return True if messages exceed the token budget."""
        total = sum(_estimate_tokens(m.get("content", "")) for m in messages)
        return total > self.token_budget
