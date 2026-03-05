"""
neuromem.scorer
~~~~~~~~~~~~~~~
Scores each message in a conversation history for importance.

Scoring factors:
  - Recency          : recent messages score higher
  - Role             : system > user > assistant (baseline weight)
  - Keyword signals  : presence of task-critical terms boosts score
  - Length           : longer messages carry more information
  - Relevance        : cosine similarity to the most recent user turn
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScoredMessage:
    index: int
    role: str
    content: str
    token_count: int
    score: float
    reasons: List[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ScoredMessage(index={self.index}, role={self.role!r}, "
            f"score={self.score:.3f}, tokens={self.token_count})"
        )


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_CRITICAL_KEYWORDS: set[str] = {
    # task management
    "goal", "objective", "task", "requirement", "constraint",
    "must", "required", "critical", "important", "priority",
    # decisions / conclusions
    "decided", "decision", "conclusion", "result", "answer",
    "solution", "final", "confirmed", "agreed",
    # errors / warnings
    "error", "bug", "fix", "issue", "problem", "warning",
    "failed", "failure", "exception", "crash",
    # instructions
    "instruction", "rule", "policy", "guideline", "step",
    "always", "never", "forbidden", "allowed",
}

_BOOSTED_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(TODO|FIXME|NOTE|IMPORTANT|WARNING|CRITICAL)\b"),
    re.compile(r"\bremember\b.*\bthis\b", re.IGNORECASE),
    re.compile(r"\bdo not\b|\bdon't\b|\bmust not\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Tiny TF-IDF-like relevance helper (no external deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z]+", text.lower())


def _tf(tokens: List[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = len(tokens)
    return {w: c / total for w, c in counts.items()}


def _cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    shared = set(a) & set(b)
    if not shared:
        return 0.0
    dot = sum(a[w] * b[w] for w in shared)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class MessageScorer:
    """
    Computes an importance score in [0, 1] for every message.

    Parameters
    ----------
    recency_decay : float
        Controls how fast scores decay with age (0 = no decay, 1 = fast decay).
    role_weights : dict
        Base score contribution per role.
    keyword_boost : float
        Extra score when critical keywords are present.
    relevance_weight : float
        Weight given to cosine similarity vs. the latest user message.
    critical_override : bool
        If True, system messages always receive score = 1.0.
    """

    DEFAULT_ROLE_WEIGHTS = {
        "system": 1.0,
        "user": 0.6,
        "assistant": 0.4,
    }

    def __init__(
        self,
        recency_decay: float = 0.05,
        role_weights: Optional[dict[str, float]] = None,
        keyword_boost: float = 0.25,
        relevance_weight: float = 0.3,
        critical_override: bool = True,
    ) -> None:
        self.recency_decay = recency_decay
        self.role_weights = role_weights or dict(self.DEFAULT_ROLE_WEIGHTS)
        self.keyword_boost = keyword_boost
        self.relevance_weight = relevance_weight
        self.critical_override = critical_override

    # ------------------------------------------------------------------
    def score_messages(
        self,
        messages: Sequence[dict],
        *,
        reference_text: Optional[str] = None,
    ) -> List[ScoredMessage]:
        """
        Score every message and return a list of :class:`ScoredMessage`.

        Parameters
        ----------
        messages : sequence of {"role": str, "content": str}
        reference_text : optional anchor for relevance scoring;
            defaults to the last user message's content.
        """
        if not messages:
            return []

        n = len(messages)

        # Determine reference for relevance
        ref_tf: dict[str, float] = {}
        if reference_text:
            ref_tf = _tf(_tokenize(reference_text))
        else:
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    ref_tf = _tf(_tokenize(msg.get("content", "")))
                    break

        results: List[ScoredMessage] = []

        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            tokens = _estimate_tokens(content)
            reasons: List[str] = []

            # --- system override ---
            if self.critical_override and role == "system":
                results.append(
                    ScoredMessage(
                        index=idx,
                        role=role,
                        content=content,
                        token_count=tokens,
                        score=1.0,
                        reasons=["system-override"],
                    )
                )
                continue

            # --- recency (exponential decay from the end) ---
            age = n - 1 - idx  # 0 = newest
            recency_score = math.exp(-self.recency_decay * age)
            reasons.append(f"recency={recency_score:.3f}")

            # --- role baseline ---
            role_score = self.role_weights.get(role, 0.3)
            reasons.append(f"role={role_score:.3f}")

            # --- keyword signals ---
            lower_content = content.lower()
            words = set(_tokenize(lower_content))
            hit_count = len(words & _CRITICAL_KEYWORDS)
            pattern_hits = sum(1 for p in _BOOSTED_PATTERNS if p.search(content))
            kw_score = min(1.0, (hit_count * 0.05) + (pattern_hits * 0.1))
            if kw_score > 0:
                reasons.append(f"keywords={kw_score:.3f}")

            # --- length signal (log-normalised, cap at 512 tokens) ---
            length_score = min(1.0, math.log1p(tokens) / math.log1p(512))
            reasons.append(f"length={length_score:.3f}")

            # --- relevance to reference ---
            msg_tf = _tf(_tokenize(content))
            sim = _cosine_sim(msg_tf, ref_tf) if ref_tf else 0.0
            reasons.append(f"relevance={sim:.3f}")

            # --- combine ---
            base = (
                0.35 * recency_score
                + 0.20 * role_score
                + 0.15 * length_score
                + self.relevance_weight * sim
            )
            base = min(1.0, base + kw_score * self.keyword_boost)

            results.append(
                ScoredMessage(
                    index=idx,
                    role=role,
                    content=content,
                    token_count=tokens,
                    score=round(base, 4),
                    reasons=reasons,
                )
            )

        return results


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token (GPT-style)."""
    return max(1, len(text) // 4)
