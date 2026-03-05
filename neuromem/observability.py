from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional


@dataclass
class MemoryMetrics:
    prune_count: int
    total_tokens_removed: int
    total_messages_removed: int
    avg_compression_ratio: float
    summary_count: int


_lock = Lock()
_prune_count = 0
_total_tokens_removed = 0
_total_messages_removed = 0
_summary_count = 0
_compression_ratio_total = 0.0
_compression_ratio_samples = 0


def _record_event(
    event_type: str,
    *,
    removed_tokens: int = 0,
    removed_messages: int = 0,
    compression_ratio: Optional[float] = None,
) -> None:
    global _prune_count
    global _total_tokens_removed
    global _total_messages_removed
    global _summary_count
    global _compression_ratio_total
    global _compression_ratio_samples

    with _lock:
        if event_type == "prune_triggered":
            _prune_count += 1
            _total_tokens_removed += max(0, int(removed_tokens))
            _total_messages_removed += max(0, int(removed_messages))
        if event_type == "summary_inserted":
            _summary_count += 1
        if compression_ratio is not None:
            _compression_ratio_total += max(0.0, float(compression_ratio))
            _compression_ratio_samples += 1


def get_metrics() -> MemoryMetrics:
    with _lock:
        avg = (
            _compression_ratio_total / _compression_ratio_samples
            if _compression_ratio_samples
            else 0.0
        )
        return MemoryMetrics(
            prune_count=_prune_count,
            total_tokens_removed=_total_tokens_removed,
            total_messages_removed=_total_messages_removed,
            avg_compression_ratio=round(avg, 6),
            summary_count=_summary_count,
        )


def reset_metrics() -> None:
    global _prune_count
    global _total_tokens_removed
    global _total_messages_removed
    global _summary_count
    global _compression_ratio_total
    global _compression_ratio_samples

    with _lock:
        _prune_count = 0
        _total_tokens_removed = 0
        _total_messages_removed = 0
        _summary_count = 0
        _compression_ratio_total = 0.0
        _compression_ratio_samples = 0


def export_prometheus() -> str:
    m = get_metrics()
    lines = [
        "# HELP neuromem_prune_count Total number of prune events.",
        "# TYPE neuromem_prune_count counter",
        f"neuromem_prune_count {m.prune_count}",
        "# HELP neuromem_total_tokens_removed Total tokens removed by pruning.",
        "# TYPE neuromem_total_tokens_removed counter",
        f"neuromem_total_tokens_removed {m.total_tokens_removed}",
        "# HELP neuromem_total_messages_removed Total messages removed by pruning.",
        "# TYPE neuromem_total_messages_removed counter",
        f"neuromem_total_messages_removed {m.total_messages_removed}",
        "# HELP neuromem_avg_compression_ratio Average summary compression ratio.",
        "# TYPE neuromem_avg_compression_ratio gauge",
        f"neuromem_avg_compression_ratio {m.avg_compression_ratio}",
        "# HELP neuromem_summary_count Total summaries inserted.",
        "# TYPE neuromem_summary_count counter",
        f"neuromem_summary_count {m.summary_count}",
    ]
    return "\n".join(lines)


class MemoryLogger:
    def __init__(self, logger_name: str = "neuromem.observability") -> None:
        self._logger = logging.getLogger(logger_name)

    def log_event(
        self,
        event_type: str,
        *,
        token_count: int,
        budget: int,
        message_count: int,
        compression_ratio: float = 0.0,
        **extra: Any,
    ) -> None:
        payload = {
            "timestamp": time.time(),
            "event_type": event_type,
            "token_count": int(token_count),
            "budget": int(budget),
            "message_count": int(message_count),
            "compression_ratio": float(compression_ratio),
        }
        if extra:
            payload.update(extra)

        self._logger.info(json.dumps(payload, sort_keys=True))
        _record_event(
            event_type,
            removed_tokens=int(extra.get("removed_tokens", 0)),
            removed_messages=int(extra.get("removed_messages", 0)),
            compression_ratio=float(compression_ratio) if compression_ratio else None,
        )
