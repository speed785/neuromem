import json
import logging

from neuromem.context_manager import ContextManager
from neuromem.observability import (
    MemoryLogger,
    export_prometheus,
    get_metrics,
    reset_metrics,
)
from neuromem.pruner import Pruner


def test_memory_logger_emits_structured_json_and_updates_metrics(caplog):
    reset_metrics()
    logger = MemoryLogger("neuromem.test.observability")

    with caplog.at_level(logging.INFO, logger="neuromem.test.observability"):
        logger.log_event(
            "prune_triggered",
            token_count=90,
            budget=100,
            message_count=4,
            compression_ratio=0.5,
            removed_tokens=40,
            removed_messages=2,
        )

    payload = json.loads(caplog.records[0].message)
    assert payload["event_type"] == "prune_triggered"
    assert "timestamp" in payload
    assert payload["token_count"] == 90
    assert payload["budget"] == 100
    assert payload["message_count"] == 4
    assert payload["compression_ratio"] == 0.5

    metrics = get_metrics()
    assert metrics.prune_count == 1
    assert metrics.total_tokens_removed == 40
    assert metrics.total_messages_removed == 2
    assert metrics.avg_compression_ratio == 0.5

    prom = export_prometheus()
    assert "neuromem_prune_count 1" in prom
    assert "neuromem_total_tokens_removed 40" in prom


def test_pruner_logs_prune_and_summary_events(caplog):
    reset_metrics()
    pruner = Pruner(
        token_budget=120,
        min_score_threshold=0.95,
        always_keep_last_n=1,
        summarize_before_prune=True,
        memory_logger=MemoryLogger("neuromem.test.pruner"),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "filler " * 80},
        {"role": "assistant", "content": "filler " * 80},
        {"role": "user", "content": "critical requirement keep this for decisions"},
    ]

    with caplog.at_level(logging.INFO, logger="neuromem.test.pruner"):
        result = pruner.prune(messages, force=True)

    assert result.summary_inserted is True
    event_types = [json.loads(r.message)["event_type"] for r in caplog.records]
    assert "scoring_complete" in event_types
    assert "summary_inserted" in event_types
    assert "prune_triggered" in event_types

    metrics = get_metrics()
    assert metrics.prune_count >= 1
    assert metrics.summary_count >= 1


def test_context_manager_debug_prints_scoring_breakdown(capsys):
    reset_metrics()
    cm = ContextManager(token_budget=25, auto_prune=False, always_keep_last_n=0, debug=True)
    for _ in range(4):
        cm.add_user("x" * 80)

    _ = cm.get_messages(force_prune=True)
    captured = capsys.readouterr().out
    assert "scoring breakdown before prune" in captured
    assert "idx=" in captured
