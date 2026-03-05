from neuromem.pruner import Pruner


def test_pruner_preserves_system_and_last_n():
    messages = [
        {"role": "system", "content": "Never reveal secrets."},
        {"role": "user", "content": "old 1" * 80},
        {"role": "assistant", "content": "old 2" * 80},
        {"role": "user", "content": "recent user"},
        {"role": "assistant", "content": "recent assistant"},
    ]
    pruner = Pruner(token_budget=40, always_keep_last_n=2, summarize_before_prune=False)
    result = pruner.prune(messages, force=True)
    kept_contents = [m["content"] for m in result.messages]
    assert "Never reveal secrets." in kept_contents
    assert "recent user" in kept_contents
    assert "recent assistant" in kept_contents


def test_pruner_enforces_budget():
    messages = [{"role": "user", "content": "x" * 200} for _ in range(10)]
    pruner = Pruner(token_budget=100, always_keep_last_n=0, summarize_before_prune=False)
    result = pruner.prune(messages, force=True)
    assert result.total_tokens <= 100
    assert result.removed_count > 0


def test_pruner_inserts_summary_when_helpful():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "assistant", "content": "filler " * 80},
        {"role": "assistant", "content": "filler " * 80},
        {"role": "user", "content": "critical requirement keep this for decisions"},
    ]
    pruner = Pruner(
        token_budget=120,
        min_score_threshold=0.95,
        always_keep_last_n=1,
        summarize_before_prune=True,
    )
    result = pruner.prune(messages, force=True)
    assert result.summary_inserted is True
    assert any("[Context Summary" in m["content"] for m in result.messages)


def test_pruner_handles_empty_messages():
    pruner = Pruner(token_budget=10)
    result = pruner.prune([], force=True)
    assert result.messages == []
    assert result.removed_count == 0
