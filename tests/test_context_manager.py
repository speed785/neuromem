from neuromem.context_manager import ContextManager
from neuromem.token_counter import TokenCounter


class FixedCounter(TokenCounter):
    def count(self, text: str) -> int:
        return 10


def test_context_manager_lifecycle_and_mutation_helpers():
    cm = ContextManager(token_budget=200, auto_prune=False)
    cm.add_system("sys")
    cm.add_user("u1")
    cm.add_assistant("a1")

    assert cm.message_count == 3
    assert len(cm.get_messages()) == 3

    popped = cm.pop()
    assert popped == {"role": "assistant", "content": "a1"}

    cm.replace_system("sys2")
    assert cm.get_raw_history()[0]["content"] == "sys2"

    cm.clear(keep_system=True)
    assert cm.get_raw_history() == [{"role": "system", "content": "sys2"}]


def test_context_manager_auto_prunes_when_threshold_reached():
    cm = ContextManager(token_budget=25, auto_prune=True, prune_threshold=0.5, always_keep_last_n=1)
    cm.add_system("system prompt")
    cm.add_user("x" * 200)
    stats = cm.stats()
    assert stats["prune_events"] >= 1


def test_context_manager_uses_custom_token_counter():
    cm = ContextManager(token_budget=1000, auto_prune=False, token_counter=FixedCounter())
    cm.add_user("small")
    cm.add_assistant("tiny")
    assert cm.token_count == 20


def test_context_manager_force_prune_path_returns_list():
    cm = ContextManager(token_budget=20, auto_prune=False, always_keep_last_n=0)
    for _ in range(6):
        cm.add_user("x" * 80)
    pruned = cm.get_messages(force_prune=True)
    assert isinstance(pruned, list)
    assert cm.stats()["prune_events"] >= 1
