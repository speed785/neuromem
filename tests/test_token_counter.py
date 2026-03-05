from neuromem.token_counter import (
    ClaudeTokenCounter,
    GPTTokenCounter,
    TiktokenCounter,
    get_token_counter,
)


def test_gpt_counter_ratio():
    counter = GPTTokenCounter()
    assert counter.count("abcd" * 10) == 10


def test_claude_counter_ratio():
    counter = ClaudeTokenCounter()
    assert counter.count("a" * 35) == 10


def test_tiktoken_counter_fallback_without_dependency():
    counter = TiktokenCounter(model="gpt-4o-mini")
    assert counter.count("a" * 40) >= 1


def test_get_token_counter_for_claude_model():
    counter = get_token_counter("claude-3-5-sonnet")
    assert isinstance(counter, ClaudeTokenCounter)


def test_get_token_counter_for_gpt_model():
    counter = get_token_counter("gpt-4o")
    assert isinstance(counter, GPTTokenCounter)
