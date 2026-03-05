from types import SimpleNamespace

from neuromem.summarizer import Summarizer


def test_extractive_summarize_non_empty_output():
    summarizer = Summarizer(mode="extractive", target_ratio=0.5)
    messages = [
        {"role": "user", "content": "Critical requirement: always validate inputs. Do this for safety."},
        {"role": "assistant", "content": "Understood. I will always validate and sanitize user input."},
    ]
    result = summarizer.summarize(messages)
    assert result.original_message_count == 2
    assert result.summary_text
    assert result.summary_token_count <= result.original_token_count


def test_summarize_empty_messages():
    summarizer = Summarizer()
    result = summarizer.summarize([])
    assert result.summary_text == ""
    assert result.original_message_count == 0


def test_abstractive_mode_uses_client_when_provided():
    class MockCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="compact summary"))])

    mock_client = SimpleNamespace(chat=SimpleNamespace(completions=MockCompletions()))
    summarizer = Summarizer(mode="abstractive", client=mock_client, model="gpt-4o-mini")
    result = summarizer.summarize([
        {"role": "user", "content": "Long message one."},
        {"role": "assistant", "content": "Long message two."},
    ])
    assert "compact summary" in result.summary_text


def test_should_summarize_threshold_logic():
    summarizer = Summarizer()
    messages = [{"role": "user", "content": "x" * 200}]
    assert summarizer.should_summarize(messages, token_budget=60, summarize_threshold=0.8)
