from types import SimpleNamespace

from neuromem.integrations.anthropic import ContextAwareAnthropic
from neuromem.integrations.openai import ContextAwareOpenAI


class MockOpenAICompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))]),
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="world"))]),
            ]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="openai reply"))]
        )


class MockOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=MockOpenAICompletions())


class MockAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="anthropic reply")]
        )

    def stream(self, **kwargs):
        self.calls.append(kwargs)

        class _Stream:
            text_stream = ["stream ", "reply"]

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Stream()


class MockAnthropicClient:
    def __init__(self):
        self.messages = MockAnthropicMessages()


def test_openai_wrapper_chat_and_stream_paths():
    client = MockOpenAIClient()
    wrapped = ContextAwareOpenAI(openai_client=client, token_budget=400, system_prompt="sys")

    response = wrapped.chat("hello")
    assert response == "openai reply"

    streamed = wrapped.chat("hello again", stream=True)
    assert streamed == "Hello world"

    assert wrapped.stats()["message_count"] >= 3
    assert len(client.chat.completions.calls) == 2


def test_anthropic_wrapper_converts_system_and_history():
    client = MockAnthropicClient()
    wrapped = ContextAwareAnthropic(
        anthropic_client=client,
        model="claude-3-5-sonnet",
        token_budget=400,
        system_prompt="system instructions",
    )

    response = wrapped.chat("user prompt")
    assert response == "anthropic reply"

    call = client.messages.calls[0]
    assert call["system"] == "system instructions"
    assert call["messages"][0]["role"] == "user"
    assert call["messages"][0]["content"] == "user prompt"


def test_anthropic_wrapper_stream_path():
    client = MockAnthropicClient()
    wrapped = ContextAwareAnthropic(anthropic_client=client)
    response = wrapped.chat("hello", stream=True)
    assert response == "stream reply"
