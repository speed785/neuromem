from types import SimpleNamespace

from neuromem.integrations.anthropic import ContextAwareAnthropic
from neuromem.integrations.crewai import NeuromemCrewMemory
from neuromem.integrations.llamaindex import NeuromemChatMemoryBuffer
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


def test_llamaindex_memory_buffer_put_get_reset(monkeypatch):
    import neuromem.integrations.llamaindex as li_module

    class FakeMessageRole:
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"

    class FakeChatMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    monkeypatch.setattr(li_module, "MessageRole", FakeMessageRole)
    monkeypatch.setattr(li_module, "ChatMessage", FakeChatMessage)

    memory = NeuromemChatMemoryBuffer(token_budget=60, system_prompt="system")
    memory.put(SimpleNamespace(role="user", content="hello"))
    memory.put(SimpleNamespace(role="assistant", content="hi"))

    messages = memory.get()
    assert len(messages) == 3
    assert messages[0].content == "system"
    assert messages[1].role == "user"

    memory.reset()
    assert memory.get_all() == []


def test_crewai_memory_save_search_reset():
    memory = NeuromemCrewMemory(token_budget=60)
    memory.save("first memory")
    memory.save({"role": "assistant", "content": "second memory"})
    memory.save(SimpleNamespace(role="assistant", content="third memory"))

    assert memory.search("memory", limit=2)[0]["content"] == "third memory"

    memory.reset(keep_system=False)
    assert memory.search("memory") == []
