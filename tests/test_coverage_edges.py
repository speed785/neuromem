import importlib
import sys
import types
from types import SimpleNamespace

import pytest

import neuromem.integrations.langchain as langchain_integration
import neuromem.integrations.llamaindex as llamaindex_integration
import neuromem.summarizer as summarizer_module
from neuromem.context_manager import ContextManager
from neuromem.integrations.anthropic import ContextAwareAnthropic
from neuromem.integrations.crewai import NeuromemCrewMemory
from neuromem.integrations.llamaindex import NeuromemChatMemoryBuffer
from neuromem.integrations.openai import ContextAwareOpenAI
from neuromem.pruner import Pruner
from neuromem.scorer import MessageScorer, ScoredMessage, _cosine_sim, _tf
from neuromem.summarizer import Summarizer, SummaryResult
from neuromem.token_counter import GPTTokenCounter, TiktokenCounter, get_token_counter


class _MockOpenAICompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])


class _MockOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_MockOpenAICompletions())


class _MockAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[])


class _MockAnthropicClient:
    def __init__(self):
        self.messages = _MockAnthropicMessages()


def test_context_manager_protocol_and_helper_edges(capsys):
    cm = ContextManager(token_budget=200, auto_prune=False)
    with cm as same:
        assert same is cm

    cm.add_messages([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}])
    assert len(cm) == 2
    assert cm.score_current()

    cm.clear(keep_system=False)
    assert cm.get_raw_history() == []
    assert cm.pop() is None

    cm.replace_system("new system")
    assert cm.get_raw_history()[0]["content"] == "new system"

    debug_cm = ContextManager(
        token_budget=20,
        auto_prune=True,
        prune_threshold=0.1,
        always_keep_last_n=0,
        debug=True,
    )
    debug_cm.add_user("x" * 120)
    captured = capsys.readouterr().out
    assert "scoring breakdown before prune" in captured


def test_anthropic_wrapper_edge_paths():
    client = _MockAnthropicClient()
    ctx = ContextManager(token_budget=100, auto_prune=False)
    wrapped = ContextAwareAnthropic(anthropic_client=client, context_manager=ctx)
    assert wrapped.context is ctx

    converted, _ = wrapped._convert_messages(
        [{"role": "system", "content": "sys"}, {"role": "tool", "content": "tool output"}]
    )
    assert converted == [{"role": "user", "content": "tool output"}]

    converted_empty, system_text = wrapped._convert_messages([{"role": "system", "content": "only system"}])
    assert converted_empty == [{"role": "user", "content": ""}]
    assert system_text == "only system"

    response = SimpleNamespace(content=[SimpleNamespace(type="tool_result", text="ignored")])
    assert wrapped._extract_text(response) == ""

    wrapped.set_system("system replaced")
    wrapped.reset(keep_system=False)
    assert wrapped.stats()["message_count"] == 0

    raw = wrapped.raw_create(model="claude", messages=[{"role": "user", "content": "hi"}])
    assert raw.content == []


def test_openai_wrapper_edge_paths():
    client = _MockOpenAIClient()
    ctx = ContextManager(token_budget=100, auto_prune=False)
    wrapped = ContextAwareOpenAI(openai_client=client, context_manager=ctx)
    assert wrapped.context is ctx

    wrapped.set_system("sys")
    wrapped.reset(keep_system=False)
    assert wrapped.stats()["message_count"] == 0

    raw = wrapped.raw_create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
    assert raw.choices[0].message.content == "ok"


def test_pruner_early_return_when_under_budget_without_force():
    pruner = Pruner(token_budget=500)
    messages = [{"role": "user", "content": "short"}]
    result = pruner.prune(messages, force=False)
    assert result.messages == messages
    assert result.removed_count == 0


def test_pruner_summary_inserted_before_first_non_protected_message():
    class StubScorer(MessageScorer):
        def score_messages(self, messages, *, reference_text=None):
            scored = []
            for i, m in enumerate(messages):
                score = 1.0 if m.get("role") == "system" else (0.9 if i == 1 else 0.1)
                scored.append(
                    ScoredMessage(
                        index=i,
                        role=m.get("role", "user"),
                        content=m.get("content", ""),
                        token_count=50,
                        score=score,
                        reasons=[],
                    )
                )
            return scored

    class StubSummarizer(Summarizer):
        def summarize(self, messages):
            return SummaryResult(
                summary_text="[Context Summary stub]",
                original_message_count=1,
                original_token_count=100,
                summary_token_count=10,
                compression_ratio=0.1,
            )

    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "keep this"},
        {"role": "assistant", "content": "remove this"},
    ]
    pruner = Pruner(
        token_budget=60,
        min_score_threshold=0.5,
        always_keep_last_n=0,
        summarize_before_prune=True,
        scorer=StubScorer(),
        summarizer=StubSummarizer(),
        token_counter=GPTTokenCounter(),
    )

    result = pruner.prune(messages, force=True)
    assert result.summary_inserted is True
    assert "[Context Summary stub]" in result.messages[1]["content"]


def test_summarizer_and_scorer_edge_branches(monkeypatch):
    assert summarizer_module._score_sentence("!!!", {"must"}) == 0.0
    assert _tf([]) == {}
    assert _cosine_sim({"x": 0.0}, {"x": 1.0}) == 0.0

    original_split = summarizer_module._split_sentences

    def fake_split(text):
        if text == "content without sentences":
            return []
        return original_split(text)

    monkeypatch.setattr(summarizer_module, "_split_sentences", fake_split)
    summary_text = summarizer_module._extractive_summarize(
        [
            {"role": "user", "content": "   "},
            {"role": "assistant", "content": "content without sentences"},
        ]
    )
    assert summary_text == ""

    with pytest.raises(ValueError):
        Summarizer(mode="invalid")

    abstractive = Summarizer(mode="abstractive", client=None)
    with pytest.raises(RuntimeError):
        abstractive._abstractive([{"role": "user", "content": "hi"}])

    scorer = MessageScorer(critical_override=False)
    scores = scorer.score_messages([{"role": "system", "content": "policy"}])
    assert scores and "system-override" not in scores[0].reasons


def test_tiktoken_and_optional_import_edge_paths(monkeypatch):
    class FakeEncoding:
        def encode(self, _text):
            return []

    class FakeTiktokenModule:
        @staticmethod
        def encoding_for_model(_model):
            raise RuntimeError("unknown model")

        @staticmethod
        def get_encoding(_name):
            return FakeEncoding()

    monkeypatch.setattr(
        "neuromem.token_counter.importlib.import_module",
        lambda name: FakeTiktokenModule() if name == "tiktoken" else None,
    )
    counter = TiktokenCounter(model="unknown-model")
    assert counter.count("hello") == 1
    assert isinstance(get_token_counter("mistral-large"), GPTTokenCounter)

    monkeypatch.setattr(langchain_integration, "_LANGCHAIN_AVAILABLE", False)
    with pytest.raises(ImportError):
        langchain_integration._require_langchain()


def test_llamaindex_and_crewai_edge_paths(monkeypatch):
    monkeypatch.setattr(llamaindex_integration, "ChatMessage", None)
    monkeypatch.setattr(llamaindex_integration, "MessageRole", None)

    memory = NeuromemChatMemoryBuffer(token_limit=50)
    memory.put(SimpleNamespace(role="SYSTEM", content="sys"))
    memory.put(SimpleNamespace(role="assistant", content="assistant response"))
    memory.put(SimpleNamespace(role="custom", content="user fallback"))
    all_messages = memory.get_all()
    assert all_messages[0]["role"] == "system"
    assert all_messages[2]["role"] == "user"

    crew_memory = NeuromemCrewMemory(token_budget=50)
    crew_memory.save({"role": "assistant", "text": "stored with text key"})
    crew_memory.save(123)
    assert crew_memory.search("text key")[0]["role"] == "assistant"

    with_system = NeuromemCrewMemory(token_budget=50, system_prompt="system prompt")
    assert with_system.search("system prompt")[0]["role"] == "system"


def test_llamaindex_import_available_and_role_fallback(monkeypatch):
    fake_memory_module = types.ModuleType("llama_index.core.memory")

    class FakeBaseChatMemoryBuffer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    setattr(fake_memory_module, "BaseChatMemoryBuffer", FakeBaseChatMemoryBuffer)

    fake_llm_types_module = types.ModuleType("llama_index.core.base.llms.types")

    class FakeChatMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    class FakeMessageRole:
        USER = "user"
        ASSISTANT = "assistant"
        SYSTEM = "system"

    setattr(fake_llm_types_module, "ChatMessage", FakeChatMessage)
    setattr(fake_llm_types_module, "MessageRole", FakeMessageRole)

    monkeypatch.setitem(sys.modules, "llama_index", types.ModuleType("llama_index"))
    monkeypatch.setitem(sys.modules, "llama_index.core", types.ModuleType("llama_index.core"))
    monkeypatch.setitem(sys.modules, "llama_index.core.base", types.ModuleType("llama_index.core.base"))
    monkeypatch.setitem(sys.modules, "llama_index.core.base.llms", types.ModuleType("llama_index.core.base.llms"))
    monkeypatch.setitem(sys.modules, "llama_index.core.memory", fake_memory_module)
    monkeypatch.setitem(sys.modules, "llama_index.core.base.llms.types", fake_llm_types_module)

    reloaded = importlib.reload(llamaindex_integration)
    memory = reloaded.NeuromemChatMemoryBuffer(token_limit=20)
    memory.put(SimpleNamespace(role="assistant", content="reply"))
    assert memory.get_all()[0].role == "assistant"

    monkeypatch.setattr(reloaded, "MessageRole", None)
    monkeypatch.setattr(reloaded, "ChatMessage", FakeChatMessage)
    fallback = reloaded.NeuromemChatMemoryBuffer(token_limit=20)
    fallback.put(SimpleNamespace(role="assistant", content="again"))
    assert fallback.get_all()[0].role == "assistant"

    importlib.reload(reloaded)
