<div align="center">
  <img src="assets/neuromem-icon.svg" width="120" height="120" alt="neuromem" />
  <h1>neuromem</h1>
  <p><strong>Smart context management — never lose critical memory again</strong></p>

  [![CI](https://github.com/speed785/neuromem/actions/workflows/ci.yml/badge.svg)](https://github.com/speed785/neuromem/actions/workflows/ci.yml)
  [![Coverage](https://codecov.io/gh/speed785/neuromem/branch/main/graph/badge.svg)](https://codecov.io/gh/speed785/neuromem)
  [![PyPI](https://img.shields.io/pypi/v/neuromem)](https://pypi.org/project/neuromem/)
  [![npm](https://img.shields.io/npm/v/neuromem)](https://www.npmjs.com/package/neuromem)
  [![Python](https://img.shields.io/badge/python-3.9+-blue)](https://python.org)
  [![TypeScript](https://img.shields.io/badge/typescript-5.0+-blue)](https://typescriptlang.org)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

  [Installation](#installation) · [Quick Start](#quick-start) · [Integrations](#integrations) · [API Reference](#api-reference)
</div>

---

## Why neuromem?

Every LLM has a context window limit. As conversations grow, you face an ugly choice:

- **Truncate from the top** — lose system instructions, early decisions, critical constraints
- **Send everything** — hit token limits, pay for redundant tokens, degrade performance
- **Hand-manage** — brittle, doesn't scale, breaks under real agent complexity

neuromem solves this automatically. It scores every message for importance using recency, role, keyword signals, and semantic relevance. Then it summarizes old turns and prunes only the least-important content. Your agent keeps its working memory sharp, no matter how long the conversation runs.

```python
from neuromem import ContextManager

cm = ContextManager(token_budget=8000)
cm.add_system("You are a financial analysis assistant.")
cm.add_user("My critical requirement: always cite sources.")
# ... 200 more turns ...
messages = cm.get_messages()  # budget-aware, importance-ranked, ready for your LLM
```

---

## Features

**Core (zero dependencies)**
- **Multi-factor importance scoring** — recency decay, role weights, 50+ keyword signals, semantic relevance, all combined into a single score per message
- **Extractive summarization** — compresses old turns to short digests with no API key required
- **Abstractive summarization** — optional LLM-powered compression for higher fidelity
- **Safe pruning** — system messages and recent turns are always protected; only low-importance content gets dropped
- **Token budget enforcement** — hard ceiling on context size, auto-triggers at a configurable threshold

**Integrations**
- **OpenAI** — drop-in `ContextAwareOpenAI` wrapper; just swap your client
- **Anthropic Claude** — same pattern, works with `claude-3-5-sonnet` and all Claude models
- **LangChain** — `NeuromemMemory` plugs directly into any `ConversationChain`
- **LlamaIndex** — `NeuromemChatMemoryBuffer` for `BaseChatMemoryBuffer` workflows
- **CrewAI** — `NeuromemCrewMemory` for agent memory save/search/reset

**Pluggable token counters**
- `TiktokenCounter` for exact GPT token counts (requires `tiktoken`)
- `ClaudeTokenCounter` for Claude-calibrated estimates
- `GPTTokenCounter` as a fast zero-dep fallback
- Bring your own by subclassing `TokenCounter`

**Both Python and TypeScript** — identical API surface in both languages

---

## Installation

### Python

```bash
# Core (zero deps)
pip install neuromem

# With OpenAI integration
pip install "neuromem[openai]"

# With Anthropic integration
pip install "neuromem[anthropic]"

# With LangChain integration
pip install "neuromem[langchain]"

# With LlamaIndex integration
pip install "neuromem[llamaindex]"

# With CrewAI integration
pip install "neuromem[crewai]"

# Everything
pip install "neuromem[all]"
```

### TypeScript / Node.js

```bash
npm install neuromem
# or
yarn add neuromem
```

---

## Quick Start

### Python

```python
from neuromem import ContextManager

cm = ContextManager(token_budget=8000)
cm.add_system("You are a helpful assistant.")
cm.add_user("What is quantum entanglement?")
cm.add_assistant("Quantum entanglement is...")
cm.add_user("Can it enable FTL communication?")

# Returns a pruned, budget-aware message list — pass directly to your LLM
messages = cm.get_messages()
```

### TypeScript

```typescript
import { ContextManager } from "neuromem";

const cm = new ContextManager({ tokenBudget: 8000 });
await cm.addSystem("You are a helpful assistant.");
await cm.addUser("What is quantum entanglement?");
await cm.addAssistant("Quantum entanglement is...");

const messages = await cm.getMessages(); // pruned, ready for API
```

---

## Integrations

### OpenAI (Python)

```python
import openai
from neuromem.integrations.openai import ContextAwareOpenAI

client = ContextAwareOpenAI(
    openai_client=openai.OpenAI(),
    model="gpt-4o",
    token_budget=8000,
    system_prompt="You are a financial analysis assistant.",
    summarize_mode="extractive",  # or "abstractive" for LLM-powered compression
)

reply = client.chat("What's the P/E ratio for AAPL?")
reply = client.chat("Compare it to the sector average.")  # history is auto-managed

print(client.stats())
# {'message_count': 5, 'token_count': 342, 'token_budget': 8000,
#  'utilization': 0.043, 'prune_events': 0, ...}
```

### Anthropic Claude (Python)

```python
import anthropic
from neuromem.integrations.anthropic import ContextAwareAnthropic

client = ContextAwareAnthropic(
    anthropic_client=anthropic.Anthropic(),
    model="claude-3-5-sonnet-latest",
    token_budget=8000,
    system_prompt="You are a senior software architect.",
)

reply = client.chat("Review this system design for a distributed cache.")
reply = client.chat("What are the failure modes?")  # context auto-managed

print(client.stats())
```

### LangChain (Python)

```python
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from neuromem.integrations.langchain import NeuromemMemory

memory = NeuromemMemory(
    token_budget=6000,
    system_prompt="You are a helpful coding assistant.",
)

chain = ConversationChain(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    memory=memory,
)

chain.predict(input="How do I reverse a list in Python?")
chain.predict(input="What about in TypeScript?")
print(memory.context_stats)
```

### LlamaIndex (Python)

```python
from neuromem.integrations.llamaindex import NeuromemChatMemoryBuffer

memory = NeuromemChatMemoryBuffer(token_budget=6000)
memory.put({"role": "user", "content": "Track all architecture decisions."})
messages = memory.get_all()
```

### CrewAI (Python)

```python
from neuromem.integrations.crewai import NeuromemCrewMemory

memory = NeuromemCrewMemory(token_budget=6000)
memory.save({"role": "assistant", "content": "We selected PostgreSQL for analytics."})
hits = memory.search("PostgreSQL")
```

### OpenAI (TypeScript)

```typescript
import OpenAI from "openai";
import { ContextAwareOpenAI } from "neuromem/integrations/openai";

const client = new ContextAwareOpenAI({
  openaiClient: new OpenAI(),
  model: "gpt-4o",
  tokenBudget: 8000,
  systemPrompt: "You are a helpful assistant.",
});

const reply1 = await client.chat("Tell me about black holes.");
const reply2 = await client.chat("How do they form?");  // context auto-managed

console.log(client.stats());
```

---

## Token Counting

neuromem ships with three token counters out of the box. Pick the one that matches your model, or bring your own.

```python
from neuromem import TiktokenCounter, ClaudeTokenCounter, GPTTokenCounter, TokenCounter

# Exact counts for GPT models (requires tiktoken)
counter = TiktokenCounter(model="gpt-4o")

# Claude-calibrated estimates (no deps)
counter = ClaudeTokenCounter()

# Fast zero-dep fallback for any model
counter = GPTTokenCounter()

# Plug into ContextManager directly
from neuromem import ContextManager
cm = ContextManager(token_budget=8000, token_counter=counter)
```

**Custom counter** — subclass `TokenCounter` and implement one method:

```python
from neuromem import TokenCounter

class MyCounter(TokenCounter):
    def count(self, text: str) -> int:
        return my_tokenizer.encode(text).length

cm = ContextManager(token_budget=8000, token_counter=MyCounter())
```

---

## How It Works

### Scoring formula

Every message gets a score from 0 to 1:

```
score = 0.35 × recency
      + 0.20 × role_weight
      + 0.15 × length_signal
      + 0.30 × semantic_relevance
      + keyword_hits × 0.25   (capped at 1.0)
```

System messages always receive `score = 1.0` and are never pruned.

### Pruning strategy

1. **Protect** system messages and the most recent N turns (configurable)
2. **Score** remaining candidates
3. **Summarize** low-scoring messages first (extractive by default)
4. **Hard-prune** only if still over budget after summarization

### Manual scoring and pruning

You can use the scorer and pruner directly if you want full control:

```python
from neuromem import MessageScorer, Pruner

scorer = MessageScorer(recency_decay=0.05, keyword_boost=0.3)
messages = [
    {"role": "system",    "content": "You are an AI assistant."},
    {"role": "user",      "content": "My critical requirement: always use type hints."},
    {"role": "assistant", "content": "Understood, I will always include type hints."},
    {"role": "user",      "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "4."},
]

scored = scorer.score_messages(messages)
for s in scored:
    print(f"[{s.role:9s}] score={s.score:.4f}  {s.content[:50]}")

# [system   ] score=1.0000  You are an AI assistant.
# [user     ] score=0.6123  My critical requirement: always use type hints.
# [assistant] score=0.5800  Understood, I will always include type hints.
# [user     ] score=0.7954  What is 2 + 2?
# [assistant] score=0.4467  4.

pruner = Pruner(token_budget=50, summarize_before_prune=True)
result = pruner.prune(messages, force=True)
print(f"Kept {len(result.messages)} of {len(messages)} messages")
print(f"Summary inserted: {result.summary_inserted}")
```

---

## API Reference

### `ContextManager`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `token_budget` | `int` | `4096` | Max context tokens |
| `auto_prune` | `bool` | `True` | Prune automatically on add |
| `prune_threshold` | `float` | `0.9` | Budget fraction that triggers auto-prune |
| `always_keep_last_n` | `int` | `4` | Always keep this many recent turns |
| `token_counter` | `TokenCounter` | `GPTTokenCounter()` | Pluggable token counter |

### `MessageScorer`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `recency_decay` | `float` | `0.05` | Exponential decay rate (higher = faster decay) |
| `keyword_boost` | `float` | `0.25` | Score boost for critical keyword hits |
| `relevance_weight` | `float` | `0.3` | Weight of semantic relevance component |
| `critical_override` | `bool` | `True` | System messages always score 1.0 |

### `Pruner`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `token_budget` | `int` | `4096` | Hard token ceiling |
| `min_score_threshold` | `float` | `0.3` | Below this, message is a candidate for summarization |
| `always_keep_last_n` | `int` | `4` | Recent turns protected from pruning |
| `summarize_before_prune` | `bool` | `True` | Try summarization before hard drops |

---

## Architecture

```
neuromem/
├── context_manager.py     Core orchestration
├── scorer.py              Multi-factor importance scoring
├── summarizer.py          Extractive + abstractive compression
├── pruner.py              Safe pruning with summary-first strategy
├── token_counter.py       Pluggable token counters (GPT, Claude, tiktoken)
├── observability.py       Metrics, logging, Prometheus export
└── integrations/
    ├── openai.py           Drop-in OpenAI chat wrapper
    ├── anthropic.py        Drop-in Anthropic Claude wrapper
    ├── langchain.py        LangChain BaseChatMemory subclass
    ├── llamaindex.py       LlamaIndex BaseChatMemoryBuffer adapter
    └── crewai.py           CrewAI memory adapter

typescript/src/
├── contextManager.ts
├── scorer.ts
├── summarizer.ts
├── pruner.ts
└── integrations/
    └── openai.ts
```

---

## Running the Examples

```bash
git clone https://github.com/speed785/neuromem
cd neuromem

# Python examples (no API key required)
python3 examples/example_python_basic.py
python3 examples/example_agent_loop.py

# TypeScript example (after build)
cd typescript && npm install && npm run build
node dist/examples/example_typescript_basic.js
```

---

## Contributing

Issues and pull requests are welcome. Open an issue before large changes so we can align on direction.

```bash
# Python dev setup
pip install -e ".[dev]"
pytest

# TypeScript dev setup
cd typescript
npm install
npm run build
npm run typecheck
```

---

## License

MIT — see [LICENSE](LICENSE)
