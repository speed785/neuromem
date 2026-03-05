# neuromem

**Smart Context Manager for LLM Agents**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-blue)](https://www.typescriptlang.org/)
[![Zero dependencies](https://img.shields.io/badge/core-zero%20deps-brightgreen)](https://github.com/speed785/neuromem)

> **Stop losing critical context. Stop paying for redundant tokens.**  
> neuromem intelligently manages your LLM conversation window — scoring message importance, summarizing older turns, and pruning safely — so your agent always has the right context, never the wrong one.

---

## Why neuromem?

Every LLM has a context window limit. As conversations grow, you face an ugly choice:

- **Truncate from the top** → lose system instructions, early decisions, critical constraints  
- **Send everything** → hit token limits, pay for redundant tokens, degrade performance  
- **Hand-manage** → brittle, error-prone, doesn't scale to complex agents

neuromem solves this automatically. It scores every message for importance using recency, role, keyword signals, and semantic relevance — then summarizes old turns and prunes only the least-important content. Your agent keeps its "working memory" sharp.

---

## Features

- **Importance scoring** — multi-factor scoring: recency decay, role weights, keyword signals (critical/task/requirement/error/etc.), semantic relevance
- **Extractive summarization** — no API key needed; compresses old turns to short digests
- **Abstractive summarization** — optional LLM-powered compression for higher quality
- **Safe pruning** — always preserves system messages and the most recent N turns; never drops high-importance messages
- **Token budget enforcement** — hard limit on context size; auto-triggers at configurable threshold
- **Drop-in integrations** — OpenAI Chat wrapper and LangChain memory class
- **Zero core dependencies** — pure Python stdlib / native TypeScript; optional deps for integrations
- **Both Python and TypeScript** — identical API surface in both languages

---

## Installation

### Python

```bash
# Core (zero deps)
pip install neuromem

# With OpenAI integration
pip install "neuromem[openai]"

# With LangChain integration
pip install "neuromem[langchain]"

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
cm.add_assistant("Quantum entanglement is…")
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
await cm.addAssistant("Quantum entanglement is…");

const messages = await cm.getMessages(); // pruned, ready for API
```

---

## Usage Examples

### OpenAI drop-in wrapper (Python)

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

### LangChain memory (Python)

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

### OpenAI integration (TypeScript)

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

### Manual scoring and pruning

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

## Architecture

```
neuromem/
├── context_manager.py     Core orchestration class
├── scorer.py              Multi-factor importance scoring
│     Factors:
│       • Recency decay (exponential)
│       • Role baseline (system > user > assistant)
│       • Keyword signals (50+ critical terms)
│       • Log-normalised message length
│       • Cosine similarity to latest user turn
├── summarizer.py          Extractive + abstractive compression
├── pruner.py              Safe pruning with summary-first strategy
└── integrations/
    ├── openai.py           Drop-in OpenAI chat wrapper
    └── langchain.py        LangChain BaseChatMemory subclass

typescript/src/
├── contextManager.ts
├── scorer.ts
├── summarizer.ts
├── pruner.ts
└── integrations/
    └── openai.ts
```

### Scoring formula

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

---

## Configuration Reference

### `ContextManager`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `token_budget` | `int` | `4096` | Max context tokens |
| `auto_prune` | `bool` | `True` | Prune automatically on add |
| `prune_threshold` | `float` | `0.9` | Budget fraction that triggers auto-prune |
| `always_keep_last_n` | `int` | `4` | Always keep this many recent turns |

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
| `min_score_threshold` | `float` | `0.3` | Below this → candidate for summarization |
| `always_keep_last_n` | `int` | `4` | Recent turns protected from pruning |
| `summarize_before_prune` | `bool` | `True` | Try summarization before hard drops |

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

## Development

```bash
# Python
pip install -e ".[dev]"
pytest

# TypeScript
cd typescript
npm install
npm run build
npm run typecheck
```

---

## Roadmap

- [ ] Anthropic Claude integration
- [ ] Streaming token counter
- [ ] Semantic chunking (sentence-transformer based)
- [ ] Redis-backed persistent memory
- [ ] Benchmark suite (compression quality vs. information retention)
- [ ] `neuromem inspect` CLI tool

---

## License

MIT — see [LICENSE](LICENSE)

---

## Contributing

Issues and pull requests welcome. Please open an issue before large changes.
