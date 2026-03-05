# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [0.1.0] - 2026-03-05

### Added

- `ContextManager` for budget-aware conversation memory management
- Multi-factor message scoring (recency, role, keyword, semantic relevance)
- Safe pruning pipeline to retain critical context while dropping low-value turns
- Extractive and abstractive summarization support
- OpenAI integration (`ContextAwareOpenAI`)
- Anthropic integration (`ContextAwareAnthropic`)
- LangChain integration (`NeuromemMemory`)
- Pluggable token counting interfaces and model-aware token counters
- Built-in observability via context statistics and pruning metrics
- Test suite with 100% coverage target enforcement
