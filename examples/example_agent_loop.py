"""
examples/example_agent_loop.py
--------------------------------
Simulates a long-running agent loop where neuromem automatically manages
the context window, triggering summarization and pruning as the conversation
grows beyond the token budget.

This example does NOT require an API key. It mocks the LLM responses to
demonstrate the full neuromem lifecycle:

  1. System message preserved at all costs
  2. Early turns get compressed / pruned as new ones arrive
  3. Critical messages (containing keywords like "requirement", "must",
     "never") are scored higher and retained longer
  4. A mid-conversation summary is automatically inserted

Run from the repo root:
    python3 examples/example_agent_loop.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from neuromem import ContextManager, MessageScorer
from neuromem.summarizer import Summarizer


# ---------------------------------------------------------------------------
# Simulated agent responses (no API key needed)
# ---------------------------------------------------------------------------

MOCK_TURNS = [
    ("user",      "Hello! I'm building a financial analysis tool."),
    ("assistant", "Great! I can help you build that. What data sources will you be using?"),
    ("user",      "We'll use Bloomberg, Refinitiv, and some internal databases."),
    ("assistant", "Good choices. Bloomberg and Refinitiv provide excellent market data coverage."),
    ("user",      "The critical requirement is: all financial data must be encrypted at rest."),
    ("assistant", "Understood. Encryption at rest is non-negotiable for financial data. "
                  "I'll make sure all storage recommendations include AES-256 encryption."),
    ("user",      "What Python libraries do you recommend for data processing?"),
    ("assistant", "For financial data: pandas for time series, numpy for computation, "
                  "polars for high-performance processing, and sqlalchemy for DB access."),
    ("user",      "We also must comply with SOX regulations."),
    ("assistant", "SOX compliance requires audit trails, access controls, and data integrity "
                  "verification. I'll factor this into all architecture decisions."),
    ("user",      "What's the best way to handle missing market data?"),
    ("assistant", "Common approaches: forward-fill for short gaps, interpolation for medium "
                  "gaps, and flagging/excluding for large gaps. Document your methodology."),
    ("user",      "How should we structure the database schema for tick data?"),
    ("assistant", "For tick data: use time-series databases like TimescaleDB or InfluxDB. "
                  "Partition by symbol and date. Include: timestamp, bid, ask, volume, exchange."),
    ("user",      "What about real-time streaming?"),
    ("assistant", "Use Apache Kafka or Redis Streams for ingestion, with consumer groups for "
                  "parallel processing. Consider Faust or Bytewax for stream processing in Python."),
    ("user",      "Never store raw PII in the financial tables — this is a hard rule."),
    ("assistant", "Confirmed. PII must be tokenized or stored separately with proper access "
                  "controls. Financial tables reference only anonymized IDs."),
    ("user",      "Can you summarize the key architectural decisions so far?"),
    ("assistant", "Key decisions: (1) Bloomberg+Refinitiv data sources, (2) AES-256 encryption "
                  "at rest, (3) SOX compliance with audit trails, (4) TimescaleDB for tick data, "
                  "(5) Kafka for real-time streaming, (6) No PII in financial tables."),
    ("user",      "Good. Now let's talk about the API layer."),
    ("assistant", "For the API layer: FastAPI is ideal — async, auto-docs, strong typing. "
                  "Add rate limiting, JWT authentication, and field-level encryption for PII."),
    ("user",      "What are the latency requirements for the real-time API?"),
    ("assistant", "For real-time financial APIs, target <10ms p99 for market data endpoints. "
                  "Use connection pooling, async I/O, and consider gRPC for internal services."),
]


def run_agent_loop():
    print("=" * 65)
    print("Neuromem Agent Loop Simulation")
    print("=" * 65)
    print()

    # Very tight budget to show pruning in action
    cm = ContextManager(
        token_budget=400,
        auto_prune=True,
        prune_threshold=0.85,
        always_keep_last_n=3,
    )

    cm.add_system(
        "You are a senior software architect. "
        "You must always follow security and compliance requirements stated by the user. "
        "Never skip mentioned requirements even if the conversation grows long."
    )

    scorer = MessageScorer(recency_decay=0.08)

    print(f"{'Turn':<5} {'Role':<10} {'History msgs':<14} {'Tokens':<8} {'Prune events'}")
    print("-" * 65)

    for i, (role, content) in enumerate(MOCK_TURNS):
        cm.add(role, content)
        stats = cm.stats()
        print(
            f"{i+1:<5} {role:<10} {stats['message_count']:<14} "
            f"{stats['token_count']:<8} {stats['prune_events']}"
        )

    print()
    print("=" * 65)
    print("Final context snapshot (what the LLM would receive):")
    print("=" * 65)

    final = cm.get_messages()
    for m in final:
        preview = m["content"][:80].replace("\n", " ")
        marker = " *** SUMMARY ***" if "[Context Summary" in m["content"] else ""
        print(f"  [{m['role']:9s}] {preview}{marker}")

    print()
    stats = cm.stats()
    print(f"Final stats:")
    print(f"  Total turns added:    {len(MOCK_TURNS)}")
    print(f"  Messages in window:   {stats['message_count']}")
    print(f"  Tokens used:          {stats['token_count']} / {stats['token_budget']}")
    print(f"  Utilization:          {stats['utilization']:.1%}")
    print(f"  Prune events:         {stats['prune_events']}")
    print(f"  Total removed:        {stats['total_removed']}")
    print(f"  Summaries inserted:   {stats['total_summaries']}")

    print()
    print("=" * 65)
    print("Message importance scores (current window):")
    print("=" * 65)
    scored = cm.score_current()
    print(f"{'#':<4} {'Role':<10} {'Score':<7} Content preview")
    print("-" * 65)
    for s in scored:
        preview = s.content[:50].replace("\n", " ")
        bar = "█" * int(s.score * 10)
        print(f"{s.index:<4} {s.role:<10} {s.score:.4f} [{bar:<10}] {preview}")

    print()
    print("Simulation complete.")


if __name__ == "__main__":
    run_agent_loop()
