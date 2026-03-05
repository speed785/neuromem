"""
examples/example_python_basic.py
---------------------------------
Basic usage of the neuromem Python package.

Run from the repo root:
    python -m examples.example_python_basic
    # or:
    PYTHONPATH=. python examples/example_python_basic.py

This example does NOT require an API key — it uses extractive summarization
and demonstrates the core scoring + pruning pipeline with simulated messages.
"""

import sys
import os

# Make sure the repo root is on the path so "neuromem" package is importable
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from neuromem import ContextManager, MessageScorer, Summarizer, Pruner


# ---------------------------------------------------------------------------
# 1. Basic ContextManager usage
# ---------------------------------------------------------------------------

print("=" * 60)
print("1. Basic ContextManager")
print("=" * 60)

# Small budget so we can see pruning trigger
cm = ContextManager(token_budget=300, always_keep_last_n=2)
cm.add_system("You are a helpful assistant specialized in science topics.")

topics = [
    ("user", "What is quantum entanglement?"),
    ("assistant",
     "Quantum entanglement is a phenomenon where two particles become correlated "
     "such that the state of one instantly influences the other, regardless of distance."),
    ("user", "How is it different from classical correlations?"),
    ("assistant",
     "Classical correlations are based on pre-shared information. "
     "Quantum correlations are fundamentally different: the states are undetermined until measured."),
    ("user", "Can entanglement enable faster-than-light communication?"),
    ("assistant",
     "No. You cannot use entanglement to transmit classical information faster than light "
     "because measuring one particle yields a random result."),
    ("user", "What is the EPR paradox?"),
    ("assistant",
     "The EPR paradox, proposed by Einstein, Podolsky, and Rosen in 1935, argued that "
     "quantum mechanics was incomplete because entangled particles appeared to require "
     "'spooky action at a distance'."),
    ("user", "What experiment resolved the EPR paradox?"),
    ("assistant",
     "Bell's theorem and Aspect's 1982 experiment showed that quantum correlations violate "
     "Bell inequalities, ruling out local hidden variable theories."),
    ("user", "This is a critical requirement: please summarize all key facts so far."),
]

for role, content in topics:
    cm.add(role, content)

print(f"Messages in history: {cm.message_count}")
print(f"Estimated tokens:    {cm.token_count} / {cm.token_budget}")
print(f"Stats: {cm.stats()}")
print()

messages = cm.get_messages()
print(f"Messages after get_messages(): {len(messages)}")
for m in messages:
    preview = m["content"][:60].replace("\n", " ")
    print(f"  [{m['role']:9s}] {preview}…")

print()


# ---------------------------------------------------------------------------
# 2. Scoring individual messages
# ---------------------------------------------------------------------------

print("=" * 60)
print("2. Message Scoring")
print("=" * 60)

scorer = MessageScorer(recency_decay=0.1)
sample_messages = [
    {"role": "system",    "content": "You are an AI assistant."},
    {"role": "user",      "content": "What is my current goal?"},
    {"role": "assistant", "content": "I don't know your specific goal."},
    {"role": "user",      "content": "My critical requirement is to finish the report by Friday."},
    {"role": "assistant", "content": "Understood! I'll help you finish the report by Friday."},
    {"role": "user",      "content": "What's 2 + 2?"},
    {"role": "assistant", "content": "4."},
]

scored = scorer.score_messages(sample_messages)
print(f"{'#':<3} {'Role':<10} {'Score':<7} Content preview")
print("-" * 70)
for s in scored:
    preview = s.content[:45].replace("\n", " ")
    print(f"{s.index:<3} {s.role:<10} {s.score:<7.4f} {preview}")

print()


# ---------------------------------------------------------------------------
# 3. Summarizer demo
# ---------------------------------------------------------------------------

print("=" * 60)
print("3. Extractive Summarizer")
print("=" * 60)

summarizer = Summarizer(mode="extractive", target_ratio=0.4)
to_summarize = [m for m in sample_messages if m["role"] != "system"]
result = summarizer.summarize(to_summarize)

print(f"Original messages:  {result.original_message_count}")
print(f"Original tokens:    {result.original_token_count}")
print(f"Summary tokens:     {result.summary_token_count}")
print(f"Compression ratio:  {result.compression_ratio:.2%}")
print()
print("Summary text:")
print(result.summary_text)
print()


# ---------------------------------------------------------------------------
# 4. Pruner demo (force-prune a large context)
# ---------------------------------------------------------------------------

print("=" * 60)
print("4. Pruner — force prune to 150 tokens")
print("=" * 60)

pruner = Pruner(
    token_budget=150,
    min_score_threshold=0.35,
    always_keep_last_n=2,
    summarize_before_prune=True,
)

big_context = [
    {"role": "system",    "content": "You are a coding assistant. Never expose secrets."},
    {"role": "user",      "content": "How do I reverse a string in Python?"},
    {"role": "assistant", "content": "Use slicing: my_string[::-1] — this is the idiomatic Pythonic way."},
    {"role": "user",      "content": "What about reversing in JavaScript?"},
    {"role": "assistant", "content": "Use: str.split('').reverse().join('') or the spread syntax."},
    {"role": "user",      "content": "What is 5 + 5?"},
    {"role": "assistant", "content": "10."},
    {"role": "user",      "content": "This is a critical requirement: always include type hints in Python code."},
    {"role": "assistant", "content": "Understood. I will always include type hints in Python code examples from now on."},
    {"role": "user",      "content": "How do I write a generic function in TypeScript using type parameters?"},
    {"role": "assistant", "content": "Use angle brackets: function identity<T>(arg: T): T { return arg; }"},
    {"role": "user",      "content": "Great. Now, what is my main requirement again?"},
]

input_tokens = sum(len(m["content"]) // 4 for m in big_context)
print(f"Input:  {len(big_context)} messages, ~{input_tokens} tokens")

prune_result = pruner.prune(big_context, force=True)

print(f"Output: {len(prune_result.messages)} messages, ~{prune_result.total_tokens} tokens")
print(f"Removed: {prune_result.removed_count} messages ({prune_result.removed_tokens} tokens)")
print(f"Summary inserted: {prune_result.summary_inserted}")
print()
print("Resulting messages:")
for m in prune_result.messages:
    preview = m["content"][:70].replace("\n", " ")
    print(f"  [{m['role']:9s}] {preview}")

print()
print("All Python examples completed successfully.")
