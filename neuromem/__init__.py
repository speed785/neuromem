"""
neuromem — Smart Context Manager for LLM agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilities for managing LLM context windows intelligently: auto-summarizing
older turns, scoring message importance, and pruning safely without losing
critical information.

Quick start::

    from neuromem import ContextManager

    cm = ContextManager(token_budget=8000)
    cm.add_system("You are a helpful assistant.")
    cm.add_user("What is the capital of France?")
    cm.add_assistant("The capital of France is Paris.")

    messages = cm.get_messages()  # pass directly to your LLM

Version: 0.1.0
"""

from .context_manager import ContextManager
from .scorer import MessageScorer, ScoredMessage
from .summarizer import Summarizer, SummaryResult
from .pruner import Pruner, PruneResult
from .token_counter import (
    TokenCounter,
    GPTTokenCounter,
    TiktokenCounter,
    ClaudeTokenCounter,
    get_token_counter,
)
from .observability import (
    MemoryLogger,
    MemoryMetrics,
    get_metrics,
    reset_metrics,
    export_prometheus,
)
from .integrations import (
    ContextAwareOpenAI,
    ContextAwareAnthropic,
    NeuromemMemory,
    NeuromemChatMemoryBuffer,
    NeuromemCrewMemory,
)

__all__ = [
    "ContextManager",
    "MessageScorer",
    "ScoredMessage",
    "Summarizer",
    "SummaryResult",
    "Pruner",
    "PruneResult",
    "TokenCounter",
    "GPTTokenCounter",
    "TiktokenCounter",
    "ClaudeTokenCounter",
    "get_token_counter",
    "MemoryLogger",
    "MemoryMetrics",
    "get_metrics",
    "reset_metrics",
    "export_prometheus",
    "ContextAwareOpenAI",
    "ContextAwareAnthropic",
    "NeuromemMemory",
    "NeuromemChatMemoryBuffer",
    "NeuromemCrewMemory",
]

__version__ = "0.1.0"
__author__ = "speed785"
__license__ = "MIT"
