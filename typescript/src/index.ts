/**
 * neuromem — Smart Context Manager for LLM agents
 * ------------------------------------------------
 * Main entry point.
 */

export { ContextManager } from "./contextManager.js";
export type { ContextManagerOptions, ContextStats } from "./contextManager.js";

export { MessageScorer, estimateTokens } from "./scorer.js";
export type { Message, ScoredMessage, ScorerOptions } from "./scorer.js";

export { Summarizer, summaryAsMessage } from "./summarizer.js";
export type { SummaryResult, SummarizerOptions } from "./summarizer.js";

export { Pruner } from "./pruner.js";
export type { PruneResult, PrunerOptions } from "./pruner.js";

export { MemoryLogger, getMetrics, resetMetrics, exportPrometheus } from "./observability.js";
export type { MemoryMetrics } from "./observability.js";
