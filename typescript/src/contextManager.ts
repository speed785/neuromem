/**
 * neuromem/contextManager
 * -----------------------
 * Core class that orchestrates message history, scoring, summarization,
 * and pruning within a configurable token budget.
 *
 * @example
 * ```ts
 * import { ContextManager } from "neuromem";
 *
 * const cm = new ContextManager({ tokenBudget: 8000 });
 * cm.addSystem("You are a helpful assistant.");
 * cm.addUser("Tell me about quantum computing.");
 * cm.addAssistant("Quantum computing uses qubits…");
 *
 * const messages = await cm.getMessages();
 * ```
 */

import { Message, MessageScorer, ScoredMessage, estimateTokens } from "./scorer.js";
import { Summarizer } from "./summarizer.js";
import { Pruner, PruneResult } from "./pruner.js";

// ---------------------------------------------------------------------------
// Internal record
// ---------------------------------------------------------------------------

interface MessageRecord extends Message {
  timestamp: number;
  metadata: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

export interface ContextManagerOptions {
  /** Maximum total tokens (default: 4096) */
  tokenBudget?: number;
  /** Automatically prune when adding a message would exceed budget (default: true) */
  autoPrune?: boolean;
  /** Fraction of budget at which auto-pruning triggers (default: 0.9) */
  pruneThreshold?: number;
  scorer?: MessageScorer;
  summarizer?: Summarizer;
  pruner?: Pruner;
  alwaysKeepLastN?: number;
}

export interface ContextStats {
  messageCount: number;
  tokenCount: number;
  tokenBudget: number;
  utilization: number;
  pruneEvents: number;
  totalRemoved: number;
  totalSummaries: number;
}

// ---------------------------------------------------------------------------
// ContextManager
// ---------------------------------------------------------------------------

export class ContextManager {
  readonly tokenBudget: number;
  private autoPrune: boolean;
  private pruneThreshold: number;

  private _scorer: MessageScorer;
  private _summarizer: Summarizer;
  private _pruner: Pruner;

  private _history: MessageRecord[] = [];
  private _pruneHistory: PruneResult[] = [];

  constructor(options: ContextManagerOptions = {}) {
    this.tokenBudget = options.tokenBudget ?? 4096;
    this.autoPrune = options.autoPrune ?? true;
    this.pruneThreshold = options.pruneThreshold ?? 0.9;

    this._scorer = options.scorer ?? new MessageScorer();
    this._summarizer = options.summarizer ?? new Summarizer();
    this._pruner =
      options.pruner ??
      new Pruner({
        tokenBudget: this.tokenBudget,
        scorer: this._scorer,
        summarizer: this._summarizer,
        alwaysKeepLastN: options.alwaysKeepLastN ?? 4,
      });
  }

  // ------------------------------------------------------------------
  // Adding messages
  // ------------------------------------------------------------------

  async add(
    role: string,
    content: string,
    metadata: Record<string, unknown> = {},
  ): Promise<void> {
    this._history.push({ role, content, timestamp: Date.now(), metadata });
    if (this.autoPrune) await this._maybePrune();
  }

  async addSystem(content: string): Promise<void> {
    await this.add("system", content);
  }

  async addUser(content: string): Promise<void> {
    await this.add("user", content);
  }

  async addAssistant(content: string): Promise<void> {
    await this.add("assistant", content);
  }

  async addMessages(messages: Message[]): Promise<void> {
    for (const m of messages) await this.add(m.role, m.content);
  }

  // ------------------------------------------------------------------
  // Retrieving messages
  // ------------------------------------------------------------------

  /**
   * Return the current message list, pruned to fit the budget.
   * Pass the result directly to your LLM call.
   */
  async getMessages(forcePrune = false): Promise<Message[]> {
    const raw = this._history.map(({ role, content }) => ({ role, content }));
    if (forcePrune || this._pruner.needsPruning(raw)) {
      const result = await this._pruner.prune(raw);
      this._pruneHistory.push(result);
      return result.messages;
    }
    return raw;
  }

  getRawHistory(): Message[] {
    return this._history.map(({ role, content }) => ({ role, content }));
  }

  // ------------------------------------------------------------------
  // Stats & inspection
  // ------------------------------------------------------------------

  get tokenCount(): number {
    return this._history.reduce((s, r) => s + estimateTokens(r.content), 0);
  }

  get messageCount(): number {
    return this._history.length;
  }

  stats(): ContextStats {
    return {
      messageCount: this.messageCount,
      tokenCount: this.tokenCount,
      tokenBudget: this.tokenBudget,
      utilization: Math.round((this.tokenCount / this.tokenBudget) * 1000) / 1000,
      pruneEvents: this._pruneHistory.length,
      totalRemoved: this._pruneHistory.reduce((s, p) => s + p.removedCount, 0),
      totalSummaries: this._pruneHistory.filter((p) => p.summaryInserted).length,
    };
  }

  scoreCurrent(): ScoredMessage[] {
    const raw = this._history.map(({ role, content }) => ({ role, content }));
    return this._scorer.scoreMessages(raw);
  }

  // ------------------------------------------------------------------
  // Mutation helpers
  // ------------------------------------------------------------------

  clear(keepSystem = true): void {
    if (keepSystem) {
      this._history = this._history.filter((r) => r.role === "system");
    } else {
      this._history = [];
    }
  }

  pop(): Message | undefined {
    const record = this._history.pop();
    return record ? { role: record.role, content: record.content } : undefined;
  }

  replaceSystem(content: string): void {
    const idx = this._history.findIndex((r) => r.role === "system");
    if (idx >= 0) {
      this._history[idx].content = content;
    } else {
      this._history.unshift({ role: "system", content, timestamp: Date.now(), metadata: {} });
    }
  }

  // ------------------------------------------------------------------
  // Internal
  // ------------------------------------------------------------------

  private async _maybePrune(): Promise<void> {
    const raw = this._history.map(({ role, content }) => ({ role, content }));
    const used = raw.reduce((s, m) => s + estimateTokens(m.content), 0);
    if (used >= this.tokenBudget * this.pruneThreshold) {
      const result = await this._pruner.prune(raw);
      this._pruneHistory.push(result);
      this._history = result.messages.map((m) => ({
        role: m.role,
        content: m.content,
        timestamp: Date.now(),
        metadata: {},
      }));
    }
  }
}
