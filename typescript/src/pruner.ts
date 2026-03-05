/**
 * neuromem/pruner
 * ---------------
 * Safely prunes low-importance messages while respecting token budgets
 * and preserving critical content.
 */

import { Message, MessageScorer, ScoredMessage, estimateTokens } from "./scorer.js";
import { Summarizer, SummaryResult, summaryAsMessage } from "./summarizer.js";

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

export interface PruneResult {
  messages: Message[];
  removedCount: number;
  removedTokens: number;
  summaryInserted: boolean;
  summaryResult?: SummaryResult;
  scores: ScoredMessage[];
}

// ---------------------------------------------------------------------------
// PrunerOptions
// ---------------------------------------------------------------------------

export interface PrunerOptions {
  /** Max total tokens. Default: 4096 */
  tokenBudget?: number;
  /** Messages below this score are candidates for pruning. Default: 0.3 */
  minScoreThreshold?: number;
  /** Always preserve the most recent N turns. Default: 4 */
  alwaysKeepLastN?: number;
  /** Try summarizing before hard-dropping. Default: true */
  summarizeBeforePrune?: boolean;
  scorer?: MessageScorer;
  summarizer?: Summarizer;
  /** Roles that must never be pruned. Default: ["system"] */
  preserveRoles?: string[];
}

// ---------------------------------------------------------------------------
// Pruner
// ---------------------------------------------------------------------------

export class Pruner {
  private tokenBudget: number;
  private minScoreThreshold: number;
  private alwaysKeepLastN: number;
  private summarizeBeforePrune: boolean;
  private scorer: MessageScorer;
  private summarizer: Summarizer;
  private preserveRoles: Set<string>;

  constructor(options: PrunerOptions = {}) {
    this.tokenBudget = options.tokenBudget ?? 4096;
    this.minScoreThreshold = options.minScoreThreshold ?? 0.3;
    this.alwaysKeepLastN = options.alwaysKeepLastN ?? 4;
    this.summarizeBeforePrune = options.summarizeBeforePrune ?? true;
    this.scorer = options.scorer ?? new MessageScorer();
    this.summarizer = options.summarizer ?? new Summarizer();
    this.preserveRoles = new Set(options.preserveRoles ?? ["system"]);
  }

  async prune(messages: Message[], force = false): Promise<PruneResult> {
    let msgs = [...messages];

    const currentTokens = () =>
      msgs.reduce((s, m) => s + estimateTokens(m.content), 0);

    if (!force && currentTokens() <= this.tokenBudget) {
      return {
        messages: msgs,
        removedCount: 0,
        removedTokens: 0,
        summaryInserted: false,
        scores: [],
      };
    }

    const [protected_, candidates] = this._partition(msgs);
    const scored = this.scorer.scoreMessages(msgs);
    const scoreByIdx = new Map(scored.map((s) => [s.index, s]));

    const candidateScored = candidates
      .map((i) => scoreByIdx.get(i)!)
      .filter(Boolean)
      .sort((a, b) => a.score - b.score);

    let removedCount = 0;
    let removedTokens = 0;
    let summaryInserted = false;
    let summaryResult: SummaryResult | undefined;

    // --- try summarization first ---
    if (this.summarizeBeforePrune && candidateScored.length) {
      const lowScored = candidateScored.filter(
        (s) => s.score < this.minScoreThreshold,
      );
      if (lowScored.length) {
        const chunk = lowScored.map((s) => msgs[s.index]);
        summaryResult = await this.summarizer.summarize(chunk);

        const saved =
          summaryResult.originalTokenCount - summaryResult.summaryTokenCount;
        if (saved > 0) {
          const summaryMsg = summaryAsMessage(summaryResult);
          const removeSet = new Set(lowScored.map((s) => s.index));
          const newMsgs: Message[] = [];
          let inserted = false;

          // Find the first non-protected, non-removed index for insertion
          const firstCandidate = Math.min(
            ...[...Array(msgs.length).keys()].filter(
              (i) => !protected_.has(i) && !removeSet.has(i),
            ),
          );

          for (let i = 0; i < msgs.length; i++) {
            if (removeSet.has(i)) {
              removedCount++;
              removedTokens += estimateTokens(msgs[i].content);
            } else {
              if (!inserted && i === firstCandidate) {
                newMsgs.push(summaryMsg);
                inserted = true;
              }
              newMsgs.push(msgs[i]);
            }
          }

          if (!inserted) {
            // insert after last system message
            let insertAt = 0;
            for (let i = 0; i < newMsgs.length; i++) {
              if (newMsgs[i].role === "system") insertAt = i + 1;
            }
            newMsgs.splice(insertAt, 0, summaryMsg);
          }

          msgs = newMsgs;
          summaryInserted = true;
        }
      }
    }

    // --- hard prune if still over budget ---
    if (currentTokens() > this.tokenBudget) {
      const scored2 = this.scorer.scoreMessages(msgs);
      const scoreByIdx2 = new Map(scored2.map((s) => [s.index, s]));
      const [, candidates2] = this._partition(msgs);
      const candsSort = candidates2
        .map((i) => scoreByIdx2.get(i)!)
        .filter(Boolean)
        .sort((a, b) => a.score - b.score);

      const removeSet2 = new Set<number>();
      let tokens = currentTokens();
      for (const sm of candsSort) {
        if (tokens <= this.tokenBudget) break;
        tokens -= sm.tokenCount;
        removeSet2.add(sm.index);
        removedCount++;
        removedTokens += sm.tokenCount;
      }

      msgs = msgs.filter((_, i) => !removeSet2.has(i));
    }

    return {
      messages: msgs,
      removedCount,
      removedTokens,
      summaryInserted,
      summaryResult,
      scores: scored,
    };
  }

  needsPruning(messages: Message[]): boolean {
    const total = messages.reduce((s, m) => s + estimateTokens(m.content), 0);
    return total > this.tokenBudget;
  }

  private _partition(msgs: Message[]): [Set<number>, number[]] {
    const n = msgs.length;
    const protected_ = new Set<number>();

    msgs.forEach((m, i) => {
      if (this.preserveRoles.has(m.role)) protected_.add(i);
    });

    const nonSystem = Array.from({ length: n }, (_, i) => i).filter(
      (i) => !protected_.has(i),
    );
    nonSystem.slice(-this.alwaysKeepLastN).forEach((i) => protected_.add(i));

    const candidates = Array.from({ length: n }, (_, i) => i).filter(
      (i) => !protected_.has(i),
    );
    return [protected_, candidates];
  }
}
