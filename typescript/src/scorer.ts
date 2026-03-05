/**
 * neuromem/scorer
 * ---------------
 * Scores each message in a conversation history for importance.
 *
 * Scoring factors:
 *   - Recency      : recent messages score higher (exponential decay)
 *   - Role         : system > user > assistant (baseline weight)
 *   - Keywords     : presence of task-critical terms boosts score
 *   - Length       : log-normalised token count
 *   - Relevance    : cosine similarity to the latest user turn
 */

export interface Message {
  role: "system" | "user" | "assistant" | string;
  content: string;
}

export interface ScoredMessage extends Message {
  index: number;
  tokenCount: number;
  score: number;
  reasons: string[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CRITICAL_KEYWORDS = new Set([
  "goal", "objective", "task", "requirement", "constraint",
  "must", "required", "critical", "important", "priority",
  "decided", "decision", "conclusion", "result", "answer",
  "solution", "final", "confirmed", "agreed",
  "error", "bug", "fix", "issue", "problem", "warning",
  "failed", "failure", "exception", "crash",
  "instruction", "rule", "policy", "guideline", "step",
  "always", "never", "forbidden", "allowed",
]);

const BOOSTED_PATTERNS: RegExp[] = [
  /\b(TODO|FIXME|NOTE|IMPORTANT|WARNING|CRITICAL)\b/,
  /\bremember\b.*\bthis\b/i,
  /\bdo not\b|\bdon't\b|\bmust not\b/i,
];

const DEFAULT_ROLE_WEIGHTS: Record<string, number> = {
  system: 1.0,
  user: 0.6,
  assistant: 0.4,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function estimateTokens(text: string): number {
  return Math.max(1, Math.floor(text.length / 4));
}

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z]+/g) ?? [];
}

function termFrequency(tokens: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1);
  const total = tokens.length || 1;
  const tf = new Map<string, number>();
  counts.forEach((c, w) => tf.set(w, c / total));
  return tf;
}

function cosineSim(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0;
  let magA = 0;
  let magB = 0;
  a.forEach((v, w) => {
    magA += v * v;
    if (b.has(w)) dot += v * (b.get(w)!);
  });
  b.forEach((v) => (magB += v * v));
  if (magA === 0 || magB === 0) return 0;
  return dot / (Math.sqrt(magA) * Math.sqrt(magB));
}

// ---------------------------------------------------------------------------
// ScorerOptions
// ---------------------------------------------------------------------------

export interface ScorerOptions {
  /** Controls how fast scores decay with age (0 = no decay). Default: 0.05 */
  recencyDecay?: number;
  roleWeights?: Record<string, number>;
  /** Extra score when critical keywords are present. Default: 0.25 */
  keywordBoost?: number;
  /** Weight given to cosine similarity vs. the latest user message. Default: 0.3 */
  relevanceWeight?: number;
  /** If true, system messages always receive score = 1. Default: true */
  criticalOverride?: boolean;
}

// ---------------------------------------------------------------------------
// MessageScorer
// ---------------------------------------------------------------------------

export class MessageScorer {
  private recencyDecay: number;
  private roleWeights: Record<string, number>;
  private keywordBoost: number;
  private relevanceWeight: number;
  private criticalOverride: boolean;

  constructor(options: ScorerOptions = {}) {
    this.recencyDecay = options.recencyDecay ?? 0.05;
    this.roleWeights = options.roleWeights ?? { ...DEFAULT_ROLE_WEIGHTS };
    this.keywordBoost = options.keywordBoost ?? 0.25;
    this.relevanceWeight = options.relevanceWeight ?? 0.3;
    this.criticalOverride = options.criticalOverride ?? true;
  }

  /**
   * Score every message and return an array of ScoredMessage objects.
   * @param messages   Conversation messages.
   * @param referenceText  Optional anchor for relevance scoring;
   *                       defaults to the last user message's content.
   */
  scoreMessages(messages: Message[], referenceText?: string): ScoredMessage[] {
    if (!messages.length) return [];

    const n = messages.length;

    // Build reference TF vector
    let refTf: Map<string, number> = new Map();
    if (referenceText) {
      refTf = termFrequency(tokenize(referenceText));
    } else {
      for (let i = n - 1; i >= 0; i--) {
        if (messages[i].role === "user") {
          refTf = termFrequency(tokenize(messages[i].content));
          break;
        }
      }
    }

    const results: ScoredMessage[] = [];

    for (let idx = 0; idx < n; idx++) {
      const msg = messages[idx];
      const { role, content } = msg;
      const tokenCount = estimateTokens(content);
      const reasons: string[] = [];

      // System override
      if (this.criticalOverride && role === "system") {
        results.push({ ...msg, index: idx, tokenCount, score: 1.0, reasons: ["system-override"] });
        continue;
      }

      // Recency
      const age = n - 1 - idx;
      const recencyScore = Math.exp(-this.recencyDecay * age);
      reasons.push(`recency=${recencyScore.toFixed(3)}`);

      // Role
      const roleScore = this.roleWeights[role] ?? 0.3;
      reasons.push(`role=${roleScore.toFixed(3)}`);

      // Keywords
      const words = new Set(tokenize(content));
      let hitCount = 0;
      words.forEach((w) => { if (CRITICAL_KEYWORDS.has(w)) hitCount++; });
      const patternHits = BOOSTED_PATTERNS.filter((p) => p.test(content)).length;
      const kwScore = Math.min(1.0, hitCount * 0.05 + patternHits * 0.1);
      if (kwScore > 0) reasons.push(`keywords=${kwScore.toFixed(3)}`);

      // Length
      const lengthScore = Math.min(1.0, Math.log1p(tokenCount) / Math.log1p(512));
      reasons.push(`length=${lengthScore.toFixed(3)}`);

      // Relevance
      const msgTf = termFrequency(tokenize(content));
      const sim = refTf.size ? cosineSim(msgTf, refTf) : 0;
      reasons.push(`relevance=${sim.toFixed(3)}`);

      // Combine
      let base =
        0.35 * recencyScore +
        0.20 * roleScore +
        0.15 * lengthScore +
        this.relevanceWeight * sim;
      base = Math.min(1.0, base + kwScore * this.keywordBoost);

      results.push({
        ...msg,
        index: idx,
        tokenCount,
        score: Math.round(base * 10000) / 10000,
        reasons,
      });
    }

    return results;
  }
}
