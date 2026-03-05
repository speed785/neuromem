/**
 * neuromem/summarizer
 * -------------------
 * Summarizes older conversation chunks to free context space.
 *
 * Two modes:
 *  1. extractive (default) — picks top sentences; zero external deps
 *  2. abstractive          — calls an OpenAI-compatible client
 */

import { Message, estimateTokens } from "./scorer.js";

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

export interface SummaryResult {
  summaryText: string;
  originalMessageCount: number;
  originalTokenCount: number;
  summaryTokenCount: number;
  /** summaryTokens / originalTokens */
  compressionRatio: number;
}

export function summaryAsMessage(result: SummaryResult): Message {
  return {
    role: "system",
    content:
      `[Context Summary — ${result.originalMessageCount} messages compressed]\n` +
      result.summaryText,
  };
}

// ---------------------------------------------------------------------------
// Extractive helpers
// ---------------------------------------------------------------------------

const CRITICAL_KW = new Set([
  "goal", "objective", "task", "requirement", "decided", "decision",
  "conclusion", "result", "answer", "solution", "error", "bug", "fix",
  "must", "never", "always", "important", "critical", "instruction",
  "rule", "policy",
]);

function splitSentences(text: string): string[] {
  return text
    .trim()
    .split(/(?<=[.!?])\s+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

function scoreSentence(sentence: string, kwSet: Set<string>): number {
  const words = sentence.toLowerCase().match(/[a-z]+/g) ?? [];
  if (!words.length) return 0;
  const hits = words.filter((w) => kwSet.has(w)).length;
  const lengthBonus = Math.min(1, words.length / 20);
  return hits * 0.15 + lengthBonus * 0.5;
}

function extractiveSummarize(
  messages: Message[],
  targetRatio = 0.35,
  maxSentencesPerMsg = 3,
): string {
  const lines: string[] = [];

  for (const msg of messages) {
    const content = msg.content.trim();
    if (!content) continue;

    const sentences = splitSentences(content);
    if (!sentences.length) continue;

    const scored = [...sentences].sort(
      (a, b) => scoreSentence(b, CRITICAL_KW) - scoreSentence(a, CRITICAL_KW),
    );

    const keep = Math.min(
      maxSentencesPerMsg,
      Math.max(1, Math.round(sentences.length * targetRatio)),
    );
    const picked = scored.slice(0, keep);

    // Re-order to original sequence
    const order = new Map(sentences.map((s, i) => [s, i]));
    picked.sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));

    let snippet = picked.join(" ");
    if (snippet.length > 240) snippet = snippet.slice(0, 237) + "…";

    lines.push(`[${msg.role.toUpperCase()}] ${snippet}`);
  }

  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// SummarizerOptions
// ---------------------------------------------------------------------------

export interface SummarizerOptions {
  /** "extractive" | "abstractive". Default: "extractive" */
  mode?: "extractive" | "abstractive";
  /** OpenAI-compatible client (required for abstractive mode) */
  client?: {
    chat: {
      completions: {
        create(params: Record<string, unknown>): Promise<{
          choices: Array<{ message: { content: string | null } }>;
        }>;
      };
    };
  };
  model?: string;
  targetRatio?: number;
  maxAbstractiveWords?: number;
}

// ---------------------------------------------------------------------------
// Summarizer
// ---------------------------------------------------------------------------

export class Summarizer {
  private mode: "extractive" | "abstractive";
  private client: SummarizerOptions["client"];
  private model: string;
  private targetRatio: number;
  private maxAbstractiveWords: number;

  constructor(options: SummarizerOptions = {}) {
    this.mode = options.mode ?? "extractive";
    this.client = options.client;
    this.model = options.model ?? "gpt-4o-mini";
    this.targetRatio = options.targetRatio ?? 0.35;
    this.maxAbstractiveWords = options.maxAbstractiveWords ?? 200;
  }

  async summarize(messages: Message[]): Promise<SummaryResult> {
    if (!messages.length) {
      return {
        summaryText: "",
        originalMessageCount: 0,
        originalTokenCount: 0,
        summaryTokenCount: 0,
        compressionRatio: 1,
      };
    }

    const originalTokenCount = messages.reduce(
      (sum, m) => sum + estimateTokens(m.content),
      0,
    );

    let summaryText: string;
    if (this.mode === "abstractive" && this.client) {
      summaryText = await this._abstractive(messages);
    } else {
      summaryText = extractiveSummarize(messages, this.targetRatio);
    }

    const summaryTokenCount = estimateTokens(summaryText);
    const compressionRatio =
      originalTokenCount > 0 ? summaryTokenCount / originalTokenCount : 1;

    return {
      summaryText,
      originalMessageCount: messages.length,
      originalTokenCount,
      summaryTokenCount,
      compressionRatio: Math.round(compressionRatio * 10000) / 10000,
    };
  }

  private async _abstractive(messages: Message[]): Promise<string> {
    const conversation = messages
      .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
      .join("\n");

    const prompt =
      `You are a concise conversation summarizer. Summarize the following ` +
      `conversation segment in no more than ${this.maxAbstractiveWords} words, ` +
      `preserving all decisions, goals, constraints, and key facts. ` +
      `Do not add commentary.\n\nConversation:\n${conversation}`;

    const response = await this.client!.chat.completions.create({
      model: this.model,
      messages: [{ role: "user", content: prompt }],
      max_tokens: this.maxAbstractiveWords * 2,
      temperature: 0.3,
    });

    return response.choices[0].message.content?.trim() ?? "";
  }

  shouldSummarize(
    messages: Message[],
    tokenBudget: number,
    threshold = 0.8,
  ): boolean {
    const used = messages.reduce((s, m) => s + estimateTokens(m.content), 0);
    return used >= tokenBudget * threshold;
  }
}
