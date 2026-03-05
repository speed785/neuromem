/**
 * neuromem/integrations/openai
 * ----------------------------
 * Drop-in context-aware wrapper around the OpenAI Chat Completions API.
 *
 * @example
 * ```ts
 * import OpenAI from "openai";
 * import { ContextAwareOpenAI } from "neuromem/integrations/openai";
 *
 * const client = new ContextAwareOpenAI({
 *   openaiClient: new OpenAI(),
 *   tokenBudget: 8000,
 *   model: "gpt-4o",
 *   systemPrompt: "You are a helpful assistant.",
 * });
 *
 * const reply = await client.chat("Tell me about relativity.");
 * console.log(reply);
 * console.log(client.stats());
 * ```
 */

import { ContextManager, ContextManagerOptions } from "../contextManager.js";
import { Summarizer } from "../summarizer.js";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Minimal interface matching OpenAI SDK's ChatCompletion client */
export interface OpenAILike {
  chat: {
    completions: {
      create(params: Record<string, unknown>): Promise<{
        choices: Array<{
          message: { content: string | null };
          delta?: { content?: string | null };
        }>;
      }>;
    };
  };
}

export interface ContextAwareOpenAIOptions {
  openaiClient: OpenAILike;
  model?: string;
  tokenBudget?: number;
  systemPrompt?: string;
  /** "extractive" | "abstractive". Default: "extractive" */
  summarizeMode?: "extractive" | "abstractive";
  contextManager?: ContextManager;
}

export interface ChatOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
  extraParams?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// ContextAwareOpenAI
// ---------------------------------------------------------------------------

export class ContextAwareOpenAI {
  private _client: OpenAILike;
  readonly model: string;
  readonly context: ContextManager;

  constructor(options: ContextAwareOpenAIOptions) {
    this._client = options.openaiClient;
    this.model = options.model ?? "gpt-4o-mini";

    if (options.contextManager) {
      this.context = options.contextManager;
    } else {
      const summarizer = new Summarizer({
        mode: options.summarizeMode ?? "extractive",
        client:
          options.summarizeMode === "abstractive"
            ? (options.openaiClient as any)
            : undefined,
        model: this.model,
      });

      this.context = new ContextManager({
        tokenBudget: options.tokenBudget ?? 4096,
        summarizer,
      });
    }

    if (options.systemPrompt) {
      // synchronously add (no await needed for system message — won't trigger prune alone)
      void this.context.addSystem(options.systemPrompt);
    }
  }

  // ------------------------------------------------------------------

  async chat(
    userMessage: string,
    options: ChatOptions = {},
  ): Promise<string> {
    await this.context.addUser(userMessage);
    const messages = await this.context.getMessages();

    const params: Record<string, unknown> = {
      model: options.model ?? this.model,
      messages,
      temperature: options.temperature ?? 0.7,
      max_tokens: options.maxTokens ?? 1024,
      ...(options.extraParams ?? {}),
    };

    const response = await this._client.chat.completions.create(params);
    const reply = response.choices[0].message.content ?? "";

    await this.context.addAssistant(reply);
    return reply;
  }

  // ------------------------------------------------------------------

  async setSystem(prompt: string): Promise<void> {
    this.context.replaceSystem(prompt);
  }

  async reset(keepSystem = true): Promise<void> {
    this.context.clear(keepSystem);
  }

  stats() {
    return this.context.stats();
  }

  /** Direct pass-through to the underlying OpenAI client */
  async rawCreate(params: Record<string, unknown>) {
    return this._client.chat.completions.create(params);
  }
}
