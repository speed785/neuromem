export interface MemoryMetrics {
  pruneCount: number;
  totalTokensRemoved: number;
  totalMessagesRemoved: number;
  avgCompressionRatio: number;
  summaryCount: number;
}

let pruneCount = 0;
let totalTokensRemoved = 0;
let totalMessagesRemoved = 0;
let summaryCount = 0;
let compressionRatioTotal = 0;
let compressionRatioSamples = 0;

function recordEvent(
  eventType: string,
  removedTokens = 0,
  removedMessages = 0,
  compressionRatio?: number,
): void {
  if (eventType === "prune_triggered") {
    pruneCount += 1;
    totalTokensRemoved += Math.max(0, removedTokens);
    totalMessagesRemoved += Math.max(0, removedMessages);
  }
  if (eventType === "summary_inserted") {
    summaryCount += 1;
  }
  if (typeof compressionRatio === "number") {
    compressionRatioTotal += Math.max(0, compressionRatio);
    compressionRatioSamples += 1;
  }
}

export class MemoryLogger {
  constructor(private sink: (line: string) => void = (line) => console.info(line)) {}

  logEvent(
    eventType: string,
    tokenCount: number,
    budget: number,
    messageCount: number,
    compressionRatio = 0,
    extra: Record<string, unknown> = {},
  ): void {
    const payload: Record<string, unknown> = {
      timestamp: Date.now() / 1000,
      event_type: eventType,
      token_count: tokenCount,
      budget,
      message_count: messageCount,
      compression_ratio: compressionRatio,
      ...extra,
    };

    this.sink(JSON.stringify(payload));
    recordEvent(
      eventType,
      Number(extra.removed_tokens ?? 0),
      Number(extra.removed_messages ?? 0),
      compressionRatio || undefined,
    );
  }
}

export function getMetrics(): MemoryMetrics {
  return {
    pruneCount,
    totalTokensRemoved,
    totalMessagesRemoved,
    avgCompressionRatio:
      compressionRatioSamples > 0
        ? Math.round((compressionRatioTotal / compressionRatioSamples) * 1_000_000) /
          1_000_000
        : 0,
    summaryCount,
  };
}

export function resetMetrics(): void {
  pruneCount = 0;
  totalTokensRemoved = 0;
  totalMessagesRemoved = 0;
  summaryCount = 0;
  compressionRatioTotal = 0;
  compressionRatioSamples = 0;
}

export function exportPrometheus(): string {
  const metrics = getMetrics();
  return [
    "# HELP neuromem_prune_count Total number of prune events.",
    "# TYPE neuromem_prune_count counter",
    `neuromem_prune_count ${metrics.pruneCount}`,
    "# HELP neuromem_total_tokens_removed Total tokens removed by pruning.",
    "# TYPE neuromem_total_tokens_removed counter",
    `neuromem_total_tokens_removed ${metrics.totalTokensRemoved}`,
    "# HELP neuromem_total_messages_removed Total messages removed by pruning.",
    "# TYPE neuromem_total_messages_removed counter",
    `neuromem_total_messages_removed ${metrics.totalMessagesRemoved}`,
    "# HELP neuromem_avg_compression_ratio Average summary compression ratio.",
    "# TYPE neuromem_avg_compression_ratio gauge",
    `neuromem_avg_compression_ratio ${metrics.avgCompressionRatio}`,
    "# HELP neuromem_summary_count Total summaries inserted.",
    "# TYPE neuromem_summary_count counter",
    `neuromem_summary_count ${metrics.summaryCount}`,
  ].join("\n");
}
