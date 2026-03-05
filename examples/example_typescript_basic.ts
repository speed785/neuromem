/**
 * examples/example_typescript_basic.ts
 * --------------------------------------
 * Basic usage of the neuromem TypeScript package.
 *
 * Run with:
 *   cd typescript && npm run build
 *   node --input-type=module < ../examples/example_typescript_basic.ts
 *
 * Or compile this file directly and run with ts-node.
 */

import { ContextManager, MessageScorer, Summarizer, Pruner } from "../typescript/src/index.js";

// ---------------------------------------------------------------------------
// 1. ContextManager — simulate a conversation
// ---------------------------------------------------------------------------

console.log("=".repeat(60));
console.log("1. Basic ContextManager (TypeScript)");
console.log("=".repeat(60));

const cm = new ContextManager({ tokenBudget: 500, alwaysKeepLastN: 2 });
await cm.addSystem("You are a helpful assistant specialized in science topics.");

const turns: Array<[string, string]> = [
  ["user", "What is quantum entanglement?"],
  ["assistant",
    "Quantum entanglement is a phenomenon where two particles become correlated " +
    "such that the state of one instantly influences the other, regardless of distance."],
  ["user", "How is it different from classical correlations?"],
  ["assistant",
    "Classical correlations are based on pre-shared information. Quantum correlations " +
    "are fundamentally different — the states are undetermined until measured."],
  ["user", "Can entanglement enable faster-than-light communication?"],
  ["assistant",
    "No. You cannot use entanglement to transmit classical information faster than light " +
    "because measuring one particle yields a random result."],
  ["user", "What is the EPR paradox?"],
  ["assistant",
    "The EPR paradox, proposed by Einstein, Podolsky, and Rosen in 1935, argued that " +
    "quantum mechanics was incomplete because entangled particles appeared to require " +
    "'spooky action at a distance'."],
  ["user", "What experiment resolved the EPR paradox?"],
  ["assistant",
    "Bell's theorem and Aspect's 1982 experiment showed that quantum correlations violate " +
    "Bell inequalities, ruling out local hidden variable theories."],
  ["user", "This is a critical requirement: summarize everything so far."],
];

for (const [role, content] of turns) {
  await cm.add(role, content);
}

console.log(`Messages added:  ${cm.messageCount}`);
console.log(`Token count:     ${cm.tokenCount}`);
console.log(`Stats:          `, cm.stats());
console.log();

const messages = await cm.getMessages();
console.log(`Messages returned after getMessages(): ${messages.length}`);
for (const m of messages) {
  const preview = m.content.slice(0, 60).replace(/\n/g, " ");
  console.log(`  [${m.role.padEnd(9)}] ${preview}…`);
}
console.log();


// ---------------------------------------------------------------------------
// 2. Scoring
// ---------------------------------------------------------------------------

console.log("=".repeat(60));
console.log("2. Message Scoring");
console.log("=".repeat(60));

const scorer = new MessageScorer({ recencyDecay: 0.1 });
const sample = [
  { role: "system",    content: "You are an AI assistant." },
  { role: "user",      content: "What is my current goal?" },
  { role: "assistant", content: "I don't know your specific goal." },
  { role: "user",      content: "My critical requirement is to finish the report by Friday." },
  { role: "assistant", content: "Understood! I'll help you finish the report by Friday." },
  { role: "user",      content: "What's 2 + 2?" },
  { role: "assistant", content: "4." },
];

const scored = scorer.scoreMessages(sample);
console.log(`${"#".padEnd(3)} ${"Role".padEnd(10)} ${"Score".padEnd(7)} Content preview`);
console.log("-".repeat(70));
for (const s of scored) {
  const preview = s.content.slice(0, 45).replace(/\n/g, " ");
  console.log(`${String(s.index).padEnd(3)} ${s.role.padEnd(10)} ${s.score.toFixed(4).padEnd(7)} ${preview}`);
}
console.log();


// ---------------------------------------------------------------------------
// 3. Summarizer
// ---------------------------------------------------------------------------

console.log("=".repeat(60));
console.log("3. Extractive Summarizer");
console.log("=".repeat(60));

const summarizer = new Summarizer({ mode: "extractive", targetRatio: 0.4 });
const toSummarize = sample.filter((m) => m.role !== "system");
const result = await summarizer.summarize(toSummarize);

console.log(`Original messages:  ${result.originalMessageCount}`);
console.log(`Original tokens:    ${result.originalTokenCount}`);
console.log(`Summary tokens:     ${result.summaryTokenCount}`);
console.log(`Compression ratio:  ${(result.compressionRatio * 100).toFixed(1)}%`);
console.log();
console.log("Summary:");
console.log(result.summaryText);
console.log();


// ---------------------------------------------------------------------------
// 4. Pruner
// ---------------------------------------------------------------------------

console.log("=".repeat(60));
console.log("4. Pruner — force prune to 200 tokens");
console.log("=".repeat(60));

const pruner = new Pruner({
  tokenBudget: 200,
  minScoreThreshold: 0.25,
  alwaysKeepLastN: 2,
  summarizeBeforePrune: true,
});

const bigContext = [
  { role: "system",    content: "You are a coding assistant. Never expose secrets." },
  { role: "user",      content: "How do I reverse a string in Python?" },
  { role: "assistant", content: "Use slicing: my_string[::-1]" },
  { role: "user",      content: "What about in JavaScript?" },
  { role: "assistant", content: "Use: str.split('').reverse().join('')" },
  { role: "user",      content: "What is 5 + 5?" },
  { role: "assistant", content: "10." },
  { role: "user",      content: "Critical requirement: always use type hints in Python." },
  { role: "assistant", content: "Understood. I will always include type hints in Python code." },
  { role: "user",      content: "How do I write a generic function in TypeScript?" },
];

const inputTokens = bigContext.reduce((s, m) => s + Math.floor(m.content.length / 4), 0);
console.log(`Input: ${bigContext.length} messages, ~${inputTokens} tokens`);

const pruneResult = await pruner.prune(bigContext, true);
const outputTokens = pruneResult.messages.reduce(
  (s, m) => s + Math.floor(m.content.length / 4), 0
);

console.log(`Output: ${pruneResult.messages.length} messages, ~${outputTokens} tokens`);
console.log(`Removed: ${pruneResult.removedCount} messages (${pruneResult.removedTokens} tokens)`);
console.log(`Summary inserted: ${pruneResult.summaryInserted}`);
console.log();
console.log("Resulting messages:");
for (const m of pruneResult.messages) {
  const preview = m.content.slice(0, 70).replace(/\n/g, " ");
  console.log(`  [${m.role.padEnd(9)}] ${preview}`);
}

console.log();
console.log("Done! All TypeScript examples ran without errors.");
