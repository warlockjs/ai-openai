import type { Usage } from "@warlock.js/ai";

/** The already-extracted token counts of one provider usage block. */
export type UsageCounts = {
  input: number;
  output: number;
  total: number;
  /** Subset of `input` served from the prompt cache. */
  cachedTokens?: number;
  /** Hidden reasoning channel, already counted within `output`. */
  reasoningTokens?: number;
  /** Subset of `input` written to the prompt cache. */
  cacheWriteTokens?: number;
};

/**
 * Build the neutral `Usage` from extracted counts. The optional detail
 * counters are emitted only when the provider reports a positive value, so
 * non-reasoning / uncached calls keep the lean `{ input, output, total }`
 * shape. Shared by the Chat Completions and Responses adapters so both
 * apply the same rule.
 */
export function buildUsage(counts: UsageCounts): Usage {
  const { input, output, total, cachedTokens, reasoningTokens, cacheWriteTokens } = counts;

  return {
    input,
    output,
    total,
    ...(cachedTokens !== undefined && cachedTokens > 0 ? { cachedTokens } : {}),
    ...(reasoningTokens !== undefined && reasoningTokens > 0 ? { reasoningTokens } : {}),
    ...(cacheWriteTokens !== undefined && cacheWriteTokens > 0 ? { cacheWriteTokens } : {}),
  };
}
