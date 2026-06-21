import type { FinishReason } from "@warlock.js/ai";

const finishReasonMap: Record<string, FinishReason> = {
  stop: "stop",
  tool_calls: "tool_calls",
  length: "length",
};

/**
 * Map the raw OpenAI `finish_reason` string to the normalized FinishReason union.
 * Unknown/unexpected values fall through to "error".
 *
 * @example
 * mapFinishReason("stop");        // "stop"
 * mapFinishReason("tool_calls");  // "tool_calls"
 * mapFinishReason(null);          // "error"
 */
export function mapFinishReason(raw: string | null | undefined): FinishReason {
  return finishReasonMap[raw ?? ""] ?? "error";
}
