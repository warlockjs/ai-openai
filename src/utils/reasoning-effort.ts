import type { ReasoningEffort } from "@warlock.js/ai";
import type OpenAI from "openai";

/** The effort strings the OpenAI wire accepts (shared by Chat Completions and Responses). */
export type OpenAIReasoningEffort = NonNullable<
  OpenAI.Chat.Completions.ChatCompletionCreateParams["reasoning_effort"]
>;

/**
 * The installed OpenAI SDK 7.23.0 declares `reasoning_effort` as
 * `none | minimal | low | medium | high | xhigh | max` in
 * `openai/src/resources/shared.ts:367`. Its type notes that model support
 * varies but does not provide a model-family-specific accepted set, so map by
 * that union alone. This exhaustive record makes a future core level a
 * typecheck failure until its OpenAI clamp is consciously selected.
 *
 * Both the Chat Completions adapter (`reasoning_effort`) and the Responses
 * adapter (`reasoning.effort`) read this one table, so the two can never
 * disagree on how a neutral level reaches the wire.
 */
const EFFORT_TO_OPENAI_EFFORT: Record<ReasoningEffort, OpenAIReasoningEffort> = {
  none: "none",
  minimal: "minimal",
  low: "low",
  medium: "medium",
  high: "high",
  xhigh: "xhigh",
  max: "max",
};

/**
 * Map a neutral `ReasoningEffort` to the value OpenAI accepts.
 *
 * @example
 * toOpenAIReasoningEffort("high"); // "high"
 */
export function toOpenAIReasoningEffort(effort: ReasoningEffort): OpenAIReasoningEffort {
  return EFFORT_TO_OPENAI_EFFORT[effort];
}
