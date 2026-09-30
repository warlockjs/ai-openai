import type { ToolConfig } from "@warlock.js/ai";
import type OpenAI from "openai";
import { toToolParameters } from "./to-openai-tools";

/**
 * Convert vendor-neutral `ToolConfig[]` to Responses function tools.
 *
 * Responses tools are FLAT (`name` / `description` / `parameters` sit beside
 * `type`, not under a nested `function` object) and `strict` is required by
 * the SDK type (`FunctionTool.strict: boolean | null`). The Chat path never
 * sets `strict` (non-strict is the OpenAI default there), so every tool is
 * sent with `strict: false` to keep the same behavior.
 *
 * @example
 * toResponsesTools([weatherTool]);
 * // [{ type: "function", name: "getWeather", description: "...", parameters: {...}, strict: false }]
 */
export function toResponsesTools(
  tools: ToolConfig<unknown, unknown>[] | undefined,
): OpenAI.Responses.FunctionTool[] | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  return tools.map((tool) => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: toToolParameters(tool.input),
    strict: false,
  }));
}

/**
 * Read an optional `toolChoice` call option into the Responses `tool_choice`.
 *
 * `ModelCallOptions` does not declare `toolChoice` today (it only arrives
 * through the open index signature), so the value is validated here rather
 * than trusted. Accepted: `"none" | "auto" | "required"`, or a forced
 * function as `{ name }` / `{ type: "function", name }` (the Chat-style
 * nested `{ type: "function", function: { name } }` is also accepted). The
 * Responses form is flat: `{ type: "function", name }`. Anything else yields
 * `undefined`, leaving the provider default.
 */
export function toResponsesToolChoice(
  toolChoice: unknown,
): OpenAI.Responses.ToolChoiceOptions | OpenAI.Responses.ToolChoiceFunction | undefined {
  if (toolChoice === "none" || toolChoice === "auto" || toolChoice === "required") {
    return toolChoice;
  }

  if (typeof toolChoice !== "object" || toolChoice === null) {
    return undefined;
  }

  const record = toolChoice as { name?: unknown; function?: { name?: unknown } };
  const name = typeof record.name === "string" ? record.name : record.function?.name;

  if (typeof name === "string" && name !== "") {
    return { type: "function", name };
  }

  return undefined;
}
