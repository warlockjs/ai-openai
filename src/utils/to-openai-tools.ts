import { extractJsonSchema, type ToolConfig } from "@warlock.js/ai";
import type OpenAI from "openai";

/**
 * Convert vendor-neutral ToolConfig[] to OpenAI's tools array.
 * Uses the shared `extractJsonSchema` helper; falls back to an empty-object
 * schema when extraction fails so the tool still registers with the provider.
 *
 * @example
 * const tools = toOpenAITools([weatherTool, calculatorTool]);
 * await client.chat.completions.create({ model, messages, tools });
 */
export function toOpenAITools(
  tools: ToolConfig<unknown, unknown>[] | undefined,
): OpenAI.Chat.Completions.ChatCompletionTool[] | undefined {
  if (!tools || tools.length === 0) {
    return undefined;
  }

  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: toParameters(tool.input),
    },
  }));
}

/**
 * Resolve a tool's input schema to a JSON-Schema object. OpenAI's
 * function `parameters` expects an object root; anything else (or a
 * failed extraction) degrades to an empty-object schema so the tool
 * still registers and the model simply sees no parameters.
 */
function toParameters(input: ToolConfig<unknown, unknown>["input"]): Record<string, unknown> {
  const schema = extractJsonSchema(input);

  if (schema && schema.type === "object") {
    return schema;
  }

  return { type: "object", properties: {} };
}
