import type { OpenAIResponseFormat } from "../config.type";

/**
 * How a `responseSchema` should reach the wire, decided once for both the
 * Chat Completions adapter (`response_format`) and the Responses adapter
 * (`text.format`). Only the wire location differs between the two.
 */
export type StructuredOutputPlan =
  | { mode: "none" }
  | { mode: "json_object" }
  | { mode: "json_schema"; schema: Record<string, unknown> };

/**
 * Map an explicit `responseFormat` override to the default
 * `structuredOutput` capability. Loose wire modes (`"json_object"`,
 * `"text"`) don't enforce shape, so the agent needs to see the soft
 * schema hint in the system prompt — that only happens when the
 * capability is `false`. Default (no override) stays `true` to
 * preserve the prior assumption that OpenAI models support strict
 * structured output.
 */
export function inferStructuredOutput(
  responseFormat: OpenAIResponseFormat | undefined,
): boolean {
  if (responseFormat === "json_object" || responseFormat === "text") {
    return false;
  }

  return true;
}

/**
 * Decide the structured-output mode for a call.
 *
 * When `override` (the `responseFormat` config) is set, it wins: `"text"`
 * emits nothing, `"json_object"` always picks the loose mode, and
 * `"json_schema"` picks strict mode (with the same strict-compatibility
 * safety check - a malformed schema still degrades to `json_object` rather
 * than 400). The override exists because some targets (older OpenAI models,
 * OpenRouter routes, Ollama OpenAI-compat) reject strict `json_schema`
 * outright.
 *
 * When the override is omitted, strict `json_schema` mode (token-level
 * enforcement) is used only when the schema is a proper root-object JSON
 * Schema whose every object lists all of its properties in `required`. For
 * anything else it falls back to loose `json_object` mode. Client-side
 * Standard Schema `validate()` still enforces the full shape.
 */
export function planStructuredOutput(
  responseSchema: Record<string, unknown> | undefined,
  override: OpenAIResponseFormat | undefined,
): StructuredOutputPlan {
  if (!responseSchema || override === "text") {
    return { mode: "none" };
  }

  if (override === "json_object") {
    return { mode: "json_object" };
  }

  // Either auto-select (no override) or explicit `"json_schema"`. The
  // strict-compat check still applies in the explicit case - a malformed or
  // non-object schema would 400 before sampling, so degrade to `json_object`.
  if (isStrictCompatible(responseSchema)) {
    return { mode: "json_schema", schema: responseSchema };
  }

  return { mode: "json_object" };
}

/**
 * OpenAI strict `json_schema` mode requires the root to be a JSON Schema
 * object type (`{ type: "object", properties: ... }`). Anything else
 * (top-level arrays, primitives, unknown shapes) is rejected with a 400
 * before a token is sampled. Checked structurally here so the first call
 * doesn't crash on a malformed extraction.
 */
export function isStrictCompatible(schema: Record<string, unknown>): boolean {
  return (
    schema.type === "object" &&
    typeof schema.properties === "object" &&
    schema.properties !== null &&
    isStrictSafeNode(schema)
  );
}

/**
 * Recursively check the one strict-mode rule schemas most often trip on:
 * every object must list ALL of its `properties` in `required` (OpenAI
 * strict has no notion of optional - optional fields must be expressed as
 * nullable, e.g. `type: ["string", "null"]`, and still appear in
 * `required`). A schema that violates this anywhere in the tree is NOT sent
 * in strict mode, so a hand-built or optional-bearing schema can't 400 the
 * call ("'required' ... must include every key in properties").
 */
function isStrictSafeNode(node: unknown): boolean {
  if (!node || typeof node !== "object") {
    return true;
  }

  const record = node as Record<string, unknown>;

  if (record.type === "object" && record.properties && typeof record.properties === "object") {
    const properties = record.properties as Record<string, unknown>;
    const keys = Object.keys(properties);
    const required = Array.isArray(record.required) ? (record.required as unknown[]) : [];

    if (keys.some((key) => !required.includes(key))) {
      return false;
    }

    for (const key of keys) {
      if (!isStrictSafeNode(properties[key])) {
        return false;
      }
    }
  }

  if (record.items !== undefined && !isStrictSafeNode(record.items)) {
    return false;
  }

  for (const branch of ["anyOf", "allOf", "oneOf"] as const) {
    const value = record[branch];

    if (Array.isArray(value) && value.some((sub) => !isStrictSafeNode(sub))) {
      return false;
    }
  }

  return true;
}
