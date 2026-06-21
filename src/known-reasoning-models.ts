/**
 * Model-name prefixes for OpenAI families that expose internal
 * reasoning / thinking tokens and accept the `reasoning_effort`
 * request parameter on the Chat Completions API.
 *
 * Matched as a prefix so dated variants (`o3-2025-04-16`) and
 * `-mini` / `-pro` suffixes (`o4-mini`, `gpt-5-pro`) are covered
 * without listing every release tag explicitly.
 *
 * Maintenance: append a new prefix when OpenAI ships a reasoning
 * model family that doesn't already match. Devs can always override
 * per-model via `openai.model({ name, reasoning: true | false })` —
 * explicit config wins over inference in either direction.
 */
const REASONING_CAPABLE_PREFIXES = ["o1", "o3", "o4", "gpt-5"];

/**
 * Infer whether a given OpenAI model name is a reasoning model (o-series
 * and the gpt-5 family) based on the known-prefix list. Unknown models
 * default to `false` so the adapter never forwards an unsupported
 * `reasoning_effort` param to a non-reasoning model (which would 400).
 *
 * @example
 * inferReasoningCapability("o3-mini");          // → true
 * inferReasoningCapability("o4-mini");          // → true
 * inferReasoningCapability("gpt-5-pro");        // → true
 * inferReasoningCapability("gpt-4o");           // → false
 * inferReasoningCapability("custom-llm");       // → false
 */
export function inferReasoningCapability(modelName: string): boolean {
  const normalized = modelName.toLowerCase();

  return REASONING_CAPABLE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}
