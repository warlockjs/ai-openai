/**
 * Model-name prefixes for OpenAI families that support vision input
 * (image attachments) on the Chat Completions API.
 *
 * Matched as a prefix so dated variants (`gpt-4o-2024-08-06`) and
 * `-mini` / `-preview` suffixes (`gpt-4o-mini`, `gpt-4-turbo-preview`)
 * are covered without listing every release tag explicitly.
 *
 * Maintenance: append a new prefix when OpenAI ships a vision-capable
 * model family that doesn't already match. Devs can always override
 * per-model via `openai.model({ name, vision: true | false })` —
 * explicit config wins over inference in either direction.
 */
const VISION_CAPABLE_PREFIXES = [
  "gpt-4o",
  "gpt-4-turbo",
  "gpt-4.1",
  "o1",
  "o3",
  "chatgpt-4o",
];

/**
 * Infer whether a given OpenAI model name supports vision based on the
 * known-prefix list. Unknown models default to `false` so that passing
 * an image attachment to an unsupported model surfaces a clear,
 * agent-side capability error instead of an opaque OpenAI 400.
 *
 * @example
 * inferVisionCapability("gpt-4o-mini");          // → true
 * inferVisionCapability("gpt-4o-2024-08-06");    // → true
 * inferVisionCapability("gpt-3.5-turbo");        // → false
 * inferVisionCapability("custom-llm");           // → false
 */
export function inferVisionCapability(modelName: string): boolean {
  const normalized = modelName.toLowerCase();

  return VISION_CAPABLE_PREFIXES.some((prefix) => normalized.startsWith(prefix));
}
