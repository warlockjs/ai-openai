/**
 * Model-id prefixes OpenAI exposes through the **Images** API
 * (`client.images.generate`). The two live families:
 *
 * - `gpt-image-*` — token-metered, always returns base64 bytes (no
 *   `response_format` knob), supports `output_format` + `background`.
 * - `dall-e-*` — per-image-metered, returns a URL or base64 via
 *   `response_format`.
 *
 * Used by {@link isOpenAIImageModel} for the construction-time guard so
 * `openai.image({ name: "gpt-4o" })` fails fast with a curated error
 * instead of a downstream 400 — mirroring the embedder/vision guards.
 */
export const OPENAI_IMAGE_MODEL_PREFIXES = ["gpt-image", "dall-e"] as const;

/**
 * True when `name` is a recognized OpenAI image-generation model. A
 * prefix match (not an exact list) so dated snapshots
 * (`gpt-image-1-mini`, `dall-e-3`) are covered without a maintenance
 * burden every time OpenAI ships a point release.
 *
 * @example
 * isOpenAIImageModel("gpt-image-1"); // true
 * isOpenAIImageModel("dall-e-3");    // true
 * isOpenAIImageModel("gpt-4o");      // false
 */
export function isOpenAIImageModel(name: string): boolean {
  return OPENAI_IMAGE_MODEL_PREFIXES.some((prefix) => name.startsWith(prefix));
}
