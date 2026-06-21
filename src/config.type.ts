import type { EmbedderConfig, ModelConfig, ModelPricing } from "@warlock.js/ai";
import type { ClientOptions } from "openai";

/**
 * Configuration for the OpenAI SDK adapter.
 *
 * `provider` lets you label the SDK with the actual upstream you're
 * pointing at — useful when the same OpenAI-compatible client is wired
 * to OpenRouter, Azure, Ollama, or a local gateway. The label flows
 * through to `ModelContract.provider` and surfaces on `AgentReport.model`,
 * logs, and any middleware that branches on provider identity. Defaults
 * to `"openai"` when omitted.
 *
 * `pricing` is an optional SDK-level registry keyed by model name. One
 * source of truth per provider; matches how providers publish pricing
 * tables. Resolution at `model()` call time: per-model `pricing` (on
 * `OpenAIModelConfig`) > this SDK registry > `undefined`. When neither
 * is set, `Usage.cost` stays `undefined` on every report this SDK's
 * models produce.
 *
 * @example
 * new OpenAISDK({ apiKey, baseURL: "https://openrouter.ai/api/v1", provider: "openrouter" });
 *
 * @example
 * // SDK-level pricing registry — pricing in USD per million tokens.
 * new OpenAISDK({
 *   apiKey,
 *   pricing: {
 *     "gpt-4o-mini": { input: 0.15, output: 0.6, cachedInput: 0.075 },
 *     "gpt-4o":      { input: 2.5,  output: 10,  cachedInput: 1.25  },
 *   },
 * });
 */
export type OpenAISDKConfig = ClientOptions & {
  provider?: string;
  /**
   * Per-model USD pricing registry, keyed by model name. Surfaced onto
   * every `OpenAIModel` produced by `model()`; per-model
   * `OpenAIModelConfig.pricing` still wins when both are set.
   */
  pricing?: Record<string, ModelPricing>;
};

/**
 * Per-model override for the OpenAI `response_format` parameter.
 *
 * - `"json_schema"` — strict token-level shape enforcement. Only modern
 *   OpenAI models (and a few compatible gateways) accept it.
 * - `"json_object"` — guarantees valid JSON output but does NOT enforce
 *   shape; shape is re-communicated via the agent's soft prompt hint.
 * - `"text"` — no `response_format` on the wire at all. Relies entirely
 *   on the agent's soft prompt hint to coax JSON out of the model.
 *
 * @example
 * // Route through OpenRouter to a model that rejects strict json_schema:
 * openai.model({ name: "some-legacy-model", responseFormat: "json_object" });
 */
export type OpenAIResponseFormat = "json_schema" | "json_object" | "text";

/**
 * Per-model configuration for `OpenAISDK.model()`. Extends the neutral
 * `ModelConfig` with OpenAI-specific capability overrides.
 *
 * @example
 * openai.model({ name: "gpt-4o-mini" });               // vision auto-true
 * openai.model({ name: "fine-tuned-x", vision: true }); // dev override
 */
export type OpenAIModelConfig = ModelConfig & {
  /**
   * Override the auto-inferred vision capability. When omitted, the
   * adapter checks the model name against a known-prefix list (see
   * `known-vision-models.ts`). Setting `true` or `false` explicitly
   * always wins over inference — useful for fine-tuned models, custom
   * gateways, or testing capability-degraded behavior.
   */
  vision?: boolean;
  /**
   * Override the wire-level `response_format` the adapter emits when
   * the caller supplies a response schema. Use when the target model
   * cannot handle strict `json_schema` mode — common for older
   * OpenAI models and many OpenAI-compatible gateways (OpenRouter,
   * Ollama, fine-tunes). Omitted = auto-select based on schema shape
   * (current default).
   *
   * Setting this to `"json_object"` or `"text"` also downgrades the
   * inferred `structuredOutput` capability to `false` so the agent
   * re-injects a soft schema hint into the system prompt. Pass
   * `structuredOutput` explicitly to override that downgrade.
   */
  responseFormat?: OpenAIResponseFormat;
  /**
   * Override the inferred `structuredOutput` capability. When omitted,
   * the adapter treats the model as capable unless `responseFormat`
   * forces a loose mode (`"json_object"` or `"text"`), in which case
   * it downgrades to `false` so the agent injects the soft schema
   * hint. Set explicitly to pin the capability regardless of
   * `responseFormat`.
   */
  structuredOutput?: boolean;
  /**
   * Override the auto-inferred `reasoning` capability. When omitted,
   * the adapter checks the model name against a known-prefix list (see
   * `known-reasoning-models.ts`) — `true` for the o-series (`o1*`,
   * `o3*`, `o4*`) and the `gpt-5*` family, `false` otherwise. Setting
   * this explicitly always wins over inference — useful for fine-tuned
   * reasoning models or gateways exposing reasoning under a custom
   * name. When `false`, `ModelCallOptions.reasoning` is ignored rather
   * than forwarded as an unsupported `reasoning_effort` param.
   */
  reasoning?: boolean;
};

/**
 * Per-embedder configuration for `OpenAISDK.embedder()`. Mirrors the
 * neutral `EmbedderConfig` — `dimensions` is forwarded as-is to the
 * OpenAI embeddings API, enabling output truncation for models that
 * support it (e.g. `text-embedding-3-*`).
 *
 * @example
 * openai.embedder({ name: "text-embedding-3-small" });
 * openai.embedder({ name: "text-embedding-3-large", dimensions: 256 });
 */
export type OpenAIEmbedderConfig = EmbedderConfig;
