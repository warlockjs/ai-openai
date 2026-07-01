import type {
  EmbedderConfig,
  ImageModelConfig,
  ModelConfig,
  ModelPricing,
  SpeechModelConfig,
  TranscriptionModelConfig,
} from "@warlock.js/ai";
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
  /**
   * Opt into PDF / document **input**. Default `false` — OpenAI accepts
   * PDF `file` content parts only on specific models (the `gpt-4o`
   * family on Chat Completions), so the capability is conservative and
   * honest by default rather than auto-inferred. When `true`, the agent
   * lets `{ type: "pdf" }` attachments through and the adapter maps them
   * to OpenAI `file` parts (base64 `file_data`).
   */
  pdf?: boolean;
  /**
   * Opt into audio **input**. Default `false` — only the
   * `gpt-4o-audio-preview` family accepts `input_audio`, so the
   * capability is off unless you set it. When `true`, the agent lets
   * `{ type: "audio" }` attachments through and the adapter maps them to
   * OpenAI `input_audio` parts (`wav` / `mp3`, base64).
   */
  audio?: boolean;
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

/**
 * Per-model configuration for `OpenAISDK.image()`. Mirrors the neutral
 * {@link ImageModelConfig} — `name` is a `gpt-image-*` or `dall-e-*`
 * model id, and `pricing` is the optional per-model USD override
 * (per-token for gpt-image, `perImage` for DALL·E).
 *
 * @example
 * openai.image({ name: "gpt-image-1" });
 * openai.image({ name: "dall-e-3", pricing: { perImage: 0.04 } });
 */
export type OpenAIImageConfig = ImageModelConfig;

/**
 * Per-model configuration for `OpenAISDK.speech()`. Mirrors the neutral
 * {@link SpeechModelConfig} — `name` is a `tts-1` / `gpt-4o-mini-tts`
 * model id, `voice` a default voice, `pricing` the per-character
 * (`tts-1`) or per-token (`gpt-4o-mini-tts`) USD override.
 *
 * @example
 * openai.speech({ name: "tts-1", voice: "alloy", pricing: { perMillionCharacters: 15 } });
 */
export type OpenAISpeechConfig = SpeechModelConfig;

/**
 * Per-model configuration for `OpenAISDK.transcribe()`. Mirrors the
 * neutral {@link TranscriptionModelConfig} — `name` is a `whisper-1` /
 * `gpt-4o-transcribe` model id, `pricing` the per-minute (`whisper-1`)
 * or per-token (`gpt-4o-transcribe`) USD override.
 *
 * @example
 * openai.transcribe({ name: "whisper-1", pricing: { perMinute: 0.006 } });
 */
export type OpenAITranscriptionConfig = TranscriptionModelConfig;
