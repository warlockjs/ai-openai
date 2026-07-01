import OpenAI from "openai";
import type {
  EmbedderContract,
  ImageModelContract,
  ModelContract,
  ModelPricing,
  SDKAdapterContract,
  SpeechModelContract,
  TranscriptionModelContract,
} from "@warlock.js/ai";
import { approximateTokenCount } from "@warlock.js/ai";
import type {
  OpenAIEmbedderConfig,
  OpenAIImageConfig,
  OpenAIModelConfig,
  OpenAISDKConfig,
  OpenAISpeechConfig,
  OpenAITranscriptionConfig,
} from "./config.type";
import { OpenAIEmbedder } from "./embedder";
import { OpenAIImageModel } from "./image";
import { OpenAIModel } from "./model";
import { OpenAISpeechModel } from "./speech";
import { OpenAITranscriptionModel } from "./transcription";

/**
 * OpenAI-backed implementation of `SDKAdapterContract`.
 *
 * **Role.** The package entry point for any OpenAI-compatible provider
 * (OpenAI, Azure OpenAI, OpenRouter, local gateways speaking the Chat
 * Completions protocol). A single `OpenAISDK` instance holds one live
 * `OpenAI` client, shared by every `ModelContract` it produces via
 * `model()`. Users construct one SDK per provider/account and reuse it
 * across all agents, workflows, and supervisors that target that
 * provider.
 *
 * **Responsibility.**
 * - Owns: a long-lived `OpenAI` client (authentication, base URL) and
 *   its lifetime scope. Factory for `OpenAIModel` instances — each
 *   model call gets a reference to the same client.
 * - Does NOT own: anything per-call (tool execution, message history,
 *   streaming loop) — those live in `OpenAIModel` and the agent runtime.
 *
 * Modeled as a class (see §4.2 of code-style.md — "long-lived state
 * across many calls"): the `OpenAI` client is heavy to construct and
 * designed to be reused; keeping it on `this` makes that reuse
 * explicit and aligns with the PascalCase naming convention readers
 * expect from a constructor.
 *
 * @example
 * const openai = new OpenAISDK({ apiKey: process.env.OPENAI_API_KEY! });
 * const model = openai.model({ name: "gpt-4o", temperature: 0.7 });
 * const tokens = await openai.count("Hello world");
 *
 * @example
 * // Compose into an `ai.openai` namespace for ergonomic agent wiring
 * const ai = { agent, tool, systemPrompt, persona, instruction, openai: new OpenAISDK({ apiKey }) };
 * const myAgent = ai.agent({ model: ai.openai.model({ name: "gpt-4o-mini" }) });
 */
export class OpenAISDK implements SDKAdapterContract {
  private readonly client: OpenAI;
  private readonly provider: string;
  private readonly pricing?: Record<string, ModelPricing>;

  public constructor(config: OpenAISDKConfig) {
    // Peel off the framework-only keys and forward every other upstream
    // `ClientOptions` (timeout, maxRetries, defaultHeaders, fetch,
    // organization, project, …) verbatim — they type-check, so dropping them
    // is a silent footgun. Mirrors the Bedrock/Google/Ollama adapters.
    const { provider, pricing, ...clientOptions } = config;

    this.client = new OpenAI(clientOptions);
    this.provider = provider ?? "openai";
    this.pricing = pricing;
  }

  /**
   * Build an `OpenAIModel` bound to this SDK's client. Each call returns
   * a fresh model instance, but all instances share the underlying
   * `OpenAI` client — connection pools, rate limits, and authentication
   * state stay unified across every model produced here. The SDK's
   * `provider` label is forwarded so every model self-identifies as
   * coming from the same upstream.
   *
   * Pricing resolution: per-model `config.pricing` wins; otherwise the
   * SDK-level registry entry keyed by `config.name`; otherwise
   * `undefined` (no cost computed).
   */
  public model(config: OpenAIModelConfig): ModelContract {
    const resolvedPricing = config.pricing ?? this.pricing?.[config.name];
    const resolvedConfig: OpenAIModelConfig =
      resolvedPricing === config.pricing ? config : { ...config, pricing: resolvedPricing };

    return new OpenAIModel(this.client, resolvedConfig, this.provider);
  }

  /**
   * Rough token-count estimate for a given text. Uses a
   * character-heuristic (`approximateTokenCount`) from the core package
   * — good enough for budgeting and quota guards, not for billing.
   * Accepts an optional model id for future per-model tokenizer
   * dispatch; currently ignored.
   */
  public async count(text: string, _model?: string): Promise<number> {
    return approximateTokenCount(text);
  }

  /**
   * Build an `OpenAIEmbedder` bound to this SDK's client. Each call
   * returns a fresh embedder instance sharing the same underlying
   * `OpenAI` client — connection pools and authentication stay unified
   * across every embedder produced here.
   *
   * @example
   * const embedder = openai.embedder({ name: "text-embedding-3-small" });
   * const { vector } = await embedder.embed("Hello world");
   */
  public embedder(config: OpenAIEmbedderConfig): EmbedderContract {
    return new OpenAIEmbedder(this.client, config);
  }

  /**
   * Build an `OpenAIImageModel` bound to this SDK's client for use with
   * `ai.image({ model, prompt })`. Accepts the `gpt-image-*` (token-metered)
   * and `dall-e-*` (per-image-metered) families; a non-image model id
   * is rejected at construction.
   *
   * Pricing resolution mirrors `model()`: per-model `config.pricing`
   * wins, otherwise the SDK-level registry entry keyed by `config.name`,
   * otherwise `undefined` (no cost computed). A token-priced
   * `gpt-image-1` entry can live in the same SDK registry as the chat
   * models.
   *
   * @example
   * const model = openai.image({ name: "gpt-image-1" });
   * const { data } = await ai.image({ model, prompt: "a red bicycle" });
   */
  public image(config: OpenAIImageConfig): ImageModelContract {
    const resolvedPricing = config.pricing ?? this.pricing?.[config.name];
    const resolvedConfig: OpenAIImageConfig =
      resolvedPricing === config.pricing ? config : { ...config, pricing: resolvedPricing };

    return new OpenAIImageModel(this.client, resolvedConfig, this.provider);
  }

  /**
   * Build an `OpenAISpeechModel` (text-to-speech) bound to this SDK's
   * client, for use with `ai.speech({ model, text })`. Accepts the
   * `tts-1` / `gpt-4o-mini-tts` families; a non-TTS model id is rejected
   * at construction.
   *
   * @example
   * const tts = openai.speech({ name: "tts-1", voice: "alloy" });
   * const { data } = await ai.speech({ model: tts, text: "Hello" });
   */
  public speech(config: OpenAISpeechConfig): SpeechModelContract {
    return new OpenAISpeechModel(this.client, config, this.provider);
  }

  /**
   * Build an `OpenAITranscriptionModel` (speech-to-text) bound to this
   * SDK's client, for use with `ai.transcribe({ model, audio })`.
   * Accepts the `whisper-1` / `gpt-4o-transcribe` families; a non-STT
   * model id is rejected at construction.
   *
   * @example
   * const stt = openai.transcribe({ name: "whisper-1" });
   * const { data } = await ai.transcribe({ model: stt, audio });
   */
  public transcribe(config: OpenAITranscriptionConfig): TranscriptionModelContract {
    return new OpenAITranscriptionModel(this.client, config, this.provider);
  }
}
