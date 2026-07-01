import {
  InvalidRequestError,
  type SpeechGenerationResponse,
  type SpeechModelContract,
  type SpeechModelPricing,
  type SpeechOptions,
} from "@warlock.js/ai";
import { log, type Logger } from "@warlock.js/logger";
import type OpenAI from "openai";
import type { OpenAISpeechConfig } from "./config.type";
import { wrapOpenAIError } from "./utils";

const LOG_MODULE = "ai.openai";

/** Model-id prefixes OpenAI exposes through the **Speech** (TTS) API. */
const SPEECH_MODEL_PREFIXES = ["tts-1", "gpt-4o-mini-tts", "gpt-audio"] as const;

/** True when `name` is a recognized OpenAI text-to-speech model. */
export function isOpenAISpeechModel(name: string): boolean {
  return SPEECH_MODEL_PREFIXES.some((prefix) => name.startsWith(prefix));
}

/** Map a neutral output container hint to its IANA audio media type. */
function audioMediaType(format: string | undefined): string {
  switch (format) {
    case "opus":
      return "audio/opus";
    case "aac":
      return "audio/aac";
    case "flac":
      return "audio/flac";
    case "wav":
      return "audio/wav";
    case "pcm":
      return "audio/pcm";
    default:
      return "audio/mpeg";
  }
}

/**
 * OpenAI-backed implementation of `SpeechModelContract` (text-to-speech)
 * via `audio.speech.create`. Standalone primitive — no relation to chat
 * completions or the agent loop. Consumed by the `ai.speech()` verb.
 *
 * **Capability guard.** The constructor rejects a non-TTS model id up
 * front (`tts-1` / `gpt-4o-mini-tts` only) so the mistake surfaces at
 * wiring time, mirroring the embedder / image guards.
 *
 * @example
 * const tts = new OpenAISpeechModel(client, { name: "tts-1", voice: "alloy" }, "openai");
 * const { audio } = await tts.generate("Welcome aboard.");
 */
export class OpenAISpeechModel implements SpeechModelContract {
  public readonly name: string;
  public readonly provider: string;
  public readonly pricing?: SpeechModelPricing;

  private readonly client: OpenAI;
  private readonly defaultVoice?: string;
  private readonly logger: Logger = log;

  public constructor(client: OpenAI, config: OpenAISpeechConfig, provider: string = "openai") {
    if (!isOpenAISpeechModel(config.name)) {
      throw new InvalidRequestError(
        `"${config.name}" is not a known OpenAI text-to-speech model. ` +
          "Use a `tts-1` / `tts-1-hd` / `gpt-4o-mini-tts` model with openai.speech({ name }).",
      );
    }

    this.client = client;
    this.name = config.name;
    this.provider = provider;
    this.pricing = config.pricing;
    this.defaultVoice = config.voice;
  }

  public async generate(text: string, options?: SpeechOptions): Promise<SpeechGenerationResponse> {
    const format = options?.format ?? "mp3";

    this.logger.debug(LOG_MODULE, "speech.request", "audio.speech.create", {
      model: this.name,
      characters: text.length,
    });

    let response: Response;

    try {
      response = await this.client.audio.speech.create(
        {
          model: this.name,
          input: text,
          voice: options?.voice ?? this.defaultVoice ?? "alloy",
          response_format: format as OpenAI.Audio.SpeechCreateParams["response_format"],
          ...(options?.speed !== undefined ? { speed: options.speed } : {}),
          ...(options?.instructions !== undefined ? { instructions: options.instructions } : {}),
        },
        options?.signal ? { signal: options.signal } : undefined,
      );
    } catch (thrown) {
      const wrapped = wrapOpenAIError(thrown);
      this.logger.error(LOG_MODULE, "speech.error", wrapped.message, {
        code: wrapped.code,
        context: wrapped.context,
      });
      throw wrapped;
    }

    const base64 = Buffer.from(await response.arrayBuffer()).toString("base64");

    return {
      audio: { type: "base64", base64, mediaType: audioMediaType(format) },
      // The Speech API reports no token usage; spend is priced per
      // character (or per token for gpt-4o-mini-tts) by `ai.speech()`.
      usage: { input: 0, output: 0, total: 0 },
      characters: text.length,
    };
  }
}
