import {
  InvalidRequestError,
  type AudioInput,
  type TranscribeOptions,
  type TranscriptionModelContract,
  type TranscriptionModelPricing,
  type TranscriptionResponse,
  type TranscriptionSegment,
} from "@warlock.js/ai";
import { log, type Logger } from "@warlock.js/logger";
import OpenAI, { toFile } from "openai";
import type { OpenAITranscriptionConfig } from "./config.type";
import { wrapOpenAIError } from "./utils";

const LOG_MODULE = "ai.openai";

/** Model-id prefixes OpenAI exposes through the **Transcription** (STT) API. */
const TRANSCRIPTION_MODEL_PREFIXES = ["whisper", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"] as const;

/** True when `name` is a recognized OpenAI speech-to-text model. */
export function isOpenAITranscriptionModel(name: string): boolean {
  return TRANSCRIPTION_MODEL_PREFIXES.some((prefix) => name.startsWith(prefix));
}

/** Defensive view over the response, whose shape varies by `response_format`. */
type RawTranscription = {
  text: string;
  duration?: number;
  language?: string;
  segments?: Array<{ text: string; start?: number; end?: number }>;
  usage?: {
    type?: string;
    seconds?: number;
    input_tokens?: number;
    output_tokens?: number;
    total_tokens?: number;
  };
};

/**
 * OpenAI-backed implementation of `TranscriptionModelContract`
 * (speech-to-text) via `audio.transcriptions.create`. Consumed by the
 * `ai.transcribe()` verb.
 *
 * **Response format.** Defaults to `verbose_json` for `whisper-1` (so
 * the run gets a `duration` + timestamped `segments`) and `json` for
 * the token-metered `gpt-4o-transcribe` family. Base64 audio is wrapped
 * in an uploadable via the SDK's `toFile`.
 *
 * @example
 * const stt = new OpenAITranscriptionModel(client, { name: "whisper-1" }, "openai");
 * const { text } = await stt.transcribe({ base64, mediaType: "audio/mpeg" });
 */
export class OpenAITranscriptionModel implements TranscriptionModelContract {
  public readonly name: string;
  public readonly provider: string;
  public readonly pricing?: TranscriptionModelPricing;

  private readonly client: OpenAI;
  private readonly logger: Logger = log;

  public constructor(
    client: OpenAI,
    config: OpenAITranscriptionConfig,
    provider: string = "openai",
  ) {
    if (!isOpenAITranscriptionModel(config.name)) {
      throw new InvalidRequestError(
        `"${config.name}" is not a known OpenAI transcription model. ` +
          "Use a `whisper-1` / `gpt-4o-transcribe` / `gpt-4o-mini-transcribe` model with openai.transcribe({ name }).",
      );
    }

    this.client = client;
    this.name = config.name;
    this.provider = provider;
    this.pricing = config.pricing;
  }

  public async transcribe(
    audio: AudioInput,
    options?: TranscribeOptions,
  ): Promise<TranscriptionResponse> {
    const isWhisper = this.name.startsWith("whisper");
    const format = options?.format ?? (isWhisper ? "verbose_json" : "json");

    const file = await toFile(Buffer.from(audio.base64, "base64"), audio.filename ?? "audio", {
      type: audio.mediaType,
    });

    this.logger.debug(LOG_MODULE, "transcription.request", "audio.transcriptions.create", {
      model: this.name,
      format,
    });

    let raw: unknown;

    try {
      raw = await this.client.audio.transcriptions.create(
        {
          model: this.name,
          file,
          response_format: format as OpenAI.Audio.TranscriptionCreateParams["response_format"],
          ...(options?.language ? { language: options.language } : {}),
          ...(options?.prompt ? { prompt: options.prompt } : {}),
        } as OpenAI.Audio.TranscriptionCreateParamsNonStreaming,
        options?.signal ? { signal: options.signal } : undefined,
      );
    } catch (thrown) {
      const wrapped = wrapOpenAIError(thrown);
      this.logger.error(LOG_MODULE, "transcription.error", wrapped.message, {
        code: wrapped.code,
        context: wrapped.context,
      });
      throw wrapped;
    }

    const response = raw as RawTranscription;

    const segments: TranscriptionSegment[] | undefined = response.segments?.map((segment) => ({
      text: segment.text,
      ...(segment.start !== undefined ? { start: segment.start } : {}),
      ...(segment.end !== undefined ? { end: segment.end } : {}),
    }));

    const durationSeconds =
      response.duration ?? (response.usage?.type === "duration" ? response.usage.seconds : undefined);

    const usage =
      response.usage?.type === "tokens"
        ? {
            input: response.usage.input_tokens ?? 0,
            output: response.usage.output_tokens ?? 0,
            total: response.usage.total_tokens ?? 0,
          }
        : { input: 0, output: 0, total: 0 };

    return {
      text: response.text,
      ...(segments && segments.length > 0 ? { segments } : {}),
      ...(durationSeconds !== undefined ? { durationSeconds } : {}),
      usage,
    };
  }
}
