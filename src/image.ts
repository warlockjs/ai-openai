import {
  InvalidRequestError,
  ProviderError,
  type GeneratedImage,
  type ImageGenerationOptions,
  type ImageGenerationResponse,
  type ImageModelContract,
  type ImageModelPricing,
} from "@warlock.js/ai";
import { log, type Logger } from "@warlock.js/logger";
import type OpenAI from "openai";
import type { OpenAIImageConfig } from "./config.type";
import { isOpenAIImageModel } from "./known-image-models";
import { wrapOpenAIError } from "./utils";

const LOG_MODULE = "ai.openai";

/** Map a neutral output container to its IANA media type. */
function mediaTypeFor(format: string | undefined): string {
  switch (format) {
    case "jpeg":
    case "jpg":
      return "image/jpeg";
    case "webp":
      return "image/webp";
    default:
      return "image/png";
  }
}

/**
 * OpenAI-backed implementation of `ImageModelContract`.
 *
 * **Role.** Bridges the vendor-neutral `ai.image()` verb to OpenAI's
 * **Images** API for the two image families OpenAI ships: the
 * token-metered `gpt-image-*` models (always return base64 bytes) and
 * the per-image-metered `dall-e-*` models (URL or base64). Like
 * `OpenAIEmbedder`, it's a standalone primitive — no relationship to
 * chat completions, tools, or the agent loop.
 *
 * **Capability guard.** The constructor rejects a non-image model id
 * up front (`gpt-4o` → typed `InvalidRequestError`) so the mistake
 * surfaces at wiring time, not as a downstream provider 400 — the
 * "fail fast at construction" rule shared with the embedder/vision
 * guards.
 *
 * **Error handling.** Raw OpenAI SDK errors are wrapped into the typed
 * `@warlock.js/ai` `AIError` hierarchy via `wrapOpenAIError`, so a
 * caller catches `ProviderRateLimitError` / `ContentFilterError` /
 * `ProviderAuthError` rather than OpenAI's own classes. `ai.image()`
 * turns those throws into `result.error`.
 *
 * @example
 * const model = new OpenAIImageModel(client, { name: "gpt-image-1" }, "openai");
 * const { images, usage } = await model.generate("a teal ceramic mug, studio light");
 */
export class OpenAIImageModel implements ImageModelContract {
  public readonly name: string;
  public readonly provider: string;
  public readonly pricing?: ImageModelPricing;

  private readonly client: OpenAI;
  private readonly logger: Logger = log;

  public constructor(client: OpenAI, config: OpenAIImageConfig, provider: string = "openai") {
    if (!isOpenAIImageModel(config.name)) {
      throw new InvalidRequestError(
        `"${config.name}" is not a known OpenAI image-generation model. ` +
          "Use a `gpt-image-*` or `dall-e-*` model with openai.image({ name }).",
      );
    }

    this.client = client;
    this.name = config.name;
    this.provider = provider;
    this.pricing = config.pricing;
  }

  public async generate(
    prompt: string,
    options?: ImageGenerationOptions,
  ): Promise<ImageGenerationResponse> {
    const isGptImage = this.name.startsWith("gpt-image");
    // gpt-image always returns base64 bytes (no `response_format` knob);
    // DALL·E defaults to self-contained base64 here (URLs expire in ~60
    // min), but the caller can ask for a URL via `options.responseFormat`.
    const responseFormat =
      (options?.responseFormat as "url" | "b64_json" | undefined) ??
      (isGptImage ? undefined : "b64_json");

    const body: OpenAI.Images.ImageGenerateParamsNonStreaming = {
      model: this.name,
      prompt,
    };

    if (options?.count !== undefined) body.n = options.count;
    if (options?.size !== undefined) body.size = options.size;
    if (options?.quality !== undefined) {
      body.quality = options.quality as OpenAI.Images.ImageGenerateParamsBase["quality"];
    }
    if (!isGptImage && responseFormat) body.response_format = responseFormat;
    if (isGptImage && options?.format !== undefined) {
      body.output_format = options.format as OpenAI.Images.ImageGenerateParamsBase["output_format"];
    }
    if (options?.background !== undefined) {
      body.background = options.background as OpenAI.Images.ImageGenerateParamsBase["background"];
    }

    this.logger.debug(LOG_MODULE, "image.request", "images.generate", {
      model: this.name,
      count: options?.count ?? 1,
    });

    let response: OpenAI.Images.ImagesResponse;

    try {
      response = await this.client.images.generate(
        body,
        options?.signal ? { signal: options.signal } : undefined,
      );
    } catch (thrown) {
      const wrapped = wrapOpenAIError(thrown);

      this.logger.error(LOG_MODULE, "image.error", wrapped.message, {
        code: wrapped.code,
        context: wrapped.context,
      });

      throw wrapped;
    }

    const images = (response.data ?? []).map((image) =>
      this.toGeneratedImage(image, options?.format),
    );

    const usage = response.usage
      ? {
          input: response.usage.input_tokens,
          output: response.usage.output_tokens,
          total: response.usage.total_tokens,
        }
      : { input: 0, output: 0, total: 0 };

    this.logger.debug(LOG_MODULE, "image.response", "images.generate succeeded", {
      images: images.length,
      usage,
    });

    return { images, usage };
  }

  /**
   * Normalize one OpenAI `Image` into the neutral discriminated shape.
   * Base64 wins when present (gpt-image, and DALL·E in b64 mode);
   * otherwise a hosted URL. A response carrying neither is a provider
   * contract violation — surface it as a typed `ProviderError` rather
   * than emitting a half-formed part.
   */
  private toGeneratedImage(image: OpenAI.Images.Image, format: string | undefined): GeneratedImage {
    if (image.b64_json) {
      return {
        type: "base64",
        base64: image.b64_json,
        mediaType: mediaTypeFor(format),
        ...(image.revised_prompt ? { revisedPrompt: image.revised_prompt } : {}),
      };
    }

    if (image.url) {
      return {
        type: "url",
        url: image.url,
        ...(image.revised_prompt ? { revisedPrompt: image.revised_prompt } : {}),
      };
    }

    throw new ProviderError("OpenAI image response contained neither base64 bytes nor a URL.");
  }
}
