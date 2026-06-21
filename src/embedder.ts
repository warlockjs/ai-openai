import {
  type EmbeddingBatchResult,
  type EmbeddingResult,
  type EmbeddingUsage,
  type EmbedderContract,
} from "@warlock.js/ai";
import { log, type Logger } from "@warlock.js/logger";
import type OpenAI from "openai";
import type { OpenAIEmbedderConfig } from "./config.type";
import { wrapOpenAIError } from "./utils";

const LOG_MODULE = "ai.openai";

type EmbeddingsResponse = Awaited<
  ReturnType<OpenAI["embeddings"]["create"]>
>;

/**
 * OpenAI-backed implementation of `EmbedderContract`.
 *
 * **Role.** Converts text (or a batch of texts) into floating-point
 * vectors via OpenAI's Embeddings API. Standalone primitive — no
 * relationship to chat completions, tools, or the agent loop.
 *
 * **Dimensions.** When no `dimensions` override is supplied in config,
 * `this.dimensions` starts at `0` and is populated from the first
 * response's vector length, then cached for all subsequent calls —
 * even if a later response were to return a different length, the
 * first value wins so batches stay dimensionally consistent. Passing
 * `dimensions` in config both forwards the truncation hint to the API
 * (for models like `text-embedding-3-*`) and sets the initial value.
 *
 * **Error handling.** Raw OpenAI SDK errors are wrapped into the
 * typed `@warlock.js/ai` `AIError` hierarchy via `wrapOpenAIError` —
 * callers catch `AIError` subclasses (`ProviderRateLimitError`,
 * `ProviderAuthError`, etc.) instead of OpenAI's own classes.
 *
 * @example
 * const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });
 * const { vector, dimensions, usage } = await embedder.embed("Hello world");
 * const { vectors } = await embedder.embedMany(["doc 1", "doc 2"]);
 */
export class OpenAIEmbedder implements EmbedderContract {
  public readonly name: string;
  public readonly provider = "openai";
  public dimensions: number;

  private readonly client: OpenAI;

  /**
   * User-specified truncation hint, or `undefined` if omitted.
   * Forwarded to the API on every call so OpenAI can truncate the
   * embedding server-side for models that support it.
   */
  private readonly configuredDimensions: number | undefined;
  private readonly logger: Logger = log;

  public constructor(client: OpenAI, config: OpenAIEmbedderConfig) {
    this.client = client;
    this.name = config.name;
    this.configuredDimensions = config.dimensions;
    this.dimensions = config.dimensions ?? 0;
  }

  public async embed(input: string): Promise<EmbeddingResult> {
    const { response, usage } = await this.request(input);

    return {
      vector: response.data[0].embedding,
      dimensions: this.dimensions,
      usage,
    };
  }

  public async embedMany(inputs: string[]): Promise<EmbeddingBatchResult> {
    const { response, usage } = await this.request(inputs);

    return {
      vectors: response.data.map((d) => d.embedding),
      dimensions: this.dimensions,
      usage,
    };
  }

  /**
   * Shared transport for both `embed()` and `embedMany()` — issues the
   * `embeddings.create` call, wraps provider errors, caches dimensions
   * on the first successful response, and returns the raw response
   * plus a camelCase usage object for the caller to shape.
   */
  private async request(input: string | string[]): Promise<{
    response: EmbeddingsResponse;
    usage: EmbeddingUsage;
  }> {
    this.logger.debug(LOG_MODULE, "embedder.request", "embeddings.create", {
      model: this.name,
      batch: Array.isArray(input),
      count: Array.isArray(input) ? input.length : 1,
    });

    let response: EmbeddingsResponse;

    try {
      response = await this.client.embeddings.create({
        model: this.name,
        input,
        ...(this.configuredDimensions !== undefined
          ? { dimensions: this.configuredDimensions }
          : {}),
      });
    } catch (thrown) {
      const wrapped = wrapOpenAIError(thrown);

      this.logger.error(LOG_MODULE, "embedder.error", wrapped.message, {
        code: wrapped.code,
        context: wrapped.context,
      });

      throw wrapped;
    }

    this.logger.debug(LOG_MODULE, "embedder.response", "embeddings.create returned", {
      dimensions: response.data[0]?.embedding.length,
      usage: {
        promptTokens: response.usage.prompt_tokens,
        totalTokens: response.usage.total_tokens,
      },
    });

    // Cache dimensions on the first response. Once set, stays set —
    // we trust the first call to define the shape for this embedder.
    if (this.dimensions === 0) {
      this.dimensions = response.data[0].embedding.length;
    }

    const usage: EmbeddingUsage = {
      promptTokens: response.usage.prompt_tokens,
      totalTokens: response.usage.total_tokens,
    };

    return { response, usage };
  }
}
