import {
  InvalidRequestError,
  safeJsonParse,
  type Message,
  type ModelCallOptions,
  type ModelCapabilities,
  type ModelContract,
  type ModelPricing,
  type ModelResponse,
  type ModelStreamChunk,
  type ModelToolCallRequest,
  type Usage,
} from "@warlock.js/ai";
import { log, type Logger } from "@warlock.js/logger";
import type OpenAI from "openai";
import type { OpenAIModelConfig } from "./config.type";
import { inferReasoningCapability } from "./known-reasoning-models";
import { inferVisionCapability } from "./known-vision-models";
import {
  buildPromptCacheParams,
  buildUsage,
  inferStructuredOutput,
  mapResponsesFinishReason,
  planStructuredOutput,
  toOpenAIReasoningEffort,
  toResponseFailure,
  toResponsesInput,
  toResponsesToolChoice,
  toResponsesTools,
  wrapOpenAIError,
} from "./utils";

const LOG_MODULE = "ai.openai";

/**
 * OpenAI Responses API implementation of `ModelContract` (`api: "responses"`).
 *
 * **Role.** A sibling of `OpenAIModel` that speaks `POST /v1/responses`
 * instead of Chat Completions. Its point is letting a tool-using agent run a
 * reasoning model with reasoning ON: the Chat path has to force
 * `reasoning_effort: "none"` whenever tools are attached, Responses does not.
 *
 * **Why a separate class.** Only `OpenAISDK.model()` constructs it, and only
 * for the direct `"openai"` provider. The wrapper packages (deepseek, groq,
 * xai, mistral) build `OpenAIModel` and cannot reach this class, so an
 * endpoint that does not implement `/v1/responses` is never sent there. The
 * constructor re-checks the provider label as a second lock.
 *
 * **Stateless.** Every request sends `store: false`; nothing is kept on the
 * server and no `previous_response_id` is used. Full history is re-sent each
 * call, exactly like the Chat path.
 *
 * **Scope of this class.** `complete()` is implemented. `stream()` throws a
 * clear "not implemented yet" error until the streaming translator lands.
 * Reasoning-item replay across tool turns is not implemented yet either:
 * without it the model cannot reuse its own reasoning on the next tool turn
 * (quality, not correctness).
 */
export class OpenAIResponsesModel implements ModelContract {
  public readonly name: string;
  public readonly provider: string;
  public readonly capabilities: ModelCapabilities;
  public readonly pricing?: ModelPricing;

  private readonly client: OpenAI;
  private readonly config: OpenAIModelConfig;
  private readonly logger: Logger = log;

  public constructor(client: OpenAI, config: OpenAIModelConfig, provider: string = "openai") {
    if (provider !== "openai") {
      throw new InvalidRequestError(
        `api: "responses" is supported only on the direct "openai" provider; got provider "${provider}".`,
        { context: { provider, model: config.name, api: "responses" } },
      );
    }

    this.client = client;
    this.config = config;
    this.name = config.name;
    this.provider = provider;
    this.pricing = config.pricing;
    this.capabilities = {
      structuredOutput: config.structuredOutput ?? inferStructuredOutput(config.responseFormat),
      vision: config.vision ?? inferVisionCapability(config.name),
      reasoning: config.reasoning ?? inferReasoningCapability(config.name),
      // Read-side cache accounting (`cachedTokens`, `cacheWriteTokens`) works
      // and `prompt_cache_key` is forwarded. Explicit write breakpoints are not
      // mapped onto Responses yet, so a `cacheControl.breakpoints` hint is
      // ignored rather than rejected.
      promptCaching: true,
      // PDF and audio input are off under Responses in this adapter: the
      // `input_file` / `input_audio` field-level mapping is unverified. The
      // `pdf` / `audio` config flags are deliberately NOT honoured here.
      pdf: false,
      audio: false,
    };
  }

  /**
   * Single-shot completion against `client.responses.create`. Sends the full
   * history as Responses `input` items, walks `response.output`, and reshapes
   * it into a vendor-neutral `ModelResponse`. Per-call `options` override the
   * instance defaults for this call only.
   */
  public async complete(messages: Message[], options?: ModelCallOptions): Promise<ModelResponse> {
    this.logger.debug(LOG_MODULE, "request", "Starting call to responses.create", {
      model: this.name,
      messageCount: messages.length,
      streaming: false,
      toolCount: options?.tools?.length ?? 0,
    });

    let response: OpenAI.Responses.Response;

    try {
      response = await this.client.responses.create(
        this.buildRequest(messages, options),
        options?.signal ? { signal: options.signal } : undefined,
      );
    } catch (thrown) {
      throw this.logAndWrap(thrown);
    }

    if (response.status === "failed") {
      throw this.logAndWrap(toResponseFailure(response, this.name));
    }

    const output = Array.isArray(response.output) ? response.output : [];
    const { content, toolCalls } = this.readOutput(output);
    const finishReason = mapResponsesFinishReason({
      status: response.status,
      incomplete_details: response.incomplete_details,
      hasToolCalls: toolCalls !== undefined,
    });
    const usage = this.extractUsage(response.usage);

    this.logger.debug(LOG_MODULE, "response", "call to responses.create succeeded", {
      finishReason,
      usage,
    });

    return { content, finishReason, usage, ...(toolCalls ? { toolCalls } : {}) };
  }

  /**
   * Streaming is not implemented for `api: "responses"` yet; it lands with
   * the streaming translator. Fails on the first iteration with a typed
   * error so a caller that streams by default gets a clear message.
   */
  public async *stream(
    _messages: Message[],
    _options?: ModelCallOptions,
  ): AsyncIterable<ModelStreamChunk> {
    throw new InvalidRequestError(
      'Streaming for api: "responses" is not implemented yet and lands next. Use complete(), or the default api: "chat" for streaming.',
      { context: { provider: this.provider, model: this.name, api: "responses" } },
    );
  }

  /**
   * Assemble the `responses.create` body.
   *
   * - System prompt -> `instructions`; history -> `input` items.
   * - `store: false` always (the API default is `true` when omitted).
   * - `max_output_tokens` caps reasoning and visible output together.
   * - `temperature` only for non-reasoning models (same rule as Chat).
   * - `tools` flat with `strict: false`; `tool_choice` only with tools.
   * - `reasoning.effort` only when the caller gave one, through the same
   *   clamp table as Chat. With no explicit effort the provider default
   *   applies EVEN with tools attached - that is the point of this API.
   */
  private buildRequest(
    messages: Message[],
    options: ModelCallOptions | undefined,
  ): OpenAI.Responses.ResponseCreateParamsNonStreaming {
    const { instructions, input } = toResponsesInput(messages);
    const tools = toResponsesTools(options?.tools);
    const toolChoice = tools ? toResponsesToolChoice(options?.toolChoice) : undefined;
    const maxTokens = options?.maxTokens ?? this.config.maxTokens;
    const temperature = options?.temperature ?? this.config.temperature;

    return {
      model: this.name,
      ...(instructions !== undefined ? { instructions } : {}),
      input,
      store: false,
      ...(maxTokens !== undefined ? { max_output_tokens: maxTokens } : {}),
      ...(!this.capabilities.reasoning && temperature !== undefined ? { temperature } : {}),
      ...(tools ? { tools } : {}),
      ...(toolChoice !== undefined ? { tool_choice: toolChoice } : {}),
      ...this.buildTextFormat(options?.responseSchema),
      ...this.buildReasoningParams(options?.reasoning),
      ...buildPromptCacheParams(this.provider, this.config),
    };
  }

  /**
   * Structured output on Responses lives in `text.format`, and `name` /
   * `schema` / `strict` sit directly on the format object (Chat nests them
   * under `json_schema`). The strict-or-downgrade decision is the shared
   * `planStructuredOutput`.
   */
  private buildTextFormat(responseSchema: Record<string, unknown> | undefined): {
    text?: OpenAI.Responses.ResponseTextConfig;
  } {
    const plan = planStructuredOutput(responseSchema, this.config.responseFormat);

    if (plan.mode === "json_schema") {
      return {
        text: { format: { type: "json_schema", name: "response", schema: plan.schema, strict: true } },
      };
    }

    if (plan.mode === "json_object") {
      return { text: { format: { type: "json_object" } } };
    }

    return {};
  }

  /** `reasoning.effort` for reasoning-capable models, only when the caller set one. */
  private buildReasoningParams(reasoning: ModelCallOptions["reasoning"]): {
    reasoning?: { effort: ReturnType<typeof toOpenAIReasoningEffort> };
  } {
    if (!this.capabilities.reasoning || !reasoning?.effort) {
      return {};
    }

    return { reasoning: { effort: toOpenAIReasoningEffort(reasoning.effort) } };
  }

  /**
   * Walk `response.output`: `message` items contribute their `output_text`
   * text and any `refusal` text (surfaced as ordinary content, decision 5);
   * each completed `function_call` item becomes one tool call with
   * `id = call_id`. `reasoning` items carry no visible content and are
   * ignored here. A `function_call` the API marked `incomplete` (truncated
   * arguments) is skipped rather than dispatched with empty input.
   */
  private readOutput(output: OpenAI.Responses.ResponseOutputItem[]): {
    content: string;
    toolCalls: ModelToolCallRequest[] | undefined;
  } {
    let content = "";
    const toolCalls: ModelToolCallRequest[] = [];

    for (const item of output) {
      if (item.type === "message") {
        for (const part of item.content) {
          content += part.type === "refusal" ? part.refusal : part.text;
        }

        continue;
      }

      if (item.type === "function_call" && item.status !== "incomplete") {
        toolCalls.push({
          id: item.call_id,
          name: item.name,
          input: safeJsonParse<Record<string, unknown>>(item.arguments, {}),
        });
      }
    }

    return { content, toolCalls: toolCalls.length > 0 ? toolCalls : undefined };
  }

  /**
   * Normalize `usage` (absent on some partials) into the neutral `Usage`.
   * `input_tokens_details.cached_tokens` -> `cachedTokens`,
   * `cache_write_tokens` -> `cacheWriteTokens`,
   * `output_tokens_details.reasoning_tokens` -> `reasoningTokens`. The
   * detail blocks are typed non-optional but read defensively.
   */
  private extractUsage(raw: OpenAI.Responses.ResponseUsage | undefined): Usage {
    if (!raw) {
      return { input: 0, output: 0, total: 0 };
    }

    return buildUsage({
      input: raw.input_tokens,
      output: raw.output_tokens,
      total: raw.total_tokens,
      cachedTokens: raw.input_tokens_details?.cached_tokens,
      cacheWriteTokens: raw.input_tokens_details?.cache_write_tokens,
      reasoningTokens: raw.output_tokens_details?.reasoning_tokens,
    });
  }

  /** Wrap (idempotently), log and return a failure for the caller to throw. */
  private logAndWrap(thrown: unknown): ReturnType<typeof wrapOpenAIError> {
    const wrapped = wrapOpenAIError(thrown);

    this.logger.error(LOG_MODULE, "error", wrapped.message, {
      code: wrapped.code,
      context: wrapped.context,
    });

    return wrapped;
  }
}
