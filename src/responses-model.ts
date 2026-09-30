import {
  InvalidRequestError,
  ProviderError,
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
  attachReasoningReplay,
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
 * **Reasoning replay.** Every request asks for
 * `include: ["reasoning.encrypted_content"]`. The `reasoning` items of a
 * tool-calling turn are stored on the FIRST tool call as
 * `providerMetadata.openaiResponses = { model, reasoningItems }` and sent back
 * before that turn's `function_call` items on the next request (only when the
 * stored model equals the current one). Text-only turns are not replayed.
 *
 * **Streaming.** `stream()` uses `responses.create({ stream: true })` and yields
 * the same chunk sequence as the Chat path: text `delta`s, then the
 * `tool-call`s, then one `done`.
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
   * Incremental streaming completion via `responses.create({ stream: true })`
   * (the SDK's `Stream<ResponseStreamEvent>`, `responses.d.ts:55`). Yields the
   * same neutral `ModelStreamChunk` sequence as the Chat path:
   *
   * - `response.output_text.delta` / `response.refusal.delta` -> `delta`.
   * - `response.output_item.done` with a completed `function_call` item ->
   *   buffered; `reasoning` items are captured for replay. The `done` copy is
   *   used because the `added` copy may be incomplete (`responses.d.ts:5711`).
   * - After the loop: the buffered `tool-call`s (the first carries the
   *   replay metadata), then a single `done` with the finish reason and usage
   *   from `response.completed` / `response.incomplete`.
   * - `response.failed` and `error` events throw a typed `AIError`; a stream
   *   that ends with no terminal event throws `ProviderError`.
   * - Abort goes through the request `signal`; the SDK rejects the read and
   *   the error is wrapped like any other.
   */
  public async *stream(
    messages: Message[],
    options?: ModelCallOptions,
  ): AsyncIterable<ModelStreamChunk> {
    this.logger.debug(LOG_MODULE, "request", "Starting streaming call to responses.create", {
      model: this.name,
      messageCount: messages.length,
      streaming: true,
      toolCount: options?.tools?.length ?? 0,
    });

    let events: AsyncIterable<OpenAI.Responses.ResponseStreamEvent>;

    try {
      events = await this.client.responses.create(
        { ...this.buildRequest(messages, options), stream: true },
        options?.signal ? { signal: options.signal } : undefined,
      );
    } catch (thrown) {
      throw this.logAndWrap(thrown);
    }

    const toolCalls: ModelToolCallRequest[] = [];
    const reasoningItems: OpenAI.Responses.ResponseReasoningItem[] = [];
    let terminal: OpenAI.Responses.Response | undefined;

    try {
      for await (const event of events) {
        if (event.type === "response.output_text.delta" || event.type === "response.refusal.delta") {
          if (event.delta) {
            yield { type: "delta", content: event.delta };
          }

          continue;
        }

        if (event.type === "response.output_item.done") {
          const { item } = event;

          if (item.type === "reasoning") {
            reasoningItems.push(item);
          } else if (item.type === "function_call" && item.status !== "incomplete") {
            toolCalls.push(this.toToolCall(item));
          }

          continue;
        }

        if (event.type === "response.completed" || event.type === "response.incomplete") {
          terminal = event.response;
          continue;
        }

        if (event.type === "response.failed") {
          throw toResponseFailure(event.response, this.name);
        }

        if (event.type === "error") {
          throw wrapOpenAIError({ code: event.code, message: event.message, param: event.param });
        }
      }

      if (terminal === undefined) {
        throw new ProviderError("OpenAI Responses stream ended before response.completed.", {
          context: { provider: this.provider, model: this.name },
        });
      }
    } catch (thrown) {
      throw this.logAndWrap(thrown);
    }

    for (const call of attachReasoningReplay(toolCalls, this.name, reasoningItems)) {
      yield {
        type: "tool-call",
        id: call.id,
        name: call.name,
        input: call.input,
        ...(call.providerMetadata ? { providerMetadata: call.providerMetadata } : {}),
      };
    }

    const finishReason = mapResponsesFinishReason({
      status: terminal.status,
      incomplete_details: terminal.incomplete_details,
      hasToolCalls: toolCalls.length > 0,
    });
    const usage = this.extractUsage(terminal.usage);

    this.logger.debug(LOG_MODULE, "response", "Streaming call to responses.create succeeded", {
      finishReason,
      usage,
    });

    yield { type: "done", finishReason, usage };
  }

  /**
   * Assemble the `responses.create` body.
   *
   * - System prompt -> `instructions`; history -> `input` items.
   * - `store: false` always (the API default is `true` when omitted), plus
   *   `include: ["reasoning.encrypted_content"]` for stateless reasoning replay.
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
    const { instructions, input } = toResponsesInput(messages, this.name);
    const tools = toResponsesTools(options?.tools);
    const toolChoice = tools ? toResponsesToolChoice(options?.toolChoice) : undefined;
    const maxTokens = options?.maxTokens ?? this.config.maxTokens;
    const temperature = options?.temperature ?? this.config.temperature;

    return {
      model: this.name,
      ...(instructions !== undefined ? { instructions } : {}),
      input,
      store: false,
      // Stateless replay: ask for the encrypted reasoning blobs so they can be
      // sent back on the next tool turn (decision 3).
      include: ["reasoning.encrypted_content"],
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
   * not content. A `function_call` the API marked `incomplete` (truncated
   * arguments) is skipped rather than dispatched with empty input.
   * `reasoning` items are captured and attached to the first tool call for
   * replay (see `attachReasoningReplay`).
   */
  private readOutput(output: OpenAI.Responses.ResponseOutputItem[]): {
    content: string;
    toolCalls: ModelToolCallRequest[] | undefined;
  } {
    let content = "";
    const toolCalls: ModelToolCallRequest[] = [];
    const reasoningItems: OpenAI.Responses.ResponseReasoningItem[] = [];

    for (const item of output) {
      if (item.type === "reasoning") {
        reasoningItems.push(item);
        continue;
      }

      if (item.type === "message") {
        for (const part of item.content) {
          content += part.type === "refusal" ? part.refusal : part.text;
        }

        continue;
      }

      if (item.type === "function_call" && item.status !== "incomplete") {
        toolCalls.push(this.toToolCall(item));
      }
    }

    return {
      content,
      toolCalls:
        toolCalls.length > 0
          ? attachReasoningReplay(toolCalls, this.name, reasoningItems)
          : undefined,
    };
  }

  /** One completed `function_call` item as a neutral tool call (`call_id` -> `id`). */
  private toToolCall(item: OpenAI.Responses.ResponseFunctionToolCall): ModelToolCallRequest {
    return {
      id: item.call_id,
      name: item.name,
      input: safeJsonParse<Record<string, unknown>>(item.arguments, {}),
    };
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
