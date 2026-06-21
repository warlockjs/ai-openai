import {
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
import type { OpenAIModelConfig, OpenAIResponseFormat } from "./config.type";
import { inferReasoningCapability } from "./known-reasoning-models";
import { inferVisionCapability } from "./known-vision-models";
import { mapFinishReason, toOpenAIMessages, toOpenAITools, wrapOpenAIError } from "./utils";

const LOG_MODULE = "ai.openai";

/**
 * Map an explicit `responseFormat` override to the default
 * `structuredOutput` capability. Loose wire modes (`"json_object"`,
 * `"text"`) don't enforce shape, so the agent needs to see the soft
 * schema hint in the system prompt — that only happens when the
 * capability is `false`. Default (no override) stays `true` to
 * preserve the prior assumption that OpenAI models support strict
 * structured output.
 */
function inferStructuredOutput(responseFormat: OpenAIResponseFormat | undefined): boolean {
  if (responseFormat === "json_object" || responseFormat === "text") {
    return false;
  }

  return true;
}

/**
 * OpenAI-backed implementation of `ModelContract`.
 *
 * **Role.** The provider-facing bridge between the vendor-neutral
 * `@warlock.js/ai` agent runtime and the official `openai` SDK. Agents,
 * workflows, and supervisors never talk to OpenAI directly — they hold a
 * `ModelContract`, and this class is what makes that contract concrete for
 * any OpenAI-compatible endpoint (OpenAI, Azure OpenAI, OpenRouter, local
 * gateways that speak the Chat Completions protocol).
 *
 * **Responsibility.**
 * - Owns: a long-lived `OpenAI` client + frozen `ModelConfig` (name,
 *   temperature, maxTokens) used as defaults for every call.
 * - Owns: translating vendor-neutral `Message[]` and
 *   `ToolContract[]` into OpenAI wire shapes on the way out, and
 *   translating OpenAI's response (content, finish reason, tool calls,
 *   usage) back into the neutral shapes on the way in.
 * - Does NOT own: dispatching tools, deciding whether to loop, tracking
 *   conversation history, or retrying on failure — those are agent
 *   concerns. The model is a stateless (per-call) protocol adapter.
 *
 * Because it holds a live client and shared defaults, it is modeled as a
 * class (see §4.2 of code-style.md — "long-lived state across calls").
 *
 * @example
 * import OpenAI from "openai";
 * const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
 * const model = new OpenAIModel(client, { name: "gpt-4o", temperature: 0.3 });
 *
 * const myAgent = agent({
 *   model,
 *   systemPrompt: "You are a helpful assistant.",
 *   tools: [searchTool],
 * });
 *
 * const result = await myAgent.execute("Summarize today's news.");
 */
export class OpenAIModel implements ModelContract {
  public readonly name: string;
  public readonly provider: string;
  public readonly capabilities: ModelCapabilities;
  public readonly pricing?: ModelPricing;

  private readonly client: OpenAI;
  private readonly config: OpenAIModelConfig;
  private readonly logger: Logger = log;

  public constructor(client: OpenAI, config: OpenAIModelConfig, provider: string = "openai") {
    this.client = client;
    this.config = config;
    this.name = config.name;
    this.provider = provider;
    this.pricing = config.pricing;
    this.capabilities = {
      structuredOutput: config.structuredOutput ?? inferStructuredOutput(config.responseFormat),
      vision: config.vision ?? inferVisionCapability(config.name),
      // o-series + gpt-5 models surface a reasoning channel and accept
      // the `reasoning_effort` param. Explicit config wins over the
      // name-prefix inference.
      reasoning: config.reasoning ?? inferReasoningCapability(config.name),
      // OpenAI prompt caching is automatic on the Chat Completions API
      // (no caller-supplied breakpoints — the platform caches long
      // prompt prefixes server-side and reports the hit count via
      // `prompt_tokens_details.cached_tokens`). We therefore advertise
      // the read-side accounting capability as always available while
      // treating `ModelCallOptions.cacheControl` write breakpoints as a
      // no-op (see `buildReasoningParams` siblings — there is no cache
      // param to emit).
      promptCaching: true,
    };
  }

  /**
   * Single-shot completion. Sends the full message list to the Chat
   * Completions endpoint, waits for the terminal response, and reshapes it
   * into a vendor-neutral `ModelResponse`. Per-call `options` override the
   * instance's `ModelConfig` defaults for this call only.
   */
  public async complete(messages: Message[], options?: ModelCallOptions): Promise<ModelResponse> {
    // Per-call request/response logs are hot-path in production agents
    // — keep them at `debug` so `info` stays reserved for lifecycle
    // events (agent starting/completed, etc.). Operators who need to
    // audit every LLM call can raise log-level at runtime.
    this.logger.debug(LOG_MODULE, "request", "Starting call to chat.completions", {
      model: this.name,
      messageCount: messages.length,
      streaming: false,
      toolCount: options?.tools?.length ?? 0,
    });

    let response: OpenAI.Chat.Completions.ChatCompletion;

    try {
      response = await this.client.chat.completions.create(
        {
          model: this.name,
          messages: toOpenAIMessages(messages),
          temperature: options?.temperature ?? this.config.temperature,
          max_tokens: options?.maxTokens ?? this.config.maxTokens,
          tools: toOpenAITools(options?.tools),
          ...this.buildResponseFormat(options?.responseSchema),
          ...this.buildReasoningParams(options?.reasoning),
        },
        options?.signal ? { signal: options.signal } : undefined,
      );
    } catch (thrown) {
      const wrapped = wrapOpenAIError(thrown);

      this.logger.error(LOG_MODULE, "error", wrapped.message, {
        code: wrapped.code,
        context: wrapped.context,
      });

      throw wrapped;
    }

    const choice = response.choices[0];
    const finishReason = mapFinishReason(choice.finish_reason);
    const usage = this.extractUsage(response.usage);

    this.logger.debug(LOG_MODULE, "response", "call to chat.completions succeeded", {
      finishReason,
      usage,
    });

    return {
      content: choice.message.content ?? "",
      finishReason,
      usage,
      toolCalls: this.extractToolCalls(choice.message.tool_calls),
    };
  }

  /**
   * Incremental streaming completion. Yields neutral `ModelStreamChunk`s —
   * `delta` for text tokens, `tool-call` when the model requests a tool,
   * and a terminal `done` carrying the final finish reason + usage totals.
   * Callers consume it with `for await`.
   */
  public async *stream(
    messages: Message[],
    options?: ModelCallOptions,
  ): AsyncIterable<ModelStreamChunk> {
    this.logger.debug(LOG_MODULE, "request", "Starting streaming call to chat.completions", {
      model: this.name,
      messageCount: messages.length,
      streaming: true,
      toolCount: options?.tools?.length ?? 0,
    });

    let stream: Awaited<ReturnType<typeof this.client.chat.completions.create>>;

    try {
      stream = await this.client.chat.completions.create(
        {
          model: this.name,
          messages: toOpenAIMessages(messages),
          temperature: options?.temperature ?? this.config.temperature,
          max_tokens: options?.maxTokens ?? this.config.maxTokens,
          tools: toOpenAITools(options?.tools),
          stream: true,
          stream_options: { include_usage: true },
          ...this.buildResponseFormat(options?.responseSchema),
          ...this.buildReasoningParams(options?.reasoning),
        },
        options?.signal ? { signal: options.signal } : undefined,
      );
    } catch (thrown) {
      const wrapped = wrapOpenAIError(thrown);

      this.logger.error(LOG_MODULE, "error", wrapped.message, {
        code: wrapped.code,
        context: wrapped.context,
      });

      throw wrapped;
    }

    let rawFinishReason: string = "stop";
    const usage: Usage = { input: 0, output: 0, total: 0 };
    const toolCallAccum = new Map<number, { id: string; name: string; arguments: string }>();

    try {
      for await (const chunk of stream as AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>) {
        const delta = chunk.choices[0]?.delta;
        const finish = chunk.choices[0]?.finish_reason;

        if (delta?.content) {
          yield { type: "delta", content: delta.content };
        }

        if (delta?.tool_calls) {
          for (const toolCall of delta.tool_calls) {
            const idx = toolCall.index ?? 0;
            if (!toolCallAccum.has(idx)) {
              toolCallAccum.set(idx, { id: "", name: "", arguments: "" });
            }
            const acc = toolCallAccum.get(idx)!;
            if (toolCall.id) acc.id = toolCall.id;
            if (toolCall.function?.name) acc.name = toolCall.function.name;
            if (toolCall.function?.arguments) acc.arguments += toolCall.function.arguments;
          }
        }

        if (finish) {
          rawFinishReason = finish;
        }

        if (chunk.usage) {
          usage.input = chunk.usage.prompt_tokens ?? 0;
          usage.output = chunk.usage.completion_tokens ?? 0;
          usage.total = chunk.usage.total_tokens ?? 0;
          const cached = chunk.usage.prompt_tokens_details?.cached_tokens;
          if (cached !== undefined && cached > 0) {
            usage.cachedTokens = cached;
          }
          const reasoning = chunk.usage.completion_tokens_details?.reasoning_tokens;
          if (reasoning !== undefined && reasoning > 0) {
            usage.reasoningTokens = reasoning;
          }
        }
      }

      for (const acc of toolCallAccum.values()) {
        // Skip accumulators that never received a function name — those
        // are partial fragments the model started but never identified
        // (e.g. arguments-only deltas with no originating `id`/`name`).
        // Yielding them produces nameless tool-calls the agent runtime
        // can't dispatch and would mis-attribute as a registered tool.
        if (!acc.name) continue;

        yield {
          type: "tool-call",
          id: acc.id,
          name: acc.name,
          input: safeJsonParse<Record<string, unknown>>(acc.arguments, {}),
        };
      }
    } catch (thrown) {
      const wrapped = wrapOpenAIError(thrown);

      this.logger.error(LOG_MODULE, "error", wrapped.message, {
        code: wrapped.code,
        context: wrapped.context,
      });

      throw wrapped;
    }

    const finishReason = mapFinishReason(rawFinishReason);

    this.logger.debug(LOG_MODULE, "response", "Streaming call to chat.completions succeeded", {
      finishReason,
      usage,
    });

    yield { type: "done", finishReason, usage };
  }

  /**
   * Translate the neutral `responseSchema` option into OpenAI's
   * `response_format` parameter.
   *
   * When `config.responseFormat` is set, it wins: `"text"` emits no
   * `response_format` at all, `"json_object"` always picks the loose
   * mode, and `"json_schema"` picks strict mode (with the same
   * `isStrictCompatible` safety check — a malformed schema still
   * degrades to `json_object` rather than 400). The override exists
   * because some targets (older OpenAI models, OpenRouter routes,
   * Ollama OpenAI-compat) reject strict `json_schema` outright.
   *
   * When the override is omitted, uses strict `json_schema` mode
   * (token-level enforcement) only when the schema is a proper
   * root-object JSON Schema (`{ type: "object", properties: ... }`).
   * For anything else — malformed extractor output, non-object
   * schemas, or future shapes we haven't tested — falls back to loose
   * `json_object` mode, which guarantees *some* valid JSON without
   * enforcing shape. The agent's soft instruction already embeds the
   * schema text in the system prompt when the model declares no
   * native structured-output capability, so shape validation still
   * runs client-side via the Standard Schema `validate()` call.
   *
   * Returns an empty spread when no schema was supplied, so the caller
   * can unconditionally `...buildResponseFormat(...)` into the request.
   */
  private buildResponseFormat(responseSchema: Record<string, unknown> | undefined): {
    response_format?: OpenAI.Chat.Completions.ChatCompletionCreateParams["response_format"];
  } {
    if (!responseSchema) {
      return {};
    }

    const override = this.config.responseFormat;

    if (override === "text") {
      return {};
    }

    if (override === "json_object") {
      return { response_format: { type: "json_object" } };
    }

    // Either auto-select (no override) or explicit `"json_schema"`.
    // The strict-compat check still applies in the explicit case —
    // a malformed / non-object schema would 400 before sampling, so
    // we degrade to `json_object` rather than crash.
    if (this.isStrictCompatible(responseSchema)) {
      return {
        response_format: {
          type: "json_schema",
          json_schema: {
            name: "response",
            schema: responseSchema,
            strict: true,
          },
        },
      };
    }

    return { response_format: { type: "json_object" } };
  }

  /**
   * OpenAI strict `json_schema` mode requires the root to be a JSON
   * Schema object type (`{ type: "object", properties: ... }`). Anything
   * else (top-level arrays, primitives, unknown shapes) is rejected with
   * a 400 before a token is sampled. We check structurally here so the
   * first call doesn't crash on a malformed extraction — loose
   * `json_object` mode is a safe degradation.
   */
  private isStrictCompatible(schema: Record<string, unknown>): boolean {
    return (
      schema.type === "object" &&
      typeof schema.properties === "object" &&
      schema.properties !== null &&
      this.isStrictSafeNode(schema)
    );
  }

  /**
   * Recursively check the one strict-mode rule schemas most often trip on:
   * every object must list ALL of its `properties` in `required` (OpenAI
   * strict has no notion of optional — optional fields must be expressed
   * as nullable, e.g. `type: ["string", "null"]`, and still appear in
   * `required`). A schema that violates this anywhere in the tree is NOT
   * sent in strict `json_schema` mode — it degrades to loose
   * `json_object` so a hand-built or optional-bearing schema can't 400
   * the call ("'required' ... must include every key in properties").
   * Client-side `validate()` still enforces the full shape.
   */
  private isStrictSafeNode(node: unknown): boolean {
    if (!node || typeof node !== "object") {
      return true;
    }

    const record = node as Record<string, unknown>;

    if (record.type === "object" && record.properties && typeof record.properties === "object") {
      const properties = record.properties as Record<string, unknown>;
      const keys = Object.keys(properties);
      const required = Array.isArray(record.required) ? (record.required as unknown[]) : [];

      if (keys.some((key) => !required.includes(key))) {
        return false;
      }

      for (const key of keys) {
        if (!this.isStrictSafeNode(properties[key])) {
          return false;
        }
      }
    }

    if (record.items !== undefined && !this.isStrictSafeNode(record.items)) {
      return false;
    }

    for (const branch of ["anyOf", "allOf", "oneOf"] as const) {
      const value = record[branch];
      if (Array.isArray(value) && value.some((sub) => !this.isStrictSafeNode(sub))) {
        return false;
      }
    }

    return true;
  }

  /**
   * Normalize OpenAI's `usage` block (which may be absent on some responses
   * or partials) into the neutral `Usage` shape. Missing usage collapses to
   * zeros rather than propagating `undefined`, so downstream aggregation
   * math stays safe.
   *
   * `cachedTokens` mirrors `prompt_tokens_details.cached_tokens` (the
   * subset of the prompt served from OpenAI's automatic prompt cache);
   * `reasoningTokens` mirrors `completion_tokens_details.reasoning_tokens`
   * (the hidden reasoning channel on o-series / gpt-5 models, already
   * counted within `output`). Both are emitted only when the provider
   * reports a positive value, so non-reasoning / uncached calls keep the
   * lean `{ input, output, total }` shape.
   */
  private extractUsage(raw: OpenAI.Completions.CompletionUsage | undefined): Usage {
    if (!raw) {
      return { input: 0, output: 0, total: 0 };
    }

    const cachedTokens = raw.prompt_tokens_details?.cached_tokens;
    const reasoningTokens = raw.completion_tokens_details?.reasoning_tokens;

    return {
      input: raw.prompt_tokens,
      output: raw.completion_tokens,
      total: raw.total_tokens,
      ...(cachedTokens !== undefined && cachedTokens > 0 ? { cachedTokens } : {}),
      ...(reasoningTokens !== undefined && reasoningTokens > 0 ? { reasoningTokens } : {}),
    };
  }

  /**
   * Translate the neutral `ModelCallOptions.reasoning` hint into OpenAI's
   * `reasoning_effort` request param. Only `effort` maps — OpenAI's Chat
   * Completions API exposes a discrete effort knob, not a token budget,
   * so `reasoning.maxTokens` (the Anthropic extended-thinking cap) has no
   * wire equivalent here and is silently ignored.
   *
   * No-ops in two cases so the adapter never forwards an unsupported
   * param: (1) the model is not reasoning-capable
   * (`capabilities.reasoning` is false — e.g. `gpt-4o`), or (2) the caller
   * supplied no `effort`. The neutral `ReasoningEffort`
   * (`"low" | "medium" | "high"`) is a strict subset of OpenAI's accepted
   * values, so it forwards verbatim.
   *
   * Returns an empty spread when nothing applies, so the caller can
   * unconditionally `...buildReasoningParams(...)` into the request.
   */
  private buildReasoningParams(reasoning: ModelCallOptions["reasoning"]): {
    reasoning_effort?: OpenAI.Chat.Completions.ChatCompletionCreateParams["reasoning_effort"];
  } {
    if (!this.capabilities.reasoning || !reasoning?.effort) {
      return {};
    }

    return { reasoning_effort: reasoning.effort };
  }

  /**
   * Reshape OpenAI's `tool_calls` array into the neutral
   * `ModelToolCallRequest[]`. The raw `arguments` field is a JSON string
   * per OpenAI's protocol — we parse it defensively via `safeJsonParse` so
   * malformed or empty arguments yield an empty object instead of crashing
   * the trip. Returns `undefined` when no tools were requested so callers
   * can branch on presence.
   */
  private extractToolCalls(
    rawToolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] | undefined,
  ): ModelToolCallRequest[] | undefined {
    if (!rawToolCalls || rawToolCalls.length === 0) {
      return undefined;
    }

    return rawToolCalls.map((toolCall) => ({
      id: toolCall.id,
      name: (toolCall as any).function.name,
      input: safeJsonParse<Record<string, unknown>>((toolCall as any).function.arguments, {}),
    }));
  }
}
