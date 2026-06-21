import type OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { OpenAIModel } from "./model";

type CreateCall = {
  params: OpenAI.Chat.Completions.ChatCompletionCreateParams;
};

/**
 * Build a fake OpenAI client whose `chat.completions.create()` records the
 * params it was called with and returns a scripted response (for `complete()`)
 * or an async iterable of chunks (for `stream()`). Lets us assert the exact
 * wire shape the OpenAIModel sends without hitting the network.
 */
function makeFakeClient(options: {
  completion?: OpenAI.Chat.Completions.ChatCompletion;
  streamChunks?: OpenAI.Chat.Completions.ChatCompletionChunk[];
}) {
  const calls: CreateCall[] = [];

  const create = async (params: OpenAI.Chat.Completions.ChatCompletionCreateParams) => {
    calls.push({ params });

    if (params.stream) {
      return (async function* () {
        for (const chunk of options.streamChunks ?? []) {
          yield chunk;
        }
      })();
    }

    return options.completion;
  };

  const client = {
    chat: { completions: { create } },
  } as unknown as OpenAI;

  return { client, calls };
}

const baseCompletion: OpenAI.Chat.Completions.ChatCompletion = {
  id: "x",
  object: "chat.completion",
  created: 0,
  model: "gpt-4o-mini",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "hello", refusal: null },
      finish_reason: "stop",
      logprobs: null,
    },
  ],
  usage: { prompt_tokens: 5, completion_tokens: 3, total_tokens: 8 },
};

describe("OpenAIModel.complete()", () => {
  it("forwards model name, messages, temperature, maxTokens to the create call", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, {
      name: "gpt-4o-mini",
      temperature: 0.4,
      maxTokens: 256,
    });

    await model.complete([{ role: "user", content: "hi" }]);

    expect(calls).toHaveLength(1);
    expect(calls[0].params.model).toBe("gpt-4o-mini");
    expect(calls[0].params.temperature).toBe(0.4);
    expect(calls[0].params.max_tokens).toBe(256);
    expect(calls[0].params.messages).toEqual([{ role: "user", content: "hi" }]);
  });

  it("per-call options override instance defaults", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, {
      name: "gpt-4o-mini",
      temperature: 0.4,
      maxTokens: 256,
    });

    await model.complete([{ role: "user", content: "hi" }], {
      temperature: 0.9,
      maxTokens: 64,
    });

    expect(calls[0].params.temperature).toBe(0.9);
    expect(calls[0].params.max_tokens).toBe(64);
  });

  it("normalizes the response into ModelResponse shape", async () => {
    const { client } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.content).toBe("hello");
    expect(result.finishReason).toBe("stop");
    expect(result.usage).toEqual({ input: 5, output: 3, total: 8 });
    expect(result.toolCalls).toBeUndefined();
  });

  it("does not send a temperature/max_tokens key when neither config nor option supplies one", async () => {
    // `options?.x ?? this.config.x` is `undefined ?? undefined` → undefined,
    // so the key is present-but-undefined on the params object. Assert the
    // value is undefined rather than a stray default the adapter never sets.
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await model.complete([{ role: "user", content: "hi" }]);

    expect(calls[0].params.temperature).toBeUndefined();
    expect(calls[0].params.max_tokens).toBeUndefined();
  });

  it("forwards an AbortSignal to the create call's request options", async () => {
    const recorded: { signal?: AbortSignal } = {};
    const controller = new AbortController();
    const client = {
      chat: {
        completions: {
          create: async (
            _params: OpenAI.Chat.Completions.ChatCompletionCreateParams,
            opts?: { signal?: AbortSignal },
          ) => {
            recorded.signal = opts?.signal;
            return baseCompletion;
          },
        },
      },
    } as unknown as OpenAI;
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await model.complete([{ role: "user", content: "hi" }], { signal: controller.signal });

    expect(recorded.signal).toBe(controller.signal);
  });

  it("passes undefined request options when no signal is supplied", async () => {
    let secondArg: unknown = "unset";
    const client = {
      chat: {
        completions: {
          create: async (
            _params: OpenAI.Chat.Completions.ChatCompletionCreateParams,
            opts?: unknown,
          ) => {
            secondArg = opts;
            return baseCompletion;
          },
        },
      },
    } as unknown as OpenAI;
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await model.complete([{ role: "user", content: "hi" }]);

    expect(secondArg).toBeUndefined();
  });

  it("surfaces cachedTokens from prompt_tokens_details when positive", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        usage: {
          prompt_tokens: 100,
          completion_tokens: 20,
          total_tokens: 120,
          prompt_tokens_details: { cached_tokens: 40 },
        },
      },
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({ input: 100, output: 20, total: 120, cachedTokens: 40 });
  });

  it("omits cachedTokens when prompt_tokens_details reports zero cached tokens", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        usage: {
          prompt_tokens: 100,
          completion_tokens: 20,
          total_tokens: 120,
          prompt_tokens_details: { cached_tokens: 0 },
        },
      },
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).not.toHaveProperty("cachedTokens");
    expect(result.usage).toEqual({ input: 100, output: 20, total: 120 });
  });

  it("surfaces reasoningTokens from completion_tokens_details when positive", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        usage: {
          prompt_tokens: 50,
          completion_tokens: 200,
          total_tokens: 250,
          completion_tokens_details: { reasoning_tokens: 180 },
        },
      },
    });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({ input: 50, output: 200, total: 250, reasoningTokens: 180 });
  });

  it("omits reasoningTokens when completion_tokens_details reports zero", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        usage: {
          prompt_tokens: 50,
          completion_tokens: 200,
          total_tokens: 250,
          completion_tokens_details: { reasoning_tokens: 0 },
        },
      },
    });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).not.toHaveProperty("reasoningTokens");
    expect(result.usage).toEqual({ input: 50, output: 200, total: 250 });
  });

  it("reports cachedTokens and reasoningTokens together when both present", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        usage: {
          prompt_tokens: 100,
          completion_tokens: 200,
          total_tokens: 300,
          prompt_tokens_details: { cached_tokens: 40 },
          completion_tokens_details: { reasoning_tokens: 150 },
        },
      },
    });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.usage).toEqual({
      input: 100,
      output: 200,
      total: 300,
      cachedTokens: 40,
      reasoningTokens: 150,
    });
  });

  it("forwards reasoning_effort on the wire for a reasoning-capable model", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    await model.complete([{ role: "user", content: "hi" }], {
      reasoning: { effort: "high" },
    });

    expect((calls[0].params as { reasoning_effort?: string }).reasoning_effort).toBe("high");
  });

  it("does NOT forward reasoning_effort for a non-reasoning model", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o" });

    await model.complete([{ role: "user", content: "hi" }], {
      reasoning: { effort: "high" },
    });

    expect((calls[0].params as { reasoning_effort?: string }).reasoning_effort).toBeUndefined();
  });

  it("omits reasoning_effort when no reasoning option is supplied", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    await model.complete([{ role: "user", content: "hi" }]);

    expect((calls[0].params as { reasoning_effort?: string }).reasoning_effort).toBeUndefined();
  });

  it("ignores reasoning.maxTokens (no Chat Completions equivalent) but still forwards effort", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    await model.complete([{ role: "user", content: "hi" }], {
      reasoning: { effort: "low", maxTokens: 2048 },
    });

    const params = calls[0].params as { reasoning_effort?: string; max_completion_tokens?: number };
    expect(params.reasoning_effort).toBe("low");
    expect(params).not.toHaveProperty("thinkingBudget");
    expect(params).not.toHaveProperty("reasoning");
  });

  it("does not forward reasoning_effort when reasoning is supplied without an effort", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    await model.complete([{ role: "user", content: "hi" }], {
      reasoning: { maxTokens: 1024 },
    });

    expect((calls[0].params as { reasoning_effort?: string }).reasoning_effort).toBeUndefined();
  });

  it("treats cacheControl as a no-op (OpenAI caches automatically, no write breakpoints)", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await model.complete([{ role: "user", content: "hi" }], {
      cacheControl: { breakpoints: 2 },
    });

    // No cache_control / prompt-cache param exists on the Chat Completions
    // wire shape — the option is accepted and silently dropped.
    expect(calls[0].params).not.toHaveProperty("cache_control");
    expect(calls[0].params).not.toHaveProperty("cacheControl");
  });

  it("forwards a multipart user message (image_url) through to the wire", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o" });

    await model.complete([
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          { type: "image", source: { url: "https://example.com/cat.jpg" } },
        ],
      },
    ]);

    expect(calls[0].params.messages).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          { type: "image_url", image_url: { url: "https://example.com/cat.jpg" } },
        ],
      },
    ]);
  });

  it("returns content '' when OpenAI sends null content", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        choices: [
          {
            ...baseCompletion.choices[0],
            message: { role: "assistant", content: null, refusal: null },
          },
        ],
      },
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });
    const result = await model.complete([{ role: "user", content: "hi" }]);
    expect(result.content).toBe("");
  });

  it("zeros usage when OpenAI omits the usage block", async () => {
    const { client } = makeFakeClient({
      completion: { ...baseCompletion, usage: undefined },
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);
    expect(result.usage).toEqual({ input: 0, output: 0, total: 0 });
  });

  it("extracts tool calls and parses arguments JSON via safeJsonParse", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              refusal: null,
              tool_calls: [
                {
                  id: "call_123",
                  type: "function",
                  function: { name: "getWeather", arguments: '{"city":"Cairo"}' },
                },
              ],
            },
            finish_reason: "tool_calls",
            logprobs: null,
          },
        ],
      },
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.finishReason).toBe("tool_calls");
    expect(result.toolCalls).toEqual([
      { id: "call_123", name: "getWeather", input: { city: "Cairo" } },
    ]);
  });

  it("falls back to {} when a tool's arguments fail to JSON-parse", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              refusal: null,
              tool_calls: [
                {
                  id: "call_x",
                  type: "function",
                  function: { name: "broken", arguments: '{not json' },
                },
              ],
            },
            finish_reason: "tool_calls",
            logprobs: null,
          },
        ],
      },
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);
    expect(result.toolCalls?.[0].input).toEqual({});
  });

  it("forwards tools array via toOpenAITools", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await model.complete([{ role: "user", content: "hi" }], {
      tools: [
        {
          name: "echo",
          description: "echoes",
          input: {
            "~standard": {
              version: 1,
              vendor: "test",
              validate: (v: unknown) => ({ value: v }),
            },
          },
          execute: async (v: unknown) => v,
        },
      ],
    });

    expect(calls[0].params.tools).toHaveLength(1);
    expect(calls[0].params.tools?.[0].function.name).toBe("echo");
  });

  it("uses strict json_schema response_format for object root schemas", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await model.complete([{ role: "user", content: "hi" }], {
      responseSchema: {
        type: "object",
        properties: { summary: { type: "string" } },
        required: ["summary"],
      },
    });

    expect(calls[0].params.response_format).toEqual({
      type: "json_schema",
      json_schema: {
        name: "response",
        schema: {
          type: "object",
          properties: { summary: { type: "string" } },
          required: ["summary"],
        },
        strict: true,
      },
    });
  });

  it("falls back to loose json_object response_format for non-object schemas", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await model.complete([{ role: "user", content: "hi" }], {
      responseSchema: { type: "array", items: { type: "string" } },
    });

    expect(calls[0].params.response_format).toEqual({ type: "json_object" });
  });

  it("omits response_format entirely when no responseSchema is supplied", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await model.complete([{ role: "user", content: "hi" }]);

    expect(calls[0].params.response_format).toBeUndefined();
  });

  it("config responseFormat 'json_object' forces loose mode even for object-root schemas", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, {
      name: "some-legacy-model",
      responseFormat: "json_object",
    });

    await model.complete([{ role: "user", content: "hi" }], {
      responseSchema: { type: "object", properties: { x: { type: "string" } } },
    });

    expect(calls[0].params.response_format).toEqual({ type: "json_object" });
  });

  it("config responseFormat 'text' emits no response_format even with a schema", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "some-legacy-model", responseFormat: "text" });

    await model.complete([{ role: "user", content: "hi" }], {
      responseSchema: { type: "object", properties: { x: { type: "string" } } },
    });

    expect(calls[0].params.response_format).toBeUndefined();
  });

  it("config responseFormat 'json_schema' still degrades a non-object schema to json_object", async () => {
    // Explicit json_schema keeps the strict-compat safety check: a
    // non-object root would 400 before sampling, so it degrades.
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini", responseFormat: "json_schema" });

    await model.complete([{ role: "user", content: "hi" }], {
      responseSchema: { type: "array", items: { type: "string" } },
    });

    expect(calls[0].params.response_format).toEqual({ type: "json_object" });
  });

  it("config responseFormat 'json_schema' uses strict mode for object-root schemas", async () => {
    const { client, calls } = makeFakeClient({ completion: baseCompletion });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini", responseFormat: "json_schema" });

    await model.complete([{ role: "user", content: "hi" }], {
      responseSchema: { type: "object", properties: { summary: { type: "string" } } },
    });

    expect(calls[0].params.response_format).toEqual({
      type: "json_schema",
      json_schema: {
        name: "response",
        schema: { type: "object", properties: { summary: { type: "string" } } },
        strict: true,
      },
    });
  });

  it("extracts multiple parallel tool calls preserving order", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              refusal: null,
              tool_calls: [
                {
                  id: "call_1",
                  type: "function",
                  function: { name: "getWeather", arguments: '{"city":"Cairo"}' },
                },
                {
                  id: "call_2",
                  type: "function",
                  function: { name: "getTime", arguments: '{"tz":"UTC"}' },
                },
              ],
            },
            finish_reason: "tool_calls",
            logprobs: null,
          },
        ],
      },
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.toolCalls).toEqual([
      { id: "call_1", name: "getWeather", input: { city: "Cairo" } },
      { id: "call_2", name: "getTime", input: { tz: "UTC" } },
    ]);
  });

  it("parses empty-string tool arguments into {} via safeJsonParse", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              refusal: null,
              tool_calls: [
                {
                  id: "call_empty",
                  type: "function",
                  function: { name: "noArgs", arguments: "" },
                },
              ],
            },
            finish_reason: "tool_calls",
            logprobs: null,
          },
        ],
      },
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.toolCalls?.[0].input).toEqual({});
  });

  it("maps an unknown finish_reason to 'error' on complete()", async () => {
    const { client } = makeFakeClient({
      completion: {
        ...baseCompletion,
        choices: [
          {
            ...baseCompletion.choices[0],
            finish_reason: "content_filter" as never,
          },
        ],
      },
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const result = await model.complete([{ role: "user", content: "hi" }]);

    expect(result.finishReason).toBe("error");
  });

  // Per-instance logger-injection tests removed — Phase 3.2 dropped
  // the `logger`/`log` config fields. Logging now flows through
  // @warlock.js/logger's `log` singleton, configured globally at boot.
  // The error-rethrow path (without logger inspection) is covered by
  // the wrapping tests below.
  it("rethrows wrapped error on complete() failure", async () => {
    const error = { status: 500, message: "boom" };
    const client = {
      chat: {
        completions: {
          create: async () => {
            throw error;
          },
        },
      },
    } as unknown as OpenAI;
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await expect(
      model.complete([{ role: "user", content: "hi" }]),
    ).rejects.toMatchObject({ code: "PROVIDER_ERROR" });
  });

  it("wraps a 401 failure into a ProviderAuthError, preserving context", async () => {
    const client = {
      chat: {
        completions: {
          create: async () => {
            throw { status: 401, code: "invalid_api_key", message: "bad key" };
          },
        },
      },
    } as unknown as OpenAI;
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await expect(model.complete([{ role: "user", content: "hi" }])).rejects.toMatchObject({
      code: "PROVIDER_AUTH",
      message: "bad key",
      context: { status: 401, code: "invalid_api_key" },
    });
  });
});

describe("OpenAIModel construction + capabilities", () => {
  const stubClient = {} as unknown as OpenAI;

  it("defaults the provider label to 'openai'", () => {
    const model = new OpenAIModel(stubClient, { name: "gpt-4o-mini" });
    expect(model.provider).toBe("openai");
  });

  it("forwards an explicit provider label", () => {
    const model = new OpenAIModel(stubClient, { name: "gpt-4o-mini" }, "openrouter");
    expect(model.provider).toBe("openrouter");
  });

  it("exposes the config name as the model name", () => {
    const model = new OpenAIModel(stubClient, { name: "custom-x" });
    expect(model.name).toBe("custom-x");
  });

  it("infers vision from the model name when not overridden", () => {
    expect(new OpenAIModel(stubClient, { name: "gpt-4o" }).capabilities.vision).toBe(true);
    expect(new OpenAIModel(stubClient, { name: "gpt-3.5-turbo" }).capabilities.vision).toBe(false);
  });

  it("honors an explicit vision override over inference", () => {
    expect(new OpenAIModel(stubClient, { name: "gpt-3.5-turbo", vision: true }).capabilities.vision).toBe(
      true,
    );
    expect(new OpenAIModel(stubClient, { name: "gpt-4o", vision: false }).capabilities.vision).toBe(
      false,
    );
  });

  it("defaults structuredOutput to true with no responseFormat override", () => {
    expect(new OpenAIModel(stubClient, { name: "gpt-4o-mini" }).capabilities.structuredOutput).toBe(
      true,
    );
  });

  it("downgrades structuredOutput to false when responseFormat is a loose mode", () => {
    expect(
      new OpenAIModel(stubClient, { name: "gpt-4o-mini", responseFormat: "json_object" }).capabilities
        .structuredOutput,
    ).toBe(false);
    expect(
      new OpenAIModel(stubClient, { name: "gpt-4o-mini", responseFormat: "text" }).capabilities
        .structuredOutput,
    ).toBe(false);
  });

  it("keeps structuredOutput true when responseFormat is the strict json_schema mode", () => {
    expect(
      new OpenAIModel(stubClient, { name: "gpt-4o-mini", responseFormat: "json_schema" }).capabilities
        .structuredOutput,
    ).toBe(true);
  });

  it("honors an explicit structuredOutput override regardless of responseFormat", () => {
    expect(
      new OpenAIModel(stubClient, {
        name: "gpt-4o-mini",
        responseFormat: "json_object",
        structuredOutput: true,
      }).capabilities.structuredOutput,
    ).toBe(true);
  });

  it("exposes pricing from config when supplied", () => {
    const pricing = { input: 0.15, output: 0.6 };
    const model = new OpenAIModel(stubClient, { name: "gpt-4o-mini", pricing });
    expect(model.pricing).toBe(pricing);
  });

  it("leaves pricing undefined when config omits it", () => {
    const model = new OpenAIModel(stubClient, { name: "gpt-4o-mini" });
    expect(model.pricing).toBeUndefined();
  });

  it("always advertises promptCaching (OpenAI caches automatically)", () => {
    expect(new OpenAIModel(stubClient, { name: "gpt-4o-mini" }).capabilities.promptCaching).toBe(
      true,
    );
    expect(new OpenAIModel(stubClient, { name: "o3-mini" }).capabilities.promptCaching).toBe(true);
  });

  it("infers reasoning from the model name when not overridden", () => {
    expect(new OpenAIModel(stubClient, { name: "o3-mini" }).capabilities.reasoning).toBe(true);
    expect(new OpenAIModel(stubClient, { name: "o4-mini" }).capabilities.reasoning).toBe(true);
    expect(new OpenAIModel(stubClient, { name: "gpt-5-pro" }).capabilities.reasoning).toBe(true);
    expect(new OpenAIModel(stubClient, { name: "gpt-4o" }).capabilities.reasoning).toBe(false);
    expect(new OpenAIModel(stubClient, { name: "gpt-4o-mini" }).capabilities.reasoning).toBe(false);
  });

  it("honors an explicit reasoning override over inference", () => {
    expect(
      new OpenAIModel(stubClient, { name: "gpt-4o", reasoning: true }).capabilities.reasoning,
    ).toBe(true);
    expect(
      new OpenAIModel(stubClient, { name: "o3-mini", reasoning: false }).capabilities.reasoning,
    ).toBe(false);
  });
});

describe("OpenAIModel.stream()", () => {
  function chunk(
    overrides: Partial<OpenAI.Chat.Completions.ChatCompletionChunk> & {
      delta?: OpenAI.Chat.Completions.ChatCompletionChunk["choices"][number]["delta"];
      finish?: OpenAI.Chat.Completions.ChatCompletionChunk["choices"][number]["finish_reason"];
    } = {},
  ): OpenAI.Chat.Completions.ChatCompletionChunk {
    const { delta, finish, ...rest } = overrides;
    return {
      id: "x",
      object: "chat.completion.chunk",
      created: 0,
      model: "gpt-4o-mini",
      choices: [
        {
          index: 0,
          delta: delta ?? {},
          finish_reason: finish ?? null,
          logprobs: null,
        },
      ],
      ...rest,
    };
  }

  it("yields delta chunks for content fragments", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({ delta: { content: "Hello" } }),
        chunk({ delta: { content: " world" } }),
        chunk({ finish: "stop", usage: { prompt_tokens: 3, completion_tokens: 2, total_tokens: 5 } }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const events: string[] = [];
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      events.push(event.type);
    }

    expect(events).toEqual(["delta", "delta", "done"]);
  });

  it("aggregates usage from a usage-bearing chunk into the terminal done event", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({ delta: { content: "x" } }),
        chunk({ finish: "stop", usage: { prompt_tokens: 9, completion_tokens: 4, total_tokens: 13 } }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    let final: { input: number; output: number; total: number } | undefined;
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "done") {
        final = event.usage;
      }
    }

    expect(final).toEqual({ input: 9, output: 4, total: 13 });
  });

  it("zeros usage when no usage chunk arrives", async () => {
    const { client } = makeFakeClient({
      streamChunks: [chunk({ delta: { content: "x" } }), chunk({ finish: "stop" })],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "done") {
        expect(event.usage).toEqual({ input: 0, output: 0, total: 0 });
      }
    }
  });

  it("maps the final finish_reason via mapFinishReason", async () => {
    const { client } = makeFakeClient({
      streamChunks: [chunk({ delta: { content: "x" } }), chunk({ finish: "length" })],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "done") {
        expect(event.finishReason).toBe("length");
      }
    }
  });

  it("falls back to 'error' when finish_reason is unknown", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({ delta: { content: "x" } }),
        chunk({ finish: "content_filter" as never }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "done") {
        expect(event.finishReason).toBe("error");
      }
    }
  });

  it("emits a tool-call chunk when the delta carries a function name", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_a",
                function: { name: "getWeather", arguments: '{"city":"Cairo"}' },
              },
            ],
          },
        }),
        chunk({ finish: "tool_calls" }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const toolEvents: Array<{ id: string; name: string; input: unknown }> = [];
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "tool-call") {
        toolEvents.push({ id: event.id, name: event.name, input: event.input });
      }
    }

    expect(toolEvents).toHaveLength(1);
    expect(toolEvents[0].id).toBe("call_a");
    expect(toolEvents[0].name).toBe("getWeather");
    expect(toolEvents[0].input).toEqual({ city: "Cairo" });
  });

  it("skips tool_call deltas that have no function name", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({
          delta: {
            tool_calls: [
              { index: 0, function: { arguments: '{"partial":' } },
            ],
          },
        }),
        chunk({ finish: "tool_calls" }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const events: string[] = [];
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      events.push(event.type);
    }

    expect(events.filter((e) => e === "tool-call")).toHaveLength(0);
  });

  it("requests stream_options.include_usage on the wire", async () => {
    const { client, calls } = makeFakeClient({
      streamChunks: [chunk({ finish: "stop" })],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    for await (const _event of model.stream([{ role: "user", content: "hi" }])) {
      // drain
    }

    expect(calls[0].params.stream).toBe(true);
    expect(
      (calls[0].params as { stream_options?: { include_usage?: boolean } }).stream_options
        ?.include_usage,
    ).toBe(true);
  });

  it("accumulates a tool call whose id/name/arguments arrive across multiple chunks", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({
          delta: {
            tool_calls: [{ index: 0, id: "call_a", function: { name: "getWeather" } }],
          },
        }),
        chunk({ delta: { tool_calls: [{ index: 0, function: { arguments: '{"city":' } }] } }),
        chunk({ delta: { tool_calls: [{ index: 0, function: { arguments: '"Cairo"}' } }] } }),
        chunk({ finish: "tool_calls" }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const toolEvents: Array<{ id: string; name: string; input: unknown }> = [];
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "tool-call") {
        toolEvents.push({ id: event.id, name: event.name, input: event.input });
      }
    }

    expect(toolEvents).toEqual([
      { id: "call_a", name: "getWeather", input: { city: "Cairo" } },
    ]);
  });

  it("accumulates two parallel tool calls keyed by their delta index", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({
          delta: {
            tool_calls: [
              { index: 0, id: "call_0", function: { name: "a", arguments: '{"x":1}' } },
              { index: 1, id: "call_1", function: { name: "b", arguments: '{"y":2}' } },
            ],
          },
        }),
        chunk({ finish: "tool_calls" }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const toolEvents: Array<{ id: string; name: string; input: unknown }> = [];
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "tool-call") {
        toolEvents.push({ id: event.id, name: event.name, input: event.input });
      }
    }

    expect(toolEvents).toEqual([
      { id: "call_0", name: "a", input: { x: 1 } },
      { id: "call_1", name: "b", input: { y: 2 } },
    ]);
  });

  it("defaults a missing tool_call delta index to 0", async () => {
    // `toolCall.index ?? 0` — OpenAI's delta type requires `index`, but a
    // real stream can omit it; cast the intentionally-incomplete entry so
    // the runtime fallback gets exercised.
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({
          delta: {
            tool_calls: [
              { id: "call_z", function: { name: "z", arguments: "{}" } },
            ] as OpenAI.Chat.Completions.ChatCompletionChunk["choices"][number]["delta"]["tool_calls"],
          },
        }),
        chunk({ finish: "tool_calls" }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const toolEvents: Array<{ name: string }> = [];
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "tool-call") {
        toolEvents.push({ name: event.name });
      }
    }

    expect(toolEvents).toEqual([{ name: "z" }]);
  });

  it("yields tool-call events after all delta events, before done", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({ delta: { content: "thinking " } }),
        chunk({
          delta: {
            tool_calls: [{ index: 0, id: "c", function: { name: "go", arguments: "{}" } }],
          },
        }),
        chunk({ finish: "tool_calls" }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const order: string[] = [];
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      order.push(event.type);
    }

    expect(order).toEqual(["delta", "tool-call", "done"]);
  });

  it("surfaces cachedTokens in the done event when the usage chunk reports them", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({ delta: { content: "x" } }),
        chunk({
          finish: "stop",
          usage: {
            prompt_tokens: 100,
            completion_tokens: 20,
            total_tokens: 120,
            prompt_tokens_details: { cached_tokens: 60 },
          },
        }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    let final: { input: number; output: number; total: number; cachedTokens?: number } | undefined;
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "done") final = event.usage;
    }

    expect(final).toEqual({ input: 100, output: 20, total: 120, cachedTokens: 60 });
  });

  it("does not set cachedTokens in the done event when the usage chunk reports zero", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({
          finish: "stop",
          usage: {
            prompt_tokens: 10,
            completion_tokens: 2,
            total_tokens: 12,
            prompt_tokens_details: { cached_tokens: 0 },
          },
        }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "done") {
        expect(event.usage).not.toHaveProperty("cachedTokens");
        expect(event.usage).toEqual({ input: 10, output: 2, total: 12 });
      }
    }
  });

  it("surfaces reasoningTokens in the done event when the usage chunk reports them", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({ delta: { content: "x" } }),
        chunk({
          finish: "stop",
          usage: {
            prompt_tokens: 50,
            completion_tokens: 200,
            total_tokens: 250,
            completion_tokens_details: { reasoning_tokens: 180 },
          },
        }),
      ],
    });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    let final: { input: number; output: number; total: number; reasoningTokens?: number } | undefined;
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "done") final = event.usage;
    }

    expect(final).toEqual({ input: 50, output: 200, total: 250, reasoningTokens: 180 });
  });

  it("does not set reasoningTokens in the done event when the usage chunk reports zero", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({
          finish: "stop",
          usage: {
            prompt_tokens: 50,
            completion_tokens: 200,
            total_tokens: 250,
            completion_tokens_details: { reasoning_tokens: 0 },
          },
        }),
      ],
    });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "done") {
        expect(event.usage).not.toHaveProperty("reasoningTokens");
        expect(event.usage).toEqual({ input: 50, output: 200, total: 250 });
      }
    }
  });

  it("forwards reasoning_effort on the streaming wire for a reasoning-capable model", async () => {
    const { client, calls } = makeFakeClient({
      streamChunks: [chunk({ finish: "stop" })],
    });
    const model = new OpenAIModel(client, { name: "o3-mini" });

    for await (const _event of model.stream([{ role: "user", content: "hi" }], {
      reasoning: { effort: "medium" },
    })) {
      // drain
    }

    expect((calls[0].params as { reasoning_effort?: string }).reasoning_effort).toBe("medium");
  });

  it("does NOT forward reasoning_effort on the streaming wire for a non-reasoning model", async () => {
    const { client, calls } = makeFakeClient({
      streamChunks: [chunk({ finish: "stop" })],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o" });

    for await (const _event of model.stream([{ role: "user", content: "hi" }], {
      reasoning: { effort: "medium" },
    })) {
      // drain
    }

    expect((calls[0].params as { reasoning_effort?: string }).reasoning_effort).toBeUndefined();
  });

  it("forwards an AbortSignal to the streaming create call's request options", async () => {
    const recorded: { signal?: AbortSignal } = {};
    const controller = new AbortController();
    const client = {
      chat: {
        completions: {
          create: async (
            _params: OpenAI.Chat.Completions.ChatCompletionCreateParams,
            opts?: { signal?: AbortSignal },
          ) => {
            recorded.signal = opts?.signal;
            return (async function* () {
              yield chunk({ finish: "stop" });
            })();
          },
        },
      },
    } as unknown as OpenAI;
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    for await (const _event of model.stream([{ role: "user", content: "hi" }], {
      signal: controller.signal,
    })) {
      // drain
    }

    expect(recorded.signal).toBe(controller.signal);
  });

  it("wraps an error thrown while opening the stream", async () => {
    const client = {
      chat: {
        completions: {
          create: async () => {
            throw { status: 429, code: "rate_limit_exceeded", message: "slow down" };
          },
        },
      },
    } as unknown as OpenAI;
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    await expect(async () => {
      for await (const _event of model.stream([{ role: "user", content: "hi" }])) {
        // drain
      }
    }).rejects.toMatchObject({ code: "PROVIDER_RATE_LIMIT" });
  });

  it("wraps an error thrown mid-iteration while consuming the stream", async () => {
    const client = {
      chat: {
        completions: {
          create: async () =>
            (async function* () {
              yield chunk({ delta: { content: "partial" } });
              throw { status: 500, message: "stream blew up" };
            })(),
        },
      },
    } as unknown as OpenAI;
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const seen: string[] = [];
    await expect(async () => {
      for await (const event of model.stream([{ role: "user", content: "hi" }])) {
        seen.push(event.type);
      }
    }).rejects.toMatchObject({ code: "PROVIDER_ERROR", message: "stream blew up" });

    // The delta before the failure was still surfaced to the consumer.
    expect(seen).toEqual(["delta"]);
  });

  it("emits a terminal done event even for an empty stream", async () => {
    const { client } = makeFakeClient({ streamChunks: [] });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const events: Array<{ type: string; usage?: unknown; finishReason?: unknown }> = [];
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      events.push(event);
    }

    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("done");
    // No finish chunk arrived → rawFinishReason defaults to "stop".
    expect(events[0].finishReason).toBe("stop");
    expect(events[0].usage).toEqual({ input: 0, output: 0, total: 0 });
  });

  it("ignores empty content deltas (no delta event for empty-string content)", async () => {
    const { client } = makeFakeClient({
      streamChunks: [
        chunk({ delta: { content: "" } }),
        chunk({ delta: { content: "real" } }),
        chunk({ finish: "stop" }),
      ],
    });
    const model = new OpenAIModel(client, { name: "gpt-4o-mini" });

    const deltas: string[] = [];
    for await (const event of model.stream([{ role: "user", content: "hi" }])) {
      if (event.type === "delta") deltas.push(event.content);
    }

    expect(deltas).toEqual(["real"]);
  });
});
