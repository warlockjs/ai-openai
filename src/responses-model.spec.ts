import type { StandardSchemaV1 } from "@standard-schema/spec";
import {
  InvalidRequestError,
  ProviderError,
  ProviderRateLimitError,
  type Message,
  type ToolConfig,
} from "@warlock.js/ai";
import type OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { OpenAIModel } from "./model";
import { OpenAIResponsesModel } from "./responses-model";

type Params = OpenAI.Responses.ResponseCreateParamsNonStreaming;
type ResponseFixture = OpenAI.Responses.Response;

/** The params of the first recorded request; fails the test when none was made. */
function firstParams(calls: readonly Params[]): Params {
  const call = calls[0];
  if (call === undefined) throw new Error("expected the fake client to have been called");
  return call;
}

/**
 * Fake OpenAI client whose `responses.create()` records the params and the
 * request options it was called with and returns a scripted response (or
 * throws a scripted error). No network.
 */
function makeFakeClient(options: { response?: ResponseFixture; error?: unknown }) {
  const calls: Params[] = [];
  const requestOptions: unknown[] = [];

  const create = async (params: Params, requestOption?: unknown) => {
    calls.push(params);
    requestOptions.push(requestOption);

    if (options.error !== undefined) {
      throw options.error;
    }

    return options.response;
  };

  const client = { responses: { create } } as unknown as OpenAI;

  return { client, calls, requestOptions };
}

/** Build a `Response` fixture; only the fields under test need to be set. */
function makeResponse(overrides: Partial<ResponseFixture> = {}): ResponseFixture {
  return {
    id: "resp_1",
    object: "response",
    created_at: 0,
    status: "completed",
    error: null,
    incomplete_details: null,
    model: "gpt-5.6",
    output: [],
    usage: {
      input_tokens: 10,
      input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
      output_tokens: 4,
      output_tokens_details: { reasoning_tokens: 0 },
      total_tokens: 14,
    },
    ...overrides,
  } as unknown as ResponseFixture;
}

function textMessage(text: string): OpenAI.Responses.ResponseOutputMessage {
  return {
    id: "msg_1",
    type: "message",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text, annotations: [] }],
  };
}

function functionCall(
  overrides: Partial<OpenAI.Responses.ResponseFunctionToolCall> = {},
): OpenAI.Responses.ResponseFunctionToolCall {
  return {
    type: "function_call",
    id: "fc_1",
    call_id: "call_1",
    name: "getWeather",
    arguments: '{"city":"Cairo"}',
    status: "completed",
    ...overrides,
  };
}

const objectInput: StandardSchemaV1<{ city: string }> & { jsonSchema: Record<string, unknown> } = {
  "~standard": { version: 1, vendor: "test", validate: (v) => ({ value: v as { city: string } }) },
  jsonSchema: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
};

/** A tool whose JSON Schema extracts to a real object schema. */
function weatherTool(): ToolConfig<unknown, unknown> {
  return {
    name: "getWeather",
    description: "Weather for a city",
    input: objectInput,
    execute: async (v: unknown) => v,
  } as ToolConfig<unknown, unknown>;
}

const userHi: Message[] = [{ role: "user", content: "hi" }];

describe("OpenAIResponsesModel request shape", () => {
  it("sends model, input and store:false, and never the Chat-only keys", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    const model = new OpenAIResponsesModel(client, { name: "gpt-4o-mini" });

    await model.complete(userHi);

    const params = firstParams(calls);
    expect(params).toEqual({
      model: "gpt-4o-mini",
      input: [{ role: "user", content: "hi" }],
      store: false,
    });
    expect(params).not.toHaveProperty("messages");
    expect(params).not.toHaveProperty("max_completion_tokens");
    expect(params).not.toHaveProperty("stream");
  });

  it("puts the system prompt in instructions and keeps later system messages inline", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    const model = new OpenAIResponsesModel(client, { name: "gpt-4o-mini" });

    await model.complete([
      { role: "system", content: "Be brief." },
      { role: "user", content: "hi" },
      { role: "system", content: "Reminder." },
    ]);

    const params = firstParams(calls);
    expect(params.instructions).toBe("Be brief.");
    expect(params.input).toEqual([
      { role: "user", content: "hi" },
      { role: "system", content: "Reminder." },
    ]);
  });

  it("omits instructions when there is no system message", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    const model = new OpenAIResponsesModel(client, { name: "gpt-4o-mini" });

    await model.complete(userHi);

    expect(firstParams(calls)).not.toHaveProperty("instructions");
  });

  it("maps a tool round trip to message, function_call and function_call_output items", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    const model = new OpenAIResponsesModel(client, { name: "gpt-5.6" });

    await model.complete(
      [
        { role: "system", content: "You are a weather bot." },
        { role: "user", content: "Weather in Cairo?" },
        {
          role: "assistant",
          content: "Checking.",
          toolCalls: [{ id: "call_1", name: "getWeather", input: { city: "Cairo" } }],
        },
        { role: "tool", toolCallId: "call_1", content: '{"temp":82}' },
      ],
      { tools: [weatherTool()], reasoning: { effort: "medium" }, maxTokens: 2000 },
    );

    expect(firstParams(calls)).toEqual({
      model: "gpt-5.6",
      instructions: "You are a weather bot.",
      input: [
        { role: "user", content: "Weather in Cairo?" },
        { role: "assistant", content: "Checking." },
        {
          type: "function_call",
          call_id: "call_1",
          name: "getWeather",
          arguments: '{"city":"Cairo"}',
        },
        { type: "function_call_output", call_id: "call_1", output: '{"temp":82}' },
      ],
      store: false,
      max_output_tokens: 2000,
      tools: [
        {
          type: "function",
          name: "getWeather",
          description: "Weather for a city",
          parameters: objectInput.jsonSchema,
          strict: false,
        },
      ],
      reasoning: { effort: "medium" },
    });
  });

  it("skips the empty assistant text item when the turn only carried tool calls", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    const model = new OpenAIResponsesModel(client, { name: "gpt-4o-mini" });

    await model.complete([
      { role: "user", content: "go" },
      { role: "assistant", content: "", toolCalls: [{ id: "c1", name: "t", input: undefined }] },
      { role: "tool", toolCallId: "c1", content: "ok" },
    ]);

    expect(firstParams(calls).input).toEqual([
      { role: "user", content: "go" },
      { type: "function_call", call_id: "c1", name: "t", arguments: "{}" },
      { type: "function_call_output", call_id: "c1", output: "ok" },
    ]);
  });

  it("maps images to input_image with the default detail and text to input_text", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    const model = new OpenAIResponsesModel(client, { name: "gpt-4o" });

    await model.complete([
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          { type: "image", source: { url: "https://example.com/cat.jpg" } },
          { type: "image", source: { base64: "QUJD", mediaType: "image/png" } },
        ],
      },
    ]);

    expect(firstParams(calls).input).toEqual([
      {
        role: "user",
        content: [
          { type: "input_text", text: "What is this?" },
          { type: "input_image", image_url: "https://example.com/cat.jpg", detail: "auto" },
          { type: "input_image", image_url: "data:image/png;base64,QUJD", detail: "auto" },
        ],
      },
    ]);
  });

  it("rejects pdf and audio parts with a typed error instead of sending them", async () => {
    const { client } = makeFakeClient({ response: makeResponse() });
    const model = new OpenAIResponsesModel(client, { name: "gpt-4o" });

    await expect(
      model.complete([
        {
          role: "user",
          content: [{ type: "pdf", source: { base64: "QUJD", mediaType: "application/pdf" } }],
        },
      ]),
    ).rejects.toBeInstanceOf(InvalidRequestError);
  });

  it("sends max_output_tokens from config and lets the per-call option win", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    const model = new OpenAIResponsesModel(client, { name: "gpt-4o-mini", maxTokens: 256 });

    await model.complete(userHi);
    await model.complete(userHi, { maxTokens: 64 });

    expect(calls[0]?.max_output_tokens).toBe(256);
    expect(calls[1]?.max_output_tokens).toBe(64);
  });

  it("sends temperature for a non-reasoning model and omits it for a reasoning one", async () => {
    const plain = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(plain.client, { name: "gpt-4o-mini", temperature: 0.4 }).complete(
      userHi,
      { temperature: 0.9 },
    );
    expect(firstParams(plain.calls).temperature).toBe(0.9);

    const reasoning = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(reasoning.client, {
      name: "gpt-5.6",
      temperature: 0.3,
    }).complete(userHi, { temperature: 0.7 });
    expect(firstParams(reasoning.calls)).not.toHaveProperty("temperature");
  });

  it("forwards the abort signal as a request option", async () => {
    const { client, requestOptions } = makeFakeClient({ response: makeResponse() });
    const controller = new AbortController();

    await new OpenAIResponsesModel(client, { name: "gpt-4o-mini" }).complete(userHi, {
      signal: controller.signal,
    });

    expect(requestOptions[0]).toEqual({ signal: controller.signal });
  });
});

describe("OpenAIResponsesModel tools and toolChoice", () => {
  it("sends flat function tools with strict:false", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, { name: "gpt-4o-mini" }).complete(userHi, {
      tools: [weatherTool()],
    });

    const tools = firstParams(calls).tools;
    expect(tools).toEqual([
      {
        type: "function",
        name: "getWeather",
        description: "Weather for a city",
        parameters: objectInput.jsonSchema,
        strict: false,
      },
    ]);
    expect(tools?.[0]).not.toHaveProperty("function");
  });

  it.each([
    ["none", "none"],
    ["auto", "auto"],
    ["required", "required"],
    [{ name: "getWeather" }, { type: "function", name: "getWeather" }],
    [{ type: "function", name: "getWeather" }, { type: "function", name: "getWeather" }],
    [
      { type: "function", function: { name: "getWeather" } },
      { type: "function", name: "getWeather" },
    ],
  ])("maps toolChoice %j to the flat Responses tool_choice", async (toolChoice, expected) => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, { name: "gpt-4o-mini" }).complete(userHi, {
      tools: [weatherTool()],
      toolChoice,
    });

    expect(firstParams(calls).tool_choice).toEqual(expected);
  });

  it("drops an unrecognized toolChoice and any toolChoice when no tools are attached", async () => {
    const bogus = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(bogus.client, { name: "gpt-4o-mini" }).complete(userHi, {
      tools: [weatherTool()],
      toolChoice: 42,
    });
    expect(firstParams(bogus.calls)).not.toHaveProperty("tool_choice");

    const noTools = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(noTools.client, { name: "gpt-4o-mini" }).complete(userHi, {
      toolChoice: "required",
    });
    expect(firstParams(noTools.calls)).not.toHaveProperty("tool_choice");
    expect(firstParams(noTools.calls)).not.toHaveProperty("tools");
  });
});

describe("OpenAIResponsesModel reasoning", () => {
  it.each(["none", "minimal", "low", "medium", "high", "xhigh", "max"] as const)(
    "sends reasoning.effort %s through the same clamp the Chat path uses",
    async (effort) => {
      const responses = makeFakeClient({ response: makeResponse() });
      await new OpenAIResponsesModel(responses.client, { name: "gpt-5.6" }).complete(userHi, {
        reasoning: { effort },
      });

      const chatCalls: OpenAI.Chat.Completions.ChatCompletionCreateParams[] = [];
      const chatClient = {
        chat: {
          completions: {
            create: async (params: OpenAI.Chat.Completions.ChatCompletionCreateParams) => {
              chatCalls.push(params);
              return {
                id: "x",
                object: "chat.completion",
                created: 0,
                model: "gpt-5.6",
                choices: [
                  {
                    index: 0,
                    message: { role: "assistant", content: "", refusal: null },
                    finish_reason: "stop",
                    logprobs: null,
                  },
                ],
              };
            },
          },
        },
      } as unknown as OpenAI;
      await new OpenAIModel(chatClient, { name: "gpt-5.6" }).complete(userHi, {
        reasoning: { effort },
      });

      expect(firstParams(responses.calls).reasoning).toEqual({ effort });
      expect(chatCalls[0]?.reasoning_effort).toBe(firstParams(responses.calls).reasoning?.effort);
    },
  );

  it("keeps the provider-default reasoning even with tools attached (no forced 'none')", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi, {
      tools: [weatherTool()],
    });

    expect(firstParams(calls)).not.toHaveProperty("reasoning");
  });

  it("still honours an explicit effort 'none' with tools attached", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi, {
      tools: [weatherTool()],
      reasoning: { effort: "none" },
    });

    expect(firstParams(calls).reasoning).toEqual({ effort: "none" });
  });

  it("ignores reasoning for a non-reasoning model and ignores reasoning.maxTokens", async () => {
    const plain = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(plain.client, { name: "gpt-4o-mini" }).complete(userHi, {
      reasoning: { effort: "high" },
    });
    expect(firstParams(plain.calls)).not.toHaveProperty("reasoning");

    const capped = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(capped.client, { name: "gpt-5.6" }).complete(userHi, {
      reasoning: { effort: "low", maxTokens: 2048 },
    });
    expect(firstParams(capped.calls).reasoning).toEqual({ effort: "low" });
  });
});

describe("OpenAIResponsesModel structured output", () => {
  const strictSchema = {
    type: "object",
    properties: { summary: { type: "string" } },
    required: ["summary"],
  };

  it("sends strict json_schema under text.format with name and schema on the format", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi, {
      responseSchema: strictSchema,
    });

    expect(firstParams(calls).text).toEqual({
      format: { type: "json_schema", name: "response", schema: strictSchema, strict: true },
    });
    expect(firstParams(calls)).not.toHaveProperty("response_format");
  });

  it("downgrades a schema with optional properties to json_object, like the Chat path", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi, {
      responseSchema: { type: "object", properties: { a: { type: "string" } } },
    });

    expect(firstParams(calls).text).toEqual({ format: { type: "json_object" } });
  });

  it("downgrades a non-object root schema to json_object", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi, {
      responseSchema: { type: "array", items: { type: "string" } },
    });

    expect(firstParams(calls).text).toEqual({ format: { type: "json_object" } });
  });

  it("honours the responseFormat overrides", async () => {
    const asObject = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(asObject.client, {
      name: "gpt-5.6",
      responseFormat: "json_object",
    }).complete(userHi, { responseSchema: strictSchema });
    expect(firstParams(asObject.calls).text).toEqual({ format: { type: "json_object" } });

    const asText = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(asText.client, {
      name: "gpt-5.6",
      responseFormat: "text",
    }).complete(userHi, { responseSchema: strictSchema });
    expect(firstParams(asText.calls)).not.toHaveProperty("text");
  });

  it("omits text when no responseSchema is supplied", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(firstParams(calls)).not.toHaveProperty("text");
  });
});

describe("OpenAIResponsesModel prompt cache passthrough", () => {
  it("forwards prompt_cache_key and prompt_cache_retention from config", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, {
      name: "gpt-5.6",
      promptCacheKey: "tenant-7",
      promptCacheRetention: "24h",
    }).complete(userHi);

    expect(firstParams(calls).prompt_cache_key).toBe("tenant-7");
    expect(firstParams(calls).prompt_cache_retention).toBe("24h");
  });

  it("omits them when not configured", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(firstParams(calls)).not.toHaveProperty("prompt_cache_key");
    expect(firstParams(calls)).not.toHaveProperty("prompt_cache_retention");
  });
});

describe("OpenAIResponsesModel capabilities", () => {
  it("keeps pdf and audio off even when the config opts in", () => {
    const { client } = makeFakeClient({});
    const model = new OpenAIResponsesModel(client, { name: "gpt-5.6", pdf: true, audio: true });

    expect(model.capabilities).toMatchObject({ pdf: false, audio: false });
  });

  it("infers vision, reasoning and structuredOutput like the Chat model", () => {
    const { client } = makeFakeClient({});
    const responses = new OpenAIResponsesModel(client, { name: "gpt-5.6" });
    const chat = new OpenAIModel(client, { name: "gpt-5.6" });

    expect(responses.capabilities?.vision).toBe(chat.capabilities?.vision);
    expect(responses.capabilities?.reasoning).toBe(true);
    expect(responses.capabilities?.structuredOutput).toBe(true);
    expect(responses.capabilities?.promptCaching).toBe(true);
  });

  it("refuses any provider label other than openai", () => {
    const { client } = makeFakeClient({});

    expect(() => new OpenAIResponsesModel(client, { name: "m" }, "deepseek")).toThrow(
      InvalidRequestError,
    );
  });
});

describe("OpenAIResponsesModel response mapping", () => {
  it("maps a completed text response", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({ output: [textMessage("hello")] }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-4o-mini" }).complete(userHi);

    expect(result).toEqual({
      content: "hello",
      finishReason: "stop",
      usage: { input: 10, output: 4, total: 14 },
    });
    expect(result).not.toHaveProperty("toolCalls");
  });

  it("maps a completed tool-call response: call_id becomes id, arguments are parsed, finish is tool_calls", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({
        output: [
          { id: "rs_1", type: "reasoning", summary: [] },
          functionCall({ call_id: "call_abc", name: "getWeather", arguments: '{"city":"Cairo"}' }),
        ],
      }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi, {
      tools: [weatherTool()],
    });

    expect(result.finishReason).toBe("tool_calls");
    expect(result.content).toBe("");
    expect(result.toolCalls).toEqual([
      { id: "call_abc", name: "getWeather", input: { city: "Cairo" } },
    ]);
  });

  it("keeps parallel tool calls in order and text alongside them", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({
        output: [
          textMessage("Checking both."),
          functionCall({ call_id: "c1", name: "a", arguments: "{}" }),
          functionCall({ call_id: "c2", name: "b", arguments: '{"x":1}' }),
        ],
      }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(result.content).toBe("Checking both.");
    expect(result.toolCalls?.map((call) => call.id)).toEqual(["c1", "c2"]);
    expect(result.finishReason).toBe("tool_calls");
  });

  it("parses malformed tool arguments to an empty object instead of throwing", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({ output: [functionCall({ arguments: "{not json" })] }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(result.toolCalls?.[0]?.input).toEqual({});
  });

  it("surfaces refusal text as normal content with finish 'stop'", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({
        output: [
          {
            id: "msg_r",
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "refusal", refusal: "I can't help with that." }],
          },
        ],
      }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(result.content).toBe("I can't help with that.");
    expect(result.finishReason).toBe("stop");
  });

  it("maps incomplete/max_output_tokens to length and keeps the partial text", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({
        status: "incomplete",
        incomplete_details: { reason: "max_output_tokens" },
        output: [textMessage("partial")],
      }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(result.finishReason).toBe("length");
    expect(result.content).toBe("partial");
  });

  it("maps incomplete/content_filter to error (no neutral content_filter value)", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({
        status: "incomplete",
        incomplete_details: { reason: "content_filter" },
      }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(result.finishReason).toBe("error");
  });

  it("skips a function_call the API marked incomplete (truncated arguments)", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({
        status: "incomplete",
        incomplete_details: { reason: "max_output_tokens" },
        output: [functionCall({ status: "incomplete", arguments: '{"city":"Ca' })],
      }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(result.finishReason).toBe("length");
    expect(result).not.toHaveProperty("toolCalls");
  });

  it("tolerates a response without an output array", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({ output: undefined as unknown as ResponseFixture["output"] }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(result).toMatchObject({ content: "", finishReason: "stop" });
  });
});

describe("OpenAIResponsesModel usage", () => {
  it("reports reasoning, cached and cache-write tokens when positive", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({
        output: [textMessage("ok")],
        usage: {
          input_tokens: 2000,
          input_tokens_details: { cached_tokens: 1536, cache_write_tokens: 64 },
          output_tokens: 300,
          output_tokens_details: { reasoning_tokens: 180 },
          total_tokens: 2300,
        },
      }),
    });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi);

    expect(result.usage).toEqual({
      input: 2000,
      output: 300,
      total: 2300,
      cachedTokens: 1536,
      reasoningTokens: 180,
      cacheWriteTokens: 64,
    });
  });

  it("omits zero-valued detail counters", async () => {
    const { client } = makeFakeClient({ response: makeResponse({ output: [textMessage("ok")] }) });

    const result = await new OpenAIResponsesModel(client, { name: "gpt-4o-mini" }).complete(userHi);

    expect(result.usage).toEqual({ input: 10, output: 4, total: 14 });
  });

  it("collapses absent usage to zeros, and tolerates missing detail blocks", async () => {
    const none = makeFakeClient({ response: makeResponse({ usage: undefined }) });
    expect(
      (await new OpenAIResponsesModel(none.client, { name: "m" }).complete(userHi)).usage,
    ).toEqual({ input: 0, output: 0, total: 0 });

    const partial = makeFakeClient({
      response: makeResponse({
        usage: { input_tokens: 5, output_tokens: 1, total_tokens: 6 } as ResponseFixture["usage"],
      }),
    });
    expect(
      (await new OpenAIResponsesModel(partial.client, { name: "m" }).complete(userHi)).usage,
    ).toEqual({ input: 5, output: 1, total: 6 });
  });
});

describe("OpenAIResponsesModel errors", () => {
  it("maps a failed response to ProviderError keeping the code, response id and request id", async () => {
    const response = makeResponse({
      status: "failed",
      error: { code: "server_error", message: "The model crashed." },
    });
    (response as { _request_id?: string })._request_id = "req_123";
    const { client } = makeFakeClient({ response });

    const error = await new OpenAIResponsesModel(client, { name: "gpt-5.6" })
      .complete(userHi)
      .catch((thrown: unknown) => thrown);

    expect(error).toBeInstanceOf(ProviderError);
    expect(error).toMatchObject({
      message: "The model crashed.",
      context: {
        code: "server_error",
        requestId: "req_123",
        responseId: "resp_1",
        model: "gpt-5.6",
        provider: "openai",
      },
    });
  });

  it("maps a failed rate_limit_exceeded response to ProviderRateLimitError", async () => {
    const { client } = makeFakeClient({
      response: makeResponse({
        status: "failed",
        error: { code: "rate_limit_exceeded", message: "slow down" },
      }),
    });

    await expect(
      new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi),
    ).rejects.toBeInstanceOf(ProviderRateLimitError);
  });

  it("maps a failed response with no error object to ProviderError", async () => {
    const { client } = makeFakeClient({ response: makeResponse({ status: "failed", error: null }) });

    await expect(
      new OpenAIResponsesModel(client, { name: "gpt-5.6" }).complete(userHi),
    ).rejects.toBeInstanceOf(ProviderError);
  });

  it("wraps a rejected request like the Chat path: rejected parameter and request id kept", async () => {
    const { client } = makeFakeClient({
      error: {
        name: "BadRequestError",
        status: 400,
        code: "unsupported_parameter",
        param: "max_output_tokens",
        message: "Unsupported parameter: max_output_tokens",
        type: "invalid_request_error",
        request_id: "req_abc",
      },
    });

    const error = await new OpenAIResponsesModel(client, { name: "gpt-5.6", maxTokens: 10 })
      .complete(userHi)
      .catch((thrown: unknown) => thrown);

    expect(error).toBeInstanceOf(InvalidRequestError);
    expect(error).toMatchObject({
      context: {
        status: 400,
        code: "unsupported_parameter",
        param: "max_output_tokens",
        requestId: "req_abc",
      },
    });
  });

  it("maps a 429 to ProviderRateLimitError with the retry-after delay", async () => {
    const { client } = makeFakeClient({
      error: { status: 429, code: "rate_limit_exceeded", message: "slow", headers: { "retry-after": "2" } },
    });

    const error = await new OpenAIResponsesModel(client, { name: "gpt-5.6" })
      .complete(userHi)
      .catch((thrown: unknown) => thrown);

    expect(error).toBeInstanceOf(ProviderRateLimitError);
    expect(error).toMatchObject({ retryAfter: 2000 });
  });
});

describe("OpenAIResponsesModel.stream()", () => {
  it("throws a clear 'lands next' error and never calls the API", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse() });
    const model = new OpenAIResponsesModel(client, { name: "gpt-5.6" });

    const consume = async () => {
      for await (const _chunk of model.stream(userHi)) {
        // never reached
      }
    };

    await expect(consume()).rejects.toThrow(/Streaming for api: "responses" is not implemented yet/);
    await expect(consume()).rejects.toBeInstanceOf(InvalidRequestError);
    expect(calls).toHaveLength(0);
  });
});
