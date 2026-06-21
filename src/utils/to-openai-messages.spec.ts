import type { Message } from "@warlock.js/ai";
import { describe, expect, it } from "vitest";
import { toOpenAIMessages } from "./to-openai-messages";

describe("toOpenAIMessages", () => {
  it("passes plain text user messages through unchanged", () => {
    const messages: Message[] = [{ role: "user", content: "Hi" }];

    expect(toOpenAIMessages(messages)).toEqual([{ role: "user", content: "Hi" }]);
  });

  it("preserves system and assistant text messages", () => {
    const messages: Message[] = [
      { role: "system", content: "Be concise." },
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello!" },
    ];

    expect(toOpenAIMessages(messages)).toEqual([
      { role: "system", content: "Be concise." },
      { role: "user", content: "Hi" },
      { role: "assistant", content: "Hello!" },
    ]);
  });

  it("converts tool messages to OpenAI's tool role with tool_call_id", () => {
    const messages: Message[] = [{ role: "tool", toolCallId: "call_1", content: '{"ok":true}' }];

    expect(toOpenAIMessages(messages)).toEqual([
      { role: "tool", content: '{"ok":true}', tool_call_id: "call_1" },
    ]);
  });

  it("falls back to empty tool_call_id when missing", () => {
    const messages: Message[] = [{ role: "tool", content: "{}" }];

    expect(toOpenAIMessages(messages)).toEqual([{ role: "tool", content: "{}", tool_call_id: "" }]);
  });

  it("emits assistant messages carrying tool calls into OpenAI's tool_calls shape", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "call_1", name: "getWeather", input: { city: "Cairo" } }],
      },
    ];

    expect(toOpenAIMessages(messages)).toEqual([
      {
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "getWeather", arguments: '{"city":"Cairo"}' },
          },
        ],
      },
    ]);
  });

  it("converts a multipart user message to OpenAI's content-parts shape", () => {
    const messages: Message[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          { type: "image", source: { url: "https://example.com/cat.jpg" } },
        ],
      },
    ];

    expect(toOpenAIMessages(messages)).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "What is this?" },
          { type: "image_url", image_url: { url: "https://example.com/cat.jpg" } },
        ],
      },
    ]);
  });

  it("renders inline base64 image sources as data URLs", () => {
    const messages: Message[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "Compare" },
          { type: "image", source: { base64: "iVBORw0KGgo=", mediaType: "image/png" } },
        ],
      },
    ];

    expect(toOpenAIMessages(messages)).toEqual([
      {
        role: "user",
        content: [
          { type: "text", text: "Compare" },
          {
            type: "image_url",
            image_url: { url: "data:image/png;base64,iVBORw0KGgo=" },
          },
        ],
      },
    ]);
  });

  it("collapses multipart content to text on non-user roles", () => {
    // Defensive: shouldn't normally happen, but parts on system/assistant/tool
    // collapse to concatenated text rather than producing an invalid wire shape.
    const messages: Message[] = [
      {
        role: "system",
        content: [
          { type: "text", text: "Hello " },
          { type: "text", text: "world" },
        ],
      },
    ];

    expect(toOpenAIMessages(messages)).toEqual([{ role: "system", content: "Hello world" }]);
  });

  it("drops non-text parts when collapsing multipart content on a non-user role", () => {
    // Only text parts survive the collapse — image parts on a system role
    // are silently filtered, never rendered as a data/url string.
    const messages: Message[] = [
      {
        role: "system",
        content: [
          { type: "text", text: "Look: " },
          { type: "image", source: { url: "https://example.com/a.png" } },
          { type: "text", text: "done" },
        ],
      },
    ];

    expect(toOpenAIMessages(messages)).toEqual([{ role: "system", content: "Look: done" }]);
  });

  it("treats an assistant message with an empty toolCalls array as plain text", () => {
    // `m.toolCalls.length > 0` guards the tool_calls branch — an empty
    // array falls through to the plain assistant-text shape (no tool_calls
    // key on the wire).
    const messages: Message[] = [{ role: "assistant", content: "done", toolCalls: [] }];

    expect(toOpenAIMessages(messages)).toEqual([{ role: "assistant", content: "done" }]);
  });

  it("collapses multipart assistant content to text when tool calls are present", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: [
          { type: "text", text: "calling " },
          { type: "text", text: "tool" },
        ],
        toolCalls: [{ id: "call_1", name: "go", input: { a: 1 } }],
      },
    ];

    expect(toOpenAIMessages(messages)).toEqual([
      {
        role: "assistant",
        content: "calling tool",
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "go", arguments: '{"a":1}' } },
        ],
      },
    ]);
  });

  it("serializes assistant tool-call input as '{}' when input is undefined", () => {
    const messages: Message[] = [
      {
        role: "assistant",
        content: "",
        toolCalls: [{ id: "call_1", name: "noArgs", input: undefined }],
      },
    ];

    expect(toOpenAIMessages(messages)).toEqual([
      {
        role: "assistant",
        content: "",
        tool_calls: [
          { id: "call_1", type: "function", function: { name: "noArgs", arguments: "{}" } },
        ],
      },
    ]);
  });

  it("stringifies multipart tool message content to its concatenated text", () => {
    const messages: Message[] = [
      {
        role: "tool",
        toolCallId: "call_1",
        content: [
          { type: "text", text: '{"ok":' },
          { type: "text", text: "true}" },
        ],
      },
    ];

    expect(toOpenAIMessages(messages)).toEqual([
      { role: "tool", content: '{"ok":true}', tool_call_id: "call_1" },
    ]);
  });

  it("maps an image-only multipart user message to a single image_url part", () => {
    const messages: Message[] = [
      {
        role: "user",
        content: [{ type: "image", source: { url: "https://example.com/only.jpg" } }],
      },
    ];

    expect(toOpenAIMessages(messages)).toEqual([
      {
        role: "user",
        content: [{ type: "image_url", image_url: { url: "https://example.com/only.jpg" } }],
      },
    ]);
  });
});
