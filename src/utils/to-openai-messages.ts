import type { ContentPart, Message } from "@warlock.js/ai";
import type OpenAI from "openai";

/**
 * Convert vendor-neutral Message[] to OpenAI's chat message shape.
 * Handles the `tool` role (requires `tool_call_id`) and assistant messages
 * that carry `toolCalls` from a prior model response.
 *
 * Multipart `content` (a `ContentPart[]`) is mapped into OpenAI's user-message
 * content-parts shape: text becomes `{ type: "text", text }`, images become
 * `{ type: "image_url", image_url: { url } }` — with base64 sources rendered
 * as `data:` URLs inline.
 *
 * @example
 * const openaiMessages = toOpenAIMessages([
 *   { role: "user", content: "Hi" },
 *   { role: "tool", toolCallId: "call_1", content: '{"ok":true}' },
 * ]);
 *
 * @example
 * toOpenAIMessages([
 *   { role: "user", content: [
 *     { type: "text", text: "What is this?" },
 *     { type: "image", source: { url: "https://example.com/cat.jpg" } },
 *   ]},
 * ]);
 */
export function toOpenAIMessages(
  messages: Message[],
): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
  return messages.map((m) => {
    if (m.role === "tool") {
      return {
        role: "tool",
        content: stringifyContent(m.content),
        tool_call_id: m.toolCallId ?? "",
      };
    }
    if (m.role === "assistant" && m.toolCalls && m.toolCalls.length > 0) {
      return {
        role: "assistant",
        content: stringifyContent(m.content),
        tool_calls: m.toolCalls.map((tc) => ({
          id: tc.id,
          type: "function" as const,
          function: { name: tc.name, arguments: JSON.stringify(tc.input ?? {}) },
        })),
      };
    }

    if (m.role === "user" && Array.isArray(m.content)) {
      return {
        role: "user",
        content: m.content.map(toOpenAIContentPart),
      };
    }

    return { role: m.role, content: stringifyContent(m.content) } as
      | OpenAI.Chat.Completions.ChatCompletionUserMessageParam
      | OpenAI.Chat.Completions.ChatCompletionSystemMessageParam
      | OpenAI.Chat.Completions.ChatCompletionAssistantMessageParam;
  });
}

/**
 * Multipart content is only meaningful on user messages — for any other
 * role (system / assistant text / tool), collapse a `ContentPart[]` to
 * its concatenated text so OpenAI's wire format stays valid. Plain
 * strings pass through unchanged.
 */
function stringifyContent(content: string | ContentPart[]): string {
  if (typeof content === "string") {
    return content;
  }

  return content
    .filter((part): part is { type: "text"; text: string } => part.type === "text")
    .map((part) => part.text)
    .join("");
}

function toOpenAIContentPart(part: ContentPart): OpenAI.Chat.Completions.ChatCompletionContentPart {
  if (part.type === "text") {
    return { type: "text", text: part.text };
  }

  // TODO: Allow other types for urls not just images
  const url =
    "url" in part.source
      ? part.source.url
      : `data:${part.source.mediaType};base64,${part.source.base64}`;

  return { type: "image_url", image_url: { url } };
}
