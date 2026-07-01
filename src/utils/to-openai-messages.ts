import { InvalidRequestError, type ContentPart, type Message } from "@warlock.js/ai";
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

/**
 * Map a resolved `ContentPart` to an OpenAI chat content part — one
 * branch per modality, each to its real wire shape:
 *
 * - `text` → `{ type: "text" }`.
 * - `image` → `{ type: "image_url" }` (remote URL, or a `data:` URL for
 *   inlined base64 bytes).
 * - `pdf` → `{ type: "file", file: { file_data } }` (OpenAI document
 *   input; base64 only — there is no remote-URL file source).
 * - `audio` → `{ type: "input_audio", input_audio: { data, format } }`
 *   (base64 only; `wav` / `mp3` are the only formats OpenAI accepts).
 *
 * PDF and audio reach this point ONLY when the model declared the
 * matching capability (`openai.model({ name, pdf: true })` /
 * `{ audio: true }`) — the agent's modality gate throws upfront
 * otherwise, so capability and behavior stay in lockstep. A remote-URL
 * pdf/audio source raises a typed `InvalidRequestError` here rather
 * than a downstream provider fault.
 */
function toOpenAIContentPart(part: ContentPart): OpenAI.Chat.Completions.ChatCompletionContentPart {
  if (part.type === "text") {
    return { type: "text", text: part.text };
  }

  if (part.type === "image") {
    const url =
      "url" in part.source
        ? part.source.url
        : `data:${part.source.mediaType};base64,${part.source.base64}`;

    return { type: "image_url", image_url: { url } };
  }

  if (part.type === "pdf") {
    if ("url" in part.source) {
      throw new InvalidRequestError(
        "OpenAI chat completions cannot fetch a remote-URL PDF; supply base64 document bytes instead.",
      );
    }

    return {
      type: "file",
      file: {
        filename: "document.pdf",
        file_data: `data:${part.source.mediaType};base64,${part.source.base64}`,
      },
    };
  }

  // Audio — the remaining `ContentPart` variant.
  if ("url" in part.source) {
    throw new InvalidRequestError(
      "OpenAI chat completions cannot fetch remote-URL audio; supply base64 audio bytes instead.",
    );
  }

  return {
    type: "input_audio",
    input_audio: {
      data: part.source.base64,
      format: toOpenAIAudioFormat(part.source.mediaType),
    },
  };
}

/**
 * Narrow a neutral audio media type to the two formats OpenAI's
 * `input_audio` accepts (`wav` / `mp3`). An unsupported type raises a
 * typed `InvalidRequestError` up front rather than a provider 400.
 */
function toOpenAIAudioFormat(mediaType: string): "wav" | "mp3" {
  if (mediaType === "audio/wav" || mediaType === "audio/x-wav" || mediaType === "audio/wave") {
    return "wav";
  }

  if (mediaType === "audio/mp3" || mediaType === "audio/mpeg" || mediaType === "audio/mpga") {
    return "mp3";
  }

  throw new InvalidRequestError(
    `OpenAI input_audio supports only "wav" and "mp3"; got "${mediaType}".`,
  );
}
