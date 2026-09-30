import { InvalidRequestError, type ContentPart, type Message } from "@warlock.js/ai";
import type OpenAI from "openai";
import { stringifyContent, toImageUrl } from "./to-openai-messages";

/** One item of the Responses `input` array. */
export type ResponsesInputItem = OpenAI.Responses.ResponseInputItem;

/** What `toResponsesInput` produces: the `instructions` string plus the `input` items. */
export type ResponsesInputPlan = {
  /** The system prompt, sent as the top-level `instructions` field. */
  instructions?: string;
  input: ResponsesInputItem[];
};

/**
 * Convert vendor-neutral `Message[]` into the Responses request pieces.
 *
 * - The FIRST system message becomes `instructions` (decision 9: it avoids
 *   the `system` vs `developer` role question on reasoning models). Any
 *   later system message stays inline as a `system` role item.
 * - `user` / `assistant` text becomes an `EasyInputMessage` (`{ role,
 *   content }`). Multipart user content becomes `input_text` / `input_image`
 *   parts; images always carry `detail: "auto"` because `ResponseInputImage.detail`
 *   is required by the SDK types.
 * - An assistant message that requested tools becomes its text (when
 *   non-empty) followed by one top-level `function_call` item per call; our
 *   `ModelToolCallRequest.id` is the wire `call_id`. No item `id` is sent:
 *   with `store: false` nothing was persisted for it to refer to.
 * - A `tool` message becomes a top-level `function_call_output` item.
 *
 * PDF and audio parts are rejected with a typed `InvalidRequestError`: the
 * Responses adapter declares those capabilities off, so they only arrive
 * here if a caller bypasses the agent's capability gate.
 *
 * @example
 * toResponsesInput([
 *   { role: "system", content: "Be brief." },
 *   { role: "user", content: "Hi" },
 * ]);
 * // { instructions: "Be brief.", input: [{ role: "user", content: "Hi" }] }
 */
export function toResponsesInput(messages: Message[]): ResponsesInputPlan {
  const input: ResponsesInputItem[] = [];
  let instructions: string | undefined;
  let instructionsTaken = false;

  for (const message of messages) {
    if (message.role === "system") {
      const text = stringifyContent(message.content);

      if (!instructionsTaken) {
        instructionsTaken = true;
        instructions = text === "" ? undefined : text;
        continue;
      }

      input.push({ role: "system", content: text });
      continue;
    }

    if (message.role === "tool") {
      input.push({
        type: "function_call_output",
        call_id: message.toolCallId ?? "",
        output: stringifyContent(message.content),
      });
      continue;
    }

    if (message.role === "assistant" && message.toolCalls && message.toolCalls.length > 0) {
      const text = stringifyContent(message.content);

      if (text !== "") {
        input.push({ role: "assistant", content: text });
      }

      for (const call of message.toolCalls) {
        input.push({
          type: "function_call",
          call_id: call.id,
          name: call.name,
          arguments: JSON.stringify(call.input ?? {}),
        });
      }

      continue;
    }

    if (message.role === "user" && Array.isArray(message.content)) {
      input.push({ role: "user", content: message.content.map(toResponsesContentPart) });
      continue;
    }

    input.push({ role: message.role, content: stringifyContent(message.content) });
  }

  return { ...(instructions !== undefined ? { instructions } : {}), input };
}

/** Map a resolved `ContentPart` to a Responses input content part (text and images only). */
function toResponsesContentPart(
  part: ContentPart,
): OpenAI.Responses.ResponseInputText | OpenAI.Responses.ResponseInputImage {
  if (part.type === "text") {
    return { type: "input_text", text: part.text };
  }

  if (part.type === "image") {
    return { type: "input_image", image_url: toImageUrl(part.source), detail: "auto" };
  }

  throw new InvalidRequestError(
    `OpenAI api: "responses" does not accept ${part.type} input in this adapter; use api: "chat" for ${part.type} attachments.`,
  );
}
