import type { AIError, FinishReason } from "@warlock.js/ai";
import type OpenAI from "openai";
import { wrapOpenAIError } from "./wrap-openai-error";

/** The slice of a Responses `Response` the status mapper reads. */
export type ResponsesTerminalState = Pick<
  OpenAI.Responses.Response,
  "status" | "incomplete_details"
> & {
  /** True when `output` holds at least one `function_call` item the model completed. */
  hasToolCalls: boolean;
};

/**
 * Derive our `FinishReason` from a Responses terminal state. Responses has
 * no `finish_reason`, so it comes from `status`, `incomplete_details` and the
 * output items (design section 2, decision 5):
 *
 * - `completed` with a function call -> `tool_calls`; without -> `stop`
 *   (a refusal is ordinary text and also `stop`).
 * - `incomplete` / `max_output_tokens` and `max_messages` -> `length`.
 * - `incomplete` / `content_filter` and `steered` (or an unknown / missing
 *   reason) -> `error`. There is no neutral `content_filter` value.
 * - `cancelled`, `queued`, `in_progress` at a terminal read -> `error`.
 * - `failed` is not mapped here: callers throw it via {@link toResponseFailure}.
 *
 * A missing `status` is treated as `completed` (the SDK types it optional).
 */
export function mapResponsesFinishReason(state: ResponsesTerminalState): FinishReason {
  const status = state.status ?? "completed";

  if (status === "completed") {
    return state.hasToolCalls ? "tool_calls" : "stop";
  }

  if (status === "incomplete") {
    const reason = state.incomplete_details?.reason;

    return reason === "max_output_tokens" || reason === "max_messages" ? "length" : "error";
  }

  return "error";
}

/**
 * Turn a `failed` Responses payload into the same typed `AIError` the Chat
 * path throws for a provider failure, by running it through
 * `wrapOpenAIError` (so `rate_limit_exceeded` becomes a
 * `ProviderRateLimitError`, everything else a `ProviderError`, with `code`
 * and request id kept on `context`). The response id and model are added so
 * the failure is traceable.
 */
export function toResponseFailure(
  response: OpenAI.Responses.Response,
  model: string,
): AIError {
  const error = response.error;
  const requestId = (response as { _request_id?: string | null })._request_id;

  const wrapped = wrapOpenAIError({
    code: error?.code,
    message: error?.message ?? "OpenAI response failed without an error object.",
    ...(requestId ? { request_id: requestId } : {}),
  });

  if (wrapped.context) {
    Object.assign(wrapped.context, { provider: "openai", model, responseId: response.id });
  }

  return wrapped;
}
