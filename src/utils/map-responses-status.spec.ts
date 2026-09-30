import { ProviderError, ProviderRateLimitError } from "@warlock.js/ai";
import type OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { mapResponsesFinishReason, toResponseFailure } from "./map-responses-status";

type Status = OpenAI.Responses.ResponseStatus;
type Reason = NonNullable<OpenAI.Responses.Response["incomplete_details"]>["reason"];

describe("mapResponsesFinishReason", () => {
  it.each<[Status, Reason | undefined, boolean, string]>([
    ["completed", undefined, false, "stop"],
    ["completed", undefined, true, "tool_calls"],
    ["incomplete", "max_output_tokens", false, "length"],
    ["incomplete", "max_messages", false, "length"],
    ["incomplete", "content_filter", false, "error"],
    ["incomplete", "steered", false, "error"],
    ["incomplete", undefined, false, "error"],
    ["cancelled", undefined, false, "error"],
    ["queued", undefined, false, "error"],
    ["in_progress", undefined, false, "error"],
  ])("status %s / reason %s / toolCalls %s -> %s", (status, reason, hasToolCalls, expected) => {
    expect(
      mapResponsesFinishReason({
        status,
        incomplete_details: reason === undefined ? null : { reason },
        hasToolCalls,
      }),
    ).toBe(expected);
  });

  it("treats a missing status as completed", () => {
    expect(
      mapResponsesFinishReason({ status: undefined, incomplete_details: null, hasToolCalls: false }),
    ).toBe("stop");
  });
});

describe("toResponseFailure", () => {
  const failed = (code: string, message: string) =>
    ({
      id: "resp_9",
      status: "failed",
      error: { code, message },
    }) as unknown as OpenAI.Responses.Response;

  it("returns ProviderError with code, provider, model and response id on context", () => {
    const error = toResponseFailure(failed("invalid_prompt", "bad prompt"), "gpt-5.6");

    expect(error).toBeInstanceOf(ProviderError);
    expect(error.context).toMatchObject({
      code: "invalid_prompt",
      provider: "openai",
      model: "gpt-5.6",
      responseId: "resp_9",
    });
  });

  it("returns ProviderRateLimitError for rate_limit_exceeded", () => {
    expect(toResponseFailure(failed("rate_limit_exceeded", "slow"), "gpt-5.6")).toBeInstanceOf(
      ProviderRateLimitError,
    );
  });
});
