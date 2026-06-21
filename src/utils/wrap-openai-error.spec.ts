import {
  AIError,
  ContentFilterError,
  ContextLengthExceededError,
  InvalidRequestError,
  ProviderAuthError,
  ProviderError,
  ProviderRateLimitError,
  ProviderTimeoutError,
  QuotaExceededError,
} from "@warlock.js/ai";
import OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { wrapOpenAIError } from "./wrap-openai-error";

/**
 * Build a duck-typed object that looks like an `OpenAI.APIError`
 * without going through the real constructor (which wants a Headers
 * instance and a private response body we don't control in tests).
 */
function fakeOpenAIError(shape: {
  status?: number;
  code?: string;
  message?: string;
  headers?: Record<string, string>;
  name?: string;
  requestId?: string;
}): unknown {
  return {
    name: shape.name ?? "APIError",
    status: shape.status,
    code: shape.code,
    message: shape.message ?? "failed",
    type: "api_error",
    headers: shape.headers,
    request_id: shape.requestId,
  };
}

describe("wrapOpenAIError", () => {
  it("passes AIError through untouched", () => {
    const original = new ProviderRateLimitError("slow", { retryAfter: 1000 });

    expect(wrapOpenAIError(original)).toBe(original);
  });

  it("maps 401 to ProviderAuthError", () => {
    const wrapped = wrapOpenAIError(fakeOpenAIError({ status: 401, message: "bad key" }));

    expect(wrapped).toBeInstanceOf(ProviderAuthError);
    expect(wrapped.code).toBe("PROVIDER_AUTH");
    expect(wrapped.message).toBe("bad key");
  });

  it("maps code=invalid_api_key to ProviderAuthError", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 403, code: "invalid_api_key", message: "nope" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderAuthError);
  });

  it("maps 429 to ProviderRateLimitError and parses retry-after", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 429, headers: { "retry-after": "2" } }),
    );

    expect(wrapped).toBeInstanceOf(ProviderRateLimitError);
    expect((wrapped as ProviderRateLimitError).retryAfter).toBe(2000);
  });

  it("accepts Retry-After header (capitalized)", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 429, headers: { "Retry-After": "5" } }),
    );

    expect((wrapped as ProviderRateLimitError).retryAfter).toBe(5000);
  });

  it("leaves retryAfter undefined when header missing", () => {
    const wrapped = wrapOpenAIError(fakeOpenAIError({ status: 429 }));

    expect((wrapped as ProviderRateLimitError).retryAfter).toBeUndefined();
  });

  it("leaves retryAfter undefined when header unparseable", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 429, headers: { "retry-after": "soon" } }),
    );

    expect((wrapped as ProviderRateLimitError).retryAfter).toBeUndefined();
  });

  it("maps rate_limit_exceeded code to ProviderRateLimitError", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 400, code: "rate_limit_exceeded" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderRateLimitError);
  });

  it("maps insufficient_quota to QuotaExceededError (not rate-limit)", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 429, code: "insufficient_quota" }),
    );

    expect(wrapped).toBeInstanceOf(QuotaExceededError);
    expect(wrapped).not.toBeInstanceOf(ProviderRateLimitError);
    expect(wrapped.code).toBe("PROVIDER_QUOTA_EXCEEDED");
  });

  it("insufficient_quota still a ProviderError for broad catches", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 429, code: "insufficient_quota" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderError);
  });

  it("maps context_length_exceeded to ContextLengthExceededError", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({
        status: 400,
        code: "context_length_exceeded",
        message: "too long",
      }),
    );

    expect(wrapped).toBeInstanceOf(ContextLengthExceededError);
    expect(wrapped.code).toBe("CONTEXT_LENGTH_EXCEEDED");
  });

  it("maps content_filter code to ContentFilterError", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 400, code: "content_filter", message: "blocked" }),
    );

    expect(wrapped).toBeInstanceOf(ContentFilterError);
    expect((wrapped as ContentFilterError).reason).toBe("blocked");
  });

  it("maps 400 (generic) to InvalidRequestError", () => {
    const wrapped = wrapOpenAIError(fakeOpenAIError({ status: 400, message: "bad" }));

    expect(wrapped).toBeInstanceOf(InvalidRequestError);
  });

  it("maps 422 to InvalidRequestError", () => {
    const wrapped = wrapOpenAIError(fakeOpenAIError({ status: 422 }));

    expect(wrapped).toBeInstanceOf(InvalidRequestError);
  });

  it("maps 404 to InvalidRequestError", () => {
    const wrapped = wrapOpenAIError(fakeOpenAIError({ status: 404 }));

    expect(wrapped).toBeInstanceOf(InvalidRequestError);
  });

  it("maps 500 to plain ProviderError", () => {
    const wrapped = wrapOpenAIError(fakeOpenAIError({ status: 500, message: "server err" }));

    expect(wrapped).toBeInstanceOf(ProviderError);
    expect(wrapped).not.toBeInstanceOf(InvalidRequestError);
    expect(wrapped.code).toBe("PROVIDER_ERROR");
  });

  it("maps APIConnectionTimeoutError-by-name to ProviderTimeoutError", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ name: "APIConnectionTimeoutError", message: "timeout" }),
    );

    expect(wrapped).toBeInstanceOf(ProviderTimeoutError);
    expect(wrapped.code).toBe("PROVIDER_TIMEOUT");
  });

  it("maps ETIMEDOUT to ProviderTimeoutError", () => {
    const wrapped = wrapOpenAIError({
      code: "ETIMEDOUT",
      message: "socket timeout",
    });

    expect(wrapped).toBeInstanceOf(ProviderTimeoutError);
  });

  it("maps ECONNABORTED to ProviderTimeoutError", () => {
    const wrapped = wrapOpenAIError({ code: "ECONNABORTED", message: "aborted" });

    expect(wrapped).toBeInstanceOf(ProviderTimeoutError);
  });

  it("real APIConnectionTimeoutError instance is categorized as timeout", () => {
    const real = new OpenAI.APIConnectionTimeoutError({ message: "timed out" });

    const wrapped = wrapOpenAIError(real);

    expect(wrapped).toBeInstanceOf(ProviderTimeoutError);
  });

  it("preserves cause", () => {
    const raw = fakeOpenAIError({ status: 500, message: "boom" });

    const wrapped = wrapOpenAIError(raw);

    expect((wrapped as unknown as { cause: unknown }).cause).toBe(raw);
  });

  it("attaches status, code, and type to context", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 429, code: "rate_limit_exceeded" }),
    );

    expect(wrapped.context).toMatchObject({
      status: 429,
      code: "rate_limit_exceeded",
      type: "api_error",
    });
  });

  it("attaches request_id (snake_case) to context as requestId", () => {
    const wrapped = wrapOpenAIError(
      fakeOpenAIError({ status: 500, requestId: "req_abc" }),
    );

    expect(wrapped.context?.requestId).toBe("req_abc");
  });

  it("reads camelCase requestId too", () => {
    const wrapped = wrapOpenAIError({
      status: 500,
      message: "x",
      requestId: "req_camel",
    });

    expect(wrapped.context?.requestId).toBe("req_camel");
  });

  it("wraps non-Error thrown values into a generic ProviderError", () => {
    const wrapped = wrapOpenAIError("random string");

    expect(wrapped).toBeInstanceOf(ProviderError);
    expect(wrapped.message).toBe("random string");
  });

  it("wraps numeric thrown values", () => {
    const wrapped = wrapOpenAIError(42);

    expect(wrapped).toBeInstanceOf(ProviderError);
    expect(wrapped.message).toBe("42");
  });

  it("wraps plain Error without status as ProviderError", () => {
    const raw = new Error("something bad");
    const wrapped = wrapOpenAIError(raw);

    expect(wrapped).toBeInstanceOf(ProviderError);
    expect(wrapped).not.toBeInstanceOf(InvalidRequestError);
    expect(wrapped.message).toBe("something bad");
  });

  it("every wrapped error is an AIError", () => {
    const samples = [
      fakeOpenAIError({ status: 401 }),
      fakeOpenAIError({ status: 429 }),
      fakeOpenAIError({ status: 429, code: "insufficient_quota" }),
      fakeOpenAIError({ status: 400, code: "context_length_exceeded" }),
      fakeOpenAIError({ status: 400, code: "content_filter" }),
      fakeOpenAIError({ status: 400 }),
      fakeOpenAIError({ status: 500 }),
      fakeOpenAIError({ name: "APIConnectionTimeoutError" }),
      "plain string",
      new Error("plain error"),
    ];

    for (const sample of samples) {
      expect(wrapOpenAIError(sample)).toBeInstanceOf(AIError);
    }
  });
});
