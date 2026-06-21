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

/**
 * Raw-error fields the wrapper reads off an OpenAI SDK error.
 *
 * `APIError` exposes `status`, `code`, `message`, `type`, `headers` —
 * we duck-type because wrapped retries, proxied errors, and custom
 * error subclasses sometimes lose the `instanceof` relationship.
 */
type OpenAIErrorShape = {
  status?: number;
  code?: string | null;
  message?: string;
  type?: string | null;
  headers?: Record<string, string> | undefined;
  name?: string;
};

/**
 * Wrap any thrown value caught inside the OpenAI adapter into the
 * appropriate `@warlock.js/ai` `AIError` subclass.
 *
 * **Dispatch strategy.** Prefers `APIError.code` when present (stable
 * machine identifier across SDK versions), falls back to `status` when
 * `code` is missing (common with proxied deployments that strip the
 * field). Name-based detection (`APIConnectionTimeoutError`) catches
 * transport-layer errors that never produced an HTTP response.
 *
 * `AIError` instances are returned unchanged — callers can pass the
 * error through `try/catch/throw wrap(e)` pipelines without accidental
 * double-wrapping.
 *
 * @example
 * try {
 *   return await this.client.chat.completions.create(...);
 * } catch (thrown) {
 *   throw wrapOpenAIError(thrown);
 * }
 */
export function wrapOpenAIError(thrown: unknown): AIError {
  if (thrown instanceof AIError) {
    return thrown;
  }

  const shape = toShape(thrown);
  const context = buildContext(thrown, shape);
  const message = shape.message ?? (thrown instanceof Error ? thrown.message : String(thrown));

  if (isTimeout(thrown, shape)) {
    return new ProviderTimeoutError(message, { cause: thrown, context });
  }

  if (shape.status === 401 || shape.code === "invalid_api_key") {
    return new ProviderAuthError(message, { cause: thrown, context });
  }

  if (shape.code === "insufficient_quota") {
    return new QuotaExceededError(message, { cause: thrown, context });
  }

  if (shape.status === 429 || shape.code === "rate_limit_exceeded") {
    return new ProviderRateLimitError(message, {
      cause: thrown,
      context,
      retryAfter: parseRetryAfter(shape.headers),
    });
  }

  if (shape.code === "context_length_exceeded") {
    return new ContextLengthExceededError(message, { cause: thrown, context });
  }

  if (shape.code === "content_filter") {
    return new ContentFilterError(message, {
      cause: thrown,
      context,
      reason: message,
    });
  }

  if (typeof shape.status === "number" && shape.status >= 400 && shape.status < 500) {
    return new InvalidRequestError(message, { cause: thrown, context });
  }

  return new ProviderError(message, { cause: thrown, context });
}

/**
 * Read the raw error shape without depending on `instanceof APIError`
 * — some consumers wrap the SDK, and proxies sometimes strip the
 * prototype chain. Duck-typing on the visible fields is resilient to
 * both.
 */
function toShape(thrown: unknown): OpenAIErrorShape {
  if (thrown instanceof OpenAI.APIError) {
    return {
      status: thrown.status,
      code: thrown.code,
      message: thrown.message,
      type: thrown.type,
      headers: thrown.headers as Record<string, string> | undefined,
      name: thrown.name,
    };
  }

  if (typeof thrown === "object" && thrown !== null) {
    const raw = thrown as Record<string, unknown>;

    return {
      status: typeof raw.status === "number" ? raw.status : undefined,
      code: typeof raw.code === "string" ? raw.code : undefined,
      message: typeof raw.message === "string" ? raw.message : undefined,
      type: typeof raw.type === "string" ? raw.type : undefined,
      headers:
        typeof raw.headers === "object" && raw.headers !== null
          ? (raw.headers as Record<string, string>)
          : undefined,
      name: typeof raw.name === "string" ? raw.name : undefined,
    };
  }

  return {};
}

/**
 * Decide whether the thrown value represents a timeout. OpenAI's SDK
 * throws `APIConnectionTimeoutError` for transport-level timeouts, and
 * Node surfaces `ETIMEDOUT` / `ECONNABORTED` on the lower socket
 * layer. Either signal counts.
 */
function isTimeout(thrown: unknown, shape: OpenAIErrorShape): boolean {
  if (thrown instanceof OpenAI.APIConnectionTimeoutError) {
    return true;
  }

  if (shape.name === "APIConnectionTimeoutError") {
    return true;
  }

  if (shape.code === "ETIMEDOUT" || shape.code === "ECONNABORTED") {
    return true;
  }

  return false;
}

/**
 * Attach the raw diagnostic fields to `error.context` so consumers
 * have everything the provider surfaced without each subclass having
 * to redeclare them. Never includes `cause` — that lives on
 * `error.cause`.
 */
function buildContext(
  thrown: unknown,
  shape: OpenAIErrorShape,
): Record<string, unknown> {
  const context: Record<string, unknown> = {};

  if (shape.status !== undefined) {
    context.status = shape.status;
  }

  if (shape.code) {
    context.code = shape.code;
  }

  if (shape.type) {
    context.type = shape.type;
  }

  const requestId = readRequestId(thrown);

  if (requestId) {
    context.requestId = requestId;
  }

  return context;
}

/**
 * OpenAI puts the request id on `APIError.request_id`. Extract
 * defensively — both camel and snake keys exist across SDK versions.
 */
function readRequestId(thrown: unknown): string | undefined {
  if (typeof thrown !== "object" || thrown === null) {
    return undefined;
  }

  const raw = thrown as Record<string, unknown>;

  if (typeof raw.request_id === "string") {
    return raw.request_id;
  }

  if (typeof raw.requestId === "string") {
    return raw.requestId;
  }

  return undefined;
}

/**
 * Parse the `Retry-After` response header (seconds per HTTP spec)
 * into milliseconds so consumers can feed it straight to `setTimeout`.
 * Returns `undefined` when missing or unparseable.
 */
function parseRetryAfter(headers: Record<string, string> | undefined): number | undefined {
  if (!headers) {
    return undefined;
  }

  const raw = headers["retry-after"] ?? headers["Retry-After"];

  if (!raw) {
    return undefined;
  }

  const seconds = Number(raw);

  if (!Number.isFinite(seconds) || seconds < 0) {
    return undefined;
  }

  return Math.round(seconds * 1000);
}
