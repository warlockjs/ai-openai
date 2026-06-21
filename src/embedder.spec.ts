import type OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { OpenAIEmbedder } from "./embedder";

type CreateCall = {
  params: OpenAI.Embeddings.EmbeddingCreateParams;
};

/**
 * Build a fake OpenAI client whose `embeddings.create()` records the
 * params it was called with and returns a scripted response. Mirrors
 * the `makeFakeClient` pattern from `model.spec.ts`.
 */
function makeFakeClient(options: { response: OpenAI.Embeddings.CreateEmbeddingResponse }) {
  const calls: CreateCall[] = [];

  const create = async (params: OpenAI.Embeddings.EmbeddingCreateParams) => {
    calls.push({ params });
    return options.response;
  };

  const client = {
    embeddings: { create },
  } as unknown as OpenAI;

  return { client, calls };
}

function makeResponse(
  embeddings: number[][],
  usage: { prompt_tokens: number; total_tokens: number } = { prompt_tokens: 5, total_tokens: 5 },
): OpenAI.Embeddings.CreateEmbeddingResponse {
  return {
    object: "list",
    model: "text-embedding-3-small",
    data: embeddings.map((embedding, index) => ({
      object: "embedding",
      index,
      embedding,
    })),
    usage,
  };
}

const VECTOR = [0.1, 0.2, 0.3, 0.4];

describe("OpenAIEmbedder.embed() — single input", () => {
  it("returns { vector, dimensions, usage } with correct values", async () => {
    const { client } = makeFakeClient({
      response: makeResponse([VECTOR], { prompt_tokens: 3, total_tokens: 3 }),
    });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    const result = await embedder.embed("Hello world");

    expect(result).toEqual({
      vector: VECTOR,
      dimensions: 4,
      usage: { promptTokens: 3, totalTokens: 3 },
    });
  });

  it("resolves dimensions from response length on first call when no override given", async () => {
    const { client } = makeFakeClient({ response: makeResponse([[1, 2, 3, 4, 5, 6]]) });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    expect(embedder.dimensions).toBe(0);
    await embedder.embed("text");
    expect(embedder.dimensions).toBe(6);
  });

  it("caches dimensions after first resolve and does not re-compute them", async () => {
    // Two scripted responses with different vector lengths. In practice
    // OpenAI always returns the same dimension for a given model+config,
    // but we assert the cache sticks so batches stay consistent even if
    // a provider ever misbehaves.
    let callIndex = 0;
    const client = {
      embeddings: {
        create: async () => {
          const embedding = callIndex++ === 0 ? [1, 2, 3, 4] : [9, 9];
          return makeResponse([embedding]);
        },
      },
    } as unknown as OpenAI;
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    await embedder.embed("first");
    expect(embedder.dimensions).toBe(4);

    await embedder.embed("second");
    expect(embedder.dimensions).toBe(4); // cached from first call, not re-resolved to 2
  });

  it("uses config dimensions as initial value and forwards it in the request", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse([[0.1, 0.2]]) });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small", dimensions: 2 });

    expect(embedder.dimensions).toBe(2);
    await embedder.embed("text");

    expect(calls[0].params.dimensions).toBe(2);
  });

  it("omits dimensions from request payload when not set in config", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse([VECTOR]) });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    await embedder.embed("text");

    expect(calls[0].params.dimensions).toBeUndefined();
  });

  it("maps usage from snake_case to camelCase", async () => {
    const { client } = makeFakeClient({
      response: makeResponse([VECTOR], { prompt_tokens: 7, total_tokens: 7 }),
    });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    const result = await embedder.embed("text");

    expect(result.usage).toEqual({ promptTokens: 7, totalTokens: 7 });
  });

  it("request payload includes model and input", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse([VECTOR]) });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    await embedder.embed("hello");

    expect(calls[0].params.model).toBe("text-embedding-3-small");
    expect(calls[0].params.input).toBe("hello");
  });

  it("provider is 'openai'", () => {
    const { client } = makeFakeClient({ response: makeResponse([VECTOR]) });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });
    expect(embedder.provider).toBe("openai");
  });

  it("wraps client errors into a typed AIError subclass", async () => {
    const err = new Error("API error");
    const client = {
      embeddings: {
        create: async () => {
          throw err;
        },
      },
    } as unknown as OpenAI;
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    await expect(embedder.embed("text")).rejects.toMatchObject({
      name: "ProviderError",
      code: "PROVIDER_ERROR",
      message: "API error",
    });
  });
});

describe("OpenAIEmbedder.embedMany() — batch input", () => {
  it("returns { vectors, dimensions, usage } with vectors.length === 2", async () => {
    const v1 = [0.1, 0.2];
    const v2 = [0.3, 0.4];
    const { client } = makeFakeClient({
      response: makeResponse([v1, v2], { prompt_tokens: 4, total_tokens: 4 }),
    });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    const result = await embedder.embedMany(["a", "b"]);

    expect(result).toEqual({
      vectors: [v1, v2],
      dimensions: 2,
      usage: { promptTokens: 4, totalTokens: 4 },
    });
    expect(result.vectors).toHaveLength(2);
  });

  it("forwards the array as the input field in the request", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse([[0.1], [0.2]]) });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    await embedder.embedMany(["doc 1", "doc 2"]);

    expect(calls[0].params.input).toEqual(["doc 1", "doc 2"]);
  });

  it("forwards configured dimensions in the batch request payload", async () => {
    const { client, calls } = makeFakeClient({ response: makeResponse([[0.1, 0.2], [0.3, 0.4]]) });
    const embedder = new OpenAIEmbedder(client, {
      name: "text-embedding-3-large",
      dimensions: 2,
    });

    await embedder.embedMany(["a", "b"]);

    expect(calls[0].params.dimensions).toBe(2);
  });

  it("resolves dimensions from the first vector's length on a batch call", async () => {
    const { client } = makeFakeClient({
      response: makeResponse([[1, 2, 3], [4, 5, 6]]),
    });
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    const result = await embedder.embedMany(["a", "b"]);

    expect(result.dimensions).toBe(3);
    expect(embedder.dimensions).toBe(3);
  });

  it("wraps client errors thrown during embedMany into a typed AIError", async () => {
    const client = {
      embeddings: {
        create: async () => {
          throw { status: 429, code: "rate_limit_exceeded", message: "slow" };
        },
      },
    } as unknown as OpenAI;
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    await expect(embedder.embedMany(["a", "b"])).rejects.toMatchObject({
      name: "ProviderRateLimitError",
      code: "PROVIDER_RATE_LIMIT",
    });
  });

  it("maps a 401 embedder failure to ProviderAuthError", async () => {
    const client = {
      embeddings: {
        create: async () => {
          throw { status: 401, message: "bad key" };
        },
      },
    } as unknown as OpenAI;
    const embedder = new OpenAIEmbedder(client, { name: "text-embedding-3-small" });

    await expect(embedder.embed("x")).rejects.toMatchObject({
      name: "ProviderAuthError",
      code: "PROVIDER_AUTH",
    });
  });
});
