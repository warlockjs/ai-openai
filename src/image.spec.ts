import { InvalidRequestError, ProviderError, ProviderRateLimitError } from "@warlock.js/ai";
import type OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { OpenAIImageModel } from "./image";
import { isOpenAIImageModel } from "./known-image-models";
import { OpenAISDK } from "./sdk";

type GenerateCall = {
  body: OpenAI.Images.ImageGenerateParams;
  requestOptions: { signal?: AbortSignal } | undefined;
};

/**
 * Fake OpenAI client whose `images.generate()` records its arguments
 * and returns a scripted response (or throws). Mirrors the
 * `makeFakeClient` pattern in `embedder.spec.ts` / `model.spec.ts`.
 */
function makeFakeClient(options: { response?: OpenAI.Images.ImagesResponse; error?: unknown }) {
  const calls: GenerateCall[] = [];

  const generate = async (
    body: OpenAI.Images.ImageGenerateParams,
    requestOptions?: { signal?: AbortSignal },
  ) => {
    calls.push({ body, requestOptions });
    if (options.error) throw options.error;
    return options.response;
  };

  const client = { images: { generate } } as unknown as OpenAI;

  return { client, calls };
}

function gptImageResponse(): OpenAI.Images.ImagesResponse {
  return {
    created: 1,
    data: [{ b64_json: "QUJD", revised_prompt: "a tidy red bicycle" }],
    usage: {
      input_tokens: 10,
      output_tokens: 1000,
      total_tokens: 1010,
      input_tokens_details: { text_tokens: 10, image_tokens: 0 },
    },
  } as OpenAI.Images.ImagesResponse;
}

describe("isOpenAIImageModel", () => {
  it("recognizes the gpt-image and dall-e families, rejects chat models", () => {
    expect(isOpenAIImageModel("gpt-image-1")).toBe(true);
    expect(isOpenAIImageModel("gpt-image-1-mini")).toBe(true);
    expect(isOpenAIImageModel("dall-e-3")).toBe(true);
    expect(isOpenAIImageModel("gpt-4o")).toBe(false);
    expect(isOpenAIImageModel("text-embedding-3-small")).toBe(false);
  });
});

describe("OpenAIImageModel — construction guard", () => {
  it("throws InvalidRequestError for a non-image model id", () => {
    const { client } = makeFakeClient({ response: gptImageResponse() });
    expect(() => new OpenAIImageModel(client, { name: "gpt-4o" })).toThrow(InvalidRequestError);
  });

  it("rejects a chat model through the SDK factory too", () => {
    const sdk = new OpenAISDK({ apiKey: "test" });
    expect(() => sdk.image({ name: "gpt-4o" })).toThrow(InvalidRequestError);
  });
});

describe("OpenAIImageModel.generate() — gpt-image", () => {
  it("returns base64 images with token usage and no response_format", async () => {
    const { client, calls } = makeFakeClient({ response: gptImageResponse() });
    const model = new OpenAIImageModel(client, { name: "gpt-image-1" });

    const { images, usage } = await model.generate("a red bicycle", { format: "png" });

    expect(images).toEqual([
      {
        type: "base64",
        base64: "QUJD",
        mediaType: "image/png",
        revisedPrompt: "a tidy red bicycle",
      },
    ]);
    expect(usage).toEqual({ input: 10, output: 1000, total: 1010 });
    // gpt-image rejects `response_format`; we must not send it.
    expect(calls[0].body.response_format).toBeUndefined();
    expect(calls[0].body.output_format).toBe("png");
  });

  it("forwards count / size / quality and the abort signal", async () => {
    const { client, calls } = makeFakeClient({ response: gptImageResponse() });
    const model = new OpenAIImageModel(client, { name: "gpt-image-1" });
    const controller = new AbortController();

    await model.generate("x", {
      count: 2,
      size: "1024x1024",
      quality: "high",
      signal: controller.signal,
    });

    expect(calls[0].body).toMatchObject({ n: 2, size: "1024x1024", quality: "high" });
    expect(calls[0].requestOptions?.signal).toBe(controller.signal);
  });

  it("never sends response_format to a gpt-image model, even if a caller passes one", async () => {
    const { client, calls } = makeFakeClient({ response: gptImageResponse() });
    const model = new OpenAIImageModel(client, { name: "gpt-image-1" });

    // gpt-image rejects response_format with a 400; the `!isGptImage`
    // guard must suppress a caller-supplied value.
    await model.generate("x", { responseFormat: "url" });

    expect(calls[0].body.response_format).toBeUndefined();
  });
});

describe("OpenAIImageModel.generate() — dall-e", () => {
  it("defaults to base64 (b64_json) and reports zero token usage", async () => {
    const { client, calls } = makeFakeClient({
      response: { created: 1, data: [{ b64_json: "REFM" }] } as OpenAI.Images.ImagesResponse,
    });
    const model = new OpenAIImageModel(client, { name: "dall-e-3" });

    const { images, usage } = await model.generate("a cat");

    expect(calls[0].body.response_format).toBe("b64_json");
    expect(images[0]).toEqual({ type: "base64", base64: "REFM", mediaType: "image/png" });
    expect(usage).toEqual({ input: 0, output: 0, total: 0 });
  });

  it("returns a URL image when the caller opts into url mode", async () => {
    const { client, calls } = makeFakeClient({
      response: {
        created: 1,
        data: [{ url: "https://img/cat.png" }],
      } as OpenAI.Images.ImagesResponse,
    });
    const model = new OpenAIImageModel(client, { name: "dall-e-3" });

    const { images } = await model.generate("a cat", { responseFormat: "url" });

    expect(calls[0].body.response_format).toBe("url");
    expect(images[0]).toEqual({ type: "url", url: "https://img/cat.png" });
  });

  it("carries the revised prompt on a URL image", async () => {
    const { client } = makeFakeClient({
      response: {
        created: 1,
        data: [{ url: "https://img/cat.png", revised_prompt: "a fluffy cat" }],
      } as OpenAI.Images.ImagesResponse,
    });
    const model = new OpenAIImageModel(client, { name: "dall-e-3" });

    const { images } = await model.generate("a cat", { responseFormat: "url" });

    expect(images[0]).toEqual({
      type: "url",
      url: "https://img/cat.png",
      revisedPrompt: "a fluffy cat",
    });
  });
});

describe("OpenAIImageModel.generate() — errors", () => {
  it("wraps provider errors into the typed AIError hierarchy", async () => {
    const { client } = makeFakeClient({ error: { status: 429, message: "rate limited" } });
    const model = new OpenAIImageModel(client, { name: "gpt-image-1" });

    await expect(model.generate("x")).rejects.toBeInstanceOf(ProviderRateLimitError);
  });

  it("throws ProviderError when a response carries neither bytes nor url", async () => {
    const { client } = makeFakeClient({
      response: { created: 1, data: [{}] } as OpenAI.Images.ImagesResponse,
    });
    const model = new OpenAIImageModel(client, { name: "gpt-image-1" });

    await expect(model.generate("x")).rejects.toBeInstanceOf(ProviderError);
  });
});
