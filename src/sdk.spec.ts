import { describe, expect, it } from "vitest";
import { OpenAIEmbedder } from "./embedder";
import { OpenAISDK } from "./sdk";

describe("OpenAISDK", () => {
  it("constructs successfully with an apiKey", () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    expect(sdk).toBeInstanceOf(OpenAISDK);
  });

  it("constructs successfully with apiKey + baseURL (OpenAI-compatible endpoints)", () => {
    const sdk = new OpenAISDK({
      apiKey: "test-key",
      baseURL: "https://openrouter.ai/api/v1",
    });
    expect(sdk).toBeInstanceOf(OpenAISDK);
  });

  it("forwards upstream ClientOptions (timeout, maxRetries) to the OpenAI client — C1 regression", () => {
    // Previously only apiKey + baseURL were forwarded, so every other
    // type-checked ClientOptions value was silently dropped.
    const sdk = new OpenAISDK({ apiKey: "test-key", maxRetries: 7, timeout: 1234 });
    const client = (sdk as unknown as { client: { maxRetries: number; timeout: number } }).client;

    expect(client.maxRetries).toBe(7);
    expect(client.timeout).toBe(1234);
  });

  it("model() returns a fresh ModelContract bound to this SDK each call", () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    const a = sdk.model({ name: "gpt-4o-mini" });
    const b = sdk.model({ name: "gpt-4o-mini" });

    expect(a).not.toBe(b); // fresh per call
    expect(a.name).toBe("gpt-4o-mini");
    expect(a.provider).toBe("openai");
  });

  it("model() forwards vision capability inference from the model name", () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    const visionModel = sdk.model({ name: "gpt-4o" });
    const nonVisionModel = sdk.model({ name: "gpt-3.5-turbo" });

    expect(visionModel.capabilities?.vision).toBe(true);
    expect(nonVisionModel.capabilities?.vision).toBe(false);
  });

  it("model() honors explicit vision override over inference", () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    const forced = sdk.model({ name: "gpt-3.5-turbo", vision: true });
    const disabled = sdk.model({ name: "gpt-4o", vision: false });

    expect(forced.capabilities?.vision).toBe(true);
    expect(disabled.capabilities?.vision).toBe(false);
  });

  it("model() always declares structuredOutput=true", () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    expect(sdk.model({ name: "gpt-4o-mini" }).capabilities?.structuredOutput).toBe(true);
    expect(sdk.model({ name: "gpt-3.5-turbo" }).capabilities?.structuredOutput).toBe(true);
  });

  it("count() returns an approximate token count using the core heuristic", async () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    expect(await sdk.count("")).toBe(0);
    expect(await sdk.count("Hello, world!")).toBe(4); // 13 chars / 4 → 4
    expect(await sdk.count("a".repeat(400))).toBe(100);
  });

  it("count() ignores the optional model parameter (current behavior)", async () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    const a = await sdk.count("hello world", "gpt-4o");
    const b = await sdk.count("hello world", "gpt-3.5-turbo");
    expect(a).toBe(b);
  });

  it("model() forwards the SDK-level provider label to produced models", () => {
    const sdk = new OpenAISDK({
      apiKey: "test-key",
      baseURL: "https://openrouter.ai/api/v1",
      provider: "openrouter",
    });

    expect(sdk.model({ name: "gpt-4o-mini" }).provider).toBe("openrouter");
  });
});

describe("OpenAISDK pricing resolution", () => {
  const SDK_PRICING = {
    "gpt-4o-mini": { input: 0.15, output: 0.6, cachedInput: 0.075 },
    "gpt-4o": { input: 2.5, output: 10, cachedInput: 1.25 },
  };

  it("resolves pricing from the SDK-level registry by model name", () => {
    const sdk = new OpenAISDK({ apiKey: "k", pricing: SDK_PRICING });
    const model = sdk.model({ name: "gpt-4o-mini" });

    expect(model.pricing).toEqual({ input: 0.15, output: 0.6, cachedInput: 0.075 });
  });

  it("per-model pricing wins over the SDK-level registry", () => {
    const sdk = new OpenAISDK({ apiKey: "k", pricing: SDK_PRICING });
    const override = { input: 99, output: 99 };
    const model = sdk.model({ name: "gpt-4o-mini", pricing: override });

    expect(model.pricing).toBe(override);
  });

  it("leaves pricing undefined when neither per-model nor registry covers the model", () => {
    const sdk = new OpenAISDK({ apiKey: "k", pricing: SDK_PRICING });
    expect(sdk.model({ name: "unlisted-model" }).pricing).toBeUndefined();
  });

  it("leaves pricing undefined when the SDK has no registry at all", () => {
    const sdk = new OpenAISDK({ apiKey: "k" });
    expect(sdk.model({ name: "gpt-4o-mini" }).pricing).toBeUndefined();
  });
});

describe("OpenAISDK.embedder()", () => {
  it("returns a fresh OpenAIEmbedder per call (not the same reference)", () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    const a = sdk.embedder({ name: "text-embedding-3-small" });
    const b = sdk.embedder({ name: "text-embedding-3-small" });

    expect(a).not.toBe(b);
    expect(a).toBeInstanceOf(OpenAIEmbedder);
    expect(b).toBeInstanceOf(OpenAIEmbedder);
  });

  it("forwards the name config to the embedder's name property", () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    const embedder = sdk.embedder({ name: "text-embedding-3-large" });
    expect(embedder.name).toBe("text-embedding-3-large");
  });

  it("forwards the dimensions override to the embedder's dimensions property", () => {
    const sdk = new OpenAISDK({ apiKey: "test-key" });
    const withDims = sdk.embedder({ name: "text-embedding-3-small", dimensions: 256 });
    const withoutDims = sdk.embedder({ name: "text-embedding-3-small" });

    expect(withDims.dimensions).toBe(256);
    expect(withoutDims.dimensions).toBe(0); // lazy — unresolved until first call
  });
});
