import { InvalidRequestError, ProviderRateLimitError } from "@warlock.js/ai";
import type OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { isOpenAISpeechModel, OpenAISpeechModel } from "./speech";

type SpeechCall = { body: OpenAI.Audio.SpeechCreateParams; options: unknown };

function makeFakeClient(options: { bytes?: number[]; error?: unknown }) {
  const calls: SpeechCall[] = [];
  const create = async (body: OpenAI.Audio.SpeechCreateParams, opts?: unknown) => {
    calls.push({ body, options: opts });
    if (options.error) throw options.error;
    const bytes = new Uint8Array(options.bytes ?? [1, 2, 3]);
    return { arrayBuffer: async () => bytes.buffer } as unknown as Response;
  };
  const client = { audio: { speech: { create } } } as unknown as OpenAI;
  return { client, calls };
}

describe("isOpenAISpeechModel", () => {
  it("recognizes the tts families, rejects chat models", () => {
    expect(isOpenAISpeechModel("tts-1")).toBe(true);
    expect(isOpenAISpeechModel("tts-1-hd")).toBe(true);
    expect(isOpenAISpeechModel("gpt-4o-mini-tts")).toBe(true);
    expect(isOpenAISpeechModel("gpt-4o")).toBe(false);
  });
});

describe("OpenAISpeechModel", () => {
  it("throws InvalidRequestError for a non-TTS model id", () => {
    const { client } = makeFakeClient({});
    expect(() => new OpenAISpeechModel(client, { name: "gpt-4o" })).toThrow(InvalidRequestError);
  });

  it("synthesizes base64 audio with the right media type and character count", async () => {
    const { client, calls } = makeFakeClient({ bytes: [65, 66, 67] }); // "ABC"
    const model = new OpenAISpeechModel(client, { name: "tts-1", voice: "alloy" });

    const { audio, characters, usage } = await model.generate("hello", { format: "wav" });

    expect(audio).toEqual({ type: "base64", base64: "QUJD", mediaType: "audio/wav" });
    expect(characters).toBe(5);
    expect(usage).toEqual({ input: 0, output: 0, total: 0 });
    expect(calls[0].body).toMatchObject({
      model: "tts-1",
      input: "hello",
      voice: "alloy",
      response_format: "wav",
    });
  });

  it("forwards voice / speed / instructions and honors the default voice", async () => {
    const { client, calls } = makeFakeClient({});
    const model = new OpenAISpeechModel(client, { name: "gpt-4o-mini-tts", voice: "verse" });

    await model.generate("x", { speed: 1.5, instructions: "calm and slow" });

    expect(calls[0].body).toMatchObject({ voice: "verse", speed: 1.5, instructions: "calm and slow" });
  });

  it("wraps provider errors into the typed AIError hierarchy", async () => {
    const { client } = makeFakeClient({ error: { status: 429, message: "rate limited" } });
    const model = new OpenAISpeechModel(client, { name: "tts-1" });
    await expect(model.generate("x")).rejects.toBeInstanceOf(ProviderRateLimitError);
  });
});
