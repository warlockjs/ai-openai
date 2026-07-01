import { InvalidRequestError, ProviderRateLimitError } from "@warlock.js/ai";
import type { AudioInput } from "@warlock.js/ai";
import type OpenAI from "openai";
import { describe, expect, it } from "vitest";
import { isOpenAITranscriptionModel, OpenAITranscriptionModel } from "./transcription";

type CreateCall = { body: OpenAI.Audio.TranscriptionCreateParams; options: unknown };

function makeFakeClient(options: { response?: unknown; error?: unknown }) {
  const calls: CreateCall[] = [];
  const create = async (body: OpenAI.Audio.TranscriptionCreateParams, opts?: unknown) => {
    calls.push({ body, options: opts });
    if (options.error) throw options.error;
    return options.response;
  };
  const client = { audio: { transcriptions: { create } } } as unknown as OpenAI;
  return { client, calls };
}

const AUDIO: AudioInput = { base64: "QUJD", mediaType: "audio/mpeg", filename: "clip.mp3" };

describe("isOpenAITranscriptionModel", () => {
  it("recognizes the whisper / gpt-4o-transcribe families", () => {
    expect(isOpenAITranscriptionModel("whisper-1")).toBe(true);
    expect(isOpenAITranscriptionModel("gpt-4o-transcribe")).toBe(true);
    expect(isOpenAITranscriptionModel("gpt-4o-mini-transcribe")).toBe(true);
    expect(isOpenAITranscriptionModel("gpt-4o")).toBe(false);
  });
});

describe("OpenAITranscriptionModel", () => {
  it("throws InvalidRequestError for a non-STT model id", () => {
    const { client } = makeFakeClient({});
    expect(() => new OpenAITranscriptionModel(client, { name: "gpt-4o" })).toThrow(
      InvalidRequestError,
    );
  });

  it("whisper-1 defaults to verbose_json and returns text + duration + segments", async () => {
    const { client, calls } = makeFakeClient({
      response: { text: "hello world", duration: 12, segments: [{ text: "hello world", start: 0, end: 1 }] },
    });
    const model = new OpenAITranscriptionModel(client, { name: "whisper-1" });

    const result = await model.transcribe(AUDIO);

    expect(calls[0].body.response_format).toBe("verbose_json");
    expect(result.text).toBe("hello world");
    expect(result.durationSeconds).toBe(12);
    expect(result.segments).toEqual([{ text: "hello world", start: 0, end: 1 }]);
    expect(result.usage).toEqual({ input: 0, output: 0, total: 0 });
  });

  it("gpt-4o-transcribe defaults to json and maps token usage", async () => {
    const { client, calls } = makeFakeClient({
      response: {
        text: "hi",
        usage: { type: "tokens", input_tokens: 100, output_tokens: 20, total_tokens: 120 },
      },
    });
    const model = new OpenAITranscriptionModel(client, { name: "gpt-4o-transcribe" });

    const result = await model.transcribe(AUDIO, { language: "en", prompt: "names: Acme" });

    expect(calls[0].body.response_format).toBe("json");
    expect(calls[0].body).toMatchObject({ language: "en", prompt: "names: Acme" });
    expect(result.text).toBe("hi");
    expect(result.usage).toEqual({ input: 100, output: 20, total: 120 });
  });

  it("wraps provider errors into the typed AIError hierarchy", async () => {
    const { client } = makeFakeClient({ error: { status: 429, message: "rate limited" } });
    const model = new OpenAITranscriptionModel(client, { name: "whisper-1" });
    await expect(model.transcribe(AUDIO)).rejects.toBeInstanceOf(ProviderRateLimitError);
  });
});
