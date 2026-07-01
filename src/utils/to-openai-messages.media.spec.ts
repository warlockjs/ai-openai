import { InvalidRequestError, type Message } from "@warlock.js/ai";
import { describe, expect, it } from "vitest";
import { toOpenAIMessages } from "./to-openai-messages";

/** Pull the mapped content-parts off the single user message. */
function partsOf(content: Message["content"]) {
  const [mapped] = toOpenAIMessages([{ role: "user", content }]);
  return mapped.content as Array<Record<string, unknown>>;
}

describe("toOpenAIMessages — multimodal parts", () => {
  it("maps an inlined image to an image_url data URL", () => {
    const parts = partsOf([
      { type: "image", source: { base64: "QUJD", mediaType: "image/png" } },
    ]);
    expect(parts[0]).toEqual({
      type: "image_url",
      image_url: { url: "data:image/png;base64,QUJD" },
    });
  });

  it("maps a PDF to an OpenAI file part with base64 file_data", () => {
    const parts = partsOf([
      { type: "pdf", source: { base64: "JVBER", mediaType: "application/pdf" } },
    ]);
    expect(parts[0]).toEqual({
      type: "file",
      file: { filename: "document.pdf", file_data: "data:application/pdf;base64,JVBER" },
    });
  });

  it("maps audio to an input_audio part, normalizing the format", () => {
    const mp3 = partsOf([{ type: "audio", source: { base64: "QUJD", mediaType: "audio/mpeg" } }]);
    expect(mp3[0]).toEqual({ type: "input_audio", input_audio: { data: "QUJD", format: "mp3" } });

    const wav = partsOf([{ type: "audio", source: { base64: "QUJD", mediaType: "audio/wav" } }]);
    expect(wav[0]).toEqual({ type: "input_audio", input_audio: { data: "QUJD", format: "wav" } });
  });

  it("throws for an unsupported audio format", () => {
    expect(() =>
      partsOf([{ type: "audio", source: { base64: "QUJD", mediaType: "audio/ogg" } }]),
    ).toThrow(InvalidRequestError);
  });

  it("throws for a remote-URL PDF (no remote file source on chat completions)", () => {
    expect(() =>
      partsOf([{ type: "pdf", source: { url: "https://x/doc.pdf" } }]),
    ).toThrow(InvalidRequestError);
  });

  it("throws for remote-URL audio", () => {
    expect(() =>
      partsOf([{ type: "audio", source: { url: "https://x/clip.mp3" } }]),
    ).toThrow(InvalidRequestError);
  });
});
