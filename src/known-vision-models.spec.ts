import { describe, expect, it } from "vitest";
import { inferVisionCapability } from "./known-vision-models";

describe("inferVisionCapability", () => {
  it("recognizes gpt-4o family", () => {
    expect(inferVisionCapability("gpt-4o")).toBe(true);
    expect(inferVisionCapability("gpt-4o-mini")).toBe(true);
    expect(inferVisionCapability("gpt-4o-2024-08-06")).toBe(true);
    expect(inferVisionCapability("gpt-4o-mini-2024-07-18")).toBe(true);
  });

  it("recognizes gpt-4-turbo and dated variants", () => {
    expect(inferVisionCapability("gpt-4-turbo")).toBe(true);
    expect(inferVisionCapability("gpt-4-turbo-preview")).toBe(true);
    expect(inferVisionCapability("gpt-4-turbo-2024-04-09")).toBe(true);
  });

  it("recognizes gpt-4.1 family", () => {
    expect(inferVisionCapability("gpt-4.1")).toBe(true);
    expect(inferVisionCapability("gpt-4.1-mini")).toBe(true);
  });

  it("recognizes o1 and o3 reasoning models", () => {
    expect(inferVisionCapability("o1")).toBe(true);
    expect(inferVisionCapability("o1-preview")).toBe(true);
    expect(inferVisionCapability("o3")).toBe(true);
    expect(inferVisionCapability("o3-mini")).toBe(true);
  });

  it("recognizes chatgpt-4o", () => {
    expect(inferVisionCapability("chatgpt-4o-latest")).toBe(true);
  });

  it("rejects gpt-3.5-turbo", () => {
    expect(inferVisionCapability("gpt-3.5-turbo")).toBe(false);
    expect(inferVisionCapability("gpt-3.5-turbo-1106")).toBe(false);
  });

  it("rejects unknown models", () => {
    expect(inferVisionCapability("custom-llm")).toBe(false);
    expect(inferVisionCapability("fine-tuned-7b")).toBe(false);
    expect(inferVisionCapability("davinci-002")).toBe(false);
  });

  it("is case-insensitive", () => {
    expect(inferVisionCapability("GPT-4O-MINI")).toBe(true);
    expect(inferVisionCapability("Gpt-4-Turbo")).toBe(true);
  });

  it("does not match arbitrary substrings", () => {
    // Prefix-only — "anything-gpt-4o" should not match
    expect(inferVisionCapability("custom-gpt-4o")).toBe(false);
    expect(inferVisionCapability("my-o1-fork")).toBe(false);
  });
});
