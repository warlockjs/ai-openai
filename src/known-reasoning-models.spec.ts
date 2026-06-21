import { describe, expect, it } from "vitest";
import { inferReasoningCapability } from "./known-reasoning-models";

describe("inferReasoningCapability()", () => {
  it("returns true for o-series prefixes (o1, o3, o4)", () => {
    expect(inferReasoningCapability("o1")).toBe(true);
    expect(inferReasoningCapability("o1-mini")).toBe(true);
    expect(inferReasoningCapability("o3")).toBe(true);
    expect(inferReasoningCapability("o3-mini")).toBe(true);
    expect(inferReasoningCapability("o4-mini")).toBe(true);
  });

  it("returns true for dated o-series variants", () => {
    expect(inferReasoningCapability("o3-2025-04-16")).toBe(true);
  });

  it("returns true for the gpt-5 family", () => {
    expect(inferReasoningCapability("gpt-5")).toBe(true);
    expect(inferReasoningCapability("gpt-5-pro")).toBe(true);
    expect(inferReasoningCapability("gpt-5-mini")).toBe(true);
  });

  it("is case-insensitive", () => {
    expect(inferReasoningCapability("O3-MINI")).toBe(true);
    expect(inferReasoningCapability("GPT-5-PRO")).toBe(true);
  });

  it("returns false for non-reasoning models", () => {
    expect(inferReasoningCapability("gpt-4o")).toBe(false);
    expect(inferReasoningCapability("gpt-4o-mini")).toBe(false);
    expect(inferReasoningCapability("gpt-4-turbo")).toBe(false);
    expect(inferReasoningCapability("gpt-3.5-turbo")).toBe(false);
  });

  it("returns false for unknown custom models", () => {
    expect(inferReasoningCapability("custom-llm")).toBe(false);
    expect(inferReasoningCapability("")).toBe(false);
  });
});
