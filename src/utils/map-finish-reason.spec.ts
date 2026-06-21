import { describe, expect, it } from "vitest";
import { mapFinishReason } from "./map-finish-reason";

describe("mapFinishReason", () => {
  it("maps known OpenAI finish reasons to the neutral union", () => {
    expect(mapFinishReason("stop")).toBe("stop");
    expect(mapFinishReason("tool_calls")).toBe("tool_calls");
    expect(mapFinishReason("length")).toBe("length");
  });

  it("falls back to 'error' for unknown values", () => {
    expect(mapFinishReason("content_filter")).toBe("error");
    expect(mapFinishReason("function_call")).toBe("error");
    expect(mapFinishReason("anything-else")).toBe("error");
  });

  it("falls back to 'error' for null and undefined", () => {
    expect(mapFinishReason(null)).toBe("error");
    expect(mapFinishReason(undefined)).toBe("error");
  });

  it("falls back to 'error' for an empty string", () => {
    expect(mapFinishReason("")).toBe("error");
  });
});
