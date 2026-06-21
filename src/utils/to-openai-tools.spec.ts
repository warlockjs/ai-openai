import type { StandardSchemaV1 } from "@standard-schema/spec";
import type { ToolContract } from "@warlock.js/ai";
import { describe, expect, it } from "vitest";
import { toOpenAITools } from "./to-openai-tools";

const cityInput: StandardSchemaV1<{ city: string }> & { jsonSchema: Record<string, unknown> } = {
  "~standard": {
    version: 1,
    vendor: "test",
    validate: (v) =>
      typeof v === "object" && v !== null && "city" in v
        ? { value: v as { city: string } }
        : { issues: [{ message: "bad" }] },
  },
  jsonSchema: {
    type: "object",
    properties: { city: { type: "string" } },
    required: ["city"],
  },
};

const noJsonSchema: StandardSchemaV1<{ q: string }> = {
  "~standard": {
    version: 1,
    vendor: "test",
    validate: (v) =>
      typeof v === "object" && v !== null && "q" in v
        ? { value: v as { q: string } }
        : { issues: [{ message: "bad" }] },
  },
};

/**
 * A schema whose extraction SUCCEEDS but whose JSON-Schema root is an
 * array, not an object. OpenAI's function `parameters` field demands an
 * object root, so `toParameters` must reject this and degrade to the
 * empty-object schema. This is the regression scenario for the
 * previously-fixed `toParameters` bug (it used to forward whatever
 * `extractJsonSchema` returned, including non-object roots, which the
 * provider rejects with a 400).
 */
const arrayRootInput: StandardSchemaV1<string[]> & {
  jsonSchema: Record<string, unknown>;
} = {
  "~standard": {
    version: 1,
    vendor: "test",
    validate: (v) => ({ value: v as string[] }),
  },
  jsonSchema: { type: "array", items: { type: "string" } },
};

/**
 * Extraction succeeds, but the root object has no `type` key at all —
 * `toParameters`' `schema.type === "object"` guard must still fail and
 * degrade rather than forward a typeless schema to the provider.
 */
const typelessRootInput: StandardSchemaV1<unknown> & {
  jsonSchema: Record<string, unknown>;
} = {
  "~standard": {
    version: 1,
    vendor: "test",
    validate: (v) => ({ value: v }),
  },
  jsonSchema: { properties: { city: { type: "string" } } },
};

function makeTool<T>(
  name: string,
  description: string,
  input: StandardSchemaV1<T>,
): ToolContract<T, unknown> {
  return { name, description, input, execute: async () => null };
}

describe("toOpenAITools", () => {
  it("returns undefined when tools array is undefined", () => {
    expect(toOpenAITools(undefined)).toBeUndefined();
  });

  it("returns undefined when tools array is empty", () => {
    expect(toOpenAITools([])).toBeUndefined();
  });

  it("converts a single tool with extractable JSON Schema", () => {
    const tool = makeTool("getWeather", "Look up weather for a city", cityInput);
    const result = toOpenAITools([tool as ToolContract<unknown, unknown>]);

    expect(result).toEqual([
      {
        type: "function",
        function: {
          name: "getWeather",
          description: "Look up weather for a city",
          parameters: cityInput.jsonSchema,
        },
      },
    ]);
  });

  it("falls back to an empty-object schema when extraction fails", () => {
    const tool = makeTool("search", "Generic search", noJsonSchema);
    const result = toOpenAITools([tool as ToolContract<unknown, unknown>]);

    expect(result?.[0].function.parameters).toEqual({ type: "object", properties: {} });
  });

  it("degrades a non-object (array) root schema to the empty-object schema (toParameters regression)", () => {
    // Guards the prior `toParameters` bug: extraction returns a valid
    // schema, but its root `type` is "array" — OpenAI rejects a non-object
    // `parameters` root with a 400, so the adapter must NOT forward it.
    const tool = makeTool("listCities", "List matching cities", arrayRootInput);
    const result = toOpenAITools([tool as ToolContract<unknown, unknown>]);

    expect(result?.[0].function.parameters).toEqual({ type: "object", properties: {} });
    // The array root must NOT leak through.
    expect(result?.[0].function.parameters).not.toHaveProperty("items");
  });

  it("degrades a typeless root schema (no `type` key) to the empty-object schema", () => {
    const tool = makeTool("noType", "Schema without a type", typelessRootInput);
    const result = toOpenAITools([tool as ToolContract<unknown, unknown>]);

    expect(result?.[0].function.parameters).toEqual({ type: "object", properties: {} });
  });

  it("keeps a valid object-root tool intact while degrading a sibling array-root tool", () => {
    const valid = makeTool("getWeather", "Look up weather", cityInput);
    const broken = makeTool("listCities", "List cities", arrayRootInput);

    const result = toOpenAITools([
      valid as ToolContract<unknown, unknown>,
      broken as ToolContract<unknown, unknown>,
    ]);

    expect(result?.[0].function.parameters).toEqual(cityInput.jsonSchema);
    expect(result?.[1].function.parameters).toEqual({ type: "object", properties: {} });
  });

  it("preserves order across multiple tools", () => {
    const a = makeTool("alpha", "first", cityInput);
    const b = makeTool("beta", "second", cityInput);
    const c = makeTool("gamma", "third", cityInput);

    const result = toOpenAITools([
      a as ToolContract<unknown, unknown>,
      b as ToolContract<unknown, unknown>,
      c as ToolContract<unknown, unknown>,
    ]);

    expect(result?.map((t) => t.function.name)).toEqual(["alpha", "beta", "gamma"]);
  });

  it("preserves name + description verbatim", () => {
    const tool = makeTool("X", "Y", cityInput);
    const result = toOpenAITools([tool as ToolContract<unknown, unknown>]);
    expect(result?.[0].function.name).toBe("X");
    expect(result?.[0].function.description).toBe("Y");
    expect(result?.[0].type).toBe("function");
  });
});
