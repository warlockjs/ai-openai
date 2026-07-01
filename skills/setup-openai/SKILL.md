---
name: setup-openai
description: 'Wire @warlock.js/ai-openai — new OpenAISDK({apiKey, baseURL?, provider?, pricing?}) for OpenAI / Azure / OpenRouter, .model({name, vision?, reasoning?, structuredOutput?, responseFormat?, pdf?, audio?}) for ModelContract, .embedder({name, dimensions?}) for embeddings, .image({name, pricing?}) for gpt-image-*/dall-e-* image generation (via ai.image). PDF input maps to OpenAI file parts (opt-in pdf:true), audio input to input_audio (opt-in audio:true). Triggers: `OpenAISDK`, `.model`, `.embedder`, `.image`, `.embed`, `.embedMany`, `baseURL`, `pricing`, `responseSchema`, `responseFormat`, `reasoning_effort`, `reasoningTokens`, `cachedTokens`, `gpt-image`, `dall-e`, `ai.image`, `pdf input`, `input_audio`, o-series / gpt-5 reasoning, prompt caching; "wire openai into a warlock agent", "configure gpt-4o", "use o3 / gpt-5 reasoning effort", "route through openrouter or azure openai", "openai embeddings with warlock", "generate images with gpt-image / dall-e", "send a pdf / audio to gpt-4o"; typical import `import { OpenAISDK } from "@warlock.js/ai-openai"`. Skip: the ai.image verb surface — `@warlock.js/ai/generate-images/SKILL.md`; agent wiring — `@warlock.js/ai/run-ai-agent/SKILL.md`; adapter comparison — `@warlock.js/ai/pick-ai-provider/SKILL.md`; competing adapters `@warlock.js/ai-anthropic`, `@warlock.js/ai-bedrock`, `@warlock.js/ai-google`, `@warlock.js/ai-ollama`; raw `openai` SDK, Vercel `@ai-sdk/openai`.'
---

# `@warlock.js/ai-openai`

Provider adapter that turns OpenAI Chat Completions into a vendor-neutral `ModelContract`. Pair with `@warlock.js/ai` for the agent / tool / system-prompt surface.

## Construction

```ts
import { OpenAISDK } from "@warlock.js/ai-openai";

const openai = new OpenAISDK({ apiKey: process.env.OPENAI_API_KEY! });

// OpenAI-compatible endpoints (Azure OpenAI, OpenRouter, local gateways).
// Pass `provider` to label the upstream — flows through to
// `ModelContract.provider`, `AgentReport.model.provider`, and logs.
const openrouter = new OpenAISDK({
  apiKey: process.env.OPENROUTER_API_KEY!,
  baseURL: "https://openrouter.ai/api/v1",
  provider: "openrouter",
});
```

`OpenAISDK` is a class (not a factory) — adapter entry points hold a long-lived `OpenAI` client and align with `new OpenAI(...)` upstream convention.

`provider` defaults to `"openai"`. It's an SDK-level identity (one client = one upstream), not a per-model knob.

## Producing a model

```ts
openai.model({ name: "gpt-4o-mini" })                       // common case
openai.model({ name: "gpt-4o", temperature: 0.2 })          // sampling controls
openai.model({ name: "fine-tuned-x", vision: true })        // explicit capability override
```

Returns a `ModelContract` you pass straight into `ai.agent({ model })`.

## Capabilities — what's auto-set

| Flag | Default |
| --- | --- |
| `structuredOutput` | `true`, unless `responseFormat` is set to `"json_object"` or `"text"` (loose modes) — then `false`. |
| `vision` | Inferred from model name. `true` for `gpt-4o*`, `gpt-4-turbo*`, `gpt-4.1*`, `o1*`, `o3*`, `chatgpt-4o*`; `false` otherwise. |
| `reasoning` | Inferred from model name. `true` for the o-series (`o1*`, `o3*`, `o4*`) and the `gpt-5*` family; `false` otherwise. Drives whether `reasoning_effort` is forwarded. |
| `promptCaching` | Always `true`. OpenAI caches long prompt prefixes automatically and reports hits via `usage.cachedTokens` — there are no caller-supplied write breakpoints. |
| `pdf` | `false` by default (opt-in `.model({ pdf: true })`). OpenAI accepts PDF `file` parts only on specific models (the `gpt-4o` family) — honest off until you set it. |
| `audio` | `false` by default (opt-in `.model({ audio: true })`). Only the `gpt-4o-audio-preview` family accepts `input_audio`. |

**Override `vision`, `structuredOutput`, `reasoning`, `pdf`, or `audio` explicitly** via `.model({ name, vision?, structuredOutput?, reasoning?, pdf?, audio? })` — an explicit value always wins over inference.

## Structured output

When the agent passes `responseSchema` (a JSON Schema object), the adapter converts to `response_format: json_schema, strict: true` on the wire — token-level enforcement. Non-object schemas fall back to loose `json_object` mode (still valid JSON, no shape enforcement).

Some targets reject strict `json_schema` (older OpenAI models, OpenRouter routes, Ollama OpenAI-compat). Override the wire mode per model with `responseFormat`:

```ts
openai.model({ name: "legacy-model", responseFormat: "json_object" }) // valid JSON, no strict shape
openai.model({ name: "some-route",   responseFormat: "text" })        // no response_format on the wire
```

`"json_object"` and `"text"` also flip `structuredOutput` to `false`, so the agent re-injects the schema as a soft prompt hint. Pin `structuredOutput` explicitly to override that.

## Multipart messages (image / PDF / audio input)

`ContentPart[]` user content maps per modality to OpenAI's real wire parts:

- `{ type: "text" }` → `{ type: "text", text }`
- `{ type: "image", source: { url } }` → `{ type: "image_url", image_url: { url } }`
- `{ type: "image", source: { base64, mediaType } }` → `{ type: "image_url", image_url: { url: "data:{mediaType};base64,{base64}" } }`
- `{ type: "pdf", source: { base64, mediaType } }` → `{ type: "file", file: { filename, file_data: "data:{mediaType};base64,…" } }` — requires `.model({ pdf: true })`
- `{ type: "audio", source: { base64, mediaType } }` → `{ type: "input_audio", input_audio: { data, format: "wav" | "mp3" } }` — requires `.model({ audio: true })`

PDF and audio reach the wire only when the model declares the matching capability — the agent's modality gate throws otherwise, so capability ≡ behavior. A remote-URL pdf/audio source raises a typed `InvalidRequestError` (OpenAI has no remote file/audio source); an unsupported audio media type does too (only `wav` / `mp3`). The agent prepares attachments before they reach the adapter; this package never reads files itself.

## Image generation (`gpt-image` / DALL·E)

`openai.image({ name })` returns an `ImageModelContract` for the `ai.image()` verb:

```ts
const gpt = openai.image({ name: "gpt-image-1", pricing: { input: 5, output: 40 } }); // token-metered
const dalle = openai.image({ name: "dall-e-3", pricing: { perImage: 0.04 } });        // per-image

const { data } = await ai.image({ model: gpt, prompt: "a red bicycle", size: "1024x1024" });
```

- `gpt-image-*` is token-metered (price with `{ input, output }`) and always returns base64 bytes — the adapter never sends `response_format` (the API rejects it).
- `dall-e-*` is per-image (price with `{ perImage }` / `perImageBySize`); defaults to base64, opt into a hosted URL with `options: { responseFormat: "url" }`.
- A non-image model id (`openai.image({ name: "gpt-4o" })`) throws `InvalidRequestError` at construction. The verb surface (envelope, options, cost-truth) lives in [`@warlock.js/ai/generate-images/SKILL.md`](@warlock.js/ai/generate-images/SKILL.md).

## Streaming

`model.stream()` drains `chat.completions.create({ stream: true })` and yields `{ type: "delta", content }` per token, then — after the stream closes — one consolidated `{ type: "tool-call", id, name, input }` per requested tool, then a terminal `{ type: "done", finishReason, usage }`. `stream_options: { include_usage: true }` is enabled by default.

Tool-call argument fragments arrive split across deltas; the adapter accumulates them per `tool_calls[n].index` and parses the assembled JSON once on completion, so streamed tool calls round-trip identically to non-streaming `complete()`. Accumulators that never received a function name are skipped.

## Embeddings

```ts
const embedder = openai.embedder({ name: "text-embedding-3-small" });

const { vector, dimensions, usage } = await embedder.embed("Hello world");
const { vectors } = await embedder.embedMany(["doc 1", "doc 2", "doc 3"]);
```

`dimensions` is lazy — starts at `0`, populated from the first response. Pass `dimensions` in config to request output truncation (supported by `text-embedding-3-*`):

```ts
openai.embedder({ name: "text-embedding-3-large", dimensions: 256 });
```

## Reasoning (o-series / gpt-5)

Reasoning models accept a discrete effort knob. The agent passes it through `ModelCallOptions.reasoning`:

```ts
const model = openai.model({ name: "o3-mini" });   // reasoning auto-true
await model.complete(messages, { reasoning: { effort: "high" } });  // → reasoning_effort: "high"
```

- `reasoning.effort` (`"low" | "medium" | "high"`) maps verbatim to OpenAI's `reasoning_effort` request param.
- `reasoning.maxTokens` has **no Chat Completions equivalent** (it's the Anthropic extended-thinking budget) and is silently ignored here.
- When `capabilities.reasoning` is `false` (e.g. `gpt-4o`), the option is dropped — the adapter never forwards `reasoning_effort` to a model that would 400 on it. Pin `reasoning: true` to force it for a custom/fine-tuned reasoning model.

## Token usage — what's reported

`usage` on the result carries the neutral channel breakdown, populated from OpenAI's `usage` block:

| `Usage` field | OpenAI source | Notes |
| --- | --- | --- |
| `input` / `output` / `total` | `prompt_tokens` / `completion_tokens` / `total_tokens` | always present (zeroed if OpenAI omits the block). |
| `cachedTokens` | `prompt_tokens_details.cached_tokens` | subset of `input` served from OpenAI's automatic prompt cache. Emitted only when > 0. |
| `reasoningTokens` | `completion_tokens_details.reasoning_tokens` | hidden reasoning channel on o-series / gpt-5 (already counted within `output`). Emitted only when > 0. |

Both `cachedTokens` and `reasoningTokens` are omitted (not set to `0`) when the provider reports zero, so non-reasoning / uncached calls keep the lean `{ input, output, total }` shape. Reported identically on `complete()` and the streaming `done` event.

`cacheControl` write breakpoints (`ModelCallOptions.cacheControl`) are a **no-op** — OpenAI has no caller-driven cache marker on the Chat Completions wire. The read-side `cachedTokens` accounting above works without any caller action.

## Token counting

```ts
await openai.count("some text")  // approximate (heuristic, not tiktoken)
```

Good enough for budgeting; not for billing.

## Pricing — per-model registry

`pricing` is a **registry keyed by model name** — one entry per model, all rates in **USD per 1,000,000 tokens** (`ModelPricing`: `input`, `output`, optional `cachedInput` / `cachedOutput`).

```ts
const openai = new OpenAISDK({
  apiKey,
  pricing: {
    // USD per 1M tokens.
    "gpt-4o-mini": { input: 0.15, output: 0.6, cachedInput: 0.075 },
    "gpt-4o":      { input: 2.5,  output: 10,  cachedInput: 1.25 },
  },
});

const { usage } = await ai.agent({ model: openai.model({ name: "gpt-4o-mini" }) }).execute("hi");
usage.cost;  // { input, output, cachedInput?, cachedOutput? } — per-channel USD breakdown of THIS run
```

The registry is per-model; `usage.cost` is the per-channel breakdown the framework computes from `tokens × pricing[model]`. Resolution at `model()` time: per-model `pricing` (`openai.model({ name, pricing })`) > SDK registry > `undefined` (no cost computed). See [`@warlock.js/ai/pick-ai-provider/SKILL.md`](@warlock.js/ai/pick-ai-provider/SKILL.md).

## Errors

Raw OpenAI SDK errors are wrapped into the typed `@warlock.js/ai` `AIError` hierarchy via the adapter's error wrapper. Dispatch keys on `APIError.status + code` combined — see [`@warlock.js/ai/handle-ai-errors/SKILL.md`](@warlock.js/ai/handle-ai-errors/SKILL.md).

## When NOT to use this skill

- Direct calls to the `openai` SDK without going through `@warlock.js/ai` agents.
- Anthropic models — use `@warlock.js/ai-anthropic`. Bedrock — `@warlock.js/ai-bedrock`. Gemini — `@warlock.js/ai-google`. Ollama — `@warlock.js/ai-ollama`.

## See also

- [`@warlock.js/ai/run-ai-agent/SKILL.md`](@warlock.js/ai/run-ai-agent/SKILL.md) — passing the model into `ai.agent({...})`
- [`@warlock.js/ai/embed-text/SKILL.md`](@warlock.js/ai/embed-text/SKILL.md) — embedder usage
- [`@warlock.js/ai/pick-ai-provider/SKILL.md`](@warlock.js/ai/pick-ai-provider/SKILL.md) — adapter comparison
