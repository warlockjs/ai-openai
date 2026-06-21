# @warlock.js/ai-openai

OpenAI adapter for [`@warlock.js/ai`](../ai). Works with any endpoint that speaks the OpenAI Chat Completions protocol — OpenAI proper, Azure OpenAI, OpenRouter, and OpenAI-compatible local gateways.

```bash
npm install @warlock.js/ai @warlock.js/ai-openai @warlock.js/seal openai
```

> `@warlock.js/seal` is the recommended Standard Schema library for tool inputs and structured output. Any Standard Schema V1 library works (Zod, Valibot, …).

## Quick start

```ts
import { OpenAISDK } from "@warlock.js/ai-openai";
import { ai } from "@warlock.js/ai";

const openai = new OpenAISDK({ apiKey: process.env.OPENAI_API_KEY! });

const myAgent = ai.agent({
  model: openai.model({ name: "gpt-4o-mini" }),
});

const result = await myAgent.execute("Hello!");
console.log(result.text);
```

## API surface

```ts
new OpenAISDK(config: OpenAISDKConfig)        // wraps `openai` SDK ClientOptions
  .model(config: OpenAIModelConfig)           // → ModelContract
  .embedder(config: OpenAIEmbedderConfig)     // → EmbedderContract
  .count(text, model?)                        // approximate token count

OpenAIModelConfig {
  name: string;                               // e.g. "gpt-4o-mini", "o3"
  temperature?: number;
  maxTokens?: number;
  vision?: boolean;                           // override auto-inference
  // ...any OpenAI-specific extras passed through to ModelCallOptions
}

OpenAIEmbedderConfig {
  name: string;                               // e.g. "text-embedding-3-small"
  dimensions?: number;                        // optional truncation for supported models
}
```

## Embeddings

```ts
const embedder = openai.embedder({ name: "text-embedding-3-small" });

// Single string — returns EmbeddingResult
const { vector, dimensions, usage } = await embedder.embed("Hello world");

// Batch — returns EmbeddingBatchResult
const { vectors, dimensions, usage } = await embedder.embed(["doc 1", "doc 2", "doc 3"]);
```

`dimensions` on the embedder object starts at `0` and is resolved from the first response's vector length, then cached. Pass `dimensions` in config to request output truncation and set the value immediately:

```ts
openai.embedder({ name: "text-embedding-3-large", dimensions: 256 });
```

## Capabilities

`OpenAIModel` declares:

| Capability         | Default                                                                |
| ------------------ | ---------------------------------------------------------------------- |
| `structuredOutput` | `true` — schemas forwarded as `response_format: json_schema, strict`   |
| `vision`           | Auto-inferred from model name; pass `vision: true \| false` to override |

Vision auto-inference matches model name prefixes: `gpt-4o`, `gpt-4-turbo`, `gpt-4.1`, `o1`, `o3`, `chatgpt-4o`. Unknown models default to `vision: false` so unsupported requests fail with a clear capability error rather than an opaque OpenAI 400.

## OpenAI-compatible endpoints

Pass a `baseURL` for Azure OpenAI / OpenRouter / local servers:

```ts
new OpenAISDK({
  apiKey: process.env.OPENROUTER_API_KEY!,
  baseURL: "https://openrouter.ai/api/v1",
});
```

## Tests

```bash
npm test
```

Covers vision capability inference and message conversion (including multipart user content with image attachments).

## License

MIT
