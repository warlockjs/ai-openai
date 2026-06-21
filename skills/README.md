# `@warlock.js/ai-openai` — skills index

Per-task skills. All cross-references use the form `@warlock.js/<pkg>/<skill>/SKILL.md`.

## Skills

### [`setup-openai/`](./setup-openai/SKILL.md)

Wire @warlock.js/ai-openai — new OpenAISDK({apiKey, baseURL?, provider?, pricing?}) for OpenAI / Azure / OpenRouter, .model({name, vision?, reasoning?, structuredOutput?, responseFormat?}) for ModelContract, .embedder({name, dimensions?}) for embeddings. Covers o-series / gpt-5 reasoning (reasoning_effort), automatic prompt caching with cost-truth usage (cachedTokens/reasoningTokens), and the per-model pricing registry. Load when wiring an OpenAI-backed model into a @warlock.js agent or routing via an OpenAI-compatible gateway.
