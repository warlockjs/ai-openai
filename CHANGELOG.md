# Changelog — @warlock.js/ai-openai

All notable changes to `@warlock.js/ai-openai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `@warlock.js/*` packages are released in lockstep — every package shares the same version number, so a version below may list only the changes that affected this package.

## 4.5.0 - 2026-07-01

### Fixed

- **All upstream `ClientOptions` now reach the OpenAI client.** The SDK constructor peels off the framework-only `provider` / `pricing` keys and forwards the rest (`timeout`, `maxRetries`, `defaultHeaders`, custom `fetch`, `organization`, `project`, …) verbatim, instead of dropping everything but `apiKey` / `baseURL`.

## 4.4.0 - 2026-06-21

### Fixed

- **Strict structured-output compatibility check is now recursive.** A schema that omits a `required` property anywhere in the tree degrades to loose `json_object` instead of `400`-ing; client-side validation still enforces the full shape.

## 4.3.0 - 2026-06-21

### Added

- `Usage.reasoningTokens` is populated from `completion_tokens_details.reasoning_tokens` (o-series / gpt-5 hidden reasoning channel), emitted only when `> 0`.
- `ModelCallOptions.reasoning.effort` maps to the native `reasoning_effort` param for reasoning-capable models; `reasoning.maxTokens` has no Chat Completions equivalent.
- `ModelCapabilities.reasoning` is inferred from the model name (overridable via `.model(...)`); `promptCaching` is always `true` (OpenAI caches automatically), and `cacheControl` write breakpoints are a no-op.

## 4.1.15

- Baseline — per-package changelog tracking starts at this version.
