# Changelog — @warlock.js/ai-openai

All notable changes to `@warlock.js/ai-openai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `@warlock.js/*` packages are released in lockstep — every package shares the same version number, so a version below may list only the changes that affected this package.

## [Unreleased]

### Fixed

- **Strict structured-output compatibility check is now recursive.** `json_schema` (strict) mode is used only when every object in the schema lists all of its properties in `required` — OpenAI strict has no optional fields, so a schema that omits one anywhere in the tree would `400`. Such schemas (and hand-built ones) now degrade to loose `json_object` instead of failing the call; client-side validation still enforces the full shape.

## 4.3.0 - 2026-06-21

### Added

- Cost-truth wiring (additive, non-breaking):
  - `Usage.reasoningTokens` now populated from `completion_tokens_details.reasoning_tokens` on both `complete()` and the streaming `done` event (o-series / gpt-5 hidden reasoning channel). Emitted only when > 0.
  - `ModelCallOptions.reasoning.effort` mapped to the provider-native `reasoning_effort` request param. Forwarded only for reasoning-capable models; `reasoning.maxTokens` has no Chat Completions equivalent and is ignored.
  - `ModelCapabilities.reasoning` inferred from the model name (`o1*` / `o3*` / `o4*` / `gpt-5*`), overridable via `.model({ name, reasoning })`. New `known-reasoning-models.ts` prefix helper.
  - `ModelCapabilities.promptCaching` advertised as always `true` (OpenAI caches automatically and reports hits via `Usage.cachedTokens`). `ModelCallOptions.cacheControl` write breakpoints are a graceful no-op.

## 4.1.15

- Baseline — per-package changelog tracking starts at this version.
