# Changelog — @warlock.js/ai-openai

All notable changes to `@warlock.js/ai-openai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `@warlock.js/*` packages are released in lockstep — every package shares the same version number, so a version below may list only the changes that affected this package.

## 5.2.3 - 2026-09-02

### Fixed

- Released in exact lockstep with Core's Web generator repairs so every family dependency remains installable at 5.2.3.

## 5.2.2

### Maintenance

- Restored the Warlock family to one exact, installable lockstep version.

## 5.1.0

No changes to `@warlock.js/ai-openai`. Released in lockstep with the `@warlock.js/web`
React-execution fix and the `@warlock.js/core` CLI additions — see those packages'
changelogs.

## 5.0.2 - 2026-08-25

No changes to `@warlock.js/ai-openai`. Released in lockstep with the `@warlock.js/web` SSR
fix (`ssr.noExternal`) — see that package's changelog.

## 5.0.1 - 2026-08-25

No changes to `@warlock.js/ai-openai`. Released in lockstep with the `create-warlock` vite
resolution pin and the `@warlock.js/web` peer narrowing — see those packages'
changelogs.

## 5.0.0 - 2026-08-25

### Changed

- This package is unchanged in 5.0.0; its version moved only because the Warlock family releases in lockstep.

## 4.15.0

### Changed

- **`openai` moves from `^6.34.0` to `^7.4.0`.** The runtime was unaffected: the full suite — 13 files / 208 tests — passed on 7.4.0 *before* any of the type fixes below were made, so nothing about the wire shape this adapter sends or the responses it reads changed across the major. Every fix in this release is a compile-time one.
- **`OpenAI.Images.ImageGenerateParamsBase` is no longer reachable upstream** — openai 7 split image generation into `ImageGenerateParamsNonStreaming` / `ImageGenerateParamsStreaming` and stopped re-exporting the shared `Base` interface from the `Images` namespace. `image.ts` now sources its `quality` / `output_format` / `background` value types from `ImageGenerateParamsNonStreaming`, which is what the request body was already typed as and what the non-streaming `images.generate` overload accepts. The three fields are inherited from `Base` unchanged, so the accepted values are identical.
- **`ChatCompletionTool` became a union** (`ChatCompletionFunctionTool | ChatCompletionCustomTool`) now that Chat Completions carries custom tools. `.function` is no longer reachable without the `type` discriminant, so the tool-conversion specs narrow on `type === "function"` and throw on anything else — a custom-tool regression fails loudly rather than silently skipping the assertion it used to make.

### Fixed

- **`OpenAISDKConfig.provider` is a usable `string` again.** openai 7 added its own `provider?: Provider` key to `ClientOptions` — an opaque branded object minted by `createProvider()` — and our intersection collapsed the field to `Provider & string`, a type no string literal can inhabit. The config now omits the upstream key (`Omit<ClientOptions, "provider">`) before declaring its own label. No behavior change: the constructor already peeled `provider` off and never forwarded it to the OpenAI client.

## 4.14.0

### Removed

- **BREAKING** — `isOpenAIImageModel()` and `OPENAI_IMAGE_MODEL_PREFIXES` are no longer exported; the `known-image-models` module is deleted. Nothing in the adapter read them once the construction-time gate went away, so they were a public list of model ids that enforced nothing and went stale on OpenAI's release schedule, not this package's. Import them from nowhere — branch on your own id list if you need one.
- **BREAKING** — `isOpenAISpeechModel()` and `isOpenAITranscriptionModel()` are no longer exported either, for the same reason. Once their construction-time gates went away nothing in the adapter read them, leaving two more public model-id lists that enforced nothing. No `@warlock.js/ai-*` adapter validates a model id locally, so the package no longer ships a helper that implies otherwise — branch on your own id list if you need one.

### Changed

- `openai.image({ name })`, `openai.speech({ name })` and `openai.transcribe({ name })` no longer reject an unknown model id at construction — the id is forwarded to OpenAI as given, so an id OpenAI does not serve now fails as a typed provider error instead of a local `InvalidRequestError`.

## 4.12.0

### Changed

- Declares its own test runner and pins it to an exact version (`vitest@4.1.10`). The package is its own repository, so a runner resolved from a workspace root it may not be cloned with is a runner it cannot rely on. The pin is exact rather than a range because the version moved underneath the suite mid-development on an unrelated install — a suite whose runner can change without anyone choosing it proves less than it appears to

## 4.9.0 - 2026-08-06

### Fixed

- `reasoning_effort` now defaults to `"none"` automatically on a reasoning-capable model called WITH `tools` and no explicit `reasoning.effort` — previously this required every call site to opt in (added in 4.8.0), so any agent/model config that didn't know to pass it kept hitting rejected tool calls (empty replies, or a hard 400 on newer model generations — `"Function tools with reasoning_effort are not supported ... in /v1/chat/completions"`). An explicit `reasoning.effort` still overrides the default in either direction; calls with no `tools` are unaffected.

## 4.8.0 - 2026-07-19

### Added

- **`reasoning: { effort: "none" }`** now emits `reasoning_effort: "none"` on the wire — unblocks function tools on gpt-5 / o-series reasoning models, which otherwise reject tools on Chat Completions while reasoning is active and return empty replies.

## 4.6.0

### Added

- **`openai.image({ name })`** — image generation for the `gpt-image-*` (token-metered) and `dall-e-*` (per-image) families, for use with `ai.image()`. A non-image model id is rejected at construction.
- **PDF + audio input.** `pdf` and `audio` content parts now map to OpenAI `file` (base64 `file_data`) and `input_audio` (`wav` / `mp3`) parts — opt in with `model({ pdf: true })` / `{ audio: true }`. A remote-URL pdf/audio source raises a typed `InvalidRequestError` up front.
- **`openai.speech({ name })`** — text-to-speech for the `tts-1` / `tts-1-hd` / `gpt-4o-mini-tts` families (`audio.speech.create`), for use with `ai.speech()`.
- **`openai.transcribe({ name })`** — speech-to-text for the `whisper-1` / `gpt-4o-transcribe` families (`audio.transcriptions.create`), for use with `ai.transcribe()`. `whisper-1` defaults to `verbose_json` (duration + segments); a non-TTS/STT model id is rejected at construction.

### Fixed

- **Non-text content parts are no longer coerced to `image_url`.** The message mapper now branches per modality (image → `image_url`, pdf → `file`, audio → `input_audio`) instead of forcing every attachment through the image path.

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
