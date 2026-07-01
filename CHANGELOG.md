# Changelog — @warlock.js/ai-openai

All notable changes to `@warlock.js/ai-openai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). `@warlock.js/*` packages are released in lockstep — every package shares the same version number, so a version below may list only the changes that affected this package.

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
