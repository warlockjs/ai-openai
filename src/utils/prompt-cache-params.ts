import type { OpenAIModelConfig } from "../config.type";

/** The OpenAI-only prompt-cache request fields common to Chat Completions and Responses. */
export type PromptCacheParams = {
  prompt_cache_key?: string;
  prompt_cache_retention?: NonNullable<OpenAIModelConfig["promptCacheRetention"]>;
};

/**
 * Build the OpenAI-only prompt-cache request fields from the model config.
 * Returns nothing for any provider label other than `"openai"`: wrapper
 * endpoints use the same transport but may reject these OpenAI-only fields.
 */
export function buildPromptCacheParams(
  provider: string,
  config: Pick<OpenAIModelConfig, "promptCacheKey" | "promptCacheRetention">,
): PromptCacheParams {
  if (provider !== "openai") {
    return {};
  }

  return {
    ...(config.promptCacheKey !== undefined ? { prompt_cache_key: config.promptCacheKey } : {}),
    ...(config.promptCacheRetention !== undefined
      ? { prompt_cache_retention: config.promptCacheRetention }
      : {}),
  };
}
