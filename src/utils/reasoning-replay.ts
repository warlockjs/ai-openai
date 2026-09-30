import type { ModelToolCallRequest } from "@warlock.js/ai";
import type OpenAI from "openai";

/** A `reasoning` output item as the Responses API returns it. */
export type ResponsesReasoningItem = OpenAI.Responses.ResponseReasoningItem;

/**
 * Key under `ModelToolCallRequest.providerMetadata` owned by the Responses
 * adapter. This name and the shape below are PERSISTED in users' conversation
 * memory, so they are part of the public contract: do not rename.
 */
export const RESPONSES_METADATA_KEY = "openaiResponses";

/**
 * What the Responses adapter stores on the FIRST tool call of an assistant
 * turn so the next request can replay the model's reasoning (design
 * section 4, decisions 2 and 3).
 */
export type ResponsesReplayMetadata = {
  /** Model the items were produced by; replay is skipped when it differs. */
  model: string;
  /** Reasoning items of the turn, in the order the API emitted them. */
  reasoningItems: ResponsesReasoningItem[];
};

/**
 * Keep only the reasoning items that can actually be replayed. With
 * `store: false` the server keeps nothing, so an item without
 * `encrypted_content` has nothing to restore and replaying its bare `id`
 * would be rejected as an unknown item.
 */
export function pickReplayableReasoning(
  items: readonly ResponsesReasoningItem[],
): ResponsesReasoningItem[] {
  return items.filter(
    (item) => typeof item.encrypted_content === "string" && item.encrypted_content !== "",
  );
}

/**
 * Attach the turn's reasoning items to the FIRST tool call only (parallel
 * calls do not each repeat the blobs). Returns the calls unchanged when
 * there is nothing to replay or no tool call to ride on (a text-only turn is
 * deliberately not replayed, decision 2). Does not mutate its input.
 */
export function attachReasoningReplay(
  toolCalls: readonly ModelToolCallRequest[],
  model: string,
  reasoningItems: readonly ResponsesReasoningItem[],
): ModelToolCallRequest[] {
  const replayable = pickReplayableReasoning(reasoningItems);
  const [first, ...rest] = toolCalls;

  if (first === undefined || replayable.length === 0) {
    return [...toolCalls];
  }

  const metadata: ResponsesReplayMetadata = { model, reasoningItems: replayable };

  return [
    { ...first, providerMetadata: { ...first.providerMetadata, [RESPONSES_METADATA_KEY]: metadata } },
    ...rest,
  ];
}

/**
 * Read the stored reasoning items off an assistant turn's tool calls, for
 * replay on `model`. Returns `[]` when the key is absent or malformed, or
 * when the stored `model` differs from `model` (encrypted blobs are not
 * portable across models, design section 4). De-duplicates by item `id`.
 */
export function readReasoningReplay(
  toolCalls: readonly ModelToolCallRequest[],
  model: string,
): ResponsesReasoningItem[] {
  const seen = new Set<string>();
  const items: ResponsesReasoningItem[] = [];

  for (const call of toolCalls) {
    const stored = call.providerMetadata?.[RESPONSES_METADATA_KEY];

    if (!isReplayMetadata(stored) || stored.model !== model) {
      continue;
    }

    for (const item of stored.reasoningItems) {
      if (item?.type !== "reasoning" || seen.has(item.id)) {
        continue;
      }

      seen.add(item.id);
      items.push(item);
    }
  }

  return items;
}

/**
 * Shape a stored reasoning item for the next request. Sends the four fields
 * a stateless replay needs (`type`, `id`, `summary`, `encrypted_content`) and
 * drops the response-only `status` and the optional raw `content`.
 *
 * UNVERIFIED: the SDK types accept the full item as input (`ResponseInputItem`
 * includes `ResponseReasoningItem`) and do not say which fields the API
 * refuses on replay with `store: false`. This is the conservative guess; a
 * live probe settles it.
 */
export function toReplayReasoningItem(item: ResponsesReasoningItem): ResponsesReasoningItem {
  return {
    type: "reasoning",
    id: item.id,
    summary: Array.isArray(item.summary) ? item.summary : [],
    encrypted_content: item.encrypted_content,
  };
}

function isReplayMetadata(value: unknown): value is ResponsesReplayMetadata {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const candidate = value as Partial<ResponsesReplayMetadata>;

  return typeof candidate.model === "string" && Array.isArray(candidate.reasoningItems);
}
