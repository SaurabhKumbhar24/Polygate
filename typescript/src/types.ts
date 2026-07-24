import type { KeyInput, Retry } from "./retry";

export interface Message {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface Usage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

export interface ChatResponse {
  content: string;
  role: string;
  model: string;
  provider: string;
  usage: Usage;
  raw: unknown;
}

interface ChatOptionsBase {
  model: string;
  messages: Message[];
  /**
   * A single key, a list of keys, or a reusable KeyPool. With more than one,
   * requests rotate round-robin; a rejected key is dropped and a rate-limited
   * one is parked while traffic moves on. Omitted falls back to the provider's
   * environment variable, which may itself hold a comma-separated list.
   */
  apiKey?: KeyInput;
  temperature?: number;
  maxTokens?: number;
  /**
   * Omitted means no retries: one request, one response. `{}` opts into
   * exponential backoff with jitter on transient failures while still failing
   * fast on deterministic ones (400, 404, 422).
   */
  retry?: Retry;
}

// Intersecting (rather than adding an index signature directly to the
// base interface) lets provider-specific extras pass through while
// keeping the known fields' specific types intact. Provider adapters
// don't need "provider" (that's only used for routing), so they take
// ProviderChatOptions; the public chat() function takes ChatOptions.
export type ProviderChatOptions = ChatOptionsBase & Record<string, unknown>;
export type ChatOptions = ChatOptionsBase & { provider: string } & Record<string, unknown>;

export interface ProviderAdapter {
  chat(options: ProviderChatOptions): Promise<ChatResponse>;
}
