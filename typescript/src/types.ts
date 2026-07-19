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
  apiKey?: string;
  temperature?: number;
  maxTokens?: number;
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
