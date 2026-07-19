import { getProvider } from "./providers";
import { ChatOptions, ChatResponse } from "./types";

/**
 * Send a chat request to any supported LLM provider through one
 * consistent interface.
 *
 * @param options.provider - "anthropic"/"claude", "openai"/"gpt",
 *   "gemini"/"google", or "moonshot"/"kimi".
 * @param options.model - The model name as expected by that provider's
 *   API (e.g. "claude-sonnet-4-6", "gpt-4o", "gemini-2.0-flash",
 *   "kimi-k2-0711-preview").
 * @param options.messages - A list of { role, content } objects using
 *   "system" | "user" | "assistant" roles. Adapters translate this
 *   into whatever shape the underlying provider expects.
 * @param options.apiKey - API key for the request. If omitted, falls
 *   back to the provider's standard environment variable (e.g.
 *   ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY,
 *   MOONSHOT_API_KEY).
 * @param options.temperature - Optional sampling temperature.
 * @param options.maxTokens - Optional max output tokens.
 * @param options - Any additional fields are passed straight through
 *   to the provider's request payload, for anything not normalized
 *   here (e.g. tools, topP, stop sequences).
 *
 * @returns A normalized ChatResponse with .content, .role, .model,
 *   .provider, .usage, and .raw (the untouched provider response).
 */
export async function chat(options: ChatOptions): Promise<ChatResponse> {
  const adapter = getProvider(options.provider);
  return adapter.chat(options);
}
