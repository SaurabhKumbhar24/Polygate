import { execute, resolveKeys } from "../retry";
import { ChatResponse, ProviderChatOptions } from "../types";

const API_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION = "2023-06-01";
const ENV_VAR = "ANTHROPIC_API_KEY";
const DEFAULT_MAX_TOKENS = 1024;

export async function chat(
  options: ProviderChatOptions
): Promise<ChatResponse> {
  const pool = resolveKeys(options.apiKey, ENV_VAR, "anthropic");

  // Anthropic takes "system" as a top-level field, not a message role.
  let system: string | undefined;
  const formattedMessages = options.messages
    .filter((m) => {
      if (m.role === "system") {
        system = m.content;
        return false;
      }
      return true;
    })
    .map((m) => ({ role: m.role, content: m.content }));

  const payload: Record<string, unknown> = {
    model: options.model,
    messages: formattedMessages,
    max_tokens: options.maxTokens ?? DEFAULT_MAX_TOKENS,
  };
  if (system) payload.system = system;
  if (options.temperature !== undefined) payload.temperature = options.temperature;

  const attempt = (key: string) =>
    fetch(API_URL, {
      method: "POST",
      headers: {
        "x-api-key": key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
      },
      body: JSON.stringify(payload),
    });

  const response = await execute("anthropic", pool, attempt, options.retry);


  const data = await response.json();
  const content = (data.content || [])
    .filter((block: { type: string }) => block.type === "text")
    .map((block: { text: string }) => block.text)
    .join("");

  const usageRaw = data.usage || {};
  return {
    content,
    role: "assistant",
    model: data.model ?? options.model,
    provider: "anthropic",
    usage: {
      promptTokens: usageRaw.input_tokens ?? 0,
      completionTokens: usageRaw.output_tokens ?? 0,
      totalTokens: (usageRaw.input_tokens ?? 0) + (usageRaw.output_tokens ?? 0),
    },
    raw: data,
  };
}
