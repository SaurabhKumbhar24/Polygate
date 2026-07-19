import { MissingAPIKeyError, ProviderAPIError } from "../errors";
import { ChatResponse, ProviderChatOptions } from "../types";

// Moonshot's API (used to access Kimi K2 and other Moonshot models) is
// OpenAI-compatible, so the request/response shape mirrors OpenAI's.
const API_URL = "https://api.moonshot.ai/v1/chat/completions";
const ENV_VAR = "MOONSHOT_API_KEY";

export async function chat(
  options: ProviderChatOptions
): Promise<ChatResponse> {
  const key = options.apiKey || process.env[ENV_VAR];
  if (!key) {
    throw new MissingAPIKeyError(
      `No Moonshot API key provided. Pass apiKey or set ${ENV_VAR}.`
    );
  }

  const payload: Record<string, unknown> = {
    model: options.model,
    messages: options.messages,
  };
  if (options.temperature !== undefined) payload.temperature = options.temperature;
  if (options.maxTokens !== undefined) payload.max_tokens = options.maxTokens;

  const response = await fetch(API_URL, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${key}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new ProviderAPIError("moonshot", response.status, await response.text());
  }

  const data = await response.json();
  const choice = data.choices[0];
  const usageRaw = data.usage || {};

  return {
    content: choice.message.content,
    role: choice.message.role ?? "assistant",
    model: data.model ?? options.model,
    provider: "moonshot",
    usage: {
      promptTokens: usageRaw.prompt_tokens ?? 0,
      completionTokens: usageRaw.completion_tokens ?? 0,
      totalTokens: usageRaw.total_tokens ?? 0,
    },
    raw: data,
  };
}
