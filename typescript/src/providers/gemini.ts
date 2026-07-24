import { execute, resolveKeys } from "../retry";
import { ChatResponse, Message, ProviderChatOptions } from "../types";

const API_BASE = "https://generativelanguage.googleapis.com/v1beta/models";
const ENV_VAR = "GEMINI_API_KEY";

function toGeminiContents(messages: Message[]): {
  contents: { role: string; parts: { text: string }[] }[];
  systemInstruction?: string;
} {
  const contents: { role: string; parts: { text: string }[] }[] = [];
  let systemInstruction: string | undefined;

  for (const m of messages) {
    if (m.role === "system") {
      systemInstruction = m.content;
      continue;
    }
    // Gemini calls the assistant role "model" instead of "assistant".
    const role = m.role === "assistant" ? "model" : "user";
    contents.push({ role, parts: [{ text: m.content }] });
  }

  return { contents, systemInstruction };
}

export async function chat(
  options: ProviderChatOptions
): Promise<ChatResponse> {
  const pool = resolveKeys(options.apiKey, ENV_VAR, "gemini");

  const { contents, systemInstruction } = toGeminiContents(options.messages);

  const payload: Record<string, unknown> = { contents };
  const generationConfig: Record<string, unknown> = {};
  if (options.temperature !== undefined) generationConfig.temperature = options.temperature;
  if (options.maxTokens !== undefined) generationConfig.maxOutputTokens = options.maxTokens;
  if (Object.keys(generationConfig).length) payload.generationConfig = generationConfig;
  if (systemInstruction) {
    payload.systemInstruction = { parts: [{ text: systemInstruction }] };
  }

  const attempt = (key: string) => {
    // Gemini authenticates via a query parameter rather than a header, so
    // the URL is rebuilt per attempt as the pool rotates keys.
    const url = `${API_BASE}/${options.model}:generateContent?key=${key}`;
    return fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
  };

  const response = await execute("gemini", pool, attempt, options.retry);


  const data = await response.json();
  const candidate = data.candidates[0];
  const content = candidate.content.parts
    .map((p: { text?: string }) => p.text ?? "")
    .join("");
  const usageRaw = data.usageMetadata || {};

  return {
    content,
    role: "assistant",
    model: options.model,
    provider: "gemini",
    usage: {
      promptTokens: usageRaw.promptTokenCount ?? 0,
      completionTokens: usageRaw.candidatesTokenCount ?? 0,
      totalTokens: usageRaw.totalTokenCount ?? 0,
    },
    raw: data,
  };
}
