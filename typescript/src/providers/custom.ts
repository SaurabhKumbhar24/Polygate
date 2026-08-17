/**
 * Any OpenAI-compatible endpoint you host yourself.
 *
 * Ollama, vLLM, LM Studio, llama.cpp's server, TGI, a corporate gateway, a
 * staging proxy — all speak the OpenAI chat-completions format, so none needs
 * its own adapter. What they need is somewhere to put the URL.
 *
 *     polygate.chat({
 *       provider: "custom",
 *       model: "llama3.1:8b",
 *       baseUrl: "http://localhost:11434/v1",
 *       messages: [{ role: "user", content: "hello" }],
 *     })
 *
 * Authentication is optional here, unlike every other provider: a local runtime
 * usually has none, and demanding a key would fail the most common case for a
 * reason that doesn't exist.
 */
import { UnsupportedProviderError } from "../errors";
import { ChatResponse, ProviderChatOptions } from "../types";
import { makeAdapter, resolveUrl } from "./openaiCompatible";

export const ENV_VAR = "CUSTOM_API_KEY";
export const BASE_URL_ENV = "POLYGATE_BASE_URL";

export async function chat(
  options: ProviderChatOptions
): Promise<ChatResponse> {
  const baseUrl =
    (options.baseUrl as string | undefined) ??
    (typeof process !== "undefined" ? process.env?.[BASE_URL_ENV] : undefined);

  if (!baseUrl) {
    // Thrown rather than defaulting to localhost: guessing an endpoint sends
    // the caller's prompt somewhere they did not choose.
    throw new UnsupportedProviderError(
      `provider='custom' needs an endpoint. Pass baseUrl='http://host:port/v1' ` +
        `or set $${BASE_URL_ENV}.`
    );
  }

  const adapter = makeAdapter({
    provider: "custom",
    apiUrl: resolveUrl(baseUrl),
    envVar: ENV_VAR,
    requireKey: false,
  });
  return adapter({ ...options, baseUrl: undefined });
}
