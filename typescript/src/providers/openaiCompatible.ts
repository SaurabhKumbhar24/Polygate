/**
 * Shared adapter for providers that speak the OpenAI chat-completions format.
 *
 * Most of the market does. OpenAI defined the shape, and Groq, Together,
 * Fireworks, Mistral, xAI, Perplexity, Cerebras, DeepSeek, Moonshot, and every
 * local runtime with an OpenAI-compatible mode accept the same request and
 * return the same response. Only three things differ: the URL, the environment
 * variable holding the key, and the name reported back.
 *
 * So this holds the one implementation and each provider file is configuration.
 * The alternative — a copy of the same thirty lines per provider — was already
 * in place for two of them, and would have become twelve places to fix the next
 * time a response field moved.
 */
import { MissingEndpointError } from "../errors";
import { execute, resolveKeys } from "../retry";
import { ChatResponse, ProviderChatOptions } from "../types";

/** Builds the credential headers for one request. */
export type AuthBuilder = (key: string) => Record<string, string>;

/** The near-universal case: `Authorization: Bearer <key>`. */
export const bearerAuth: AuthBuilder = (key) => ({
  Authorization: `Bearer ${key}`,
});

/**
 * Auth for hosts that carry the key in a bare header, not a Bearer token.
 * Azure OpenAI is the one that matters — a Bearer token there returns a 401
 * indistinguishable from a wrong key.
 */
export function apiKeyHeaderAuth(header = "api-key"): AuthBuilder {
  return (key) => ({ [header]: key });
}

export interface CompatibleConfig {
  provider: string;
  /** Omitted for providers whose URL contains your own account or workspace. */
  apiUrl?: string;
  envVar: string;
  /**
   * Local runtimes authenticate nothing. Demanding a key there would fail the
   * most common case — a laptop running Ollama — for a reason that doesn't exist.
   */
  requireKey?: boolean;
  /**
   * How the key becomes headers. Defaults to Bearer, because
   * "OpenAI-compatible" describes the body, not the handshake.
   */
  auth?: AuthBuilder;
  /** Builds the final URL from the root and the model, for hosts like Azure
   *  that put the deployment in the path. */
  urlBuilder?: (root: string, model: string) => string;
  /**
   * Environment variable holding this provider's endpoint. Each gets its own
   * rather than sharing a global: one variable would mean pointing it at a
   * local Ollama silently redirected every other endpoint-hungry provider.
   */
  baseUrlEnv?: string;
}

/**
 * Accept either a bare host root or a full completions URL.
 *
 * People pass both, and silently posting to the wrong path produces a 404 that
 * reads like a broken provider rather than a mistyped setting.
 */
export function resolveUrl(baseUrl: string): string {
  const url = baseUrl.replace(/\/+$/, "");
  return url.endsWith("/chat/completions") ? url : `${url}/chat/completions`;
}

/**
 * Normalize an OpenAI-shaped body into a ChatResponse.
 *
 * Defensive because "OpenAI-compatible" is a claim, not a standard: several
 * providers omit `usage` on some routes, and reasoning models return an empty
 * `content` with the text in `reasoning_content`. A missing field should cost a
 * zero in a token count, never an exception on a response that arrived.
 */
export function parse(
  data: any,
  provider: string,
  fallbackModel: string
): ChatResponse {
  const choice = data?.choices?.[0] ?? {};
  const message = choice.message ?? {};
  const usageRaw = data?.usage ?? {};

  return {
    content: message.content || message.reasoning_content || "",
    role: message.role ?? "assistant",
    model: data?.model ?? fallbackModel,
    provider,
    usage: {
      promptTokens: usageRaw.prompt_tokens ?? 0,
      completionTokens: usageRaw.completion_tokens ?? 0,
      totalTokens: usageRaw.total_tokens ?? 0,
    },
    raw: data,
  };
}

/** Build a provider module's `chat` from its distinguishing facts. */
export function makeAdapter(config: CompatibleConfig) {
  const {
    provider, apiUrl, envVar, requireKey = true,
    auth = bearerAuth, urlBuilder, baseUrlEnv,
  } = config;

  const chat = async function chat(
    options: ProviderChatOptions
  ): Promise<ChatResponse> {
    const pool = resolveKeys(
      // A placeholder keeps one code path through execute(); local servers
      // ignore the header entirely.
      requireKey ? options.apiKey : (options.apiKey ?? "local"),
      envVar,
      provider
    );

    // A caller-supplied baseUrl wins, so a proxy or regional endpoint needs no
    // new provider.
    const root =
      (options.baseUrl as string | undefined) ||
      (baseUrlEnv ? process.env[baseUrlEnv] : undefined) ||
      apiUrl;
    if (!root) {
      throw new MissingEndpointError(
        `${provider} has no fixed host — its URL contains your own account, ` +
          `workspace or project. Pass baseUrl: ...` +
          (baseUrlEnv ? ` or set $${baseUrlEnv}.` : ".")
      );
    }
    const url = urlBuilder ? urlBuilder(root, options.model) : resolveUrl(root);

    const payload: Record<string, unknown> = {
      model: options.model,
      messages: options.messages,
    };
    if (options.temperature !== undefined) payload.temperature = options.temperature;
    if (options.maxTokens !== undefined) payload.max_tokens = options.maxTokens;

    // Anything the caller passed that isn't a known option rides through to the
    // provider — tools, top_p, stop sequences, response_format.
    for (const [key, value] of Object.entries(options)) {
      if (
        ![
          "model", "messages", "apiKey", "temperature", "maxTokens",
          "retry", "provider", "baseUrl",
        ].includes(key)
      ) {
        payload[key] = value;
      }
    }

    const attempt = (key: string) =>
      fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...auth(key),
        },
        body: JSON.stringify(payload),
      });

    const response = await execute(provider, pool, attempt, options.retry);
    return parse(await response.json(), provider, options.model);
  };

  // True when the provider cannot be called without an explicit endpoint. Read
  // by UIs that need to prompt for one before offering the provider.
  (chat as any).endpointRequired = apiUrl === undefined;
  return chat;
}
