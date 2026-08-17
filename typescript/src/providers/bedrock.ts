/**
 * Amazon Bedrock, via the Converse API.
 *
 * Bedrock is the one provider here that is neither OpenAI-shaped nor
 * token-authenticated, so it gets a full adapter rather than configuration.
 *
 * **Converse, not InvokeModel.** `InvokeModel` passes a body straight through
 * to the underlying vendor, so calling Claude and Llama through Bedrock would
 * need two different request shapes — reintroducing, inside one provider,
 * exactly the problem this package exists to remove. `Converse` is Bedrock's own
 * unified interface: one message shape across every model it hosts. It is the
 * same idea as polygate, one layer down, and building on it means a new Bedrock
 * model works here the day AWS adds it.
 *
 * **Credentials are a pair.** See `splitCredentials` for why they are packed
 * into a single `ACCESS_KEY:SECRET[:TOKEN]` string.
 *
 * **The region is part of the host**, so it is a real option, defaulting to
 * `$AWS_REGION` and then `us-east-1`.
 */
import { execute, resolveKeys } from "../retry";
import { ChatResponse, ProviderChatOptions } from "../types";
import { encodePath, signRequest } from "./sigv4";

export const ENV_VAR = "AWS_ACCESS_KEY_ID";
export const SECRET_ENV = "AWS_SECRET_ACCESS_KEY";
export const SESSION_ENV = "AWS_SESSION_TOKEN";
export const REGION_ENV = "AWS_REGION";

export const DEFAULT_REGION = "us-east-1";
export const SERVICE = "bedrock";

/**
 * Let the standard AWS variables work without packing them by hand.
 *
 * Nobody has `AWS_ACCESS_KEY_ID` holding `id:secret` — they have the two real
 * variables, usually placed there by the AWS CLI or an instance role.
 */
function resolvePair(apiKey: unknown): unknown {
  if (apiKey) return apiKey;
  const access = process.env[ENV_VAR] ?? "";
  if (access.includes(":")) return access;
  const secret = process.env[SECRET_ENV] ?? "";
  if (!access || !secret) return undefined; // let resolveKeys raise its own error
  const token = process.env[SESSION_ENV] ?? "";
  return token ? `${access}:${secret}:${token}` : `${access}:${secret}`;
}

export interface ConverseSplit {
  system: { text: string }[];
  turns: { role: string; content: { text: string }[] }[];
}

/**
 * Split polygate's flat messages into Converse's `system` + `messages`.
 *
 * Converse takes system prompts in their own top-level field and every content
 * as a list of typed blocks. Consecutive same-role messages are *not* merged
 * here: Bedrock rejects them for some models, but silently repairing the
 * caller's conversation would hide a bug in their code behind a quirk of ours.
 */
export function toConverse(
  messages: { role?: string; content?: string }[]
): ConverseSplit {
  const system: { text: string }[] = [];
  const turns: { role: string; content: { text: string }[] }[] = [];
  for (const message of messages) {
    const content = message.content ?? "";
    if (message.role === "system") {
      system.push({ text: content });
    } else {
      turns.push({
        role: message.role === "assistant" ? "assistant" : "user",
        content: [{ text: content }],
      });
    }
  }
  return { system, turns };
}

/**
 * Normalize a Converse response.
 *
 * Content is a list of blocks; only the text ones are joined. A tool-use block
 * has no `text` key at all, and reaching for one would turn a valid
 * tool-calling response into undefined.
 */
export function parse(data: any, fallbackModel: string): ChatResponse {
  const message = data?.output?.message ?? {};
  const blocks: any[] = message.content ?? [];
  const usage = data?.usage ?? {};
  return {
    content: blocks.map((b) => b?.text ?? "").join(""),
    role: message.role ?? "assistant",
    model: fallbackModel,
    provider: "bedrock",
    usage: {
      promptTokens: usage.inputTokens ?? 0,
      completionTokens: usage.outputTokens ?? 0,
      totalTokens: usage.totalTokens ?? 0,
    },
    raw: data,
  };
}

/**
 * One Converse call against Bedrock.
 *
 * `model` is a Bedrock model id or inference profile
 * (`anthropic.claude-3-5-sonnet-20241022-v2:0`), not a vendor model name.
 */
export async function chat(
  options: ProviderChatOptions & { region?: string }
): Promise<ChatResponse> {
  const pool = resolveKeys(resolvePair(options.apiKey) as any, ENV_VAR, "bedrock");
  const region = options.region || process.env[REGION_ENV] || DEFAULT_REGION;
  const host = options.baseUrl
    ? String(options.baseUrl).replace("https://", "").replace(/\/+$/, "")
    : `bedrock-runtime.${region}.amazonaws.com`;

  const { system, turns } = toConverse(options.messages as any[]);
  const payload: Record<string, unknown> = { messages: turns };
  if (system.length) payload.system = system;

  const inference: Record<string, unknown> = {};
  if (options.temperature !== undefined) inference.temperature = options.temperature;
  if (options.maxTokens !== undefined) inference.maxTokens = options.maxTokens;
  if (Object.keys(inference).length) payload.inferenceConfig = inference;

  for (const [key, value] of Object.entries(options)) {
    if (
      ![
        "model", "messages", "apiKey", "temperature", "maxTokens",
        "retry", "provider", "baseUrl", "region",
      ].includes(key)
    ) {
      payload[key] = value;
    }
  }

  // Serialized once and reused: the bytes hashed for the signature must be the
  // bytes sent, so re-stringifying would risk a different key order and an
  // invalid signature.
  const body = JSON.stringify(payload);
  const path = encodePath("model", options.model, "converse");
  const url = `https://${host}${path}`;

  const attempt = (key: string) =>
    fetch(url, {
      method: "POST",
      headers: signRequest({ key, region, service: SERVICE, host, path, body }),
      body,
    });

  const response = await execute("bedrock", pool, attempt, options.retry);
  return parse(await response.json(), options.model);
}
