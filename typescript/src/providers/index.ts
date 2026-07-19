import { UnsupportedProviderError } from "../errors";
import { ProviderAdapter } from "../types";
import * as anthropic from "./anthropic";
import * as gemini from "./gemini";
import * as moonshot from "./moonshot";
import * as openai from "./openai";

// Multiple aliases map to the same adapter so callers can use whichever
// name feels natural (e.g. "claude" or "anthropic", "kimi" or "moonshot").
const REGISTRY: Record<string, ProviderAdapter> = {
  anthropic,
  claude: anthropic,
  openai,
  gpt: openai,
  gemini,
  google: gemini,
  moonshot,
  kimi: moonshot,
};

export const SUPPORTED_PROVIDERS = Object.keys(REGISTRY).sort();

export function getProvider(name: string): ProviderAdapter {
  const key = name.toLowerCase().trim();
  const adapter = REGISTRY[key];
  if (!adapter) {
    throw new UnsupportedProviderError(
      `Unknown provider '${name}'. Supported: ${SUPPORTED_PROVIDERS.join(", ")}`
    );
  }
  return adapter;
}
