import { UnsupportedProviderError } from "../errors";
import { ProviderAdapter } from "../types";
import * as anthropic from "./anthropic";
import * as azure from "./azure";
import * as baseten from "./baseten";
import * as bedrock from "./bedrock";
import * as cerebras from "./cerebras";
import * as cloudflare from "./cloudflare";
import * as custom from "./custom";
import * as databricks from "./databricks";
import * as deepinfra from "./deepinfra";
import * as deepseek from "./deepseek";
import * as fireworks from "./fireworks";
import * as gemini from "./gemini";
import * as groq from "./groq";
import * as hyperbolic from "./hyperbolic";
import * as mistral from "./mistral";
import * as moonshot from "./moonshot";
import * as nebius from "./nebius";
import * as novita from "./novita";
import * as openai from "./openai";
import * as openrouter from "./openrouter";
import * as perplexity from "./perplexity";
import * as sambanova from "./sambanova";
import * as together from "./together";
import * as vercel from "./vercel";
import * as vertex from "./vertex";
import * as xai from "./xai";
import * as zai from "./zai";

// Multiple aliases map to the same adapter so callers can use whichever
// name feels natural (e.g. "claude" or "anthropic", "kimi" or "moonshot").
//
// Aliases are also where a model family's popular name lives: people reach for
// "grok" far more often than "xai", and being told that is an unknown provider
// is a bad first minute with a library whose whole job is to hide this.
const REGISTRY: Record<string, ProviderAdapter> = {
  anthropic,
  claude: anthropic,
  openai,
  gpt: openai,
  gemini,
  google: gemini,
  moonshot,
  kimi: moonshot,
  groq,
  together,
  togetherai: together,
  fireworks,
  mistral,
  xai,
  grok: xai,
  perplexity,
  pplx: perplexity,
  sonar: perplexity,
  cerebras,
  deepseek,
  zai,
  // The dot is how the brand is written, and the lower/trim below means a
  // caller who types it that way is not punished for it.
  "z.ai": zai,
  glm: zai,
  // Zhipu is the company behind GLM and the name the mainland platform ships
  // under. Same wire format, different host — see zai.CHINA_BASE_URL.
  zhipu: zai,
  bigmodel: zai,
  // ── Clouds ────────────────────────────────────────────────────────────
  // Not token-in-a-header like everything above: Bedrock signs every request,
  // Azure uses its own header and puts the deployment in the path, and Vertex
  // takes a short-lived OAuth token.
  bedrock,
  aws: bedrock,
  azure,
  azure_openai: azure,
  vertex,
  vertexai: vertex,
  databricks,
  cloudflare,
  "workers-ai": cloudflare,

  // ── Gateways and aggregators ──────────────────────────────────────────
  // These route to other people's models, so a model id here is usually
  // "vendor/model" rather than a bare name.
  openrouter,
  vercel,
  "ai-gateway": vercel,
  baseten,
  deepinfra,
  sambanova,
  nebius,
  novita,
  hyperbolic,

  // Anything you host yourself that speaks the OpenAI format.
  custom,
  ollama: custom,
  local: custom,
  vllm: custom,
};

export const SUPPORTED_PROVIDERS = Object.keys(REGISTRY).sort();

/**
 * The canonical name per adapter, for docs and UIs that shouldn't list every
 * alias. Ordered as a person would expect to see them.
 */
export const CANONICAL_PROVIDERS = [
  "openai",
  "anthropic",
  "gemini",
  "mistral",
  "groq",
  "together",
  "fireworks",
  "perplexity",
  "xai",
  "cerebras",
  "deepseek",
  "moonshot",
  "zai",
  // Clouds.
  "bedrock",
  "azure",
  "vertex",
  "databricks",
  "cloudflare",
  // Gateways.
  "openrouter",
  "vercel",
  "baseten",
  "deepinfra",
  "sambanova",
  "nebius",
  "novita",
  "hyperbolic",
  "custom",
];

export function getProvider(name: string): ProviderAdapter {
  const key = (name ?? "").toLowerCase().trim();
  const adapter = REGISTRY[key];
  if (!adapter) {
    throw new UnsupportedProviderError(
      `Unknown provider '${name}'. Supported: ${SUPPORTED_PROVIDERS.join(", ")}`
    );
  }
  return adapter;
}
