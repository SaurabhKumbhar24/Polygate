/**
 * Nebius AI Studio — open models from a European cloud.
 *
 * The host is `api.studio.nebius.com`, not `api.nebius.com`; the latter is the
 * cloud control plane and will not serve inference.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.studio.nebius.com/v1/chat/completions";
export const ENV_VAR = "NEBIUS_API_KEY";

export const chat = makeAdapter({
  provider: "nebius",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
