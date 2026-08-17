/**
 * xAI's Grok models.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.x.ai/v1/chat/completions";
export const ENV_VAR = "XAI_API_KEY";

export const chat = makeAdapter({
  provider: "xai",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
