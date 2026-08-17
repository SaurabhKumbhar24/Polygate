/**
 * Moonshot's API, used to reach Kimi K2 and its siblings.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.moonshot.ai/v1/chat/completions";
export const ENV_VAR = "MOONSHOT_API_KEY";

export const chat = makeAdapter({
  provider: "moonshot",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
