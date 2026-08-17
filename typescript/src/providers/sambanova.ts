/**
 * SambaNova Cloud — open models on their own RDU silicon.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.sambanova.ai/v1/chat/completions";
export const ENV_VAR = "SAMBANOVA_API_KEY";

export const chat = makeAdapter({
  provider: "sambanova",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
