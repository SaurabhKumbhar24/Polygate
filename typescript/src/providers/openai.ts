/**
 * The format every provider below imitates.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.openai.com/v1/chat/completions";
export const ENV_VAR = "OPENAI_API_KEY";

export const chat = makeAdapter({
  provider: "openai",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
