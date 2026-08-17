/**
 * Groq's LPU inference.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.groq.com/openai/v1/chat/completions";
export const ENV_VAR = "GROQ_API_KEY";

export const chat = makeAdapter({
  provider: "groq",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
