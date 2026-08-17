/**
 * Fireworks AI. Note the /inference prefix - it is not the usual /v1 root.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.fireworks.ai/inference/v1/chat/completions";
export const ENV_VAR = "FIREWORKS_API_KEY";

export const chat = makeAdapter({
  provider: "fireworks",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
