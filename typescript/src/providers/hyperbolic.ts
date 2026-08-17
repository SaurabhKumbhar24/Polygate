/**
 * Hyperbolic — open models on spare GPU capacity.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.hyperbolic.xyz/v1/chat/completions";
export const ENV_VAR = "HYPERBOLIC_API_KEY";

export const chat = makeAdapter({
  provider: "hyperbolic",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
