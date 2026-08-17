/**
 * Baseten Model APIs — hosted open models, no deployment step.
 *
 * Baseten also serves *your own* deployed models, each on its own URL. Those are
 * reachable with `baseUrl`; this default is the shared Model APIs endpoint.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://inference.baseten.co/v1/chat/completions";
export const ENV_VAR = "BASETEN_API_KEY";

export const chat = makeAdapter({
  provider: "baseten",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
