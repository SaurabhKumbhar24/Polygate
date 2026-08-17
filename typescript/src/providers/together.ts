/**
 * Together AI's hosted open-weight models.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.together.xyz/v1/chat/completions";
export const ENV_VAR = "TOGETHER_API_KEY";

export const chat = makeAdapter({
  provider: "together",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
