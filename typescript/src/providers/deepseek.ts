/**
 * DeepSeek's API. Reasoning models return the answer in `reasoning_content`
 * when `content` is empty, which the shared parser handles.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.deepseek.com/chat/completions";
export const ENV_VAR = "DEEPSEEK_API_KEY";

export const chat = makeAdapter({
  provider: "deepseek",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
