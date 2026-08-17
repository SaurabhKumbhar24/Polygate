/**
 * DeepInfra — open models billed per token.
 *
 * Note the path: the OpenAI-compatible surface lives under `/v1/openai`, beside
 * DeepInfra's own native API at `/v1/inference`.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const API_URL = "https://api.deepinfra.com/v1/openai/chat/completions";
export const ENV_VAR = "DEEPINFRA_API_KEY";

export const chat = makeAdapter({
  provider: "deepinfra",
  apiUrl: API_URL,
  envVar: ENV_VAR,
});
