/**
 * Databricks Model Serving — foundation models and your own, on your workspace.
 *
 * There is no fixed host: the URL is your workspace, so `baseUrl` (or
 * `$DATABRICKS_HOST`) is required. The `model` is the **serving endpoint name**,
 * not a vendor model id — which is why two workspaces on the same underlying
 * model can call it different things.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const ENV_VAR = "DATABRICKS_TOKEN";
export const BASE_URL_ENV = "DATABRICKS_HOST";
export const SERVING_PATH = "/serving-endpoints";

/**
 * Build the completions URL from whatever shape of root was supplied.
 *
 * Databricks documents the endpoint three ways — workspace root, root plus
 * `/serving-endpoints`, and the full per-endpoint invocation URL — and all three
 * are things people paste. Accepting each beats a 404 whose cause is a missing
 * path segment.
 */
export function buildUrl(root: string, model: string): string {
  let base = root.replace(/\/+$/, "");
  if (base.endsWith("/invocations")) return base;
  if (!base.endsWith(SERVING_PATH)) base = `${base}${SERVING_PATH}`;
  return `${base}/${model}/invocations`;
}

export const chat = makeAdapter({
  provider: "databricks",
  envVar: ENV_VAR,
  baseUrlEnv: BASE_URL_ENV,
  urlBuilder: buildUrl,
});
