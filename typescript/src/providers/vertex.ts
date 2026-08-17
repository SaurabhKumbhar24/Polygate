/**
 * Google Vertex AI, through its OpenAI-compatible endpoint.
 *
 * **The credential is an OAuth access token, not an API key.** Vertex issues no
 * long-lived keys; it authenticates with a short-lived token, typically from
 * `gcloud auth print-access-token`. Those expire in about an hour, so a token
 * pasted into an environment variable will start returning 401 rather than
 * continuing to work — a property of Vertex, not a fault here, but the thing
 * that surprises people.
 *
 * Minting tokens is deliberately out of scope: doing it properly means service
 * account JWT signing and a refresh cycle, which would mean a crypto dependency
 * in a package whose premise is that it has almost none.
 *
 * An OpenAI-compatible endpoint; see `openaiCompatible` for the shared
 * implementation and why it is shared.
 */
import { makeAdapter } from "./openaiCompatible";

export const ENV_VAR = "GOOGLE_ACCESS_TOKEN";
export const BASE_URL_ENV = "VERTEX_BASE_URL";

/**
 * Build the base URL from a project and region.
 *
 * Offered because the template names the region twice, and a mismatch between
 * the two produces a 404 rather than anything that names the problem.
 */
export function endpoint(project: string, region = "us-central1"): string {
  return (
    `https://${region}-aiplatform.googleapis.com/v1/` +
    `projects/${project}/locations/${region}/endpoints/openapi`
  );
}

export const chat = makeAdapter({
  provider: "vertex",
  envVar: ENV_VAR,
  baseUrlEnv: BASE_URL_ENV,
});
