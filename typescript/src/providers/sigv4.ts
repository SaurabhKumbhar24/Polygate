/**
 * AWS Signature Version 4, in about sixty lines of `node:crypto`.
 *
 * Bedrock is the only provider here that does not authenticate with a token in
 * a header — it wants every request cryptographically signed. The obvious
 * answer is the AWS SDK, and it is the wrong one for this package:
 * `@aws-sdk/client-bedrock-runtime` pulls in dozens of transitive packages, in
 * a library whose whole premise is that it has none. Someone installing
 * polygate to call OpenAI should not pay for an AWS SDK.
 *
 * So this implements the signing directly. It is a fixed, well-specified
 * algorithm — HMAC-SHA256 four times over a canonical string — and AWS has not
 * revised it since 2012. Verified byte-for-byte against botocore's signer in
 * the Python suite.
 *
 * Reference: https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv.html
 */
import { createHash, createHmac } from "node:crypto";

export const ALGORITHM = "AWS4-HMAC-SHA256";

function hmac(key: Buffer | string, message: string): Buffer {
  return createHmac("sha256", key).update(message, "utf8").digest();
}

function sha256(data: Buffer | string): string {
  return createHash("sha256").update(data).digest("hex");
}

/**
 * The four-step derived key. Each step narrows the scope of the signature,
 * which is what stops a captured signature being replayed against another
 * region or service.
 */
function signingKey(
  secret: string,
  date: string,
  region: string,
  service: string
): Buffer {
  let key = hmac(`AWS4${secret}`, date);
  key = hmac(key, region);
  key = hmac(key, service);
  return hmac(key, "aws4_request");
}

export interface AwsCredentials {
  accessKeyId: string;
  secretAccessKey: string;
  sessionToken: string;
}

/**
 * Unpack an `ACCESS_KEY:SECRET[:SESSION_TOKEN]` string.
 *
 * Packed into one string because the key pool — which handles rotation, parking
 * a throttled credential, and dropping a rejected one — is built around a single
 * opaque string per credential. Encoding the pair keeps AWS inside that
 * machinery instead of beside it, so a list of Bedrock credentials rotates
 * exactly like a list of OpenAI keys.
 */
export function splitCredentials(key: string): AwsCredentials {
  const parts = key.split(":");
  if (parts.length === 2) {
    return { accessKeyId: parts[0], secretAccessKey: parts[1], sessionToken: "" };
  }
  if (parts.length === 3) {
    return {
      accessKeyId: parts[0],
      secretAccessKey: parts[1],
      sessionToken: parts[2],
    };
  }
  throw new Error(
    "AWS credentials must be 'ACCESS_KEY_ID:SECRET_ACCESS_KEY' " +
      "(optionally ':SESSION_TOKEN' for temporary credentials)."
  );
}

/**
 * Percent-encode path segments the way SigV4 expects.
 *
 * `encodeURIComponent` leaves `!'()*` alone where AWS wants them encoded, hence
 * the extra pass.
 */
export function encodePath(...segments: string[]): string {
  const encode = (s: string) =>
    encodeURIComponent(s).replace(
      /[!'()*]/g,
      (c) => "%" + c.charCodeAt(0).toString(16).toUpperCase()
    );
  return "/" + segments.filter(Boolean).map(encode).join("/");
}

export interface SignOptions {
  key: string;
  region: string;
  service: string;
  host: string;
  /** Already encoded once by {@link encodePath} — i.e. as it will be sent. */
  path: string;
  body: string;
  method?: string;
  now?: Date;
}

/**
 * Return the headers that authenticate one request.
 *
 * `path` is the path as it will actually be sent. This encodes it **again** for
 * the canonical request, which is not a mistake: SigV4 requires every path
 * segment to be URI-encoded twice for all services except S3, so a Bedrock model
 * id ending `v2:0` travels as `v2%3A0` in the request line and is signed as
 * `v2%253A0`.
 *
 * Getting this wrong is the easiest way to break SigV4, and it fails as a bare
 * 403 with no indication that the path was the problem.
 */
export function signRequest(options: SignOptions): Record<string, string> {
  const { key, region, service, host, path, body, method = "POST" } = options;
  const { accessKeyId, secretAccessKey, sessionToken } = splitCredentials(key);

  const stamp = options.now ?? new Date();
  const amzDate = stamp.toISOString().replace(/[:-]|\.\d{3}/g, "");
  const dateStamp = amzDate.slice(0, 8);
  const payloadHash = sha256(body);

  let canonicalHeaders =
    `content-type:application/json\n` +
    `host:${host}\n` +
    `x-amz-content-sha256:${payloadHash}\n` +
    `x-amz-date:${amzDate}\n`;
  let signedHeaders = "content-type;host;x-amz-content-sha256;x-amz-date";
  if (sessionToken) {
    canonicalHeaders += `x-amz-security-token:${sessionToken}\n`;
    signedHeaders += ";x-amz-security-token";
  }

  // The second encoding pass. "/" and "~" stay literal, matching botocore.
  const canonicalPath = path
    .split("/")
    .map((segment) =>
      encodeURIComponent(segment).replace(/[!'()*]/g, (c) =>
        "%" + c.charCodeAt(0).toString(16).toUpperCase()
      )
    )
    .join("/");

  const canonicalRequest = [
    method,
    canonicalPath,
    "",
    canonicalHeaders,
    signedHeaders,
    payloadHash,
  ].join("\n");

  const scope = `${dateStamp}/${region}/${service}/aws4_request`;
  const toSign = [ALGORITHM, amzDate, scope, sha256(canonicalRequest)].join("\n");
  const signature = createHmac(
    "sha256",
    signingKey(secretAccessKey, dateStamp, region, service)
  )
    .update(toSign, "utf8")
    .digest("hex");

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Amz-Date": amzDate,
    "X-Amz-Content-Sha256": payloadHash,
    Authorization:
      `${ALGORITHM} Credential=${accessKeyId}/${scope}, ` +
      `SignedHeaders=${signedHeaders}, Signature=${signature}`,
  };
  if (sessionToken) headers["X-Amz-Security-Token"] = sessionToken;
  return headers;
}
