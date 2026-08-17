"""AWS Signature Version 4, in about sixty lines of stdlib.

Bedrock is the only provider here that does not authenticate with a token in a
header — it wants every request cryptographically signed. The obvious answer is
boto3, and it is the wrong one for this package: boto3 plus botocore is tens of
megabytes and a dependency tree of its own, in a library whose whole premise is
``requests`` and nothing else. Someone who installs polygate to call OpenAI
should not pay for an AWS SDK.

So this implements the signing directly. It is a fixed, well-specified
algorithm — HMAC-SHA256 four times over a canonical string — and it does not
change; AWS has not revised SigV4 since 2012.

Reference: https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv.html
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
from typing import Dict, Tuple
from urllib.parse import quote

ALGORITHM = "AWS4-HMAC-SHA256"


def _sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, date: str, region: str, service: str) -> bytes:
    """The four-step derived key. Each step narrows the scope of the signature,
    which is what stops a captured signature being replayed against another
    region or service."""
    key = _sign(f"AWS4{secret}".encode("utf-8"), date)
    key = _sign(key, region)
    key = _sign(key, service)
    return _sign(key, "aws4_request")


def split_credentials(key: str) -> Tuple[str, str, str]:
    """Unpack an ``ACCESS_KEY:SECRET[:SESSION_TOKEN]`` string.

    Packed into one string because the key pool — which handles rotation,
    parking a throttled credential, and dropping a rejected one — is built
    around a single opaque string per credential. Encoding the pair keeps AWS
    inside that machinery instead of beside it, so a list of Bedrock credentials
    rotates exactly like a list of OpenAI keys.
    """
    parts = key.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], ""
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    raise ValueError(
        "AWS credentials must be 'ACCESS_KEY_ID:SECRET_ACCESS_KEY' "
        "(optionally ':SESSION_TOKEN' for temporary credentials)."
    )


def sign_request(
    *,
    key: str,
    region: str,
    service: str,
    host: str,
    path: str,
    body: bytes,
    method: str = "POST",
    now: _dt.datetime = None,
) -> Dict[str, str]:
    """Return the headers that authenticate one request.

    ``path`` is the path as it will actually be sent — i.e. already encoded once
    by :func:`encode_path`. This function encodes it **again** for the canonical
    request, which is not a mistake: SigV4 requires every path segment to be
    URI-encoded twice for all services except S3, so a Bedrock model id ending
    ``v2:0`` travels as ``v2%3A0`` in the request line and is signed as
    ``v2%253A0``.

    Getting this wrong is the single easiest way to break SigV4, and it fails as
    a bare 403 with no indication that the path was the problem. Verified
    against botocore's own signer — see ``test_matches_botocore``.
    """
    access_key, secret_key, session_token = split_credentials(key)
    stamp = now or _dt.datetime.now(_dt.timezone.utc)
    amz_date = stamp.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = stamp.strftime("%Y%m%d")

    payload_hash = hashlib.sha256(body).hexdigest()

    canonical_headers = (
        f"content-type:application/json\n"
        f"host:{host}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "content-type;host;x-amz-content-sha256;x-amz-date"
    if session_token:
        canonical_headers += f"x-amz-security-token:{session_token}\n"
        signed_headers += ";x-amz-security-token"

    # The second encoding pass. "/" and "~" stay literal, matching botocore.
    canonical_path = quote(path, safe="/~")
    canonical_request = "\n".join([
        method, canonical_path, "", canonical_headers, signed_headers, payload_hash,
    ])
    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    to_sign = "\n".join([
        ALGORITHM, amz_date, scope,
        hashlib.sha256(canonical_request.encode("utf-8")).hexdigest(),
    ])
    signature = hmac.new(
        _signing_key(secret_key, date_stamp, region, service),
        to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    headers = {
        "Content-Type": "application/json",
        "X-Amz-Date": amz_date,
        "X-Amz-Content-Sha256": payload_hash,
        "Authorization": (
            f"{ALGORITHM} Credential={access_key}/{scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        ),
    }
    if session_token:
        headers["X-Amz-Security-Token"] = session_token
    return headers


def encode_path(*segments: str) -> str:
    """Percent-encode path segments the way SigV4 expects.

    ``safe=""`` so a colon in a model id becomes ``%3A`` in both the canonical
    request and the URL actually sent. Leaving them inconsistent is the classic
    SigV4 mistake, and it fails as an opaque 403.
    """
    return "/" + "/".join(quote(s, safe="") for s in segments if s)


__all__ = ["sign_request", "split_credentials", "encode_path", "ALGORITHM"]
