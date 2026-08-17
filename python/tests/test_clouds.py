"""Clouds and gateways: Bedrock, Azure, Vertex, Databricks, Cloudflare, and the
OpenAI-compatible aggregators.

The gateways are covered by the shared table in ``test_providers.py``. What is
worth pinning here is the places where these providers *stop* being uniform —
SigV4 signing, Azure's three deviations, and the URLs that contain your own
account. Each of those fails as an opaque 401/403/404 when it is wrong, so a
test that catches it is worth more than usual.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polygate import chat  # noqa: E402
from polygate.exceptions import MissingEndpointError  # noqa: E402
from polygate.providers import azure, bedrock, cloudflare, databricks, vertex  # noqa: E402
from polygate.providers._sigv4 import (  # noqa: E402
    encode_path, sign_request, split_credentials,
)

FIXED_TIME = dt.datetime(2026, 8, 16, 12, 0, 0, tzinfo=dt.timezone.utc)


def ok(payload):
    return MagicMock(status_code=200, json=lambda: payload)


# ── SigV4 ────────────────────────────────────────────────────────────────────

def test_credentials_unpack_with_and_without_a_session_token():
    assert split_credentials("AKIA:secret") == ("AKIA", "secret", "")
    assert split_credentials("AKIA:secret:tok") == ("AKIA", "secret", "tok")


def test_a_malformed_credential_says_what_the_format_is():
    """A bare access key here would otherwise sign with an empty secret and come
    back as a 403 that reads like a permissions problem."""
    with pytest.raises(ValueError, match="ACCESS_KEY_ID:SECRET_ACCESS_KEY"):
        split_credentials("just-an-access-key")


def test_signing_is_deterministic_for_a_fixed_time():
    """Same inputs, same signature — the property that makes the rest of these
    assertions meaningful."""
    args = dict(
        key="AKIA:secret", region="us-east-1", service="bedrock",
        host="bedrock-runtime.us-east-1.amazonaws.com",
        path="/model/m/converse", body=b"{}", now=FIXED_TIME,
    )
    assert sign_request(**args)["Authorization"] == sign_request(**args)["Authorization"]


def test_the_signature_covers_the_body():
    """If the body were not hashed into the signature, a proxy could alter the
    prompt in flight and AWS would still accept it."""
    base = dict(
        key="AKIA:secret", region="us-east-1", service="bedrock",
        host="h", path="/p", now=FIXED_TIME,
    )
    one = sign_request(**base, body=b'{"a":1}')["Authorization"]
    two = sign_request(**base, body=b'{"a":2}')["Authorization"]
    assert one != two


def test_the_signature_is_scoped_to_region_and_service():
    """The derived key narrows by date, region and service precisely so a
    captured signature cannot be replayed elsewhere."""
    base = dict(key="AKIA:secret", host="h", path="/p", body=b"{}", now=FIXED_TIME)
    east = sign_request(**base, region="us-east-1", service="bedrock")["Authorization"]
    west = sign_request(**base, region="us-west-2", service="bedrock")["Authorization"]
    other = sign_request(**base, region="us-east-1", service="s3")["Authorization"]
    assert east != west and east != other
    assert "us-east-1/bedrock/aws4_request" in east


def test_the_content_hash_header_matches_the_body():
    headers = sign_request(
        key="AKIA:secret", region="us-east-1", service="bedrock",
        host="h", path="/p", body=b'{"hello":1}', now=FIXED_TIME,
    )
    assert headers["X-Amz-Content-Sha256"] == hashlib.sha256(b'{"hello":1}').hexdigest()


def test_a_session_token_is_signed_and_sent():
    """Temporary credentials are what an assumed role hands you, and omitting
    the token from SignedHeaders makes AWS reject the request."""
    headers = sign_request(
        key="AKIA:secret:tok123", region="us-east-1", service="bedrock",
        host="h", path="/p", body=b"{}", now=FIXED_TIME,
    )
    assert headers["X-Amz-Security-Token"] == "tok123"
    assert "x-amz-security-token" in headers["Authorization"]


def test_model_ids_with_a_colon_are_encoded_for_the_url():
    """Bedrock model ids end in ':0', which must be percent-encoded in the path
    that is actually sent. The *canonical* path is encoded again on top of this
    — see ``test_the_canonical_path_is_double_encoded``."""
    path = encode_path("model", "anthropic.claude-3-5-sonnet-20241022-v2:0", "converse")
    assert path == "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse"


# ── Bedrock ──────────────────────────────────────────────────────────────────

def test_system_messages_move_to_their_own_field():
    """Converse takes system prompts top-level, not as a message. Left in the
    list, the instruction would be delivered as if the user had said it."""
    system, turns = bedrock.to_converse([
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hello"},
    ])
    assert system == [{"text": "be brief"}]
    assert turns == [{"role": "user", "content": [{"text": "hello"}]}]


def test_unknown_roles_are_treated_as_user_not_dropped():
    _, turns = bedrock.to_converse([{"role": "human", "content": "hi"}])
    assert turns == [{"role": "user", "content": [{"text": "hi"}]}]


def test_converse_usage_is_normalized():
    """Bedrock reports inputTokens/outputTokens; everything else here reports
    prompt_tokens/completion_tokens. Callers must not have to know which."""
    result = bedrock.parse(
        {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 4, "totalTokens": 14},
        },
        fallback_model="anthropic.claude-3-5-sonnet-20241022-v2:0",
    )
    assert result.content == "hi"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens) == (10, 4)
    assert result.provider == "bedrock"


def test_a_tool_use_block_does_not_crash_the_parser():
    """Tool blocks carry no 'text' key at all; reaching for one would turn a
    valid response into a KeyError."""
    result = bedrock.parse(
        {"output": {"message": {"content": [
            {"toolUse": {"name": "search"}}, {"text": "and some prose"},
        ]}}},
        fallback_model="m",
    )
    assert result.content == "and some prose"


def test_an_empty_response_yields_empty_content_not_an_error():
    assert bedrock.parse({}, fallback_model="m").content == ""


@patch("polygate.providers.bedrock.requests.post")
def test_bedrock_posts_to_the_regional_converse_url(mock_post, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    mock_post.return_value = ok({"output": {"message": {"content": [{"text": "hi"}]}}})

    chat(provider="bedrock", model="anthropic.claude-3-5-sonnet-20241022-v2:0",
         messages=[{"role": "user", "content": "hi"}])

    url = mock_post.call_args[0][0]
    assert url.startswith("https://bedrock-runtime.eu-west-1.amazonaws.com/model/")
    assert url.endswith("/converse")
    assert "%3A0" in url, "the colon in the model id must be encoded"


@patch("polygate.providers.bedrock.requests.post")
def test_the_standard_aws_variables_work_unpacked(mock_post, monkeypatch):
    """Nobody has AWS_ACCESS_KEY_ID holding 'id:secret' — they have the two real
    variables the AWS CLI writes. Requiring the packed form by hand would make
    the normal setup fail."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    mock_post.return_value = ok({"output": {"message": {"content": [{"text": "x"}]}}})

    chat(provider="bedrock", model="m", messages=[{"role": "user", "content": "hi"}])
    assert mock_post.call_args[1]["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256")


@patch("polygate.providers.bedrock.requests.post")
def test_the_signed_bytes_are_the_bytes_sent(mock_post, monkeypatch):
    """Re-serializing the payload between signing and sending could reorder keys
    and invalidate the signature."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    mock_post.return_value = ok({"output": {"message": {"content": [{"text": "x"}]}}})

    chat(provider="bedrock", model="m", messages=[{"role": "user", "content": "hi"}])
    sent = mock_post.call_args[1]["data"]
    headers = mock_post.call_args[1]["headers"]
    assert isinstance(sent, bytes)
    assert headers["X-Amz-Content-Sha256"] == hashlib.sha256(sent).hexdigest()


# ── Azure ────────────────────────────────────────────────────────────────────

@patch("polygate.providers._openai_compatible.requests.post")
def test_azure_puts_the_key_in_its_own_header(mock_post, monkeypatch):
    """A Bearer token here returns a 401 indistinguishable from a wrong key."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    mock_post.return_value = ok({"choices": [{"message": {"content": "hi"}}]})

    chat(provider="azure", model="my-deployment",
         messages=[{"role": "user", "content": "hi"}],
         base_url="https://res.openai.azure.com")

    headers = mock_post.call_args[1]["headers"]
    assert headers["api-key"] == "k"
    assert "Authorization" not in headers


@patch("polygate.providers._openai_compatible.requests.post")
def test_azure_puts_the_deployment_in_the_path_with_an_api_version(mock_post, monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    mock_post.return_value = ok({"choices": [{"message": {"content": "hi"}}]})

    chat(provider="azure", model="my-deployment",
         messages=[{"role": "user", "content": "hi"}],
         base_url="https://res.openai.azure.com")

    url = mock_post.call_args[0][0]
    assert url == (
        "https://res.openai.azure.com/openai/deployments/my-deployment"
        f"/chat/completions?api-version={azure.DEFAULT_API_VERSION}"
    )


@patch("polygate.providers._openai_compatible.requests.post")
def test_the_azure_api_version_is_overridable(mock_post, monkeypatch):
    """Azure versions change model behaviour, so pinning must be possible
    without editing the library."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    mock_post.return_value = ok({"choices": [{"message": {"content": "hi"}}]})

    chat(provider="azure", model="d", messages=[{"role": "user", "content": "hi"}],
         base_url="https://res.openai.azure.com")
    assert "api-version=2025-01-01-preview" in mock_post.call_args[0][0]


def test_azure_without_an_endpoint_names_what_to_pass(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(MissingEndpointError, match="AZURE_OPENAI_ENDPOINT"):
        chat(provider="azure", model="d", messages=[{"role": "user", "content": "hi"}])


# ── Databricks ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("root", [
    "https://ws.cloud.databricks.com",
    "https://ws.cloud.databricks.com/",
    "https://ws.cloud.databricks.com/serving-endpoints",
])
def test_databricks_accepts_every_documented_root_shape(root):
    """All three are things people paste out of the docs; a missing path segment
    would 404 without saying which part was wrong."""
    assert databricks._url(root, "my-endpoint") == (
        "https://ws.cloud.databricks.com/serving-endpoints/my-endpoint/invocations"
    )


def test_databricks_leaves_a_full_invocation_url_alone():
    full = "https://ws.cloud.databricks.com/serving-endpoints/ep/invocations"
    assert databricks._url(full, "ignored") == full


def test_databricks_without_a_workspace_says_so(monkeypatch):
    monkeypatch.setenv("DATABRICKS_TOKEN", "t")
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    with pytest.raises(MissingEndpointError, match="DATABRICKS_HOST"):
        chat(provider="databricks", model="ep",
             messages=[{"role": "user", "content": "hi"}])


# ── Cloudflare ───────────────────────────────────────────────────────────────

def test_a_bare_account_id_is_expanded_into_the_full_url():
    """The account id is what the dashboard shows; making people assemble five
    URL segments around it adds a step that exists only to be got wrong."""
    assert cloudflare._url("abc123", "@cf/meta/llama-3.1-8b-instruct") == (
        "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1/chat/completions"
    )


@patch("polygate.providers._openai_compatible.requests.post")
def test_the_account_id_is_read_at_call_time_not_import_time(mock_post, monkeypatch):
    """Configuring the environment after importing polygate is the normal order
    in a notebook, a test, or anything using dotenv."""
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "t")
    monkeypatch.delenv("CLOUDFLARE_BASE_URL", raising=False)
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "set-after-import")
    mock_post.return_value = ok({"choices": [{"message": {"content": "hi"}}]})

    chat(provider="cloudflare", model="@cf/meta/llama-3.1-8b-instruct",
         messages=[{"role": "user", "content": "hi"}])
    assert "accounts/set-after-import/ai/v1" in mock_post.call_args[0][0]


# ── Vertex ───────────────────────────────────────────────────────────────────

def test_the_vertex_endpoint_helper_fills_the_region_twice():
    """The template names the region in two places and a mismatch 404s without
    naming the problem."""
    url = vertex.endpoint("my-project", "europe-west4")
    assert url.startswith("https://europe-west4-aiplatform.googleapis.com/")
    assert "/locations/europe-west4/" in url
    assert "/projects/my-project/" in url


def test_vertex_without_a_project_says_so(monkeypatch):
    monkeypatch.setenv("GOOGLE_ACCESS_TOKEN", "ya29.token")
    monkeypatch.delenv("VERTEX_BASE_URL", raising=False)
    with pytest.raises(MissingEndpointError, match="VERTEX_BASE_URL"):
        chat(provider="vertex", model="google/gemini-2.0-flash",
             messages=[{"role": "user", "content": "hi"}])


# ── The signature, checked against AWS's own implementation ──────────────────

def test_the_canonical_path_is_double_encoded():
    """SigV4 encodes path segments twice for every service except S3, so a model
    id ending ``v2:0`` is *sent* as ``v2%3A0`` and *signed* as ``v2%253A0``.

    Signing the single-encoded form produces a valid-looking request that AWS
    rejects with a bare 403 — no hint that the path was the problem. This was
    wrong until it was diffed against botocore.
    """
    path = encode_path("model", "anthropic.claude-3-5-sonnet-20241022-v2:0", "converse")
    assert path == "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0/converse"

    headers = sign_request(
        key="AKIA:secret", region="us-east-1", service="bedrock",
        host="h", path=path, body=b"{}", now=FIXED_TIME,
    )
    reference = sign_request(
        key="AKIA:secret", region="us-east-1", service="bedrock",
        host="h", path="/model/anthropic.claude-3-5-sonnet-20241022-v2%253A0/converse",
        body=b"{}", now=FIXED_TIME,
    )
    assert headers["Authorization"] != reference["Authorization"], (
        "signing must apply its own second encoding, not accept a pre-doubled path"
    )


@pytest.mark.parametrize("model,token", [
    ("anthropic.claude-3-5-sonnet-20241022-v2:0", None),
    ("us.anthropic.claude-sonnet-4-20250514-v1:0", "tok-123"),
    ("amazon.nova-pro-v1:0", None),
    ("arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.meta.llama3", None),
])
def test_matches_botocore(model, token):
    """The signature must be byte-identical to AWS's own signer.

    Hand-rolling SigV4 to avoid a 30MB dependency is only defensible if it is
    provably the same algorithm, so this diffs against botocore whenever it is
    installed. Skipped rather than vendored: polygate must not gain an AWS
    dependency even for its tests.
    """
    botocore_auth = pytest.importorskip("botocore.auth")
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials
    import hashlib as _h, json as _j

    region, service = "us-east-1", "bedrock"
    host = f"bedrock-runtime.{region}.amazonaws.com"
    secret = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY"
    body = _j.dumps({"messages": [{"role": "user", "content": [{"text": "hi"}]}]}).encode()
    path = encode_path("model", model, "converse")

    request = AWSRequest(
        method="POST", url=f"https://{host}{path}", data=body,
        headers={"Content-Type": "application/json",
                 "X-Amz-Content-Sha256": _h.sha256(body).hexdigest()},
    )
    botocore_auth.SigV4Auth(
        Credentials("AKIDEXAMPLE", secret, token), service, region,
    ).add_auth(request)
    theirs = request.headers["Authorization"]
    stamp = dt.datetime.strptime(
        request.headers["X-Amz-Date"], "%Y%m%dT%H%M%SZ",
    ).replace(tzinfo=dt.timezone.utc)

    key = f"AKIDEXAMPLE:{secret}" + (f":{token}" if token else "")
    mine = sign_request(key=key, region=region, service=service, host=host,
                        path=path, body=body, now=stamp)["Authorization"]

    assert mine.split("Signature=")[-1] == theirs.split("Signature=")[-1]
