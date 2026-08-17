"""Amazon Bedrock, via the Converse API.

Bedrock is the one provider here that is neither OpenAI-shaped nor
token-authenticated, so it gets a full adapter rather than three lines of
configuration.

**Converse, not InvokeModel.** ``InvokeModel`` passes a body straight through to
the underlying vendor, so calling Claude and Llama through Bedrock would need two
different request shapes — reintroducing, inside one provider, exactly the
problem this package exists to remove. ``Converse`` is Bedrock's own unified
interface: one message shape across every model it hosts. It is the same idea as
polygate, one layer down, and building on it means a new Bedrock model works here
the day AWS adds it.

**Credentials are a pair.** See ``_sigv4.split_credentials`` for why they are
packed into a single ``ACCESS_KEY:SECRET[:TOKEN]`` string.

**The region is part of the host**, so it is a real parameter, defaulting to
``$AWS_REGION`` and then ``us-east-1``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

from ._openai_compatible import TIMEOUT_SECONDS
from ._sigv4 import encode_path, sign_request
from ..retry import KeyInput, Retry, execute, resolve_keys
from ..types import ChatResponse, Usage

ENV_VAR = "AWS_ACCESS_KEY_ID"
SECRET_ENV = "AWS_SECRET_ACCESS_KEY"
SESSION_ENV = "AWS_SESSION_TOKEN"
REGION_ENV = "AWS_REGION"

DEFAULT_REGION = "us-east-1"
SERVICE = "bedrock"


def _resolve_pair(api_key: KeyInput) -> KeyInput:
    """Let the standard AWS variables work without packing them by hand.

    Nobody has ``AWS_ACCESS_KEY_ID`` holding ``id:secret`` — they have the two
    real variables, usually placed there by the AWS CLI or an instance role. So
    when no key is passed and the packed form is absent, assemble it.
    """
    if api_key:
        return api_key
    access = os.environ.get(ENV_VAR) or ""
    if ":" in access:  # already packed
        return access
    secret = os.environ.get(SECRET_ENV) or ""
    if not access or not secret:
        return None  # let resolve_keys raise its own, clearer error
    token = os.environ.get(SESSION_ENV) or ""
    return f"{access}:{secret}:{token}" if token else f"{access}:{secret}"


def to_converse(messages: List[Dict[str, str]]) -> tuple:
    """Split polygate's flat messages into Converse's ``system`` + ``messages``.

    Converse takes system prompts in their own top-level field and every content
    as a list of typed blocks. Consecutive same-role messages are *not* merged
    here: Bedrock rejects them for some models, but silently repairing the
    caller's conversation would hide a bug in their code behind a quirk of ours.
    """
    system: List[Dict[str, str]] = []
    turns: List[Dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content") or ""
        if role == "system":
            system.append({"text": content})
        else:
            turns.append({
                "role": "assistant" if role == "assistant" else "user",
                "content": [{"text": content}],
            })
    return system, turns


def parse(data: Dict[str, Any], *, fallback_model: str) -> ChatResponse:
    """Normalize a Converse response.

    Content is a list of blocks; only the text ones are joined. A tool-use block
    has no ``text`` key at all, and reaching for one would turn a valid
    tool-calling response into a KeyError.
    """
    message = ((data.get("output") or {}).get("message")) or {}
    blocks = message.get("content") or []
    text = "".join(b.get("text", "") for b in blocks if isinstance(b, dict))

    usage_raw = data.get("usage") or {}
    return ChatResponse(
        content=text,
        role=message.get("role", "assistant"),
        model=fallback_model,
        provider="bedrock",
        usage=Usage(
            prompt_tokens=usage_raw.get("inputTokens", 0) or 0,
            completion_tokens=usage_raw.get("outputTokens", 0) or 0,
            total_tokens=usage_raw.get("totalTokens", 0) or 0,
        ),
        raw=data,
    )


def chat(
    *,
    model: str,
    messages: List[Dict[str, str]],
    api_key: KeyInput = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    retry: Optional[Retry] = None,
    region: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs: Any,
) -> ChatResponse:
    """One Converse call against Bedrock.

    ``model`` is a Bedrock model id or inference profile
    (``anthropic.claude-3-5-sonnet-20241022-v2:0``,
    ``us.anthropic.claude-sonnet-4-20250514-v1:0``), not a vendor model name.
    """
    pool = resolve_keys(_resolve_pair(api_key), ENV_VAR, "bedrock")
    resolved_region = region or os.environ.get(REGION_ENV) or DEFAULT_REGION
    host = (
        base_url.replace("https://", "").rstrip("/")
        if base_url
        else f"bedrock-runtime.{resolved_region}.amazonaws.com"
    )

    system, turns = to_converse(messages)
    payload: Dict[str, Any] = {"messages": turns}
    if system:
        payload["system"] = system

    inference: Dict[str, Any] = {}
    if temperature is not None:
        inference["temperature"] = temperature
    if max_tokens is not None:
        inference["maxTokens"] = max_tokens
    if inference:
        payload["inferenceConfig"] = inference
    payload.update(kwargs)

    # Encoded once and reused: the bytes that get hashed for the signature must
    # be byte-identical to the bytes sent, so re-serializing would risk a
    # different key order and an invalid signature.
    body = json.dumps(payload).encode("utf-8")
    path = encode_path("model", model, "converse")
    url = f"https://{host}{path}"

    def _attempt(key: str):
        headers = sign_request(
            key=key, region=resolved_region, service=SERVICE,
            host=host, path=path, body=body,
        )
        return requests.post(url, headers=headers, data=body, timeout=TIMEOUT_SECONDS)

    response = execute("bedrock", pool, _attempt, retry)
    return parse(response.json(), fallback_model=model)
