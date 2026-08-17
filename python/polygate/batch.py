"""Offline batch inference — submit many requests, come back later, pay half.

Several providers run a second, asynchronous tier: you hand over a whole job,
they process it within a window (usually 24 hours, often much sooner), and it
costs roughly 50% of the synchronous price. For evals, backfills, classification
sweeps and anything else where nobody is waiting on the answer, that is the
difference between a run you can afford and one you cannot.

The problem this module solves is that no two providers agree on how it works:

  OpenAI-style   upload a JSONL file, reference the file id in a job, poll the
                 job, then download a second file of results.
  Anthropic      send the requests inline in the job itself, poll, then stream
                 a JSONL from a URL the job hands back.

Different verbs, different field names, different status vocabularies, and
different result envelopes. So this normalizes all of it to one shape:
:func:`submit`, :func:`status`, :func:`results`, :func:`cancel` — and
:func:`run` when you just want the answers and are happy to wait.

Not the same thing as continuous batching
-----------------------------------------
This is *offline* batching: a job queued for later. Continuous batching is a
server-side scheduling technique — vLLM and TGI slot new requests into a running
forward pass — and no client can perform it. What a client can do is keep enough
requests in flight for the server to have something to slot, which is what
:mod:`polygate.concurrent` is for. The two are complementary and often confused.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

import requests

from .exceptions import PolygateError, ProviderAPIError, UnsupportedProviderError
from .retry import KeyInput, resolve_keys
from .types import ChatResponse, Usage

TIMEOUT_SECONDS = 120

#: How long to wait between polls by default. Batches are measured in minutes to
#: hours, so a tight loop buys nothing but rate-limit errors.
DEFAULT_POLL_SECONDS = 30


class BatchNotSupportedError(UnsupportedProviderError):
    """Raised for a provider with no asynchronous batch tier.

    Its own type because the fallback is real and worth naming: everything
    polygate can reach synchronously can also be driven through
    :func:`polygate.concurrent.map_chat`, which costs full price but works
    everywhere and returns in seconds rather than hours.
    """


# ── The normalized shapes ────────────────────────────────────────────────────

#: Terminal states. Polling stops here.
TERMINAL = frozenset({"completed", "failed", "cancelled", "expired"})


@dataclass
class BatchRequest:
    """One request inside a batch.

    ``custom_id`` is how a result is matched back to its input. It is required
    rather than generated, because results come back unordered and often
    partially — matching by position would quietly mis-assign every answer in a
    job where one request failed.
    """
    custom_id: str
    model: str
    messages: List[Dict[str, str]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJob:
    """A submitted batch, in normalized terms."""
    id: str
    provider: str
    #: One of pending / running / completed / failed / cancelled / expired.
    status: str
    total: int = 0
    completed: int = 0
    failed: int = 0
    #: The provider's own untouched job body, for anything not normalized here.
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.status in TERMINAL

    @property
    def progress(self) -> float:
        """Fraction finished, 0.0–1.0. Zero rather than a division error when
        the provider has not yet reported a total."""
        return (self.completed + self.failed) / self.total if self.total else 0.0


@dataclass
class BatchResult:
    """One result from a finished batch.

    ``response`` is None when the request failed, and ``error`` is None when it
    succeeded — never both. A batch of ten thousand will contain failures, and a
    caller that has to guess which field to trust will eventually trust the
    wrong one.
    """
    custom_id: str
    response: Optional[ChatResponse] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.response is not None


# ── OpenAI-style backend (upload a file, reference it in a job) ──────────────

class _OpenAIStyle:
    """Providers that copied OpenAI's two-step file+job batch flow."""

    def __init__(self, provider: str, base_url: str, env_var: str,
                 completion_window: str = "24h"):
        self.provider = provider
        self.base = base_url.rstrip("/")
        self.env_var = env_var
        self.completion_window = completion_window

    def _headers(self, key: str) -> Dict[str, str]:
        return {"Authorization": f"Bearer {key}"}

    def _check(self, response) -> Dict[str, Any]:
        if response.status_code >= 400:
            raise ProviderAPIError(self.provider, response.status_code, response.text[:500])
        return response.json()

    def submit(self, key: str, requests_in: List[BatchRequest],
               endpoint: str = "/v1/chat/completions") -> BatchJob:
        lines = []
        for item in requests_in:
            body: Dict[str, Any] = {"model": item.model, "messages": item.messages}
            if item.temperature is not None:
                body["temperature"] = item.temperature
            if item.max_tokens is not None:
                body["max_tokens"] = item.max_tokens
            body.update(item.extra)
            lines.append(json.dumps({
                "custom_id": item.custom_id,
                "method": "POST",
                "url": endpoint,
                "body": body,
            }))
        payload = "\n".join(lines).encode("utf-8")

        uploaded = self._check(requests.post(
            f"{self.base}/files",
            headers=self._headers(key),
            files={"file": ("batch.jsonl", payload, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=TIMEOUT_SECONDS,
        ))
        created = self._check(requests.post(
            f"{self.base}/batches",
            headers={**self._headers(key), "Content-Type": "application/json"},
            json={
                "input_file_id": uploaded["id"],
                "endpoint": endpoint,
                "completion_window": self.completion_window,
            },
            timeout=TIMEOUT_SECONDS,
        ))
        return self._to_job(created)

    def status(self, key: str, batch_id: str) -> BatchJob:
        return self._to_job(self._check(requests.get(
            f"{self.base}/batches/{batch_id}",
            headers=self._headers(key), timeout=TIMEOUT_SECONDS,
        )))

    def cancel(self, key: str, batch_id: str) -> BatchJob:
        return self._to_job(self._check(requests.post(
            f"{self.base}/batches/{batch_id}/cancel",
            headers=self._headers(key), timeout=TIMEOUT_SECONDS,
        )))

    def results(self, key: str, batch_id: str) -> List[BatchResult]:
        job = self.status(key, batch_id)
        out: List[BatchResult] = []
        # Successes and failures land in two separate files, and a caller that
        # reads only the output file silently loses every failed request.
        for file_id, is_error in (
            (job.raw.get("output_file_id"), False),
            (job.raw.get("error_file_id"), True),
        ):
            if not file_id:
                continue
            response = requests.get(
                f"{self.base}/files/{file_id}/content",
                headers=self._headers(key), timeout=TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise ProviderAPIError(self.provider, response.status_code,
                                       response.text[:500])
            for line in response.text.splitlines():
                if line.strip():
                    out.append(self._to_result(json.loads(line), is_error))
        return out

    def _to_job(self, data: Dict[str, Any]) -> BatchJob:
        counts = data.get("request_counts") or {}
        return BatchJob(
            id=data.get("id", ""),
            provider=self.provider,
            status=_NORMALIZE_OPENAI.get(data.get("status", ""), "running"),
            total=counts.get("total", 0) or 0,
            completed=counts.get("completed", 0) or 0,
            failed=counts.get("failed", 0) or 0,
            raw=data,
        )

    def _to_result(self, row: Dict[str, Any], is_error: bool) -> BatchResult:
        custom_id = row.get("custom_id", "")
        if is_error or row.get("error"):
            return BatchResult(custom_id=custom_id,
                               error=json.dumps(row.get("error") or row)[:500])
        body = ((row.get("response") or {}).get("body")) or {}
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = body.get("usage") or {}
        return BatchResult(
            custom_id=custom_id,
            response=ChatResponse(
                content=message.get("content") or message.get("reasoning_content") or "",
                role=message.get("role", "assistant"),
                model=body.get("model", ""),
                provider=self.provider,
                usage=Usage(
                    prompt_tokens=usage.get("prompt_tokens", 0) or 0,
                    completion_tokens=usage.get("completion_tokens", 0) or 0,
                    total_tokens=usage.get("total_tokens", 0) or 0,
                ),
                raw=body,
            ),
        )


#: OpenAI's vocabulary → ours. "finalizing" is deliberately *not* terminal: the
#: results file does not exist yet, and treating it as done returns an empty set.
_NORMALIZE_OPENAI = {
    "validating":  "pending",
    "in_progress": "running",
    "finalizing":  "running",
    "completed":   "completed",
    "failed":      "failed",
    "expired":     "expired",
    "cancelling":  "running",
    "cancelled":   "cancelled",
    "canceled":    "cancelled",
}


# ── Anthropic backend (requests inline, results streamed from a URL) ─────────

class _AnthropicStyle:
    """Anthropic's Message Batches: no file upload, requests travel in the job."""

    provider = "anthropic"

    def __init__(self, base_url: str = "https://api.anthropic.com/v1",
                 env_var: str = "ANTHROPIC_API_KEY"):
        self.base = base_url.rstrip("/")
        self.env_var = env_var

    def _headers(self, key: str) -> Dict[str, str]:
        return {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _check(self, response) -> Dict[str, Any]:
        if response.status_code >= 400:
            raise ProviderAPIError(self.provider, response.status_code, response.text[:500])
        return response.json()

    def submit(self, key: str, requests_in: List[BatchRequest],
               endpoint: str = "") -> BatchJob:
        payload = []
        for item in requests_in:
            system, turns = _split_system(item.messages)
            params: Dict[str, Any] = {
                "model": item.model,
                "messages": turns,
                # Required by the Messages API, and a batch that omits it is
                # rejected only after the whole job has been accepted.
                "max_tokens": item.max_tokens or 1024,
            }
            if system:
                params["system"] = system
            if item.temperature is not None:
                params["temperature"] = item.temperature
            params.update(item.extra)
            payload.append({"custom_id": item.custom_id, "params": params})

        return self._to_job(self._check(requests.post(
            f"{self.base}/messages/batches",
            headers=self._headers(key), json={"requests": payload},
            timeout=TIMEOUT_SECONDS,
        )))

    def status(self, key: str, batch_id: str) -> BatchJob:
        return self._to_job(self._check(requests.get(
            f"{self.base}/messages/batches/{batch_id}",
            headers=self._headers(key), timeout=TIMEOUT_SECONDS,
        )))

    def cancel(self, key: str, batch_id: str) -> BatchJob:
        return self._to_job(self._check(requests.post(
            f"{self.base}/messages/batches/{batch_id}/cancel",
            headers=self._headers(key), timeout=TIMEOUT_SECONDS,
        )))

    def results(self, key: str, batch_id: str) -> List[BatchResult]:
        job = self.status(key, batch_id)
        url = job.raw.get("results_url")
        if not url:
            # Ended without results means the whole job failed or expired, which
            # is different from "no requests succeeded" and should not look like
            # an empty success.
            if job.status in TERMINAL and job.status != "completed":
                raise PolygateError(
                    f"Anthropic batch {batch_id} ended as {job.status} with no results."
                )
            return []
        response = requests.get(url, headers=self._headers(key), timeout=TIMEOUT_SECONDS)
        if response.status_code >= 400:
            raise ProviderAPIError(self.provider, response.status_code, response.text[:500])
        return [
            self._to_result(json.loads(line))
            for line in response.text.splitlines() if line.strip()
        ]

    def _to_job(self, data: Dict[str, Any]) -> BatchJob:
        counts = data.get("request_counts") or {}
        succeeded = counts.get("succeeded", 0) or 0
        errored = counts.get("errored", 0) or 0
        canceled = counts.get("canceled", 0) or 0
        expired = counts.get("expired", 0) or 0
        processing = counts.get("processing", 0) or 0
        return BatchJob(
            id=data.get("id", ""),
            provider=self.provider,
            status=_NORMALIZE_ANTHROPIC.get(data.get("processing_status", ""), "running"),
            total=succeeded + errored + canceled + expired + processing,
            completed=succeeded,
            # Cancelled and expired requests never ran. Counting them as
            # successes would overstate the job; leaving them out entirely would
            # make progress stall below 1.0 forever.
            failed=errored + canceled + expired,
            raw=data,
        )

    def _to_result(self, row: Dict[str, Any]) -> BatchResult:
        custom_id = row.get("custom_id", "")
        result = row.get("result") or {}
        if result.get("type") != "succeeded":
            return BatchResult(
                custom_id=custom_id,
                error=json.dumps(result.get("error") or {"type": result.get("type")})[:500],
            )
        message = result.get("message") or {}
        usage = message.get("usage") or {}
        blocks = message.get("content") or []
        return BatchResult(
            custom_id=custom_id,
            response=ChatResponse(
                content="".join(
                    b.get("text", "") for b in blocks if isinstance(b, dict)
                ),
                role=message.get("role", "assistant"),
                model=message.get("model", ""),
                provider=self.provider,
                usage=Usage(
                    prompt_tokens=usage.get("input_tokens", 0) or 0,
                    completion_tokens=usage.get("output_tokens", 0) or 0,
                    total_tokens=(usage.get("input_tokens", 0) or 0)
                    + (usage.get("output_tokens", 0) or 0),
                ),
                raw=message,
            ),
        )


#: Anthropic reports only two states. "ended" covers success, failure and
#: cancellation alike, so the per-request counts are what distinguish them.
_NORMALIZE_ANTHROPIC = {"in_progress": "running", "canceling": "running", "ended": "completed"}


def _split_system(messages: List[Dict[str, str]]) -> tuple:
    """Anthropic takes the system prompt beside the messages, not inside them."""
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    turns = [
        {"role": "assistant" if m.get("role") == "assistant" else "user",
         "content": m.get("content", "")}
        for m in messages if m.get("role") != "system"
    ]
    return "\n\n".join(p for p in system_parts if p), turns


# ── Registry ─────────────────────────────────────────────────────────────────

#: Only providers with a documented batch tier. Deliberately not every
#: OpenAI-compatible host: most of them implement the chat route and not the
#: batch route, and a request to a missing /batches returns a 404 that reads
#: like an outage rather than an unsupported feature.
_BACKENDS: Dict[str, Any] = {
    "openai": _OpenAIStyle("openai", "https://api.openai.com/v1", "OPENAI_API_KEY"),
    "anthropic": _AnthropicStyle(),
    "groq": _OpenAIStyle(
        "groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
        # Groq recommends the longer window: under load a 24h job is likelier to
        # expire than to finish, and an expired batch has to be resubmitted whole.
        completion_window="7d",
    ),
}

BATCH_PROVIDERS = sorted(_BACKENDS)


def _backend(provider: str):
    key = (provider or "").lower().strip()
    # Aliases resolve through the main registry so "claude" and "gpt" work here
    # exactly as they do for chat.
    from .providers import _REGISTRY  # local import avoids a cycle
    module = _REGISTRY.get(key)
    if module is not None:
        key = getattr(module, "__name__", "").rsplit(".", 1)[-1]
    if key not in _BACKENDS:
        raise BatchNotSupportedError(
            f"{provider!r} has no batch API. Supported: {BATCH_PROVIDERS}. "
            f"For everything else use polygate.map_chat(), which runs requests "
            f"concurrently at normal prices."
        )
    return _BACKENDS[key]


def _key_for(backend, api_key: KeyInput) -> str:
    """Batch jobs are long-lived, so a single key is used rather than a pool.

    Rotating mid-job would submit under one credential and poll under another,
    and providers scope a batch id to the key that created it.
    """
    return resolve_keys(api_key, backend.env_var, getattr(backend, "provider", "batch")).acquire()


# ── Public API ───────────────────────────────────────────────────────────────

def submit(
    provider: str,
    requests_in: Iterable[BatchRequest],
    *,
    api_key: KeyInput = None,
) -> BatchJob:
    """Queue a batch. Returns immediately with a job you can poll."""
    backend = _backend(provider)
    items = list(requests_in)
    if not items:
        raise ValueError("A batch needs at least one request.")
    seen = {item.custom_id for item in items}
    if len(seen) != len(items):
        # Providers key results by custom_id, so duplicates make some results
        # unreachable — and it fails silently at retrieval, hours later.
        raise ValueError("custom_id must be unique within a batch.")
    return backend.submit(_key_for(backend, api_key), items)


def status(provider: str, batch_id: str, *, api_key: KeyInput = None) -> BatchJob:
    """Where a batch has got to."""
    backend = _backend(provider)
    return backend.status(_key_for(backend, api_key), batch_id)


def results(provider: str, batch_id: str, *, api_key: KeyInput = None) -> List[BatchResult]:
    """Every result from a finished batch, successes and failures alike."""
    backend = _backend(provider)
    return backend.results(_key_for(backend, api_key), batch_id)


def cancel(provider: str, batch_id: str, *, api_key: KeyInput = None) -> BatchJob:
    """Ask the provider to stop. Requests already sent are still billed."""
    backend = _backend(provider)
    return backend.cancel(_key_for(backend, api_key), batch_id)


def run(
    provider: str,
    requests_in: Iterable[BatchRequest],
    *,
    api_key: KeyInput = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: Optional[float] = None,
    on_progress: Optional[Callable[[BatchJob], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> List[BatchResult]:
    """Submit, wait, and return the results.

    Convenient and occasionally the wrong tool: this blocks for as long as the
    provider takes, which can be hours. Prefer :func:`submit` plus your own
    polling for anything that must survive the process being restarted — the
    batch id is durable, this call's stack frame is not.

    ``timeout_seconds`` gives up waiting; it does **not** cancel the job, which
    keeps running and can be collected later with :func:`results`. Abandoning a
    paid job silently would be worse than reporting that we stopped watching.
    """
    job = submit(provider, requests_in, api_key=api_key)
    started = time.monotonic()
    while not job.done:
        if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
            raise TimeoutError(
                f"Batch {job.id} still {job.status} after {timeout_seconds:.0f}s. "
                f"It is still running — collect it later with "
                f"polygate.batch.results({provider!r}, {job.id!r})."
            )
        sleep(poll_seconds)
        job = status(provider, job.id, api_key=api_key)
        if on_progress:
            on_progress(job)

    if job.status != "completed":
        raise PolygateError(f"Batch {job.id} ended as {job.status}.")
    return results(provider, job.id, api_key=api_key)


__all__ = [
    "BATCH_PROVIDERS",
    "BatchJob",
    "BatchNotSupportedError",
    "BatchRequest",
    "BatchResult",
    "cancel",
    "results",
    "run",
    "status",
    "submit",
]
