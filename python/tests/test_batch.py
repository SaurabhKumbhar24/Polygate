"""Offline batch jobs and concurrent dispatch.

Two features that get confused with each other, tested together because the
distinction is the point: :mod:`polygate.batch` queues a job with the provider
and costs half price; :mod:`polygate.concurrent` keeps requests in flight so a
server's continuous batcher has something to schedule.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polygate import BatchRequest, batch  # noqa: E402
from polygate.concurrent import (  # noqa: E402
    AdaptiveLimiter, Completion, contents, map_chat, map_prompts,
)
from polygate.exceptions import ProviderAPIError  # noqa: E402


def reply(payload, status_code=200, text=None):
    return MagicMock(
        status_code=status_code,
        json=lambda: payload,
        text=text if text is not None else json.dumps(payload),
    )


REQS = [
    BatchRequest(custom_id="a", model="gpt-4o-mini", messages=[{"role": "user", "content": "1"}]),
    BatchRequest(custom_id="b", model="gpt-4o-mini", messages=[{"role": "user", "content": "2"}]),
]


# ── Guardrails that fail before a job is paid for ────────────────────────────

def test_a_provider_without_a_batch_tier_points_at_the_alternative():
    """A 404 from a missing /batches route reads like an outage. Naming the
    fallback matters because there is a real one that works everywhere."""
    with pytest.raises(batch.BatchNotSupportedError, match="map_chat"):
        batch.submit("gemini", REQS)


def test_duplicate_custom_ids_are_refused_up_front():
    """Providers key results by custom_id, so a duplicate makes some results
    unreachable — and it surfaces at retrieval, hours and dollars later."""
    dupes = [
        BatchRequest(custom_id="same", model="m", messages=[]),
        BatchRequest(custom_id="same", model="m", messages=[]),
    ]
    with pytest.raises(ValueError, match="unique"):
        batch.submit("openai", dupes)


def test_an_empty_batch_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        batch.submit("openai", [])


def test_batch_aliases_resolve_like_chat_aliases(monkeypatch):
    """"claude" works everywhere else in the package; it must work here too."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with patch("polygate.batch.requests.post") as post:
        post.return_value = reply({"id": "msgbatch_1", "processing_status": "in_progress"})
        job = batch.submit("claude", [
            BatchRequest(custom_id="a", model="claude-sonnet-4-5", messages=[]),
        ])
    assert job.provider == "anthropic"


# ── OpenAI-style: upload a file, then reference it ───────────────────────────

@patch("polygate.batch.requests.post")
def test_openai_uploads_jsonl_then_creates_the_job(post, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    post.side_effect = [
        reply({"id": "file-1"}),
        reply({"id": "batch_1", "status": "validating",
               "request_counts": {"total": 2, "completed": 0, "failed": 0}}),
    ]
    job = batch.submit("openai", REQS)

    upload = post.call_args_list[0]
    assert upload[0][0].endswith("/files")
    assert upload[1]["data"]["purpose"] == "batch"
    lines = upload[1]["files"]["file"][1].decode().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["custom_id"] == "a"
    assert first["url"] == "/v1/chat/completions"
    assert first["body"]["model"] == "gpt-4o-mini"

    create = post.call_args_list[1]
    assert create[1]["json"]["input_file_id"] == "file-1"
    assert job.id == "batch_1"
    assert job.status == "pending"
    assert job.total == 2


def test_groq_asks_for_the_longer_completion_window():
    """Groq recommends 7d: under load a 24h job is likelier to expire than to
    finish, and an expired batch must be resubmitted whole."""
    assert batch._BACKENDS["groq"].completion_window == "7d"
    assert batch._BACKENDS["openai"].completion_window == "24h"


@pytest.mark.parametrize("raw,expected", [
    ("validating", "pending"),
    ("in_progress", "running"),
    ("finalizing", "running"),
    ("completed", "completed"),
    ("failed", "failed"),
    ("expired", "expired"),
    ("cancelled", "cancelled"),
    ("canceled", "cancelled"),
])
def test_openai_statuses_normalize(raw, expected):
    job = batch._BACKENDS["openai"]._to_job({"status": raw})
    assert job.status == expected


def test_finalizing_is_not_treated_as_done():
    """The results file does not exist yet at 'finalizing'. Stopping there
    returns an empty result set that looks like a job with no output."""
    job = batch._BACKENDS["openai"]._to_job({"status": "finalizing"})
    assert not job.done


@patch("polygate.batch.requests.get")
def test_failed_requests_are_read_from_the_error_file_too(get, monkeypatch):
    """Successes and failures land in two different files. Reading only the
    output file silently loses every failure in the job."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    get.side_effect = [
        reply({"id": "batch_1", "status": "completed",
               "output_file_id": "file-out", "error_file_id": "file-err",
               "request_counts": {"total": 2, "completed": 1, "failed": 1}}),
        reply({}, text=json.dumps({
            "custom_id": "a",
            "response": {"body": {
                "model": "gpt-4o-mini",
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            }},
        })),
        reply({}, text=json.dumps({
            "custom_id": "b", "error": {"message": "context_length_exceeded"},
        })),
    ]
    out = batch.results("openai", "batch_1")
    by_id = {r.custom_id: r for r in out}
    assert by_id["a"].ok and by_id["a"].response.content == "hi"
    assert by_id["a"].response.usage.prompt_tokens == 3
    assert not by_id["b"].ok
    assert "context_length_exceeded" in by_id["b"].error


# ── Anthropic: inline requests, results streamed from a URL ──────────────────

@patch("polygate.batch.requests.post")
def test_anthropic_sends_requests_inline_with_no_file_upload(post, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    post.return_value = reply({"id": "msgbatch_1", "processing_status": "in_progress",
                               "request_counts": {"processing": 1, "succeeded": 0,
                                                  "errored": 0, "canceled": 0, "expired": 0}})
    batch.submit("anthropic", [BatchRequest(
        custom_id="a", model="claude-sonnet-4-5",
        messages=[{"role": "system", "content": "be brief"},
                  {"role": "user", "content": "hi"}],
    )])
    assert post.call_count == 1, "no file upload step for Anthropic"
    body = post.call_args[1]["json"]
    entry = body["requests"][0]
    assert entry["custom_id"] == "a"
    # System prompts travel beside the messages, not inside them.
    assert entry["params"]["system"] == "be brief"
    assert entry["params"]["messages"] == [{"role": "user", "content": "hi"}]


@patch("polygate.batch.requests.post")
def test_anthropic_always_sends_max_tokens(post, monkeypatch):
    """The Messages API requires it, and a batch missing it is rejected only
    after the whole job has been accepted."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    post.return_value = reply({"id": "b", "processing_status": "in_progress"})
    batch.submit("anthropic", [BatchRequest(custom_id="a", model="m", messages=[])])
    assert post.call_args[1]["json"]["requests"][0]["params"]["max_tokens"] > 0


def test_anthropic_counts_unrun_requests_as_failed_not_successful():
    """Cancelled and expired requests never ran. Counting them as successes
    overstates the job; omitting them stalls progress below 1.0 forever."""
    job = batch._BACKENDS["anthropic"]._to_job({
        "id": "b", "processing_status": "ended",
        "request_counts": {"processing": 0, "succeeded": 5, "errored": 2,
                           "canceled": 2, "expired": 1},
    })
    assert job.total == 10
    assert job.completed == 5
    assert job.failed == 5
    assert job.progress == 1.0


def test_anthropic_results_normalize_to_the_same_shape():
    result = batch._BACKENDS["anthropic"]._to_result({
        "custom_id": "a",
        "result": {"type": "succeeded", "message": {
            "role": "assistant", "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {"input_tokens": 7, "output_tokens": 2},
        }},
    })
    assert result.ok
    assert result.response.content == "hello"
    # Anthropic reports input/output tokens; the normalized name is prompt/completion.
    assert result.response.usage.prompt_tokens == 7
    assert result.response.usage.total_tokens == 9


@pytest.mark.parametrize("kind", ["errored", "canceled", "expired"])
def test_every_non_success_result_type_becomes_an_error(kind):
    result = batch._BACKENDS["anthropic"]._to_result(
        {"custom_id": "a", "result": {"type": kind}}
    )
    assert not result.ok and result.error


# ── run() ────────────────────────────────────────────────────────────────────

def test_run_polls_until_terminal_then_fetches(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    seen = []
    states = iter(["running", "running", "completed"])

    with patch.object(batch, "submit") as submit, \
         patch.object(batch, "status") as status_, \
         patch.object(batch, "results") as results_:
        submit.return_value = batch.BatchJob(id="b", provider="openai", status="pending")
        status_.side_effect = lambda *a, **k: batch.BatchJob(
            id="b", provider="openai", status=next(states),
        )
        results_.return_value = [batch.BatchResult(custom_id="a")]
        out = batch.run("openai", REQS, poll_seconds=0,
                        sleep=lambda s: seen.append(s))

    assert len(seen) == 3, "polled until the job reached a terminal state"
    assert len(out) == 1


def test_a_timeout_says_the_job_is_still_running(monkeypatch):
    """Giving up watching is not the same as cancelling. The batch id is still
    good, and abandoning a paid job silently would be worse than saying so."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    clock = iter([0, 0, 10_000, 10_000])
    with patch.object(batch, "submit") as submit, \
         patch.object(batch, "status") as status_, \
         patch("polygate.batch.time.monotonic", lambda: next(clock)):
        submit.return_value = batch.BatchJob(id="b7", provider="openai", status="running")
        status_.return_value = batch.BatchJob(id="b7", provider="openai", status="running")
        with pytest.raises(TimeoutError, match="b7"):
            batch.run("openai", REQS, timeout_seconds=1, poll_seconds=0,
                      sleep=lambda s: None)


# ── Concurrency: the client half of continuous batching ─────────────────────

def test_results_come_back_in_input_order():
    """Completion order is not submission order. Placing results by arrival
    attaches every answer to the wrong question — silently, with no error."""
    def fake(provider, model, messages, **kwargs):
        # Later items finish first.
        index = int(messages[0]["content"])
        return MagicMock(content=f"answer-{index}")

    with patch("polygate.client.chat", side_effect=fake):
        out = map_chat("openai", "gpt-4o-mini",
                       [[{"role": "user", "content": str(i)}] for i in range(20)],
                       api_key="k", adaptive=False)
    assert [c.index for c in out] == list(range(20))
    assert [c.response.content for c in out] == [f"answer-{i}" for i in range(20)]


def test_one_failure_does_not_discard_the_rest():
    """A single 400 in item 4,000 of 10,000 must not throw away 3,999 answers
    that have already been paid for."""
    def fake(provider, model, messages, **kwargs):
        if messages[0]["content"] == "3":
            raise ProviderAPIError("openai", 400, "bad request")
        return MagicMock(content="ok")

    with patch("polygate.client.chat", side_effect=fake):
        out = map_chat("openai", "m",
                       [[{"role": "user", "content": str(i)}] for i in range(6)],
                       api_key="k", adaptive=False)
    assert sum(1 for c in out if c.ok) == 5
    assert not out[3].ok and isinstance(out[3].error, ProviderAPIError)


def test_raise_on_error_is_available_for_all_or_nothing_runs():
    def fake(provider, model, messages, **kwargs):
        raise ProviderAPIError("openai", 400, "nope")

    with patch("polygate.client.chat", side_effect=fake):
        with pytest.raises(ProviderAPIError):
            map_chat("openai", "m", [[{"role": "user", "content": "a"}]],
                     api_key="k", raise_on_error=True, adaptive=False)


def test_concurrency_is_actually_bounded():
    """Unbounded fan-out over ten thousand prompts opens ten thousand sockets
    and earns an immediate 429."""
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def fake(provider, model, messages, **kwargs):
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        try:
            return MagicMock(content="ok")
        finally:
            with lock:
                live["now"] -= 1

    with patch("polygate.client.chat", side_effect=fake):
        map_chat("openai", "m", [[{"role": "user", "content": "x"}]] * 200,
                 api_key="k", concurrency=4, adaptive=False)
    assert live["peak"] <= 4


def test_an_empty_input_does_no_work():
    assert map_chat("openai", "m", [], api_key="k") == []


def test_map_prompts_builds_the_conversations():
    with patch("polygate.client.chat", return_value=MagicMock(content="ok")) as chat_:
        map_prompts("openai", "m", ["one", "two"], system="be brief",
                    api_key="k", adaptive=False)
    messages = chat_.call_args[1]["messages"]
    assert messages[0] == {"role": "system", "content": "be brief"}
    assert messages[1]["role"] == "user"


def test_contents_renders_failures_as_the_default():
    out = [
        Completion(index=0, response=MagicMock(content="hi")),
        Completion(index=1, error=RuntimeError("boom")),
    ]
    assert contents(out, default="<failed>") == ["hi", "<failed>"]


# ── The adaptive limiter ─────────────────────────────────────────────────────

def test_the_limiter_halves_on_a_rate_limit():
    limiter = AdaptiveLimiter(start=8)
    limiter.on_rate_limit()
    assert limiter.limit == 4
    limiter.on_rate_limit()
    assert limiter.limit == 2


def test_the_limiter_never_reaches_zero():
    """A ceiling of zero deadlocks the whole run."""
    limiter = AdaptiveLimiter(start=1)
    for _ in range(10):
        limiter.on_rate_limit()
    assert limiter.limit >= 1


def test_the_limiter_recovers_one_at_a_time_up_to_its_ceiling():
    """Additive increase, so the steady state sits just under the provider's
    limit rather than oscillating across it."""
    limiter = AdaptiveLimiter(start=8)
    limiter.on_rate_limit()          # 8 -> 4
    for _ in range(20):
        limiter.on_success()
    assert limiter.limit == 8, "recovers, but never above the configured ceiling"


def test_an_idle_limiter_still_recovers():
    """Regression: an earlier design tracked shrinkage as debt paid off by later
    releases, so a limiter with nothing in flight could never grow back — it sat
    at half speed for the rest of the run with no way to discover the provider
    had recovered."""
    limiter = AdaptiveLimiter(start=8)
    limiter.on_rate_limit()
    for _ in range(20):
        limiter.on_success()
    assert limiter.limit == 8


def test_shrinking_takes_effect_immediately_for_new_work():
    """A halved limit that still admitted the old number of requests would keep
    hitting the rate limit it was reacting to."""
    limiter = AdaptiveLimiter(start=4)
    for _ in range(4):
        limiter.acquire()
    limiter.on_rate_limit()          # 4 -> 2
    for _ in range(4):
        limiter.release()

    for _ in range(2):
        limiter.acquire()

    blocked = threading.Event()

    def third():
        limiter.acquire()
        blocked.set()

    threading.Thread(target=third, daemon=True).start()
    assert not blocked.wait(timeout=0.2), "a third slot must not exist at limit 2"


def test_in_flight_requests_are_not_cancelled_by_a_shrink():
    """Four requests already sent are already being paid for; dropping them on
    a 429 wastes work that is going to complete anyway."""
    limiter = AdaptiveLimiter(start=4)
    for _ in range(4):
        limiter.acquire()
    limiter.on_rate_limit()
    # All four can still be released without error or deadlock.
    for _ in range(4):
        limiter.release()
    assert limiter._active == 0
