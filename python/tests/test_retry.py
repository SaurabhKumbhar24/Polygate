"""Key rotation and backoff retries.

The rules under test, in order of how expensive they are to get wrong:

  * a deterministic 4xx is never retried, because retrying re-sends a request that
    cannot succeed and bills for it
  * a rejected key (401/403) is evicted, not merely skipped, so it stops
    burning one attempt on every later request
  * a rate-limited key (429) rotates *before* it sleeps, because the backoff
    window is wasted while another key sits idle
  * opting out is the default: no ``retry=`` means exactly one attempt

No network. Run:  python -m pytest tests/test_retry.py
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polygate.exceptions import MissingAPIKeyError, ProviderAPIError  # noqa: E402
from polygate.retry import KeyPool, Retry, execute, resolve_keys  # noqa: E402


class FakeResponse:
    def __init__(self, status_code, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


def recorder(*statuses):
    """An attempt fn that returns the given statuses in order, logging keys used."""
    seen = []
    seq = list(statuses)

    def attempt(key):
        seen.append(key)
        status = seq.pop(0) if seq else 200
        return status if isinstance(status, FakeResponse) else FakeResponse(status)

    attempt.keys_used = seen
    return attempt


def no_sleep(_seconds):
    """Collect sleeps instead of performing them, so tests stay instant."""
    no_sleep.calls.append(_seconds)


no_sleep.calls = []


def setup_function(_fn):
    no_sleep.calls = []


# ── opting out ────────────────────────────────────────────────────────────────

def test_no_retry_policy_means_one_attempt():
    pool = KeyPool(["k1"])
    attempt = recorder(500, 200)
    with pytest.raises(ProviderAPIError):
        execute("openai", pool, attempt, retry=None, sleep=no_sleep)
    assert len(attempt.keys_used) == 1, "retried despite no retry policy"


def test_success_returns_the_response():
    pool = KeyPool(["k1"])
    resp = execute("openai", pool, recorder(200), sleep=no_sleep)
    assert resp.status_code == 200


# ── what must not be retried ──────────────────────────────────────────────────

@pytest.mark.parametrize("status", [400, 404, 422])
def test_deterministic_errors_are_not_retried(status):
    pool = KeyPool(["k1", "k2"])
    attempt = recorder(status, 200)
    with pytest.raises(ProviderAPIError) as exc:
        execute("openai", pool, attempt, Retry(max_attempts=5), sleep=no_sleep)
    assert exc.value.status_code == status
    assert len(attempt.keys_used) == 1, f"{status} was retried"


# ── auth failures evict ───────────────────────────────────────────────────────

def test_rejected_key_is_evicted_and_the_next_key_is_tried_immediately():
    pool = KeyPool(["bad", "good"])
    attempt = recorder(401, 200)
    resp = execute("openai", pool, attempt, Retry(max_attempts=3), sleep=no_sleep)
    assert resp.status_code == 200
    assert attempt.keys_used == ["bad", "good"]
    assert no_sleep.calls == [], "slept on an auth failure instead of rotating"


def test_eviction_persists_across_calls():
    """A dead key must not burn an attempt on every subsequent request."""
    pool = KeyPool(["bad", "good"])
    execute("openai", pool, recorder(401, 200), Retry(max_attempts=3), sleep=no_sleep)

    attempt = recorder(200)
    execute("openai", pool, attempt, Retry(max_attempts=3), sleep=no_sleep)
    assert attempt.keys_used == ["good"], "evicted key came back"


def test_all_keys_rejected_raises():
    pool = KeyPool(["a", "b"])
    with pytest.raises(ProviderAPIError) as exc:
        execute("openai", pool, recorder(403, 403), Retry(max_attempts=5), sleep=no_sleep)
    assert exc.value.status_code == 403


# ── rate limits rotate before sleeping ────────────────────────────────────────

def test_rate_limited_key_rotates_without_sleeping_when_another_is_free():
    pool = KeyPool(["k1", "k2"])
    attempt = recorder(429, 200)
    resp = execute("openai", pool, attempt, Retry(max_attempts=3), sleep=no_sleep)
    assert resp.status_code == 200
    assert attempt.keys_used == ["k1", "k2"]
    assert no_sleep.calls == [], "burned a backoff window with an idle key available"


def test_single_key_rate_limited_falls_back_to_sleeping():
    pool = KeyPool(["only"])
    attempt = recorder(429, 200)
    resp = execute("openai", pool, attempt, Retry(max_attempts=3), sleep=no_sleep)
    assert resp.status_code == 200
    assert len(no_sleep.calls) == 1, "did not back off with no other key to try"


def test_retry_after_header_is_honoured():
    pool = KeyPool(["only"])

    def attempt(key):
        attempt.n = getattr(attempt, "n", 0) + 1
        if attempt.n == 1:
            return FakeResponse(429, headers={"Retry-After": "7"})
        return FakeResponse(200)

    execute("openai", pool, attempt, Retry(max_attempts=3, jitter=False), sleep=no_sleep)
    assert no_sleep.calls == [7.0]


def test_retry_after_is_capped_by_max_delay():
    pool = KeyPool(["only"])

    def attempt(key):
        attempt.n = getattr(attempt, "n", 0) + 1
        return FakeResponse(429, headers={"Retry-After": "9999"}) if attempt.n == 1 else FakeResponse(200)

    execute("openai", pool, attempt, Retry(max_attempts=3, max_delay=5.0, jitter=False), sleep=no_sleep)
    assert no_sleep.calls == [5.0], "a hostile Retry-After stalled the caller"


# ── transient server errors ───────────────────────────────────────────────────

def test_5xx_retries_then_succeeds():
    pool = KeyPool(["k1"])
    attempt = recorder(503, 200)
    resp = execute("openai", pool, attempt, Retry(max_attempts=3), sleep=no_sleep)
    assert resp.status_code == 200
    assert len(attempt.keys_used) == 2


def test_529_overloaded_is_retryable():
    """Anthropic's overloaded code: easy to omit, and it is the common one."""
    assert 529 in Retry().retry_on


def test_attempts_are_bounded():
    pool = KeyPool(["k1"])
    attempt = recorder(500, 500, 500, 500, 500)
    with pytest.raises(ProviderAPIError):
        execute("openai", pool, attempt, Retry(max_attempts=3), sleep=no_sleep)
    assert len(attempt.keys_used) == 3


# ── pool mechanics ────────────────────────────────────────────────────────────

def test_pool_rotates_round_robin():
    pool = KeyPool(["a", "b", "c"])
    assert [pool.acquire() for _ in range(4)] == ["a", "b", "c", "a"]


def test_pool_drops_duplicates():
    assert len(KeyPool(["a", "a", "b"])) == 2


def test_pool_rejects_empty():
    with pytest.raises(ValueError):
        KeyPool([])
    with pytest.raises(ValueError):
        KeyPool(["", "   "])


def test_backoff_is_exponential_and_capped():
    policy = Retry(base_delay=1.0, max_delay=10.0, jitter=False)
    assert [policy.delay_for(i) for i in range(5)] == [1.0, 2.0, 4.0, 8.0, 10.0]


def test_jitter_stays_within_bounds():
    policy = Retry(base_delay=1.0, max_delay=10.0, jitter=True)
    for i in range(4):
        assert 0.0 <= policy.delay_for(i) <= min(1.0 * (2 ** i), 10.0)


# ── key resolution ────────────────────────────────────────────────────────────

def test_resolve_accepts_a_single_key():
    assert len(resolve_keys("k1", "SOME_ENV", "openai")) == 1


def test_resolve_accepts_a_list():
    assert len(resolve_keys(["k1", "k2"], "SOME_ENV", "openai")) == 2


def test_resolve_passes_a_pool_through():
    pool = KeyPool(["k1"])
    assert resolve_keys(pool, "SOME_ENV", "openai") is pool


def test_resolve_reads_a_comma_separated_env_var(monkeypatch):
    """Lets an operator turn on rotation without a code change."""
    monkeypatch.setenv("POLYGATE_TEST_KEYS", "k1, k2 ,k3")
    assert len(resolve_keys(None, "POLYGATE_TEST_KEYS", "openai")) == 3


def test_resolve_raises_when_nothing_is_configured(monkeypatch):
    monkeypatch.delenv("POLYGATE_TEST_KEYS", raising=False)
    with pytest.raises(MissingAPIKeyError):
        resolve_keys(None, "POLYGATE_TEST_KEYS", "openai")
