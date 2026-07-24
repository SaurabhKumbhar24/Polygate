"""Optional key rotation and backoff retries.

Both features are off by default. ``chat()`` with no ``retry=`` and a single
``api_key`` behaves exactly as it did before this module existed: one request,
one response, errors raised straight through.

Why the two belong together
---------------------------
The interesting failure in high-volume LLM traffic is not the network, it is the
rate limit, and a rate limit is per *key*, not per host. Sleeping on a 429 when
another key is idle wastes the whole backoff window. So the loop below rotates
first and sleeps only when it runs out of keys, which turns a pool of N keys
into roughly N times the throughput ceiling rather than N times the retries.

Failure classes are handled differently on purpose:

  * ``401`` / ``403``  the key is bad, not busy. Evict it permanently and try
    the next one immediately. A dead key left in a pool otherwise burns one
    attempt on every subsequent request, forever.
  * ``429``  the key is rate limited. Cool it down for the ``Retry-After`` the
    provider asked for and move to another key at once.
  * ``5xx`` / ``529``  the provider is struggling. Backoff applies; rotating
    keys does not help, but it does not hurt.
  * ``400`` / ``404`` / ``422``  deterministic. Retrying re-sends a request that
    cannot succeed and bills for it. Raise immediately.
"""
from __future__ import annotations

import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Union

from .exceptions import MissingAPIKeyError, ProviderAPIError

# Status codes worth trying again. 529 is Anthropic's "overloaded".
DEFAULT_RETRY_ON = (408, 409, 429, 500, 502, 503, 504, 529)

# A key that authenticates but is rate limited comes back; a key the provider
# rejects outright does not.
_AUTH_FAILURE_CODES = (401, 403)


@dataclass
class Retry:
    """Backoff policy. Pass ``retry=Retry()`` to opt in; omit it for no retries.

    ``max_attempts`` counts total attempts, not extra ones: ``max_attempts=1``
    is equivalent to no retry at all.
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 20.0
    jitter: bool = True
    retry_on: Sequence[int] = DEFAULT_RETRY_ON
    #: Honour the provider's ``Retry-After`` header over the computed backoff.
    respect_retry_after: bool = True

    def delay_for(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Seconds to wait before ``attempt`` (0-indexed)."""
        if self.respect_retry_after and retry_after is not None:
            return min(retry_after, self.max_delay)
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter:
            # Full jitter: spreads a thundering herd of clients that all got
            # rate limited by the same upstream at the same moment.
            delay = random.uniform(0, delay)
        return delay


class KeyPool:
    """A rotating set of API keys for one provider.

    Thread-safe: a pool is typically shared by every worker thread in a process,
    and the cursor and cool-down table are mutable.
    """

    def __init__(self, keys: Sequence[str]) -> None:
        cleaned = [k.strip() for k in keys if k and k.strip()]
        if not cleaned:
            raise ValueError("KeyPool requires at least one non-empty key")
        # Preserve order, drop duplicates. A repeated key is not extra capacity.
        seen: Dict[str, None] = {}
        for k in cleaned:
            seen.setdefault(k, None)
        self._keys: List[str] = list(seen)
        self._cursor = 0
        self._cooldown: Dict[str, float] = {}
        self._dead: set = set()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._keys)

    @property
    def usable(self) -> int:
        """Keys neither evicted nor currently cooling down."""
        now = time.monotonic()
        with self._lock:
            return sum(
                1 for k in self._keys
                if k not in self._dead and self._cooldown.get(k, 0.0) <= now
            )

    def acquire(self) -> Optional[str]:
        """Next usable key, round-robin, or None if every key is unavailable.

        Falls back to the least-cooled key when all live keys are cooling down,
        so a caller that is willing to wait can still make progress instead of
        failing outright.
        """
        now = time.monotonic()
        with self._lock:
            live = [k for k in self._keys if k not in self._dead]
            if not live:
                return None
            n = len(live)
            for offset in range(n):
                key = live[(self._cursor + offset) % n]
                if self._cooldown.get(key, 0.0) <= now:
                    self._cursor = (self._cursor + offset + 1) % n
                    return key
            # Everything is cooling down: hand back whichever recovers first.
            return min(live, key=lambda k: self._cooldown.get(k, 0.0))

    def cool_down(self, key: str, seconds: float) -> None:
        with self._lock:
            self._cooldown[key] = time.monotonic() + max(0.0, seconds)

    def evict(self, key: str) -> None:
        """Permanently remove a key the provider rejected."""
        with self._lock:
            self._dead.add(key)

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return all(k in self._dead for k in self._keys)


KeyInput = Union[str, Sequence[str], KeyPool, None]


def resolve_keys(api_key: KeyInput, env_var: str, provider: str) -> KeyPool:
    """Turn whatever the caller passed into a pool.

    Accepts a single key (the original contract), a list, or a pool. Falls back
    to the provider's standard environment variable, which may itself hold a
    comma-separated list so a pool can be configured without code changes.
    """
    if isinstance(api_key, KeyPool):
        return api_key
    if isinstance(api_key, str):
        keys: List[str] = [api_key]
    elif api_key:
        keys = list(api_key)
    else:
        raw = os.environ.get(env_var) or ""
        keys = [k for k in (part.strip() for part in raw.split(",")) if k]

    if not keys:
        raise MissingAPIKeyError(
            f"No {provider} API key provided. Pass api_key= or set {env_var}."
        )
    return KeyPool(keys)


def _retry_after_seconds(response) -> Optional[float]:
    raw = None
    try:
        raw = response.headers.get("Retry-After") or response.headers.get("retry-after")
    except Exception:  # noqa: BLE001 - header access should never break the loop
        return None
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        # The HTTP-date form is legal but rare from LLM providers; the computed
        # backoff is a fine substitute.
        return None


def execute(
    provider: str,
    pool: KeyPool,
    attempt: Callable[[str], object],
    retry: Optional[Retry] = None,
    sleep: Callable[[float], None] = time.sleep,
):
    """Run ``attempt(key)`` until it returns a 2xx, or give up.

    ``attempt`` receives a key and returns a ``requests.Response``. Returns the
    successful response; raises :class:`ProviderAPIError` otherwise.
    """
    import requests  # local import keeps module import cheap

    policy = retry or Retry(max_attempts=1)
    attempts = max(1, policy.max_attempts)
    last_status: Optional[int] = None
    last_body: str = ""

    for i in range(attempts):
        key = pool.acquire()
        if key is None:
            raise ProviderAPIError(
                provider, last_status or 401,
                last_body or "every API key was rejected by the provider",
            )

        try:
            response = attempt(key)
        except requests.RequestException as exc:
            last_status, last_body = 0, str(exc)
            if i == attempts - 1:
                raise ProviderAPIError(provider, 0, f"request failed: {exc}") from None
            sleep(policy.delay_for(i))
            continue

        status = response.status_code
        if 200 <= status < 300:
            return response

        last_status, last_body = status, _safe_body(response)

        if status in _AUTH_FAILURE_CODES:
            # Not a transient condition: stop using this key entirely and try
            # the next one right away, without spending a backoff window.
            pool.evict(key)
            if pool.exhausted:
                raise ProviderAPIError(provider, status, last_body)
            continue

        if status not in tuple(policy.retry_on):
            raise ProviderAPIError(provider, status, last_body)

        retry_after = _retry_after_seconds(response)
        if status == 429:
            # Park this key for as long as the provider asked, then rotate.
            pool.cool_down(key, retry_after if retry_after is not None else policy.base_delay * (2 ** i))
            if pool.usable > 0 and i < attempts - 1:
                continue  # another key is free, so no reason to sleep

        if i == attempts - 1:
            break
        sleep(policy.delay_for(i, retry_after))

    raise ProviderAPIError(provider, last_status or 0, last_body)


def _safe_body(response) -> str:
    try:
        return response.text
    except Exception:  # noqa: BLE001
        return ""
