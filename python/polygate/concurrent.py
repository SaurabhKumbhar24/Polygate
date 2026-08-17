"""Keep many requests in flight at once.

This is the client half of continuous batching. The technique itself lives in
the server — vLLM, TGI and every hosted provider run an iteration-level
scheduler that slots new sequences into a forward pass as older ones finish, so
the GPU never idles waiting for the slowest generation in a fixed batch. No
client library can perform that; it happens inside the model server.

What a client controls is whether the scheduler has anything to schedule. A loop
that sends one request, waits, then sends the next leaves a continuous-batching
server running a batch size of one — the same GPU, a fraction of the throughput,
and you are billed per token either way. Keeping N requests outstanding is the
entire client-side contribution, and it is worth a module because doing it
naively goes wrong in three specific ways:

**Unbounded fan-out.** ``ThreadPoolExecutor(max_workers=len(items))`` on ten
thousand prompts opens ten thousand sockets and earns an immediate 429, or an
OOM, depending on which limit is nearer.

**Losing the mapping.** Completion order is not submission order. Results have to
be placed back by index or every answer is attached to the wrong question — a
failure that produces plausible output and no error.

**One failure killing the run.** A single 400 in item 4,000 of 10,000 should not
discard the 3,999 answers already paid for.

For work nobody is waiting on, :mod:`polygate.batch` is roughly half the price.
This module is for when you want the answers now.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .exceptions import ProviderAPIError
from .retry import KeyInput, Retry
from .types import ChatResponse

#: Enough to keep a server's batcher fed without looking like an attack. Hosted
#: providers throttle well below this on free tiers, which is what the adaptive
#: limiter below is for.
DEFAULT_CONCURRENCY = 8

#: Never drop below this. A limiter that can reach zero deadlocks.
MIN_CONCURRENCY = 1


@dataclass
class Completion:
    """One finished item, successful or not.

    Carries ``index`` because results are yielded as they finish when streaming,
    and the caller needs to know which input this answers.
    """
    index: int
    response: Optional[ChatResponse] = None
    error: Optional[BaseException] = None

    @property
    def ok(self) -> bool:
        return self.response is not None


@dataclass
class Progress:
    """A snapshot handed to ``on_progress``."""
    done: int
    total: int
    failed: int
    concurrency: int
    elapsed: float

    @property
    def fraction(self) -> float:
        return self.done / self.total if self.total else 0.0


class AdaptiveLimiter:
    """A concurrency ceiling that backs off when the provider pushes back.

    Additive-increase, multiplicative-decrease — the same shape as TCP
    congestion control, for the same reason: the right level is unknown, differs
    per provider and per key, and changes with load. Guessing once and holding it
    is wrong in both directions, and the cost of guessing high is a rate-limit
    ban rather than a slow run.

    Halving on a 429 and recovering by one keeps the steady state just under the
    limit instead of oscillating across it.
    """

    def __init__(self, start: int = DEFAULT_CONCURRENCY, maximum: Optional[int] = None):
        self.limit = max(MIN_CONCURRENCY, start)
        self.maximum = maximum or max(MIN_CONCURRENCY, start)
        #: A condition over an explicit in-flight count, rather than a
        #: Semaphore. A semaphore's permit count cannot be lowered, so shrinking
        #: would have to be tracked as debt paid off by later releases — and an
        #: idle or finishing run never pays it, leaving the limiter permanently
        #: stuck at its reduced level with no way back up. Counting directly
        #: makes both directions immediate and needs no bookkeeping.
        self._active = 0
        self._condition = threading.Condition()

    def acquire(self) -> None:
        with self._condition:
            while self._active >= self.limit:
                self._condition.wait()
            self._active += 1

    def release(self) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def on_success(self) -> None:
        with self._condition:
            if self.limit < self.maximum:
                self.limit += 1
                self._condition.notify()

    def on_rate_limit(self) -> None:
        # Takes effect immediately for anything not yet started; requests
        # already in flight are left to finish rather than cancelled.
        with self._condition:
            self.limit = max(MIN_CONCURRENCY, self.limit // 2)


def _is_rate_limit(error: BaseException) -> bool:
    return isinstance(error, ProviderAPIError) and error.status_code == 429


def map_chat(
    provider: str,
    model: str,
    conversations: Iterable[List[Dict[str, str]]],
    *,
    api_key: KeyInput = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    adaptive: bool = True,
    retry: Optional[Retry] = None,
    raise_on_error: bool = False,
    on_progress: Optional[Callable[[Progress], None]] = None,
    **kwargs: Any,
) -> List[Completion]:
    """Run many conversations against one model, N at a time.

    Returns a list **in input order**, one entry per conversation, each either a
    response or an error. Errors are captured rather than raised by default: a
    ten-thousand-item run will contain failures, and discarding the successes
    because of them wastes money already spent. Pass ``raise_on_error=True``
    when a partial result is useless to you.

    ``adaptive`` lets the concurrency find its own level against whatever limit
    the provider enforces. Turn it off when you have a known quota and want a
    fixed, predictable rate.
    """
    from .client import chat  # local import keeps module import cheap

    items = [list(c) for c in conversations]
    if not items:
        return []

    limiter = AdaptiveLimiter(concurrency, maximum=concurrency) if adaptive else None
    results: List[Optional[Completion]] = [None] * len(items)
    started = time.monotonic()
    state = {"done": 0, "failed": 0}
    state_lock = threading.Lock()

    def work(index: int) -> Completion:
        if limiter:
            limiter.acquire()
        try:
            response = chat(
                provider=provider, model=model, messages=items[index],
                api_key=api_key, retry=retry, **kwargs,
            )
            if limiter:
                limiter.on_success()
            return Completion(index=index, response=response)
        except BaseException as error:  # noqa: BLE001 - recorded, not swallowed
            if limiter and _is_rate_limit(error):
                limiter.on_rate_limit()
            return Completion(index=index, error=error)
        finally:
            if limiter:
                limiter.release()

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(work, i): i for i in range(len(items))}
        for future in as_completed(futures):
            completion = future.result()
            results[completion.index] = completion
            with state_lock:
                state["done"] += 1
                if not completion.ok:
                    state["failed"] += 1
                snapshot = Progress(
                    done=state["done"], total=len(items), failed=state["failed"],
                    concurrency=limiter.limit if limiter else concurrency,
                    elapsed=time.monotonic() - started,
                )
            if on_progress:
                on_progress(snapshot)

    finished = [r for r in results if r is not None]
    if raise_on_error:
        for completion in finished:
            if completion.error is not None:
                raise completion.error
    return finished


def map_prompts(
    provider: str,
    model: str,
    prompts: Iterable[str],
    *,
    system: str = "",
    **kwargs: Any,
) -> List[Completion]:
    """:func:`map_chat` for the common case of one user message each."""
    conversations = []
    for prompt in prompts:
        turns: List[Dict[str, str]] = []
        if system:
            turns.append({"role": "system", "content": system})
        turns.append({"role": "user", "content": prompt})
        conversations.append(turns)
    return map_chat(provider, model, conversations, **kwargs)


def contents(completions: Iterable[Completion], *, default: str = "") -> List[str]:
    """Just the text, with failures rendered as ``default``.

    For the very common case of feeding results into something that wants
    strings and treats a blank as "no answer".
    """
    return [c.response.content if c.ok and c.response else default for c in completions]


__all__ = [
    "AdaptiveLimiter",
    "Completion",
    "DEFAULT_CONCURRENCY",
    "Progress",
    "contents",
    "map_chat",
    "map_prompts",
]
