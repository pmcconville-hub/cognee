"""LRU cache that calls .close() on evicted entries."""

import asyncio
import logging
from collections import OrderedDict
from functools import wraps
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)


# Strong refs for fire-and-forget async close() tasks. ``asyncio.create_task``
# returns a task whose only strong reference is our local variable; without
# anchoring here, Python's gc can collect an in-flight eviction task before
# it completes (and the async close never runs). Tasks remove themselves on
# done, so this set's size tracks currently-pending close operations.
_PENDING_CLOSE_TASKS: set = set()


def _close_value(value):
    """Call close() on a value, scheduling it as a task if it returns a coroutine.

    If close() is async and no event loop is running, falls back to
    ``asyncio.run()`` to ensure cleanup is not silently skipped.
    """
    if not hasattr(value, "close"):
        return
    try:
        result = value.close()
    except Exception:
        # A raising close() must not abort the surrounding loop (eviction
        # iteration in ``cache_clear`` or a single eviction in
        # ``get_or_create``). Log and keep going — the caller already lost
        # the reference, and any partial cleanup is better than none.
        logger.warning(
            "Failed to close %s during eviction",
            type(value).__name__,
            exc_info=True,
        )
        return
    if asyncio.iscoroutine(result):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(result)
            except Exception:
                logger.warning(
                    "Failed to run async close() for %s during eviction",
                    type(value).__name__,
                    exc_info=True,
                )
            return

        task = loop.create_task(result)
        _PENDING_CLOSE_TASKS.add(task)

        def _on_close_done(done_task, _value_type=type(value).__name__):
            # Always drop the strong ref so the task can be collected.
            _PENDING_CLOSE_TASKS.discard(done_task)
            # Retrieve the result so failures surface through the same
            # structured ``logger.warning`` channel as the sync /
            # ``asyncio.run()`` branches above. Without this, an async
            # ``close()`` that raises only surfaces as Python's
            # "Task exception was never retrieved" warning at GC time,
            # which ops greps for ``Failed to run async close()`` would miss.
            try:
                done_task.result()
            except Exception:
                logger.warning(
                    "Failed to run async close() for %s during eviction",
                    _value_type,
                    exc_info=True,
                )

        task.add_done_callback(_on_close_done)


class ClosingLRUCache:
    """Thread-safe LRU cache that calls ``close()`` on evicted values.

    Unlike :func:`functools.lru_cache`, evicted entries are cleaned up
    deterministically — their ``close()`` method (if present) is called
    at the moment of eviction, while all fields are still alive.
    """

    def __init__(self, maxsize: Optional[int] = 128):
        """``maxsize`` semantics mirror ``functools.lru_cache``:

        - ``int > 0`` — bounded LRU. The least-recently-used entry is evicted
          on insert and its ``close()`` is called.
        - ``int <= 0`` — cache disabled. ``factory()`` is called on every
          request and the result is returned to the caller without being
          stored. ``close()`` is NOT called: the caller owns the lifecycle,
          just like ``functools.lru_cache(maxsize=0)`` returns a fresh value
          per call.
        - ``None`` — unbounded. Entries are never evicted.
        """
        if isinstance(maxsize, int):
            if maxsize < 0:
                maxsize = 0
        elif maxsize is not None:
            raise TypeError("maxsize must be an int or None")
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = Lock()

    def get_or_create(self, key, factory):
        # Disabled-cache mode: act as a pass-through. Caller owns the value's
        # lifecycle — matches ``functools.lru_cache(maxsize=0)``.
        if self._maxsize == 0:
            return factory()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        value = factory()

        with self._lock:
            # Re-check after releasing lock — another thread may have created it.
            if key in self._cache:
                self._cache.move_to_end(key)
                _close_value(value)
                return self._cache[key]

            # ``None`` means unbounded — skip the eviction check entirely.
            if self._maxsize is not None and len(self._cache) >= self._maxsize:
                _, evicted = self._cache.popitem(last=False)
                _close_value(evicted)

            self._cache[key] = value
            return value

    def cache_clear(self):
        """Close and remove all cached entries."""
        with self._lock:
            for value in self._cache.values():
                _close_value(value)
            self._cache.clear()

    def cache_info(self):
        """Return current size and max size."""
        with self._lock:
            return {"size": len(self._cache), "maxsize": self._maxsize}


def closing_lru_cache(maxsize: Optional[int] = 128):
    """Decorator that caches return values in a :class:`ClosingLRUCache`.

    Drop-in replacement for ``@functools.lru_cache`` that calls ``.close()``
    on values evicted from the cache. ``maxsize`` semantics match
    ``functools.lru_cache``: positive int = bounded; ``0`` (or negative) =
    disabled; ``None`` = unbounded.

    The decorated function gains ``cache_clear()`` and ``cache_info()``
    attributes, matching the ``lru_cache`` API, as well as a ``__wrapped__``
    attribute pointing to the original function.
    """

    def decorator(fn):
        cache = ClosingLRUCache(maxsize=maxsize)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            key = args + tuple(sorted(kwargs.items()))
            return cache.get_or_create(key, lambda: fn(*args, **kwargs))

        wrapper.cache_clear = cache.cache_clear
        wrapper.cache_info = cache.cache_info
        wrapper.__wrapped__ = fn
        return wrapper

    return decorator
