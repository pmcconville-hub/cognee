"""White-box tests for LanceDBAdapter lifecycle synchronization.

Pin the invariant that ``get_connection()`` and ``close()`` cannot race
into a state where a coroutine returns a connection from an adapter that
has already been closed — exactly the bug CodeRabbit flagged in the
"Serialize get_connection and close" comment.
"""

from __future__ import annotations

import asyncio

import pytest

try:
    import lancedb  # noqa: F401

    from cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter import (
        LanceDBAdapter,
    )

    HAS_LANCEDB = True
except ModuleNotFoundError:
    HAS_LANCEDB = False


class _FakeEmbeddingEngine:
    def get_vector_size(self):
        return 3

    def get_batch_size(self):
        return 100

    async def embed_text(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_close_during_pending_get_connection_blocks_resurrection(monkeypatch, tmp_path):
    """The race CodeRabbit flagged: a coroutine that's inside
    ``get_connection()`` mid-``connect_async`` while another coroutine
    runs ``close()`` must not silently overwrite the closed state with
    its newly-built connection. ``_lifecycle_lock`` + post-await
    re-check ensures the in-flight caller raises and the new connection
    is discarded.
    """
    started = asyncio.Event()
    proceed = asyncio.Event()
    real_connect = lancedb.connect_async

    async def slow_connect_async(*args, **kwargs):
        started.set()
        await proceed.wait()
        return await real_connect(*args, **kwargs)

    monkeypatch.setattr(
        "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter.lancedb.connect_async",
        slow_connect_async,
    )

    adapter = LanceDBAdapter(
        url=str(tmp_path / "lance_db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )

    fetch_task = asyncio.create_task(adapter.get_connection())
    await started.wait()
    # Sanity: adapter is now parked inside get_connection's await.
    assert adapter.connection is None
    assert adapter._permanently_closed is False

    # Race in close() while the fetch is suspended.
    await adapter.close()
    assert adapter._permanently_closed is True
    assert adapter.connection is None

    # Let the in-flight connect_async finish.
    proceed.set()

    # The pending get_connection must observe the closed state on its
    # post-await re-check and raise — NOT return a connection that
    # would resurrect the closed adapter.
    with pytest.raises(RuntimeError, match="closed"):
        await fetch_task

    # Closed state is preserved: the throwaway connection didn't sneak
    # back into self.connection.
    assert adapter.connection is None
    assert adapter._permanently_closed is True


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_close_is_idempotent(tmp_path):
    """Double close must not raise or re-shutdown anything."""
    adapter = LanceDBAdapter(
        url=str(tmp_path / "lance_db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )
    await adapter.get_connection()  # make the local connection
    assert adapter.connection is not None

    await adapter.close()
    assert adapter._permanently_closed is True
    assert adapter.connection is None

    # Second close: idempotent, no-op.
    await adapter.close()
    assert adapter._permanently_closed is True


@pytest.mark.asyncio
@pytest.mark.skipif(not HAS_LANCEDB, reason="lancedb not installed")
async def test_concurrent_first_get_connection_returns_same_object(monkeypatch, tmp_path):
    """Two coroutines reaching the lazy-create path simultaneously must
    end up returning the SAME ``self.connection`` object. Without the
    post-await re-check, both would build a fresh ``lancedb.AsyncConnection``
    and the second would clobber ``self.connection`` — leaking the first.
    """
    real_connect = lancedb.connect_async

    async def yielding_connect_async(*args, **kwargs):
        # Force both coroutines to park here before either commits.
        await asyncio.sleep(0)
        return await real_connect(*args, **kwargs)

    monkeypatch.setattr(
        "cognee.infrastructure.databases.vector.lancedb.LanceDBAdapter.lancedb.connect_async",
        yielding_connect_async,
    )

    adapter = LanceDBAdapter(
        url=str(tmp_path / "lance_db"),
        api_key=None,
        embedding_engine=_FakeEmbeddingEngine(),
    )

    c1, c2 = await asyncio.gather(adapter.get_connection(), adapter.get_connection())
    assert c1 is c2, "concurrent first get_connection must converge on one connection"
    assert adapter.connection is c1

    await adapter.close()
