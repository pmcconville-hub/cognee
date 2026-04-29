"""White-box tests for the Kuzu subprocess proxies.

Pin lifecycle invariants the harness relies on but which would otherwise
only be visible at runtime as silent state corruption — specifically that
``close()`` deregisters every replay step the proxy registered so a
post-close worker respawn can't resurrect a handle the user already closed.
"""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("kuzu")

from cognee.infrastructure.databases.graph.kuzu.subprocess.proxy import (
    KuzuSubprocessSession,
    RemoteKuzuConnection,
    RemoteKuzuDatabase,
)


def _start_proxy(tmp_path):
    """Common setup: open a worker session + open a Database against it."""
    session = KuzuSubprocessSession.start()
    db = RemoteKuzuDatabase(
        session,
        db_path=str(tmp_path / "kuzu_db"),
        buffer_pool_size=64 * 1024 * 1024,
        max_num_threads=1,
        max_db_size=128 * 1024 * 1024,
    )
    return session, db


def test_remote_kuzu_database_close_deregisters_replay_steps(tmp_path):
    """``RemoteKuzuDatabase.close()`` must remove every replay step the
    proxy registered. Otherwise a worker respawn would replay the OPEN
    step, calling ``apply_new_handle`` and re-arming a closed proxy.
    """
    session, db = _start_proxy(tmp_path)
    try:
        # The OPEN step is registered in __init__.
        assert len(db._replay_steps) == 1
        assert len(session._replay_steps) == 1

        # init_database adds another step.
        db.init_database()
        assert len(db._replay_steps) == 2
        assert len(session._replay_steps) == 2

        db.close()

        # Both steps gone from session-side state.
        assert len(session._replay_steps) == 0
        assert len(db._replay_steps) == 0
        # close() is idempotent — second call must not blow up.
        db.close()
    finally:
        session.shutdown()


def test_remote_kuzu_connection_close_deregisters_replay_steps(tmp_path):
    """``RemoteKuzuConnection.close()`` must remove the OPEN_CONNECTION
    step *and* every per-extension replay step ``load_extension`` added.
    """
    session, db = _start_proxy(tmp_path)
    try:
        db.init_database()
        # 1 (database OPEN) + 1 (DB_INIT) so far.
        steps_before_conn = len(session._replay_steps)
        assert steps_before_conn == 2

        conn = RemoteKuzuConnection(session, db)
        # OPEN_CONNECTION registered.
        assert len(conn._replay_steps) == 1
        assert len(session._replay_steps) == steps_before_conn + 1

        # Each load_extension adds a replay step.
        conn.load_extension("JSON")
        assert len(conn._replay_steps) == 2
        assert len(session._replay_steps) == steps_before_conn + 2

        conn.close()
        # All connection-side steps gone; database steps untouched.
        assert len(conn._replay_steps) == 0
        assert len(session._replay_steps) == steps_before_conn
        # Idempotent close.
        conn.close()
    finally:
        session.shutdown()


def test_post_close_respawn_does_not_resurrect_handle(tmp_path):
    """The systemic regression we're guarding against: after ``close()``,
    a respawn must NOT replay the closed proxy's OPEN step. We simulate
    a respawn by replaying the steps directly and checking that the
    closed proxy's ``_handle_id`` stays ``None``.
    """
    session, db = _start_proxy(tmp_path)
    try:
        db.init_database()
        db.close()
        assert db._handle_id is None
        assert len(session._replay_steps) == 0

        # Walk what ``_respawn`` would do — execute every remaining replay
        # step's ``make_request``. With deregistration in place, none of
        # the closed proxy's steps remain, so the loop is a no-op.
        for step in list(session._replay_steps):
            req = step.make_request()
            # Sanity: nothing should reference the closed db's open kwargs.
            assert req.kwargs.get("database_path") != str(tmp_path / "kuzu_db"), (
                "closed proxy's OPEN step must have been deregistered"
            )

        # Closed proxy stays closed.
        assert db._handle_id is None
    finally:
        session.shutdown()
