"""Tests for legacy Kuzu compatibility shims."""

import sys
from importlib.metadata import version as package_version
from pathlib import Path


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "kuzu").is_dir():
            return parent
    raise RuntimeError("Could not find repository root.")


def test_kuzu_import_shim_points_to_ladybug(monkeypatch):
    monkeypatch.syspath_prepend(str(_repo_root()))
    sys.modules.pop("kuzu", None)
    sys.modules.pop("kuzu.database", None)

    from cognee.infrastructure.databases.graph.kuzu.kuzu_migrate import kuzu_migration
    import kuzu
    import kuzu.database
    import ladybug
    import ladybug.database

    assert kuzu.__version__ == package_version("ladybug")
    assert kuzu.Database is ladybug.database.Database
    assert kuzu.database.Database is ladybug.database.Database
    assert callable(kuzu_migration)
