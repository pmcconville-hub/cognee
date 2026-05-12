"""Compatibility shim exposing Ladybug under the legacy Kuzu module name."""

from importlib.metadata import PackageNotFoundError, version as package_version

import ladybug as _ladybug
from ladybug import AsyncConnection, Connection, PreparedStatement, QueryResult, Type
from ladybug.database import Database  # noqa: F401

try:
    __version__ = package_version("ladybug")
except PackageNotFoundError:
    __version__ = getattr(_ladybug, "__version__", "unknown")

version = __version__

__all__ = [
    "AsyncConnection",
    "Connection",
    "Database",
    "PreparedStatement",
    "QueryResult",
    "Type",
    "__version__",
    "version",
    "storage_version",
]


def __getattr__(name: str):
    if name == "storage_version":
        return getattr(_ladybug, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
