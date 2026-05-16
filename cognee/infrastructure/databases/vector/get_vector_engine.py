from .config import get_vectordb_context_config
from .create_vector_engine import create_vector_engine


class _VectorEngineHandle:
    """Auto-refreshing handle that re-resolves through the cache on every access.

    If the cached engine was closed (e.g. by ``prune_system`` or
    ``cache_clear``), the next attribute access transparently gets a
    fresh one from the cache.  Callers can hold a reference across
    prune boundaries without hitting "adapter is closed" errors.
    """

    __slots__ = ("_config",)

    def __init__(self, config: dict):
        object.__setattr__(self, "_config", config)

    def _engine(self):
        return create_vector_engine(**self._config)

    def __getattr__(self, name):
        return getattr(self._engine(), name)

    def __repr__(self):
        return f"<VectorEngineHandle config={self._config!r}>"


def get_vector_engine():
    return _VectorEngineHandle(get_vectordb_context_config())
