"""Shared Kuzu/Ladybug graph database defaults.

This module is intentionally dependency-free so configuration modules can
import these values without importing graph adapters.
"""

DEFAULT_KUZU_BUFFER_POOL_SIZE = 2 * 1024 * 1024 * 1024  # 2 GB
DEFAULT_KUZU_MAX_DB_SIZE = 4 * 1024 * 1024 * 1024  # 4 GB
