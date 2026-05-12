from uuid import UUID
from typing import Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from cognee.infrastructure.databases.graph.config import get_graph_config
from cognee.infrastructure.databases.graph.get_graph_engine import (
    create_graph_engine,
    evict_graph_engine,
)
from cognee.modules.users.models import User, DatasetDatabase


def _replace_database_in_url(url: str, new_db: str) -> str:
    """Replace the database name in a postgresql+asyncpg:// connection string."""
    base = url.rsplit("/", 1)[0]
    return f"{base}/{new_db}"


class PostgresGraphDatasetDatabaseHandler:
    """Handler for per-dataset Postgres graph databases."""

    @classmethod
    async def create_dataset(cls, dataset_id: Optional[UUID], user: Optional[User]) -> dict:
        graph_config = get_graph_config()

        if graph_config.graph_database_provider != "postgres":
            raise ValueError(
                "PostgresGraphDatasetDatabaseHandler can only be used "
                "with postgres graph database provider."
            )

        graph_db_name = f"{dataset_id}"
        dataset_url = _replace_database_in_url(graph_config.graph_database_url, graph_db_name)

        new_graph_config = {
            "graph_database_provider": "postgres",
            "graph_database_url": dataset_url,
            "graph_database_name": graph_db_name,
            "graph_database_key": graph_config.graph_database_key,
            "graph_dataset_database_handler": "postgres_graph",
            "graph_database_connection_info": {},
        }

        await cls._create_pg_database(graph_config.graph_database_url, graph_db_name)

        return new_graph_config

    @classmethod
    async def _create_pg_database(cls, base_url: str, db_name: str) -> None:
        """Create the per-dataset Postgres database and initialize its tables."""
        maintenance_url = _replace_database_in_url(base_url, "postgres")

        maintenance_engine = create_async_engine(maintenance_url)
        try:
            connection = await maintenance_engine.connect()
            try:
                connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
                result = await connection.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db"),
                    {"db": db_name},
                )
                if not result.scalar():
                    await connection.execute(text(f'CREATE DATABASE "{db_name}";'))
            finally:
                await connection.close()
        finally:
            await maintenance_engine.dispose()

        dataset_url = _replace_database_in_url(base_url, db_name)
        engine = create_graph_engine(
            graph_database_provider="postgres",
            graph_file_path="",
            graph_database_url=dataset_url,
        )
        await engine.initialize()

    @classmethod
    async def resolve_dataset_connection_info(
        cls, dataset_database: DatasetDatabase
    ) -> DatasetDatabase:
        return dataset_database

    @classmethod
    async def delete_dataset(cls, dataset_database: DatasetDatabase) -> None:
        graph_url = dataset_database.graph_database_url

        evict_graph_engine(
            graph_database_provider="postgres",
            graph_file_path="",
            graph_database_url=graph_url,
        )

        maintenance_url = _replace_database_in_url(graph_url, "postgres")
        db_name = dataset_database.graph_database_name

        maintenance_engine = create_async_engine(maintenance_url)
        try:
            connection = await maintenance_engine.connect()
            try:
                connection = await connection.execution_options(isolation_level="AUTOCOMMIT")
                await connection.execute(
                    text(
                        "SELECT pg_terminate_backend(pid) "
                        "FROM pg_stat_activity "
                        "WHERE datname = :db AND pid <> pg_backend_pid()"
                    ),
                    {"db": db_name},
                )
                await connection.execute(text(f'DROP DATABASE IF EXISTS "{db_name}";'))
            finally:
                await connection.close()
        finally:
            await maintenance_engine.dispose()
