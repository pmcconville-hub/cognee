"""Ingest first-class AGENT.md and MEMORY.md profiles into the graph."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Union
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import AgentProfile, MemoryProfile, Skill
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.engine.utils.generate_node_id import generate_node_id
from cognee.modules.pipelines.models import PipelineContext
from cognee.modules.tools.ingest_skills import (
    SKILL_SOURCE_ROOTS_ENV,
    _resolve_skill_source_path,
)
from cognee.modules.tools.path_safety import trusted_is_dir, trusted_is_file, trusted_rglob
from cognee.modules.tools.profile_parser import (
    PROFILE_ENTRY_NAMES,
    ParsedProfilePackage,
    parse_profile_package,
)
from cognee.shared.logging_utils import get_logger
from cognee.tasks.storage.add_data_points import add_data_points

logger = get_logger("cognee.tools.ingest_profiles")


def _is_profile_entry(path: Path) -> bool:
    return trusted_is_file(path) and path.name.lower() in PROFILE_ENTRY_NAMES


def looks_like_profile_source(data) -> bool:
    """Return True when ``data`` is an AGENT.md/MEMORY.md file or package folder."""
    path = _resolve_skill_source_path(data)
    if path is None:
        return False
    try:
        if _is_profile_entry(path):
            return True
        if trusted_is_dir(path):
            return any(
                p.name.lower() in PROFILE_ENTRY_NAMES
                for p in trusted_rglob(path, "*")
                if trusted_is_file(p)
            )
    except OSError:
        return False
    return False


def _profile_source_data_id(dataset_id: UUID, source: Path) -> UUID:
    """Stable pseudo data id used to attach direct profile writes to dataset ACL tables."""
    return uuid5(NAMESPACE_URL, f"cognee:profiles:{dataset_id}:{source}")


def _make_storage_context(user, dataset, source: Path) -> Optional[PipelineContext]:
    if user is None or dataset is None:
        return None
    return PipelineContext(
        user=user,
        dataset=dataset,
        data_item=SimpleNamespace(id=_profile_source_data_id(dataset.id, source)),
        pipeline_name="profiles_ingest_pipeline",
    )


def _tag(dp: DataPoint, ns: NodeSet, node_set: str) -> None:
    existing = dp.belongs_to_set or []
    existing_names = {s.name if hasattr(s, "name") else s for s in existing}
    if node_set not in existing_names:
        dp.belongs_to_set = list(existing) + [ns]


def _apply_node_set(package: ParsedProfilePackage, node_set: str) -> None:
    ns = NodeSet(id=generate_node_id(f"NodeSet:{node_set}"), name=node_set)
    seen: set[str] = set()

    def tag_once(dp: DataPoint) -> None:
        if str(dp.id) in seen:
            return
        seen.add(str(dp.id))
        _tag(dp, ns, node_set)

    for agent in package.agents:
        tag_once(agent)
    for memory in package.memories:
        tag_once(memory)
    for skill in package.skills:
        tag_once(skill)
        for resource in skill.resources or []:
            tag_once(resource)
        for _edge, pattern in skill.solves or []:
            tag_once(pattern)


def _apply_dataset_scope(package: ParsedProfilePackage, dataset_id: Optional[UUID]) -> None:
    if dataset_id is None:
        return
    scope = str(dataset_id)
    scoped_items: list[AgentProfile | MemoryProfile | Skill] = [
        *package.agents,
        *package.memories,
        *package.skills,
    ]
    for item in scoped_items:
        if not item.dataset_scope:
            item.dataset_scope = [scope]
        elif scope not in item.dataset_scope:
            item.dataset_scope.append(scope)


async def add_profiles(
    source: Union[str, Path],
    source_repo: str = "",
    node_set: str = "profiles",
    user=None,
    dataset=None,
) -> ParsedProfilePackage:
    """Parse AGENT.md/MEMORY.md package files and persist first-class profile DataPoints."""
    path = _resolve_skill_source_path(source)
    if path is None:
        raise PermissionError(
            f"Profile source must be under the current working directory or a root "
            f"listed in {SKILL_SOURCE_ROOTS_ENV}: {source}"
        )

    package = parse_profile_package(path, source_repo=source_repo, base_dir=path)
    if not package.items:
        logger.warning("No AGENT.md or MEMORY.md profiles discovered under %s", source)
        return package

    _apply_dataset_scope(package, getattr(dataset, "id", None))
    _apply_node_set(package, node_set)
    await add_data_points(package.items, ctx=_make_storage_context(user, dataset, path))

    logger.info(
        "Ingested %d agent profile(s), %d memory profile(s), and %d skill(s) from %s",
        len(package.agents),
        len(package.memories),
        len(package.skills),
        source,
    )
    return package
