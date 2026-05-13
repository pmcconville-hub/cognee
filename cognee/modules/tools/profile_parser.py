"""Parser for first-class AGENT.md and MEMORY.md profile packages."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid5

import yaml

from cognee.infrastructure.engine import Edge
from cognee.modules.engine.models import AgentProfile, MemoryProfile, Skill
from cognee.modules.tools.path_safety import (
    trusted_is_dir,
    trusted_is_file,
    trusted_read_text,
    trusted_rglob,
)
from cognee.modules.tools.skill_parser import parse_skills_folder
from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

NAMESPACE = UUID("2f0e7c1a-b5a4-4f6d-9c52-38c41afac011")

AGENT_ENTRY_CANDIDATES = ["AGENT.md", "agent.md", "Agent.md"]
MEMORY_ENTRY_CANDIDATES = ["MEMORY.md", "memory.md", "Memory.md"]
PROFILE_ENTRY_NAMES = {name.lower() for name in AGENT_ENTRY_CANDIDATES + MEMORY_ENTRY_CANDIDATES}

_NAME_ALIASES = ("name", "title", "agent_name", "memory_name")
_DESCRIPTION_ALIASES = ("description", "summary", "short_description", "about")
_TOOLS_ALIASES = ("allowed-tools", "allowed_tools", "declared_tools", "tools")
_ALLOWED_SKILLS_ALIASES = ("allowed-skills", "allowed_skills", "skills")
_MEMORY_PROFILES_ALIASES = ("memory-profiles", "memory_profiles", "memories")
_TRIGGER_ALIASES = ("triggers", "activation", "routing_triggers", "routing-triggers")
_PERSONALIZATION_ALIASES = ("personalization", "personalization_policy", "personalization-policy")
_CONTEXT_POLICY_ALIASES = ("context", "context_policy", "context-policy")
_CONTEXT_BUDGET_ALIASES = (
    "context_budget_tokens",
    "context-budget-tokens",
    "context_tokens",
    "context-tokens",
)
_OBSERVER_ALIASES = (
    "observer",
    "perspective",
)
_TARGET_ALIASES = (
    "targets",
    "target",
    "target_profiles",
    "target-profiles",
)
_OBSERVE_ME_ALIASES = ("observe_me", "observe-me")
_OBSERVE_OTHERS_ALIASES = ("observe_others", "observe-others")
_REPRESENTATION_ALIASES = (
    "representation",
    "working_representation",
    "working-representation",
)
_OBSERVATIONS_ALIASES = ("observations", "memory_items", "memory-items")
_SEMANTIC_SEARCH_ALIASES = (
    "semantic_search",
    "semantic-search",
    "semantic_search_policy",
    "semantic-search-policy",
    "search_policy",
    "search-policy",
    "retrieval",
    "retrieval_policy",
    "retrieval-policy",
)
_SUMMARY_POLICY_ALIASES = ("summary_policy", "summary-policy")
_LIMIT_TO_SESSION_ALIASES = ("limit_to_session", "limit-to-session")
_SESSION_SCOPE_ALIASES = ("session_scope", "session-scope")


@dataclass
class ParsedProfilePackage:
    agents: List[AgentProfile]
    memories: List[MemoryProfile]
    skills: List[Skill]

    @property
    def items(self) -> List[AgentProfile | MemoryProfile | Skill]:
        return [*self.agents, *self.memories, *self.skills]


def _deterministic_id(namespace_key: str) -> UUID:
    return uuid5(NAMESPACE, namespace_key)


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parse_frontmatter(text: str) -> tuple[Dict[str, Any], str]:
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n?(.*)", text, re.DOTALL)
    if not match:
        return {}, text.strip()

    try:
        frontmatter = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        logger.warning("Failed to parse profile YAML frontmatter: %s", exc)
        frontmatter = {}

    if not isinstance(frontmatter, dict):
        frontmatter = {}

    return frontmatter, match.group(2).strip()


def _pop_first(d: Dict[str, Any], aliases: tuple[str, ...]) -> Any:
    for key in aliases:
        if key in d:
            return d.pop(key)
    return None


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in (_clean_list_item(item) for item in value) if item]
    if isinstance(value, tuple):
        return [item for item in (_clean_list_item(item) for item in value) if item]
    return [
        item
        for item in (_clean_list_item(item) for item in re.split(r"[,;\n]", str(value)))
        if item
    ]


def _clean_list_item(value: Any) -> str:
    item = str(value).strip()
    return re.sub(r"^(?:[-*]\s+|\d+[.)]\s+)", "", item).strip()


def _mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return {"items": _string_list(value)}
    text = str(value).strip()
    return {"instructions": text} if text else {}


def _optional_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _description(frontmatter: Dict[str, Any], body: str) -> str:
    value = _pop_first(frontmatter, _DESCRIPTION_ALIASES)
    if value:
        return str(value).strip()
    for para in re.split(r"\n{2,}", body):
        para = para.strip()
        if para and not para.startswith("#") and len(para) >= 30:
            return re.sub(r"[`*_~]", "", para)[:500]
    return ""


def _section(body: str, *headings: str) -> str:
    escaped = "|".join(re.escape(heading) for heading in headings)
    match = re.search(
        rf"^##\s+(?:{escaped})\s*\n(.*?)(?=^##\s+|\Z)",
        body,
        re.DOTALL | re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _profile_key(path: Path) -> str:
    if path.name.lower() in PROFILE_ENTRY_NAMES:
        return path.parent.name
    return path.stem


def _is_relative_to(path: Path, base_dir: Path) -> bool:
    try:
        path_str = os.path.normpath(os.path.realpath(os.path.abspath(os.fspath(path))))
        base_str = os.path.normpath(os.path.realpath(os.path.abspath(os.fspath(base_dir))))
    except (OSError, RuntimeError, ValueError):
        return False
    base_prefix = base_str if base_str.endswith(os.sep) else f"{base_str}{os.sep}"
    return path_str == base_str or path_str.startswith(base_prefix)


def _read_profile_file(
    path: Path, base_dir: Optional[Path] = None
) -> Optional[tuple[str, str, dict]]:
    if base_dir is not None and not _is_relative_to(path, base_dir):
        logger.warning("Skipping profile outside allowed base directory")
        return None
    if not trusted_is_file(path):
        return None
    raw_text = trusted_read_text(path, encoding="utf-8")
    if not raw_text.strip():
        return None
    frontmatter, body = _parse_frontmatter(raw_text)
    return raw_text, body, frontmatter


def parse_agent_file(
    agent_md: Path,
    source_repo: str = "",
    profile_key: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Optional[AgentProfile]:
    agent_md = Path(agent_md)
    loaded = _read_profile_file(agent_md, base_dir=base_dir)
    if loaded is None:
        return None
    raw_text, body, frontmatter = loaded

    profile_key = profile_key or _profile_key(agent_md)
    display_name = _pop_first(frontmatter, _NAME_ALIASES)
    name = str(display_name).strip() if display_name else profile_key
    description = _description(frontmatter, body)
    tools = _string_list(_pop_first(frontmatter, _TOOLS_ALIASES))
    allowed_skills = _string_list(_pop_first(frontmatter, _ALLOWED_SKILLS_ALIASES))
    memory_profiles = _string_list(_pop_first(frontmatter, _MEMORY_PROFILES_ALIASES))
    triggers = _string_list(_pop_first(frontmatter, _TRIGGER_ALIASES))
    if not triggers:
        triggers = _string_list(_section(body, "When to Use", "When to Activate"))

    personalization = _mapping(_pop_first(frontmatter, _PERSONALIZATION_ALIASES))
    personalization_section = _section(body, "Personalization", "Personalization Policy")
    if personalization_section:
        personalization.setdefault("instructions", personalization_section)

    context_policy = _mapping(_pop_first(frontmatter, _CONTEXT_POLICY_ALIASES))
    context_section = _section(body, "Context Policy", "Context Retrieval", "Context")
    if context_section:
        context_policy.setdefault("instructions", context_section)

    context_budget_tokens = _optional_int(_pop_first(frontmatter, _CONTEXT_BUDGET_ALIASES))
    if context_budget_tokens is None:
        context_budget_tokens = _optional_int(
            context_policy.get("tokens") or context_policy.get("context_budget_tokens")
        )

    observer = str(_pop_first(frontmatter, _OBSERVER_ALIASES) or "").strip()
    targets = _string_list(_pop_first(frontmatter, _TARGET_ALIASES))

    observation = frontmatter.pop("observation", frontmatter.pop("observe", {}))
    if not isinstance(observation, dict):
        observation = {}
    if not observer:
        observer = str(observation.pop("observer", "")).strip()
    if not targets:
        targets = _string_list(observation.pop("targets", None))
    observe_me = _optional_bool(
        _pop_first(frontmatter, _OBSERVE_ME_ALIASES)
        if any(alias in frontmatter for alias in _OBSERVE_ME_ALIASES)
        else observation.pop("observe_me", observation.pop("observe-me", None)),
        True,
    )
    observe_others = _optional_bool(
        _pop_first(frontmatter, _OBSERVE_OTHERS_ALIASES)
        if any(alias in frontmatter for alias in _OBSERVE_OTHERS_ALIASES)
        else observation.pop("observe_others", observation.pop("observe-others", None)),
        False,
    )

    model_preferences = frontmatter.pop(
        "model_preferences", frontmatter.pop("model-preferences", {})
    )
    if not isinstance(model_preferences, dict):
        model_preferences = {"preference": model_preferences}

    extra = frontmatter or None

    return AgentProfile(
        id=_deterministic_id(f"agent:{name}"),
        name=name,
        description=description,
        instructions=body,
        declared_tools=tools,
        allowed_skills=allowed_skills,
        memory_profiles=memory_profiles,
        model_preferences=model_preferences,
        routing_triggers=triggers,
        personalization=personalization,
        context_policy=context_policy,
        observer=observer,
        targets=targets,
        context_budget_tokens=context_budget_tokens,
        observe_me=observe_me,
        observe_others=observe_others,
        source_path=str(agent_md.parent),
        source_repo=source_repo,
        content_hash=_content_hash(raw_text),
        is_active=True,
        extra_metadata=extra,
    )


def parse_memory_file(
    memory_md: Path,
    source_repo: str = "",
    profile_key: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Optional[MemoryProfile]:
    memory_md = Path(memory_md)
    loaded = _read_profile_file(memory_md, base_dir=base_dir)
    if loaded is None:
        return None
    raw_text, body, frontmatter = loaded

    profile_key = profile_key or _profile_key(memory_md)
    display_name = _pop_first(frontmatter, _NAME_ALIASES)
    name = str(display_name).strip() if display_name else profile_key
    description = _description(frontmatter, body)
    facts = _string_list(frontmatter.pop("facts", None))
    preferences = _string_list(frontmatter.pop("preferences", None))
    observations = _string_list(_pop_first(frontmatter, _OBSERVATIONS_ALIASES))
    if not observations:
        observations = _string_list(_section(body, "Observations", "Memory Items"))
    representation = str(
        _pop_first(frontmatter, _REPRESENTATION_ALIASES)
        or _section(body, "Representation", "Working Representation", "Memory Representation")
    ).strip()
    observer = str(_pop_first(frontmatter, _OBSERVER_ALIASES) or "").strip()
    target = str(
        frontmatter.pop(
            "observed",
            frontmatter.pop("target", ""),
        )
    ).strip()
    semantic_search_policy = _mapping(_pop_first(frontmatter, _SEMANTIC_SEARCH_ALIASES))
    semantic_search_section = _section(body, "Semantic Search", "Search Policy", "Retrieval Policy")
    if semantic_search_section:
        semantic_search_policy.setdefault("instructions", semantic_search_section)
    limit_to_session_value = _pop_first(frontmatter, _LIMIT_TO_SESSION_ALIASES)
    if limit_to_session_value is None:
        limit_to_session_value = semantic_search_policy.get(
            "limit_to_session", semantic_search_policy.get("limit-to-session")
        )
    limit_to_session = _optional_bool(limit_to_session_value, False)
    session_scope = str(_pop_first(frontmatter, _SESSION_SCOPE_ALIASES) or "").strip()
    recall_policy = str(
        frontmatter.pop("recall_policy", frontmatter.pop("recall-policy", ""))
        or _section(body, "Recall Policy", "Retrieval Policy")
    ).strip()
    write_policy = str(
        frontmatter.pop("write_policy", frontmatter.pop("write-policy", ""))
        or _section(body, "Write Policy", "Update Policy")
    ).strip()
    retention_policy = str(
        frontmatter.pop("retention_policy", frontmatter.pop("retention-policy", ""))
        or _section(body, "Retention Policy")
    ).strip()
    privacy_policy = str(
        frontmatter.pop("privacy_policy", frontmatter.pop("privacy-policy", ""))
        or _section(body, "Privacy Policy")
    ).strip()
    summary_policy = str(
        _pop_first(frontmatter, _SUMMARY_POLICY_ALIASES) or _section(body, "Summary Policy")
    ).strip()
    scope = str(frontmatter.pop("scope", "")).strip()

    extra = frontmatter or None

    return MemoryProfile(
        id=_deterministic_id(f"memory:{name}"),
        name=name,
        description=description,
        memory=body,
        facts=facts,
        preferences=preferences,
        observations=observations,
        representation=representation,
        observer=observer,
        target=target,
        recall_policy=recall_policy,
        write_policy=write_policy,
        retention_policy=retention_policy,
        privacy_policy=privacy_policy,
        summary_policy=summary_policy,
        semantic_search_policy=semantic_search_policy,
        limit_to_session=limit_to_session,
        session_scope=session_scope,
        scope=scope,
        source_path=str(memory_md.parent),
        source_repo=source_repo,
        content_hash=_content_hash(raw_text),
        is_active=True,
        extra_metadata=extra,
    )


def _find_entry(root: Path, candidates: List[str]) -> Optional[Path]:
    for name in candidates:
        candidate = root / name
        if trusted_is_file(candidate):
            return candidate
    return None


def _has_profile_entry(root: Path) -> bool:
    return _find_entry(root, AGENT_ENTRY_CANDIDATES + MEMORY_ENTRY_CANDIDATES) is not None


def _profile_roots(root: Path) -> List[Path]:
    roots: List[Path] = []
    if _has_profile_entry(root):
        roots.append(root)

    for child in sorted(trusted_rglob(root, "*")):
        if trusted_is_dir(child) and _has_profile_entry(child):
            roots.append(child)

    return roots


def _link_profiles(package: ParsedProfilePackage) -> None:
    skills_by_name = {skill.name: skill for skill in package.skills}
    memories_by_name = {memory.name: memory for memory in package.memories}

    for memory in package.memories:
        memory.constrains = [
            (Edge(relationship_type="constrains"), skill) for skill in package.skills
        ]

    for agent in package.agents:
        selected_skill_names = agent.allowed_skills or list(skills_by_name)
        agent.uses = [
            (Edge(relationship_type="uses"), skills_by_name[name])
            for name in selected_skill_names
            if name in skills_by_name
        ]

        selected_memory_names = agent.memory_profiles or list(memories_by_name)
        agent.guided_by = [
            (Edge(relationship_type="guided_by"), memories_by_name[name])
            for name in selected_memory_names
            if name in memories_by_name
        ]


def _parse_profile_directory(
    path: Path,
    source_repo: str,
    base_dir: Path,
) -> ParsedProfilePackage:
    agents: List[AgentProfile] = []
    memories: List[MemoryProfile] = []
    skills: List[Skill] = []

    agent_entry = _find_entry(path, AGENT_ENTRY_CANDIDATES)
    if agent_entry is not None:
        agent = parse_agent_file(agent_entry, source_repo=source_repo, base_dir=base_dir)
        if agent is not None:
            agents.append(agent)

    memory_entry = _find_entry(path, MEMORY_ENTRY_CANDIDATES)
    if memory_entry is not None:
        memory = parse_memory_file(memory_entry, source_repo=source_repo, base_dir=base_dir)
        if memory is not None:
            memories.append(memory)

    skills_dir = path / "skills"
    if trusted_is_dir(skills_dir):
        skills.extend(parse_skills_folder(skills_dir, source_repo=source_repo, base_dir=base_dir))

    package = ParsedProfilePackage(agents=agents, memories=memories, skills=skills)
    _link_profiles(package)
    return package


def parse_profile_package(
    source: str | Path,
    source_repo: str = "",
    base_dir: Optional[Path] = None,
) -> ParsedProfilePackage:
    path = Path(source)
    base_dir = Path(base_dir) if base_dir is not None else path
    source_repo = source_repo or path.stem if trusted_is_file(path) else source_repo or path.name

    if trusted_is_file(path):
        agents: List[AgentProfile] = []
        memories: List[MemoryProfile] = []
        lowered = path.name.lower()
        if lowered in {name.lower() for name in AGENT_ENTRY_CANDIDATES}:
            agent = parse_agent_file(path, source_repo=source_repo, base_dir=base_dir.parent)
            if agent is not None:
                agents.append(agent)
        elif lowered in {name.lower() for name in MEMORY_ENTRY_CANDIDATES}:
            memory = parse_memory_file(path, source_repo=source_repo, base_dir=base_dir.parent)
            if memory is not None:
                memories.append(memory)
        else:
            raise ValueError(f"Unsupported profile file: {source}")
        package = ParsedProfilePackage(agents=agents, memories=memories, skills=[])
        _link_profiles(package)
        return package
    elif trusted_is_dir(path):
        aggregate = ParsedProfilePackage(agents=[], memories=[], skills=[])
        for profile_root in _profile_roots(path):
            package = _parse_profile_directory(
                profile_root,
                source_repo=source_repo,
                base_dir=base_dir,
            )
            aggregate.agents.extend(package.agents)
            aggregate.memories.extend(package.memories)
            aggregate.skills.extend(package.skills)
        return aggregate
    else:
        raise FileNotFoundError(f"Profile source not found: {source}")
