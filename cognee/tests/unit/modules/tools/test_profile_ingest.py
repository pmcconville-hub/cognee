from __future__ import annotations

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4


_AGENT_MD = """\
---
name: researcher
description: Research agent for project analysis.
allowed-tools:
  - memory_search
allowed-skills:
  - summarize
memory-profiles:
  - project-memory
model_preferences:
  provider: openai
personalization:
  tone: concise
  use_memory: true
context:
  summary: true
  search_top_k: 8
context_budget_tokens: 2000
observer: researcher
targets:
  - user
observation:
  observe_others: true
---
# Researcher

Use project memory and approved skills to answer multi-step research questions.

## When to Use
- Project analysis
- Cross-document synthesis
"""

_MEMORY_MD = """\
---
name: project-memory
description: Project memory.
facts:
  - Cognee stores memory in graph and vector databases.
preferences:
  - Prefer concise technical summaries.
observations:
  - The user prefers project-scoped answers grounded in durable graph memory.
observer: researcher
target: user
semantic_search:
  search_top_k: 10
  search_max_distance: 0.8
  include_most_frequent: true
limit_to_session: true
session_scope: active-session
summary_policy: Summarize older sessions before adding recent messages.
scope: project
---
# Project Memory

Cognee stores durable project facts in graph and vector databases.
The user prefers project-scoped answers grounded in durable graph memory.

## Representation
The user prefers project-scoped answers grounded in durable graph memory.

## Recall Policy
Recall facts from the active project dataset first.

## Write Policy
Only write durable facts that are useful across sessions.
"""

_SKILL_MD = """\
---
name: summarize
description: Summarize technical notes.
allowed-tools: memory_search
---
# Instructions

Condense the input into concise technical bullets.
"""


def _make_profile_package() -> Path:
    root = Path(tempfile.mkdtemp(dir=Path.cwd())) / "researcher"
    (root / "skills" / "summarize").mkdir(parents=True)
    (root / "AGENT.md").write_text(_AGENT_MD)
    (root / "MEMORY.md").write_text(_MEMORY_MD)
    (root / "skills" / "summarize" / "SKILL.md").write_text(_SKILL_MD)
    return root


class TestProfileIngest(unittest.TestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_profile_source_detects_agent_and_memory_files(self):
        from cognee.modules.tools.ingest_profiles import looks_like_profile_source

        root = _make_profile_package()
        try:
            assert looks_like_profile_source(str(root))
            assert looks_like_profile_source(str(root / "AGENT.md"))
            assert looks_like_profile_source(str(root / "MEMORY.md"))
        finally:
            shutil.rmtree(root.parent)

    def test_profile_package_persists_first_class_profiles_and_links_skills(self):
        from cognee.modules.engine.models import AgentProfile, MemoryProfile, Skill
        from cognee.modules.tools.ingest_profiles import add_profiles

        root = _make_profile_package()
        dataset = SimpleNamespace(id=uuid4(), name="profiles")
        user = SimpleNamespace(id=uuid4(), tenant_id=uuid4())
        try:

            async def _run():
                with patch(
                    "cognee.modules.tools.ingest_profiles.add_data_points",
                    new_callable=AsyncMock,
                ) as mock_add:
                    package = await add_profiles(
                        str(root),
                        node_set="profiles",
                        user=user,
                        dataset=dataset,
                    )
                    return package, mock_add

            package, mock_add = self._run(_run())

            assert [agent.name for agent in package.agents] == ["researcher"]
            assert [memory.name for memory in package.memories] == ["project-memory"]
            assert [skill.name for skill in package.skills] == ["summarize"]

            agent = package.agents[0]
            memory = package.memories[0]
            skill = package.skills[0]

            assert isinstance(agent, AgentProfile)
            assert isinstance(memory, MemoryProfile)
            assert isinstance(skill, Skill)
            assert agent.declared_tools == ["memory_search"]
            assert agent.personalization == {"tone": "concise", "use_memory": True}
            assert agent.context_policy == {"summary": True, "search_top_k": 8}
            assert agent.context_budget_tokens == 2000
            assert agent.observer == "researcher"
            assert agent.targets == ["user"]
            assert agent.observe_me is True
            assert agent.observe_others is True
            assert agent.uses[0][1] is skill
            assert agent.guided_by[0][1] is memory
            assert memory.memory.startswith("# Project Memory")
            assert "Cognee stores durable project facts" in memory.memory
            assert memory.observations == [
                "The user prefers project-scoped answers grounded in durable graph memory."
            ]
            assert memory.representation.startswith("The user prefers project-scoped answers")
            assert memory.observer == "researcher"
            assert memory.target == "user"
            assert memory.semantic_search_policy == {
                "search_top_k": 10,
                "search_max_distance": 0.8,
                "include_most_frequent": True,
            }
            assert memory.limit_to_session is True
            assert memory.session_scope == "active-session"
            assert memory.summary_policy.startswith("Summarize older sessions")
            assert memory.constrains[0][1] is skill
            assert agent.dataset_scope == [str(dataset.id)]
            assert memory.dataset_scope == [str(dataset.id)]
            assert skill.dataset_scope == [str(dataset.id)]

            assert mock_add.await_count == 1
            persisted = mock_add.await_args.args[0]
            assert {type(item).__name__ for item in persisted} == {
                "AgentProfile",
                "MemoryProfile",
                "Skill",
            }
            assert mock_add.await_args.kwargs["ctx"].dataset is dataset
        finally:
            shutil.rmtree(root.parent)

    def test_profile_package_parent_directory_discovers_nested_packages(self):
        from cognee.modules.tools.profile_parser import parse_profile_package

        first = _make_profile_package()
        parent = first.parent
        second = parent / "reviewer"
        (second / "skills" / "summarize").mkdir(parents=True)
        (second / "AGENT.md").write_text(_AGENT_MD.replace("researcher", "reviewer"))
        (second / "MEMORY.md").write_text(_MEMORY_MD.replace("project-memory", "review-memory"))
        (second / "skills" / "summarize" / "SKILL.md").write_text(_SKILL_MD)
        try:
            package = parse_profile_package(parent)

            assert {agent.name for agent in package.agents} == {"researcher", "reviewer"}
            assert {memory.name for memory in package.memories} == {
                "project-memory",
                "review-memory",
            }
            assert len(package.skills) == 2
            researcher = next(agent for agent in package.agents if agent.name == "researcher")
            reviewer = next(agent for agent in package.agents if agent.name == "reviewer")
            assert researcher.guided_by[0][1].name == "project-memory"
            assert reviewer.guided_by == []
        finally:
            shutil.rmtree(parent)

    def test_skill_folder_parser_does_not_treat_profile_files_as_skills(self):
        from cognee.modules.tools.skill_parser import parse_skills_folder

        root = Path(tempfile.mkdtemp(dir=Path.cwd()))
        try:
            (root / "AGENT.md").write_text(_AGENT_MD)
            (root / "MEMORY.md").write_text(_MEMORY_MD)

            assert parse_skills_folder(root) == []
        finally:
            shutil.rmtree(root)
