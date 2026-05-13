from typing import Annotated, Any, Dict, List, Optional

from pydantic import Field

from cognee.infrastructure.engine import DataPoint, Edge, Embeddable, LLMContext, Dedup
from cognee.modules.engine.models.Skill import Skill


class MemoryProfile(DataPoint):
    """Persistent memory content loaded from MEMORY.md."""

    name: Annotated[str, Dedup()]
    description: Annotated[str, Embeddable(), LLMContext()] = ""
    memory: Annotated[str, Embeddable(), LLMContext("Memory content")] = ""
    facts: List[str] = Field(default_factory=list)
    preferences: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    representation: Annotated[str, Embeddable(), LLMContext("Memory representation")] = ""
    observer: str = ""
    target: str = ""
    recall_policy: str = ""
    write_policy: str = ""
    retention_policy: str = ""
    privacy_policy: str = ""
    summary_policy: str = ""
    semantic_search_policy: Dict[str, Any] = Field(default_factory=dict)
    limit_to_session: bool = False
    session_scope: str = ""
    scope: str = ""
    dataset_scope: Optional[List[str]] = None
    source_path: str = ""
    source_repo: str = ""
    content_hash: str = ""
    is_active: bool = True
    extra_metadata: Optional[Dict[str, Any]] = None

    constrains: List[tuple[Edge, Skill]] = Field(default_factory=list)

    metadata: dict = Field(
        default_factory=lambda: {
            "index_fields": ["name", "description", "memory", "representation"]
        }
    )


class AgentProfile(DataPoint):
    """Agent behavior profile, tool policy, and profile-to-memory links from AGENT.md."""

    name: Annotated[str, Dedup()]
    description: Annotated[str, Embeddable(), LLMContext()] = ""
    instructions: Annotated[str, Embeddable(), LLMContext("Agent profile instructions")] = ""
    declared_tools: List[str] = Field(default_factory=list)
    allowed_skills: List[str] = Field(default_factory=list)
    memory_profiles: List[str] = Field(default_factory=list)
    model_preferences: Dict[str, Any] = Field(default_factory=dict)
    routing_triggers: List[str] = Field(default_factory=list)
    personalization: Dict[str, Any] = Field(default_factory=dict)
    context_policy: Dict[str, Any] = Field(default_factory=dict)
    observer: str = ""
    targets: List[str] = Field(default_factory=list)
    context_budget_tokens: Optional[int] = None
    observe_me: bool = True
    observe_others: bool = False
    dataset_scope: Optional[List[str]] = None
    source_path: str = ""
    source_repo: str = ""
    content_hash: str = ""
    is_active: bool = True
    extra_metadata: Optional[Dict[str, Any]] = None

    uses: List[tuple[Edge, Skill]] = Field(default_factory=list)
    guided_by: List[tuple[Edge, MemoryProfile]] = Field(default_factory=list)

    metadata: dict = Field(
        default_factory=lambda: {"index_fields": ["name", "description", "instructions"]}
    )


AgentProfile.model_rebuild()
MemoryProfile.model_rebuild()
