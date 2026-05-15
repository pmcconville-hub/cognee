from typing import List

from pydantic import Field

from cognee.infrastructure.engine import DataPoint


class SkillImprovementProposal(DataPoint):
    """Reviewable graph-only proposal for improving a stored Skill."""

    proposal_id: str
    skill_id: str
    skill_name: str
    dataset_scope: List[str] = Field(default_factory=list)
    old_procedure: str = ""
    proposed_procedure: str = ""
    runs_used: List[str] = Field(default_factory=list)
    model_name: str = ""
    confidence: float = 0.0
    rationale: str = ""
    status: str = "proposed"

    metadata: dict = Field(
        default={
            "index_fields": ["skill_name", "rationale"],
            "identity_fields": ["proposal_id"],
        }
    )
