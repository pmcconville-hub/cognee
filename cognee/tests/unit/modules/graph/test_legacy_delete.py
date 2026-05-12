from pydantic import Field

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.methods.legacy_delete import _get_index_fields


def test_get_index_fields_handles_literal_metadata_default():
    class LiteralMetadataPoint(DataPoint):
        metadata: dict = {"index_fields": ["name"]}

    assert _get_index_fields(LiteralMetadataPoint) == ["name"]


def test_get_index_fields_handles_factory_metadata_default():
    class FactoryMetadataPoint(DataPoint):
        metadata: dict = Field(default_factory=lambda: {"index_fields": ["description"]})

    assert _get_index_fields(FactoryMetadataPoint) == ["description"]


def test_get_index_fields_handles_required_metadata():
    class RequiredMetadataPoint(DataPoint):
        metadata: dict

    assert _get_index_fields(RequiredMetadataPoint) == []
