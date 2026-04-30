from pydantic_core import PydanticUndefined
from cognee.infrastructure.engine import DataPoint
from cognee.modules.storage.utils import copy_model


# Memoize extended-model classes across calls. ``copy_model`` returns a
# brand-new pydantic subclass on every invocation, and each one attaches
# per-class validator/serializer state to pydantic's global caches that's
# never released. Keying by ``(base_type, frozenset of field specs)``
# means a single class per unique relationship shape *regardless of the
# order edges arrive in* — without the frozenset, an incremental
# subclass-of-subclass approach would mint a new class per permutation
# even though the final shape is identical.
_EXTENDED_MODEL_CACHE: dict = {}


def _extended_model_for(base_type, field_specs):
    """Return a pydantic subclass of ``base_type`` extended with all the
    fields described by ``field_specs`` (an iterable of
    ``(edge_label, target_type, is_list)`` tuples). Cache key is
    order-independent — same set of specs always returns the same class.
    """
    spec_key = frozenset(field_specs)
    key = (base_type, spec_key)
    cached = _EXTENDED_MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    field_defs = {}
    for edge_label, target_type, is_list in spec_key:
        annotation = list[target_type] if is_list else target_type
        field_defs[edge_label] = (annotation, PydanticUndefined)
    model = copy_model(base_type, field_defs)
    _EXTENDED_MODEL_CACHE[key] = model
    return model


def get_model_instance_from_graph(nodes: list[DataPoint], edges: list, entity_id: str):
    node_map = {}

    for node in nodes:
        node_map[node.id] = node

    # Group edges by source so we can build one extended subclass per
    # source (with all its outgoing fields at once) instead of chaining
    # subclasses incrementally — the chained approach makes
    # ``type(source_node)`` an already-extended subclass on subsequent
    # iterations, and that drives the cache key, so different edge
    # orderings would mint distinct cached classes for the same final
    # shape.
    edges_by_source: dict = {}
    for edge in edges:
        edges_by_source.setdefault(edge[0], []).append(edge)

    for source_id, source_edges in edges_by_source.items():
        source_node = node_map[source_id]
        base_type = type(source_node)

        field_specs = []
        values: dict = {}
        for edge in source_edges:
            target_node = node_map[edge[1]]
            edge_label = edge[2]
            edge_properties = edge[3] if len(edge) == 4 else {}
            edge_metadata = edge_properties.get("metadata", {})
            edge_type = edge_metadata.get("type")
            is_list = edge_type == "list"

            field_specs.append((edge_label, type(target_node), is_list))

            if is_list:
                # Preserve targets already attached for this (source, edge)
                # — multi-target list relationships otherwise lose all but
                # the last iteration's target.
                existing = values.get(edge_label) or []
                values[edge_label] = existing + [target_node]
            else:
                values[edge_label] = target_node

        NewModel = _extended_model_for(base_type, field_specs)

        dump = source_node.model_dump()
        # Drop fields we're about to overwrite so the kwargs form isn't a
        # duplicate keyword, and so previously-list values on the dumped
        # dict don't collide with the new lists.
        for edge_label in values:
            dump.pop(edge_label, None)
        node_map[source_id] = NewModel(**dump, **values)

    return node_map[entity_id]
