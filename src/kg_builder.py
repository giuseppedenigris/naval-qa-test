"""Build and incrementally extend the cached ragas KnowledgeGraph.

The default ragas transform pipeline for pre-chunked input has two very
different phases:

1. Per-node extractors (SummaryExtractor, ThemesExtractor, NERExtractor,
   EmbeddingExtractor) - one LLM call per chunk, independent of every other
   chunk. This is where ~all the cost/time lives.
2. Relationship builders (CosineSimilarityBuilder, OverlapScoreBuilder) - pure
   local math over the properties the extractors produced, comparing every
   node against every other node. No LLM calls, cheap even at full-corpus
   size, but it needs the *whole* graph to connect a new document's chunks to
   previously-ingested ones.

So adding a new document only needs to re-run phase 1 on the new chunks, then
re-run phase 2 (cheap) over the merged graph. Relationship builders append
rather than replace, so relationships are cleared before phase 2 to avoid
duplicating the ones that already existed.
"""

from __future__ import annotations

from pathlib import Path

from kb_loader import Chunk
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import RelationshipBuilder, apply_transforms, default_transforms_for_prechunked
from ragas.testset.transforms.engine import Parallel


def _split_transform_phases(llm, embedding_model):
    """Split default_transforms_for_prechunked() into (extractor_phase,
    relationship_phase), asserting the last step really is relationship
    builders only - this relies on ragas' internal pipeline shape, so fail
    loudly if a ragas upgrade changes it instead of silently corrupting the
    graph."""
    transforms = default_transforms_for_prechunked(llm=llm, embedding_model=embedding_model)
    extractor_phase, relationship_phase = transforms[:-1], transforms[-1:]

    last_step = relationship_phase[0]
    steps = last_step.transformations if isinstance(last_step, Parallel) else [last_step]
    if not steps or not all(isinstance(s, RelationshipBuilder) for s in steps):
        raise RuntimeError(
            "default_transforms_for_prechunked()'s last step is no longer relationship-builders-only - "
            "update _split_transform_phases() for the installed ragas version"
        )
    return extractor_phase, relationship_phase


def get_ingested_doc_names(kg: KnowledgeGraph) -> set[str]:
    return {
        doc_name
        for node in kg.nodes
        if (doc_name := node.properties.get("document_metadata", {}).get("doc_name"))
    }


def enrich_new_chunks(chunks: list[Chunk], llm, embedding_model) -> list[Node]:
    """Build fresh nodes for `chunks` and run only the (expensive) per-node
    extractor phase on them - no relationships yet."""
    nodes = [
        Node(
            type=NodeType.CHUNK,
            properties={"page_content": chunk.text, "document_metadata": {"doc_name": chunk.doc_name}},
        )
        for chunk in chunks
        if chunk.text and chunk.text.strip()
    ]
    extractor_phase, _ = _split_transform_phases(llm, embedding_model)
    temp_kg = KnowledgeGraph(nodes=nodes)
    apply_transforms(temp_kg, extractor_phase)
    return temp_kg.nodes


def rebuild_relationships(kg: KnowledgeGraph, llm, embedding_model) -> None:
    """Recompute relationships over the whole graph in place (cheap, no LLM
    calls). Clears existing relationships first since relationship builders
    append rather than replace."""
    _, relationship_phase = _split_transform_phases(llm, embedding_model)
    kg.relationships = []
    apply_transforms(kg, relationship_phase)


def remove_docs(kg: KnowledgeGraph, doc_names: set[str]) -> None:
    """Drop all nodes (and any relationships touching them) belonging to the
    given documents - used by --force to redo specific documents cleanly."""
    keep_nodes = [
        n for n in kg.nodes if n.properties.get("document_metadata", {}).get("doc_name") not in doc_names
    ]
    keep_ids = {n.id for n in keep_nodes}
    kg.nodes = keep_nodes
    kg.relationships = [r for r in kg.relationships if r.source.id in keep_ids and r.target.id in keep_ids]


def save_knowledge_graph(kg: KnowledgeGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    kg.save(path)


def load_knowledge_graph(path: Path) -> KnowledgeGraph:
    return KnowledgeGraph.load(path) if path.exists() else KnowledgeGraph()
