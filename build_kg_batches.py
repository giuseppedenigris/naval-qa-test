"""Build the knowledge graph in batches of documents, so a failure only loses the current batch.

Usage:
    python build_kg_batches.py --docs-dir ..\pdf-ingestion\data\output --llm-model gemini-2.5-flash-lite@google_api --embedding-model gemini-embedding-001@google_api

The documents come from --docs-file (default data/selected_docs.txt, one name per line).
The KG is saved after every batch. Re-run the same command to resume: documents already
in the cache are skipped. The similarity relationships are rebuilt once, at the very end
(local and single-threaded, the slowest step); until then the saved KG has none.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from kb_loader import list_available_docs, load_chunks  # noqa: E402
from kg_builder import enrich_new_chunks, get_ingested_doc_names, load_knowledge_graph, rebuild_relationships, save_knowledge_graph  # noqa: E402
from main import DATA_DIR, DEFAULT_EMBEDDING_MODEL, EMBEDDING_HELP, KG_CACHE_PATH, build_embeddings, build_llm  # noqa: E402


def read_doc_names(docs_file: Path) -> list[str]:
    return [line.strip() for line in docs_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def run(args: argparse.Namespace) -> None:
    if not args.docs_dir.is_dir():
        raise SystemExit(f"--docs-dir not found or not a directory: {args.docs_dir}")
    if not args.docs_file.is_file():
        raise SystemExit(f"--docs-file not found: {args.docs_file}")

    wanted = read_doc_names(args.docs_file)
    missing = sorted(set(wanted) - set(list_available_docs(args.docs_dir)))
    if missing:
        raise SystemExit(f"{len(missing)} documents of {args.docs_file.name} not found in {args.docs_dir}: {missing[:5]}")

    kg = load_knowledge_graph(KG_CACHE_PATH)
    already_done = get_ingested_doc_names(kg)
    todo = [d for d in wanted if d not in already_done]
    print(f"{len(wanted)} documents selected, {len(wanted) - len(todo)} already in the cache, {len(todo)} to do")

    needs_rebuild = bool(kg.nodes) and not kg.relationships
    if not todo and not needs_rebuild:
        print("Nothing to do.")
        return

    llm = build_llm(args.llm_model)
    embeddings = build_embeddings(args.embedding_model)

    failed = []
    batches = [todo[i : i + args.batch_size] for i in range(0, len(todo), args.batch_size)]
    for number, batch in enumerate(batches, 1):
        print(f"\n=== Batch {number}/{len(batches)}: {len(batch)} documents ===")
        try:
            new_nodes = enrich_new_chunks(load_chunks(args.docs_dir, batch), llm, embeddings)
        except Exception as e:
            print(f"Batch {number} failed, skipping it: {e}")
            failed.extend(batch)
            continue
        kg.nodes.extend(new_nodes)
        kg.relationships = []  # stale until rebuilt at the end
        save_knowledge_graph(kg, KG_CACHE_PATH)
        print(f"Saved {len(kg.nodes)} chunks to {KG_CACHE_PATH}")

    if failed:
        print(f"\n{len(failed)} documents failed: {failed}")
        print("Re-run the same command to retry them (use --batch-size 1 to isolate a bad document). Relationships get rebuilt once everything is in.")
        raise SystemExit(1)

    print(f"\nRebuilding relationships over {len(kg.nodes)} chunks (local, single-threaded, can take hours)...")
    rebuild_relationships(kg, llm, embeddings)
    save_knowledge_graph(kg, KG_CACHE_PATH)
    print(f"Done: {len(kg.nodes)} chunks, {len(kg.relationships)} relationships in {KG_CACHE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the knowledge graph in resumable batches of documents")
    parser.add_argument("--docs-dir", type=Path, required=True, help="Folder containing the KB's .md documents")
    parser.add_argument("--docs-file", type=Path, default=DATA_DIR / "selected_docs.txt", help="Text file with one document name per line")
    parser.add_argument("--batch-size", type=int, default=10, help="Documents per batch; the KG is saved after each one")
    parser.add_argument("--llm-model", required=True, dest="llm_model", help="'<model>@<url>'; url='google_api' for Gemini, else OpenAI-compatible endpoint (needs /v1)")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, dest="embedding_model", help=EMBEDDING_HELP)
    args = parser.parse_args()

    load_dotenv()
    run(args)


if __name__ == "__main__":
    main()
