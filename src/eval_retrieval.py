"""Compare embedding models by retrieval recall@k against a Chroma vector store.

Assumes each collection was created with an embedding_function attached, so
collection.query(query_texts=...) embeds the query with the right model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from tqdm import tqdm


def get_chroma_client(chroma: str):
    import chromadb

    parsed = urlparse(chroma)
    if parsed.scheme in ("http", "https"):
        return chromadb.HttpClient(host=parsed.hostname, port=parsed.port or 8000, ssl=parsed.scheme == "https")
    return chromadb.PersistentClient(path=chroma)


_HOP_PREFIX_RE = re.compile(r"^<\d+-hop>\s*")


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def matches(reference_context: str, retrieved_chunk: str, threshold: float = 0.8) -> bool:
    reference_context = _HOP_PREFIX_RE.sub("", reference_context)
    ref_words = _words(reference_context)
    if not ref_words:
        return False
    return len(ref_words & _words(retrieved_chunk)) / len(ref_words) >= threshold


def recall_at_k(reference_contexts: list[str], retrieved_chunks: list[str]) -> float:
    if not reference_contexts:
        return 0.0
    hits = sum(any(matches(ref, chunk) for chunk in retrieved_chunks) for ref in reference_contexts)
    return hits / len(reference_contexts)


def run_retrieval_eval(testset_rows: list[dict], client, collections: list[str], k: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as out_f:
        for name in collections:
            collection = client.get_collection(name)
            scores = []

            for row in tqdm(testset_rows, desc=name):
                result = collection.query(query_texts=[row["user_input"]], n_results=k)
                retrieved_chunks = result["documents"][0]
                score = recall_at_k(row["reference_contexts"], retrieved_chunks)
                scores.append(score)
                out_f.write(json.dumps(
                    {
                        "collection": name,
                        "user_input": row["user_input"],
                        "recall_at_k": score,
                        "retrieved_chunks": retrieved_chunks,
                    },
                    ensure_ascii=False,
                ) + "\n")

            print(f"{name}: recall@{k} = {sum(scores) / len(scores):.3f}")
