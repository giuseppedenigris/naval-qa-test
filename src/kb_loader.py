"""Load the naval technical manuals KB and split them into section-level chunks.

Phase 1 (text-only): anchors like [[IMG:id]]/[[TABLE:id]] are left untouched in
the chunk text - they are harmless noise tokens for the LLM at this stage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc_name: str
    text: str

HEADING_RE = re.compile(r"^## \S")
MIN_BODY_WORDS = 30


def find_md_files(kb_root: Path) -> list[Path]:
    return sorted(kb_root.glob("*.md"))


def split_into_chunks(text: str) -> list[str]:
    """Split markdown text into chunks, one per top-level ("## ") section."""
    lines = text.splitlines()
    chunks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if HEADING_RE.match(line) and current:
            chunks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append(current)

    return _merge_degenerate(["\n".join(c).strip() for c in chunks])


def _body_word_count(chunk: str) -> int:
    lines = chunk.splitlines()
    body_lines = lines[1:] if lines and HEADING_RE.match(lines[0]) else lines
    return len(" ".join(body_lines).split())


def _merge_degenerate(chunks: list[str]) -> list[str]:
    """Merge chunks whose body (heading excluded) is under MIN_BODY_WORDS into
    the following chunk, so short/empty sections don't become their own
    near-empty node."""
    merged: list[str] = []
    pending = ""

    for chunk in chunks:
        combined = f"{pending}\n\n{chunk}".strip() if pending else chunk
        if _body_word_count(chunk) < MIN_BODY_WORDS:
            pending = combined
            continue
        merged.append(combined)
        pending = ""

    if pending:
        if merged:
            merged[-1] = f"{merged[-1]}\n\n{pending}".strip()
        else:
            merged.append(pending)

    return merged


def list_available_docs(kb_root: Path) -> list[str]:
    return [f.stem for f in find_md_files(kb_root)]


def load_chunks(kb_root: Path, doc_names: list[str] | None = None) -> list[Chunk]:
    """Load all (or a subset of) documents from the KB and return a flat list
    of section-level Chunks, tagged with their source document name."""
    md_files = find_md_files(kb_root)
    if doc_names is not None:
        wanted = set(doc_names)
        md_files = [f for f in md_files if f.stem in wanted or f.name in wanted]

    chunks: list[Chunk] = []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks.extend(Chunk(doc_name=md_file.stem, text=t) for t in split_into_chunks(text))
    return chunks
