"""Entrypoint for the Q&A testset generation project.

Usage:
    python main.py build-kg --docs-dir path\to\kb --docs 99817OM B170249XQ_Lensatic_Compass
    python main.py build-kg --docs-dir path\to\kb                      # every document not yet cached
    python main.py build-kg --docs-dir path\to\kb --docs 99817OM --force  # redo a document even if cached
    python main.py generate --target 40 --llm-model gemini-2.5-flash@google_api
    python main.py evaluate data/testsets/testset_xxx.jsonl --llm-model gemini-3.1-flash-lite@google_api
    python main.py eval-retrieval data/testsets/testset_xxx.jsonl --chroma ./chroma_data --collections openai,gemini

--llm-model takes <model>@<url>; url "google_api" means Gemini's cloud endpoint,
anything else is treated as a custom OpenAI-compatible endpoint (e.g. local Ollama).
--embedding-model takes <model> (local HuggingFace) or <model>@<url> with the same rules.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent / "src"))

from eval_retrieval import get_chroma_client, run_retrieval_eval  # noqa: E402
from evaluate import list_candidates, load_jsonl, run_evaluation  # noqa: E402
from kb_loader import list_available_docs, load_chunks  # noqa: E402
from kg_builder import (  # noqa: E402
    enrich_new_chunks,
    get_ingested_doc_names,
    load_knowledge_graph,
    rebuild_relationships,
    remove_docs,
    save_knowledge_graph,
)
from personas import PERSONAS  # noqa: E402

# Benign ragas noise: embeddings run in a thread executor either way, and the node filter is a no-op on pre-chunked nodes.
warnings.filterwarnings("ignore", message="Using sync embedding model")
logging.getLogger("ragas.testset.transforms.filters").setLevel(logging.ERROR)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_HELP = "'<model>' = local HuggingFace; '<model>@<url>' = url 'google_api' for Gemini, else OpenAI-compatible endpoint (needs /v1)"

DATA_DIR = Path(__file__).parent / "data"
KG_CACHE_PATH = DATA_DIR / "kg_cache.json"
CANDIDATES_DIR = Path(__file__).parent / "candidates"

# ragas' single-hop synthesizer prompt ships with exactly one built-in
# few-shot example ("What is the purpose of microservices in software
# architecture?"), and the LLM anchors on it regardless of persona - producing
# flat, generic definition-lookup questions. TestsetGenerator's llm_context
# is threaded into that prompt as explicit steering, so use it to push toward
# situational questions instead of rewriting the prompt/examples by hand.
LLM_CONTEXT = (
    "Generate practical, situational questions that a real person would naturally ask "
    "while working with this equipment - for example while diagnosing a malfunction or "
    "error, checking whether a measured or installed value meets requirements before "
    "signing off on an installation, or figuring out the exact steps to carry out a "
    "task. Phrase the question the way that person would actually ask it in that "
    "situation, using the topic naturally within it. "
    "Avoid generic definition-lookup questions such as 'What is the purpose of X?' or "
    "'What is X?' - do not ask about a term in the abstract."
)


def build_llm(llm_spec: str):
    """Build a ragas LLM from a "<model>@<url>" spec.

    url "google_api" resolves to Gemini's OpenAI-compatible endpoint (using
    provider="openai" rather than "google" sidesteps a known upstream bug
    where instructor's native google-genai integration sends invalid safety
    settings). Any other url is used as-is, e.g. a local Ollama server -
    those don't check the API key, so a placeholder is used.
    """
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    model, _, url = llm_spec.partition("@")
    if not url:
        raise SystemExit(f"--llm-model must be '<model>@<url>', got: {llm_spec!r}")

    if url == "google_api":
        url = GEMINI_OPENAI_BASE_URL
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY not set (copy .env.example to .env and fill it in)")
    else:
        api_key = "not-needed"

    _check_reachable(url)

    client = AsyncOpenAI(api_key=api_key, base_url=url)
    # ragas defaults max_tokens=1024, which Gemini 2.5's thinking tokens eat
    # into before any structured output is produced - raise the budget.
    return llm_factory(model, client=client, max_tokens=8192)


def _check_reachable(url: str) -> None:
    """ragas's executor runs every chunk to completion even if the LLM is
    unreachable, only raising after all of them fail - check the endpoint is
    up first instead of burning through hundreds of doomed calls."""
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError:
        pass  # got a response, server is up
    except Exception as e:
        raise SystemExit(f"Cannot reach {url}: {e}")


def build_embeddings(embedding_spec: str):
    """Build ragas embeddings from "<model>" or "<model>@<url>".

    No url: local HuggingFace model. url "google_api": Gemini cloud. Any other
    url: OpenAI-compatible embeddings endpoint (e.g. Ollama, vLLM).
    """
    model, _, url = embedding_spec.partition("@")

    if not url:
        from ragas.embeddings import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model=model)

    if url == "google_api":
        from google import genai
        from ragas.embeddings import GoogleEmbeddings

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("GOOGLE_API_KEY not set (copy .env.example to .env and fill it in)")
        return GoogleEmbeddings(client=genai.Client(api_key=api_key), model=model)

    from openai import OpenAI
    from ragas.embeddings import OpenAIEmbeddings

    _check_reachable(url)
    return OpenAIEmbeddings(client=OpenAI(api_key="not-needed", base_url=url), model=model)


def cmd_build_kg(args: argparse.Namespace) -> None:
    load_dotenv()

    if not args.docs_dir.is_dir():
        raise SystemExit(f"--docs-dir not found or not a directory: {args.docs_dir}")

    kg = load_knowledge_graph(KG_CACHE_PATH)
    already_ingested = get_ingested_doc_names(kg)

    requested_docs = args.docs if args.docs is not None else list_available_docs(args.docs_dir)
    if args.force:
        new_docs = requested_docs
    else:
        new_docs = [d for d in requested_docs if d not in already_ingested]

    skipped = [d for d in requested_docs if d not in new_docs]
    if skipped:
        print(f"{len(skipped)} documents already in the cache, skipping (use --force to redo)")
    if not new_docs:
        print("Nothing new to add.")
        return

    chunks = load_chunks(args.docs_dir, doc_names=new_docs)
    if not chunks:
        raise SystemExit(f"No chunks loaded for {new_docs} - check the document names")
    print(f"Loaded {len(chunks)} chunks from {len(new_docs)} files")

    llm = build_llm(args.llm_model)
    embeddings = build_embeddings(args.embedding_model)

    if args.force:
        remove_docs(kg, set(new_docs))

    print(f"Enriching {len(chunks)} new chunks (~3 LLM calls each)...")
    new_nodes = enrich_new_chunks(chunks, llm, embeddings)
    kg.nodes.extend(new_nodes)

    print(f"Rebuilding relationships over the full graph ({len(kg.nodes)} nodes, local computation, no LLM calls)...")
    rebuild_relationships(kg, llm, embeddings)

    save_knowledge_graph(kg, KG_CACHE_PATH)
    print(f"Saved knowledge graph ({len(kg.nodes)} nodes, {len(kg.relationships)} relationships) to {KG_CACHE_PATH}")


def cmd_generate(args: argparse.Namespace) -> None:
    load_dotenv()

    if not KG_CACHE_PATH.exists():
        raise SystemExit(f"No cached knowledge graph at {KG_CACHE_PATH} - run 'python main.py build-kg' first")

    from ragas.testset import TestsetGenerator

    kg = load_knowledge_graph(KG_CACHE_PATH)
    print(f"Using cached KG: {len(kg.nodes)} nodes from {len(get_ingested_doc_names(kg))} docs")

    llm = build_llm(args.llm_model)
    embeddings = build_embeddings(args.embedding_model)
    generator = TestsetGenerator(
        llm=llm,
        embedding_model=embeddings,
        knowledge_graph=kg,
        persona_list=PERSONAS,
        llm_context=LLM_CONTEXT,
    )

    testset = generator.generate(testset_size=args.target, with_debugging_logs=args.verbose)

    out_dir = DATA_DIR / "testsets"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"testset_{timestamp}.jsonl"

    # Testset.to_jsonl() opens the file without encoding="utf-8", which fails
    # on Windows (cp1252 default) for KB text containing non-ASCII characters
    # (e.g. bullet glyphs from PDF extraction). Write it out ourselves instead.
    samples = testset.to_list()
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(samples)} samples to {out_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    load_dotenv()

    if not args.testset.exists():
        raise SystemExit(f"Testset not found: {args.testset}")

    candidates = list_candidates(CANDIDATES_DIR)
    if not candidates:
        raise SystemExit(f"No candidates found in {CANDIDATES_DIR} (one .py file per candidate)")

    name = args.testset.name
    eval_name = "eval_" + name[len("testset_"):] if name.startswith("testset_") else f"eval_{name}"
    out_path = DATA_DIR / "evals" / eval_name

    testset_rows = load_jsonl(args.testset)

    llm = build_llm(args.llm_model)

    asyncio.run(run_evaluation(testset_rows, candidates, CANDIDATES_DIR, llm, out_path))
    print(f"Wrote results to {out_path}")


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    if not args.testset.exists():
        raise SystemExit(f"Testset not found: {args.testset}")

    testset_rows = load_jsonl(args.testset)
    client = get_chroma_client(args.chroma)

    name = args.testset.name
    out_name = "retrieval_evals_" + name[len("testset_"):] if name.startswith("testset_") else f"retrieval_evals_{name}"
    out_path = DATA_DIR / "retrieval_evals" / out_name

    run_retrieval_eval(testset_rows, client, args.collections, args.k, out_path)
    print(f"Wrote results to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic Q&A testset with ragas")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser("build-kg", help="Build/extend the cached knowledge graph")
    p_build.add_argument(
        "--docs-dir",
        type=Path,
        required=True,
        help="Folder containing the KB's .md documents",
    )
    p_build.add_argument(
        "--docs",
        nargs="*",
        default=None,
        help="Document names to add (stem or filename), from within --docs-dir; default: every document not yet cached",
    )
    p_build.add_argument("--llm-model", required=True, dest="llm_model", help="'<model>@<url>'; url='google_api' for Gemini, else OpenAI-compatible endpoint (needs /v1)")
    p_build.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, dest="embedding_model", help=EMBEDDING_HELP)
    p_build.add_argument("--force", action="store_true", help="Redo documents even if already in the cache")
    p_build.set_defaults(func=cmd_build_kg)

    p_generate = subparsers.add_parser("generate", help="Generate a testset from the cached knowledge graph")
    p_generate.add_argument("--target", type=int, default=40, help="Number of Q&A samples to generate")
    p_generate.add_argument("--llm-model", required=True, dest="llm_model", help="'<model>@<url>'; url='google_api' for Gemini, else OpenAI-compatible endpoint (needs /v1)")
    p_generate.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, dest="embedding_model", help=EMBEDDING_HELP)
    p_generate.add_argument("--verbose", action="store_true")
    p_generate.set_defaults(func=cmd_generate)

    p_evaluate = subparsers.add_parser("evaluate", help="Evaluate candidate RAG pipelines against a testset")
    p_evaluate.add_argument("testset", type=Path, help="Path to a testset JSONL file")
    p_evaluate.add_argument("--llm-model", required=True, dest="llm_model", help="'<model>@<url>'; url='google_api' for Gemini, else OpenAI-compatible endpoint (needs /v1)")
    p_evaluate.set_defaults(func=cmd_evaluate)

    p_eval_retrieval = subparsers.add_parser("eval-retrieval", help="Compare embedding models via retrieval recall@k")
    p_eval_retrieval.add_argument("testset", type=Path)
    p_eval_retrieval.add_argument("--chroma", required=True, help="Chroma path or http(s) URL")
    p_eval_retrieval.add_argument("--collections", type=lambda s: s.split(","), required=True)
    p_eval_retrieval.add_argument("--k", type=int, default=5)
    p_eval_retrieval.set_defaults(func=cmd_eval_retrieval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
