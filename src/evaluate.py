"""Evaluate candidate RAG pipelines against a generated testset, using ragas
metrics with a single LLM judge.

Sequential and simple on purpose: one (testset row, candidate) pair at a
time, no concurrency, no partial-row persistence. A row is written only once
fully complete; any failure (candidate call or a metric) is logged and the
pair is left out, so it gets retried on the next run.
"""

from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm


def list_candidates(candidates_dir: Path) -> list[str]:
    if not candidates_dir.is_dir():
        return []
    return sorted(f.stem for f in candidates_dir.glob("*.py") if f.stem != "__init__")


def load_candidate(candidates_dir: Path, name: str):
    import importlib.util

    path = candidates_dir / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.answer


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_done_pairs(out_path: Path) -> set[tuple[str, str, str]]:
    """(candidate, user_input, reference) triples already written to out_path."""
    done = set()
    if out_path.exists():
        for row in load_jsonl(out_path):
            done.add((row["candidate"], row["user_input"], row["reference"]))
    return done


async def run_evaluation(testset_rows: list[dict], candidates: list[str], candidates_dir: Path, llm, out_path: Path) -> None:
    from ragas.metrics.collections import ContextRecall, Faithfulness, FactualCorrectness

    fc_recall_metric = FactualCorrectness(llm=llm, mode="recall")
    context_recall_metric = ContextRecall(llm=llm)
    faithfulness_metric = Faithfulness(llm=llm)

    done = load_done_pairs(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as out_f:
        for candidate in candidates:
            answer = load_candidate(candidates_dir, candidate)

            for row in tqdm(testset_rows, desc=candidate):
                key = (candidate, row["user_input"], row["reference"])
                if key in done:
                    continue

                try:
                    result = answer(row["user_input"])
                    response = result["response"]
                    retrieved_contexts = result["contexts"]

                    fc_recall = (await fc_recall_metric.ascore(
                        response=response, reference=row["reference"]
                    )).value
                    context_recall = (await context_recall_metric.ascore(
                        user_input=row["user_input"],
                        retrieved_contexts=retrieved_contexts,
                        reference=row["reference"],
                    )).value
                    faithfulness = (await faithfulness_metric.ascore(
                        user_input=row["user_input"],
                        response=response,
                        retrieved_contexts=retrieved_contexts,
                    )).value
                except Exception as e:
                    print(f"[{candidate}] error on {row['user_input']!r}: {e}")
                    continue

                out_row = {
                    **row,
                    "candidate": candidate,
                    "response": response,
                    "retrieved_contexts": retrieved_contexts,
                    "latency_s": result["latency_s"],
                    "tokens": result["tokens"],
                    "fc_recall": fc_recall,
                    "context_recall": context_recall,
                    "faithfulness": faithfulness,
                }
                out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out_f.flush()
                done.add(key)
