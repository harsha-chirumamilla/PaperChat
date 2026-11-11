"""Run the curated retrieval, citation, and latency benchmark against PaperChat."""

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.common.config import Config  # noqa: E402
from src.rag.query.rag_pipeline import RAGSystem  # noqa: E402


ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def normalize_source_id(value: Any) -> str:
    """Normalize arXiv version suffixes while retaining other source identifiers."""
    source_id = str(value or "").strip()
    return ARXIV_VERSION.sub("", source_id)


def source_id(document: Any) -> str:
    return normalize_source_id(
        document.metadata.get("source") or document.metadata.get("arxiv_id")
    )


def corpus_fingerprint(documents: Sequence[Any]) -> str:
    """Hash indexed text and source IDs in a stable order for report comparison."""
    records = sorted((source_id(document), document.page_content) for document in documents)
    digest = hashlib.sha256()
    for source, content in records:
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    cases = []
    seen_ids = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        for field in ("id", "question", "relevant_source_ids"):
            if field not in case:
                raise ValueError(f"{path}:{line_number}: missing required field '{field}'")
        if not isinstance(case["relevant_source_ids"], list) or not case["relevant_source_ids"]:
            raise ValueError(f"{path}:{line_number}: relevant_source_ids must be a non-empty list")
        if "supporting_source_ids" in case and (
            not isinstance(case["supporting_source_ids"], list) or not case["supporting_source_ids"]
        ):
            raise ValueError(f"{path}:{line_number}: supporting_source_ids must be a non-empty list")
        if case["id"] in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate case id '{case['id']}'")
        seen_ids.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError(f"No evaluation cases found in {path}")
    return cases


def recall_at_k(ranked_source_ids: Sequence[str], relevant: Set[str]) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_source_ids) & relevant) / len(relevant)


def reciprocal_rank(ranked_source_ids: Sequence[str], relevant: Set[str]) -> float:
    for rank, candidate in enumerate(ranked_source_ids, start=1):
        if candidate in relevant:
            return 1.0 / rank
    return 0.0


def citation_scores(answer: str, citations: Sequence[Dict[str, Any]], relevant: Set[str]) -> Dict[str, Any]:
    labels = {str(citation["number"]): normalize_source_id(citation.get("arxiv_id")) for citation in citations}
    used_labels = set(CITATION_PATTERN.findall(answer))
    used_sources = {labels[label] for label in used_labels if label in labels}
    valid_count = sum(label in labels for label in used_labels)
    matched = used_sources & relevant
    # Invalid labels such as [99] remain in the precision denominator.
    precision = len(matched) / len(used_labels) if used_labels else 0.0
    recall = len(matched) / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    validity = valid_count / len(used_labels) if used_labels else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inline_citation_validity": validity,
        "inline_citation_count": len(used_labels),
        "cited_source_ids": sorted(used_sources),
    }


def percentile(values: Sequence[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile_value + 0.5)))
    return ordered[index]


def latency_summary(samples: Iterable[float]) -> Dict[str, float]:
    values = list(samples)
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "mean_ms": round(statistics.mean(values), 2),
        "p50_ms": round(percentile(values, 0.50), 2),
        "p95_ms": round(percentile(values, 0.95), 2),
    }


def evaluate_case(system: RAGSystem, case: Dict[str, Any], top_k: int) -> Dict[str, Any]:
    question = case["question"]
    relevant = {normalize_source_id(value) for value in case["relevant_source_ids"]}
    supporting = {
        normalize_source_id(value)
        for value in case.get("supporting_source_ids", case["relevant_source_ids"])
    }
    retrieval_k = max(Config.RETRIEVAL_K, top_k)

    started = time.perf_counter()
    retrieval_started = time.perf_counter()
    retrieved = system.retriever.retrieve(question, k=retrieval_k)
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000

    rerank_started = time.perf_counter()
    reranked = system.reranker.rerank(question, retrieved, top_k=top_k) if retrieved else []
    rerank_ms = (time.perf_counter() - rerank_started) * 1000

    generation_started = time.perf_counter()
    if reranked:
        prompt = system.prompter.format_rag_prompt(question, reranked)
        answer = system.llm_client.generate(prompt)
        citations = system._citations(reranked)
    else:
        answer = "No relevant documents were found. Ingest papers before querying."
        citations = []
    generation_ms = (time.perf_counter() - generation_started) * 1000
    total_ms = (time.perf_counter() - started) * 1000

    retrieved_ids = [source_id(document) for document in retrieved]
    context_ids = list(dict.fromkeys(source_id(document) for document in reranked))
    return {
        "id": case["id"],
        "question": question,
        "answer": answer,
        "reference_answer": case.get("reference_answer"),
        "relevant_source_ids": sorted(relevant),
        "supporting_source_ids": sorted(supporting),
        "retrieved_source_ids": retrieved_ids,
        "context_source_ids": context_ids,
        "retrieval_recall_at_k": recall_at_k(retrieved_ids, relevant),
        "retrieval_mrr": reciprocal_rank(retrieved_ids, relevant),
        "context_recall_at_k": recall_at_k(context_ids, relevant),
        "citations": citation_scores(answer, citations, supporting),
        "latency_ms": {
            "retrieval": round(retrieval_ms, 2),
            "reranking": round(rerank_ms, 2),
            "generation": round(generation_ms, 2),
            "end_to_end": round(total_ms, 2),
        },
    }


def mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return statistics.mean(row[key] for row in rows) if rows else 0.0


def build_report(
    rows: List[Dict[str, Any]],
    init_ms: float,
    top_k: int,
    dataset_path: Path,
    corpus_chunk_count: int,
    corpus_sha256: str,
) -> Dict[str, Any]:
    generation_model = {
        "openai": Config.OPENAI_MODEL_ID,
        "anthropic": Config.ANTHROPIC_MODEL_ID,
        "gemini": Config.GEMINI_MODEL_ID,
    }.get(Config.MODEL_PROVIDER)
    return {
        "schema_version": 1,
        "model_provider": Config.MODEL_PROVIDER,
        "generation_model": generation_model,
        "embedding_model": Config.EMBEDDING_MODEL_ID,
        "reranker_model": Config.RERANKER_MODEL_ID,
        "temperature": Config.TEMPERATURE,
        "retrieval_k": max(Config.RETRIEVAL_K, top_k),
        "context_top_k": top_k,
        "corpus_chunk_count": corpus_chunk_count,
        "corpus_sha256": corpus_sha256,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "system_initialization_ms": round(init_ms, 2),
        "case_count": len(rows),
        "metrics": {
            "retrieval_recall_at_k": mean(rows, "retrieval_recall_at_k"),
            "retrieval_mrr": mean(rows, "retrieval_mrr"),
            "context_recall_at_k": mean(rows, "context_recall_at_k"),
            "citation_precision": statistics.mean(row["citations"]["precision"] for row in rows),
            "citation_recall": statistics.mean(row["citations"]["recall"] for row in rows),
            "citation_f1": statistics.mean(row["citations"]["f1"] for row in rows),
            "inline_citation_validity": statistics.mean(
                row["citations"]["inline_citation_validity"] for row in rows
            ),
            "latency_ms": {
                name: latency_summary(row["latency_ms"][name] for row in rows)
                for name in ("retrieval", "reranking", "generation", "end_to_end")
            },
        },
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "evals/datasets/starter.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "evals/results/latest.json")
    parser.add_argument("--top-k", type=int, default=Config.RERANK_TOP_K)
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    cases = load_dataset(args.dataset)
    init_started = time.perf_counter()
    system = RAGSystem()
    init_ms = (time.perf_counter() - init_started) * 1000
    # Load the persistent sparse index and embedding model before timed cases.
    warmup_started = time.perf_counter()
    system.retriever.retrieve(cases[0]["question"], k=max(Config.RETRIEVAL_K, args.top_k))
    warmup_ms = (time.perf_counter() - warmup_started) * 1000
    rows = []
    for case in cases:
        print(f"Evaluating {case['id']}...", flush=True)
        rows.append(evaluate_case(system, case, args.top_k))
    report = build_report(
        rows,
        init_ms,
        args.top_k,
        args.dataset,
        system.vector_db.collection_version(),
        corpus_fingerprint(system.retriever._bm25_docs),
    )
    report["retrieval_warmup_ms"] = round(warmup_ms, 2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = report["metrics"]
    print(f"Results: {args.output}")
    print(f"Model initialization: {init_ms:.2f}ms; retrieval warm-up: {warmup_ms:.2f}ms")
    print(
        "Retrieval: Recall@k={:.3f}, MRR={:.3f}, context Recall@k={:.3f}".format(
            metrics["retrieval_recall_at_k"],
            metrics["retrieval_mrr"],
            metrics["context_recall_at_k"],
        )
    )
    print(
        "Citations: precision={:.3f}, recall={:.3f}, F1={:.3f}, valid inline={:.3f}".format(
            metrics["citation_precision"],
            metrics["citation_recall"],
            metrics["citation_f1"],
            metrics["inline_citation_validity"],
        )
    )
    for stage, summary in metrics["latency_ms"].items():
        print(
            "Latency {}: mean={:.2f}ms, p50={:.2f}ms, p95={:.2f}ms".format(
                stage, summary["mean_ms"], summary["p50_ms"], summary["p95_ms"]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
