"""RAGAS evaluation harness — per-document eval via REST API."""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from typing import Any

import httpx
import jwt
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from langchain_core.embeddings import Embeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from config.logging import get_logger
from config.settings import settings

try:
    from ragas import RunConfig
except ImportError:
    from ragas.run_config import RunConfig

log = get_logger()


_API_BASE = "http://localhost:8000"
_GROQ_BASE = "https://api.groq.com/openai/v1"


def _make_jwt(tenant_id: str) -> str:
    return jwt.encode({"sub": tenant_id}, settings.jwt_secret, algorithm="HS256")


def _make_ragas_llm(api_key: str | None = None) -> LangchainLLMWrapper:
    key = api_key or settings.groq_api_key
    if not key:
        raise ValueError("GROQ_API_KEY not set")
    lm = ChatOpenAI(
        base_url=_GROQ_BASE,
        api_key=key,
        model=settings.groq_eval_model,
        temperature=0,
        timeout=60,
    )
    return LangchainLLMWrapper(lm)


class _VoyageEmbeddings(Embeddings):
    """Minimal LangChain-compatible wrapper around Voyage AI embeddings REST API."""

    _URL = "https://api.voyageai.com/v1/embeddings"

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        resp = httpx.post(
            self._URL,
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json={"model": self._model, "input": texts},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


def _make_ragas_embeddings() -> LangchainEmbeddingsWrapper:
    if not settings.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY not set — required for answer_relevancy metric")
    return LangchainEmbeddingsWrapper(
        _VoyageEmbeddings(api_key=settings.voyage_api_key, model=settings.voyage_embedding_model)
    )


async def _run_query(question: str, tenant_id: str, workspace_id: int) -> dict[str, Any]:
    try:
        token = _make_jwt(tenant_id)
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{_API_BASE}/query",
                json={"question": question, "workspace_id": workspace_id, "history": [], "bypass_cache": True},
                headers={"Authorization": f"Bearer {token}"},
            )
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        log.error("eval.run_query_failed", tenant_id=tenant_id, workspace_id=workspace_id,
                  question=question[:60], error=str(exc))
        return {"answer": "", "chunks": []}
    finally:
        log.debug("eval.run_query_exit", tenant_id=tenant_id, question=question[:60])


# ---------------------------------------------------------------------------
# Retrieval metrics (no LLM needed)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def _is_relevant(chunk_text: str, ground_truth: str, threshold: float = 0.08) -> bool:
    ct = _tokenize(chunk_text)
    gt = _tokenize(ground_truth)
    if not gt:
        return False
    return len(ct & gt) / len(ct | gt) >= threshold


def _compute_retrieval_metrics(samples: list[dict], k: int = 10) -> dict[str, float]:
    try:
        precisions, recalls, mrrs, ndcgs, hit_rates = [], [], [], [], []

        for s in samples:
            gt = s.get("ground_truth", "")
            contexts = s["contexts"][:k]
            relevance = [1 if _is_relevant(c, gt) else 0 for c in contexts]
            n_relevant = sum(relevance)

            # hit rate@k — did any relevant chunk appear?
            hit_rates.append(1.0 if n_relevant > 0 else 0.0)

            # precision@k
            precisions.append(n_relevant / k if k else 0.0)

            # recall@k
            total_relevant = max(n_relevant, 1)
            recalls.append(n_relevant / total_relevant)

            # MRR
            rr = 0.0
            for rank, rel in enumerate(relevance, 1):
                if rel:
                    rr = 1.0 / rank
                    break
            mrrs.append(rr)

            # NDCG@k
            dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance, 1))
            ideal = sorted(relevance, reverse=True)
            idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal, 1))
            ndcgs.append(dcg / idcg if idcg else 0.0)

        n = len(samples)
        return {
            f"hit_rate_at_{k}": round(sum(hit_rates) / n, 4),
            f"precision_at_{k}": round(sum(precisions) / n, 4),
            f"recall_at_{k}": round(sum(recalls) / n, 4),
            "mrr": round(sum(mrrs) / n, 4),
            f"ndcg_at_{k}": round(sum(ndcgs) / n, 4),
        }
    except Exception as exc:
        log.error("eval.retrieval_metrics_failed", error=str(exc))
        return {f"hit_rate_at_{k}": 0.0, f"precision_at_{k}": 0.0, f"recall_at_{k}": 0.0, "mrr": 0.0, f"ndcg_at_{k}": 0.0}
    finally:
        log.debug("eval.retrieval_metrics_exit", n_samples=len(samples))


# ---------------------------------------------------------------------------
# Per-document eval — accepts Q&A pairs, rotates Groq keys, saves to DB
# ---------------------------------------------------------------------------

async def _eval_single_question(
    question: str,
    ground_truth: str,
    tenant_id: str,
    workspace_id: int,
    groq_key: str,
    delay_s: float = 5.0,
    idx: int = 0,
) -> dict:
    """Evaluate one Q&A pair. Applies delay to stay under Groq rate limit."""
    if idx > 0:
        await asyncio.sleep(delay_s)
    t0 = time.perf_counter()
    result = await _run_query(question, tenant_id, workspace_id)
    latency_ms = round((time.perf_counter() - t0) * 1000)
    answer = result.get("answer", "")
    chunks = result.get("chunks", [])
    contexts = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "latency_ms": latency_ms,
        "run_id": result.get("run_id"),
        "groq_key": groq_key,
    }


async def evaluate_doc(
    doc_id: int,
    tenant_id: str,
    workspace_id: int,
    questions: list[str],
    ground_truths: list[str],
    strategy_id: int | None = None,
    delay_s: float = 5.0,
) -> dict[str, Any]:
    """Run RAGAS eval for a single document with key rotation and DB persistence."""
    from db.postgres import save_ragas_eval
    from llm.groq_keys import next_groq_key

    if len(questions) != len(ground_truths):
        raise ValueError("questions and ground_truths must be same length")
    if not questions:
        raise ValueError("Need at least 1 question")

    log.info("eval.doc_start", tenant_id=tenant_id, doc_id=doc_id, workspace_id=workspace_id, n=len(questions))
    t0 = time.perf_counter()

    samples: list[dict] = []
    for i, (q, gt) in enumerate(zip(questions, ground_truths)):
        key = next_groq_key()
        sample = await _eval_single_question(q, gt, tenant_id, workspace_id, key, delay_s=delay_s, idx=i)
        samples.append(sample)
        log.info("eval.doc_q_done", tenant_id=tenant_id, doc_id=doc_id, idx=i, latency_ms=sample["latency_ms"])

    # RAGAS LLM scoring — run each metric separately with its own key + pause between
    # Two separate evaluate() calls avoids burst: 5 calls on key1, sleep 10s, 5 calls on key2
    faithfulness_score = 0.0
    context_precision_score = 0.0
    answer_relevancy_score = 0.0
    try:
        data = {
            "question": [s["question"] for s in samples],
            "answer": [s["answer"][:600] for s in samples],
            "contexts": [[c[:800] for c in s["contexts"][:3]] for s in samples],
            "ground_truth": [s["ground_truth"] for s in samples],
        }
        dataset = Dataset.from_dict(data)
        run_cfg = RunConfig(timeout=90, max_retries=2)

        faithfulness.llm = _make_ragas_llm(api_key=next_groq_key())
        if hasattr(faithfulness, "run_config"):
            faithfulness.run_config = run_cfg
        f_scores = evaluate(dataset, metrics=[faithfulness], llm=faithfulness.llm,
                            run_config=run_cfg, batch_size=1, raise_exceptions=False)
        faithfulness_score = float(f_scores.to_pandas()["faithfulness"].mean())
        log.info("eval.faithfulness_done", doc_id=doc_id, score=faithfulness_score)

        await asyncio.sleep(10)

        context_precision.llm = _make_ragas_llm(api_key=next_groq_key())
        if hasattr(context_precision, "run_config"):
            context_precision.run_config = run_cfg
        cp_scores = evaluate(dataset, metrics=[context_precision], llm=context_precision.llm,
                             run_config=run_cfg, batch_size=1, raise_exceptions=False)
        context_precision_score = float(cp_scores.to_pandas()["context_precision"].mean())
        log.info("eval.context_precision_done", doc_id=doc_id, score=context_precision_score)

        await asyncio.sleep(10)

        answer_relevancy.llm = _make_ragas_llm(api_key=next_groq_key())
        answer_relevancy.embeddings = _make_ragas_embeddings()
        if hasattr(answer_relevancy, "run_config"):
            answer_relevancy.run_config = run_cfg
        ar_scores = evaluate(dataset, metrics=[answer_relevancy], llm=answer_relevancy.llm,
                             run_config=run_cfg, batch_size=1, raise_exceptions=False)
        answer_relevancy_score = float(ar_scores.to_pandas()["answer_relevancy"].mean())
        log.info("eval.answer_relevancy_done", doc_id=doc_id, score=answer_relevancy_score)
    except Exception as exc:
        log.error("eval.doc_ragas_failed", doc_id=doc_id, error=str(exc))

    retrieval_scores = _compute_retrieval_metrics(samples)
    avg = round((faithfulness_score + context_precision_score + answer_relevancy_score) / 3, 4)

    eval_id = await save_ragas_eval(
        doc_id=doc_id,
        tenant_id=tenant_id,
        strategy_id=strategy_id,
        questions=questions,
        ground_truths=ground_truths,
        answers=[s["answer"] for s in samples],
        contexts=[s["contexts"] for s in samples],
        faithfulness=faithfulness_score,
        context_precision=context_precision_score,
        answer_relevance=answer_relevancy_score,
        retrieval_metrics=retrieval_scores,
    )

    result = {
        "eval_id": eval_id,
        "doc_id": doc_id,
        "tenant_id": tenant_id,
        "n_questions": len(questions),
        "faithfulness": round(faithfulness_score, 4),
        "context_precision": round(context_precision_score, 4),
        "answer_relevance": round(answer_relevancy_score, 4),
        "avg_score": avg,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        **retrieval_scores,
    }
    log.info("eval.doc_done", **result)
    return result
