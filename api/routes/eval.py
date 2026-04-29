"""Eval + strategy endpoints.

POST /eval/run               — run RAGAS on a doc (questions + ground truths)
GET  /eval/history/{doc_id}  — list all eval runs for a doc
GET  /eval/strategy/{doc_id} — list strategy versions for a doc
POST /eval/reingest/{doc_id} — re-ingest with refined strategy (from RAGAS)
POST /eval/revert            — revert to a previous strategy version
GET  /eval/strategies        — list all named/saved strategies for tenant
POST /eval/strategy/{id}/name — give a strategy a reusable name
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth import get_tenant
from api.models.eval import EvalRequest, NameStrategyRequest, ReIngestRequest, RevertRequest
from config.logging import get_logger
from db.postgres import (execute, fetch_all, fetch_one, get_active_strategy,
                         get_latest_ragas_eval, revert_strategy)
from eval.ragas_eval import evaluate_doc
from ingestion.strategy_refiner import refine_strategy
from ingestion.tasks import process_document

log = get_logger()
router = APIRouter(prefix="/eval", tags=["eval"])



@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_eval(
    req: EvalRequest,
    tenant_id: str = Depends(get_tenant),
) -> dict[str, Any]:
    doc = await fetch_one(
        "SELECT id, status, chunk_count, workspace_id FROM documents WHERE id=$1 AND tenant_id=$2",
        req.doc_id, tenant_id,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    has_chunks = doc.get("chunk_count") or doc["status"] == "ready"
    if not has_chunks:
        raise HTTPException(status_code=409, detail=f"Document status is '{doc['status']}' — must be 'ready' or have chunks")

    active = await get_active_strategy(req.doc_id, tenant_id)
    strategy_id = active["id"] if active else None

    try:
        result = await evaluate_doc(
            doc_id=req.doc_id,
            tenant_id=tenant_id,
            workspace_id=doc["workspace_id"],
            questions=req.questions,
            ground_truths=req.ground_truths,
            strategy_id=strategy_id,
            delay_s=req.delay_s,
        )
        return result
    except Exception as exc:
        log.error("eval.run_failed", tenant_id=tenant_id, doc_id=req.doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/history/{doc_id}", status_code=status.HTTP_200_OK)
async def eval_history(
    doc_id: int,
    tenant_id: str = Depends(get_tenant),
) -> dict[str, Any]:
    doc = await fetch_one(
        "SELECT id FROM documents WHERE id=$1 AND tenant_id=$2", doc_id, tenant_id
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    row = await fetch_one(
        """SELECT id, strategy_id, faithfulness, context_precision, answer_relevance,
                  avg_score, retrieval_metrics, questions, created_at
           FROM ragas_evals WHERE doc_id=$1 AND tenant_id=$2""",
        doc_id, tenant_id,
    )
    if not row:
        return {"doc_id": doc_id, "eval": None}

    retrieval = json.loads(row["retrieval_metrics"]) if row["retrieval_metrics"] else {}
    return {
        "doc_id": doc_id,
        "eval": {
            "eval_id": row["id"],
            "strategy_id": row["strategy_id"],
            "faithfulness": row["faithfulness"],
            "context_precision": row["context_precision"],
            "answer_relevance": row["answer_relevance"],
            "avg_score": row["avg_score"],
            "retrieval_metrics": retrieval,
            "n_questions": len(json.loads(row["questions"])) if row["questions"] else 0,
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        },
    }


@router.get("/strategy/{doc_id}", status_code=status.HTTP_200_OK)
async def strategy_history(
    doc_id: int,
    tenant_id: str = Depends(get_tenant),
) -> dict[str, Any]:
    doc = await fetch_one(
        "SELECT id FROM documents WHERE id=$1 AND tenant_id=$2", doc_id, tenant_id
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    rows = await fetch_all(
        """SELECT id, version, chunker, chunk_size, overlap, reasoning,
                  source, ragas_scores, user_feedback, is_active, created_at
           FROM ingestion_strategies WHERE doc_id=$1 AND tenant_id=$2
           ORDER BY version DESC""",
        doc_id, tenant_id,
    )
    return {
        "doc_id": doc_id,
        "strategies": [
            {
                "strategy_id": r["id"],
                "version": r["version"],
                "chunker": r["chunker"],
                "chunk_size": r["chunk_size"],
                "overlap": r["overlap"],
                "reasoning": r["reasoning"],
                "source": r["source"],
                "ragas_scores": r["ragas_scores"],
                "user_feedback": r["user_feedback"],
                "is_active": r["is_active"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ],
    }


@router.post("/reingest/{doc_id}", status_code=status.HTTP_202_ACCEPTED)
async def reingest_with_refinement(
    doc_id: int,
    req: ReIngestRequest,
    tenant_id: str = Depends(get_tenant),
) -> dict[str, Any]:
    doc = await fetch_one(
        "SELECT id, filename, path, status FROM documents WHERE id=$1 AND tenant_id=$2",
        doc_id, tenant_id,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    tenant = await fetch_one("SELECT domain FROM tenants WHERE id=$1", tenant_id)
    domain = tenant["domain"] if tenant else "general"

    latest_eval = await get_latest_ragas_eval(doc_id, tenant_id)
    if not latest_eval:
        raise HTTPException(
            status_code=409,
            detail="No RAGAS eval found for this document. Run /eval/run first.",
        )

    try:
        strategy = await refine_strategy(
            doc_id=doc_id,
            tenant_id=tenant_id,
            file_path=doc["path"],
            filename=doc["filename"],
            domain=domain,
            user_feedback=req.user_feedback,
        )
        if not strategy:
            raise HTTPException(status_code=500, detail="Strategy refinement failed")

        task = process_document.apply_async(
            args=[tenant_id, doc_id, doc["path"]],
            task_id=f"{tenant_id}-reingest-{doc_id}",
        )
        log.info("eval.reingest_enqueued", tenant_id=tenant_id, doc_id=doc_id, task_id=task.id)

        return {
            "doc_id": doc_id,
            "job_id": task.id,
            "new_strategy": strategy.to_dict(),
            "status": "reingesting",
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.error("eval.reingest_failed", tenant_id=tenant_id, doc_id=doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/revert", status_code=status.HTTP_200_OK)
async def revert_to_version(
    req: RevertRequest,
    tenant_id: str = Depends(get_tenant),
) -> dict[str, Any]:
    doc = await fetch_one(
        "SELECT id, filename, path FROM documents WHERE id=$1 AND tenant_id=$2",
        req.doc_id, tenant_id,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    ok = await revert_strategy(req.doc_id, tenant_id, req.version)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Strategy version {req.version} not found")

    task = process_document.apply_async(
        args=[tenant_id, req.doc_id, doc["path"]],
        task_id=f"{tenant_id}-revert-{req.doc_id}-v{req.version}",
    )
    log.info("eval.revert_enqueued", tenant_id=tenant_id, doc_id=req.doc_id, version=req.version)

    return {
        "doc_id": req.doc_id,
        "reverted_to_version": req.version,
        "job_id": task.id,
        "status": "reingesting",
    }


# ── Strategy library — named, reusable across docs ───────────────────────────



@router.get("/strategies", status_code=status.HTTP_200_OK)
async def list_strategies(
    tenant_id: str = Depends(get_tenant),
    named_only: bool = False,
) -> dict[str, Any]:
    """List all strategies for this tenant — optionally filter to named (reusable) ones."""
    query = """
        SELECT s.id, s.doc_id, s.version, s.name, s.chunker, s.chunk_size, s.overlap,
               s.reasoning, s.source, s.ragas_scores, s.is_active, s.created_at,
               d.filename
        FROM ingestion_strategies s
        JOIN documents d ON d.id = s.doc_id
        WHERE s.tenant_id = $1
    """
    params = [tenant_id]
    if named_only:
        query += " AND s.name IS NOT NULL"
    query += " ORDER BY s.created_at DESC"

    rows = await fetch_all(query, *params)
    return {
        "tenant_id": tenant_id,
        "strategies": [
            {
                "strategy_id": r["id"],
                "doc_id": r["doc_id"],
                "filename": r["filename"],
                "version": r["version"],
                "name": r["name"],
                "chunker": r["chunker"],
                "chunk_size": r["chunk_size"],
                "overlap": r["overlap"],
                "reasoning": r["reasoning"],
                "source": r["source"],
                "ragas_scores": r["ragas_scores"],
                "is_active": r["is_active"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
            }
            for r in rows
        ],
    }


@router.post("/strategy/{strategy_id}/name", status_code=status.HTTP_200_OK)
async def name_strategy(
    strategy_id: int,
    req: NameStrategyRequest,
    tenant_id: str = Depends(get_tenant),
) -> dict[str, Any]:
    """Give a strategy a reusable name so it can be cloned for future uploads."""
    row = await fetch_one(
        "SELECT id, doc_id, chunker FROM ingestion_strategies WHERE id=$1 AND tenant_id=$2",
        strategy_id, tenant_id,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Prevent duplicate names within tenant
    existing = await fetch_one(
        "SELECT id FROM ingestion_strategies WHERE tenant_id=$1 AND name=$2 AND id!=$3",
        tenant_id, req.name, strategy_id,
    )
    if existing:
        raise HTTPException(status_code=409, detail=f"Strategy name '{req.name}' already exists for this tenant")

    await execute(
        "UPDATE ingestion_strategies SET name=$1 WHERE id=$2 AND tenant_id=$3",
        req.name, strategy_id, tenant_id,
    )
    log.info("eval.strategy_named", tenant_id=tenant_id, strategy_id=strategy_id, name=req.name)
    return {"strategy_id": strategy_id, "name": req.name, "doc_id": row["doc_id"]}
