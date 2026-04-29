"""Hybrid retrieval: SPLADE sparse + dense + Reciprocal Rank Fusion."""
from __future__ import annotations

import re

from qdrant_client import QdrantClient
from qdrant_client.models import (FieldCondition, Filter, MatchValue,
                                  NamedSparseVector, NamedVector, SparseVector)

from config.logging import get_logger
from config.settings import settings
from ingestion.embedder import Embedder
from retrieval import splade as splade_enc

_LEGAL_REF_RE = re.compile(
    r"\b(?:sections?|sect\.?|sec\.?|article|clause|chapter)\s+(\d+(?:\.\d+)?)\b",
    re.IGNORECASE,
)

log = get_logger()

_TOP_K = 20
_RRF_K = settings.rrf_k


def _qdrant() -> QdrantClient:
    try:
        kwargs: dict = {"host": settings.qdrant_host, "port": settings.qdrant_port}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
            kwargs["https"] = True
        return QdrantClient(**kwargs)
    except Exception as exc:
        log.error("qdrant.client_failed", error=str(exc))
        raise


def _rrf_score(rank: int, k: int = _RRF_K) -> float:
    return 1.0 / (k + rank)


def _fuse(dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
    return _fuse_three([], dense_ids, sparse_ids)


def _fuse_three(direct_ids: list[str], dense_ids: list[str], sparse_ids: list[str]) -> list[str]:
    try:
        scores: dict[str, float] = {}
        # Direct hits get double weight (counted twice at rank 0)
        for rank, pid in enumerate(direct_ids):
            scores[pid] = scores.get(pid, 0.0) + _rrf_score(rank) * 2
        for rank, pid in enumerate(dense_ids):
            scores[pid] = scores.get(pid, 0.0) + _rrf_score(rank)
        for rank, pid in enumerate(sparse_ids):
            scores[pid] = scores.get(pid, 0.0) + _rrf_score(rank)
        return sorted(scores, key=lambda x: scores[x], reverse=True)
    except Exception as exc:
        log.error("retrieval.fuse_failed", error=str(exc))
        return list(dict.fromkeys(direct_ids + dense_ids + sparse_ids))
    finally:
        log.debug("retrieval.fuse_exit")


_CONTEXT_WINDOW = {
    "financial": 2,
    "legal":     2,
    "medical":   2,
    "clinical":  2,
    "general":   1,
    "scientific":1,
    "technical": 0,  # AST chunks are self-contained
}


def _detect_section_reference(query: str) -> int | None:
    """Return integer section_num if query names a specific numbered section, else None."""
    m = _LEGAL_REF_RE.search(query)
    if not m:
        return None
    raw = m.group(1)
    try:
        return int(raw.split(".")[0])
    except ValueError:
        return None


def _section_direct_search(tenant_id: str, section_num: int, top_k: int, collection_name: str | None = None) -> list[dict]:
    """Scroll Qdrant for all chunks where section_num matches — works across all domains."""
    try:
        client = _qdrant()
        coll = collection_name or f"{tenant_id}_docs"
        scroll_filter = Filter(must=[
            FieldCondition(key="section_num", match=MatchValue(value=section_num)),
        ])
        points, _ = client.scroll(
            collection_name=coll,
            scroll_filter=scroll_filter,
            limit=top_k,
            with_payload=True,
        )
        results = [
            {"id": str(p.id), "text": p.payload.get("text", ""), "score": 1.0, "payload": p.payload}
            for p in points
        ]
        log.info("retrieval.section_direct", tenant_id=tenant_id, section_num=section_num, hits=len(results))
        return results
    except Exception as exc:
        log.warning("retrieval.section_direct_failed", tenant_id=tenant_id, section_num=section_num, error=str(exc))
        return []


def retrieve(
    query: str,
    tenant_id: str,
    embedder: Embedder,
    payload_filter: dict | None = None,
    top_k: int = _TOP_K,
    domain: str = "general",
    collection_name: str | None = None,
) -> list[dict]:
    coll = collection_name or f"{tenant_id}_docs"
    try:
        qdrant_filter = _build_filter(payload_filter) if payload_filter else None

        direct_hits: list[dict] = []
        section_num = _detect_section_reference(query)
        if section_num is not None:
            direct_hits = _section_direct_search(tenant_id, section_num, top_k, coll)

        dense_hits = _dense_search(query, tenant_id, embedder, qdrant_filter, top_k, coll)
        sparse_hits = _sparse_search(query, tenant_id, qdrant_filter, top_k, dense_hits, coll)

        dense_ids = [h["id"] for h in dense_hits]
        sparse_ids = [h["id"] for h in sparse_hits]
        direct_ids = [h["id"] for h in direct_hits]

        fused_ids = _fuse_three(direct_ids, dense_ids, sparse_ids)[:top_k]

        hit_map = {h["id"]: h for h in direct_hits + dense_hits + sparse_hits}
        results = [hit_map[pid] for pid in fused_ids if pid in hit_map]

        window = _CONTEXT_WINDOW.get(domain, 1)
        if window > 0:
            results = _expand_context(results, tenant_id, window, coll)

        log.info("retrieval.hybrid", tenant_id=tenant_id, collection=coll, results=len(results), direct=len(direct_hits))
        return results
    except Exception as exc:
        log.error("retrieval.hybrid_failed", tenant_id=tenant_id, collection=coll, error=str(exc))
        return []


def _expand_context(hits: list[dict], tenant_id: str, window: int, collection_name: str | None = None) -> list[dict]:
    """Fetch prev/next neighbor chunks from Qdrant and merge into results."""
    try:
        neighbor_ids: set[int] = set()
        seen_ids = {int(h["id"]) for h in hits}

        for hit in hits:
            payload = hit.get("payload", {})
            for _ in range(window):
                prev_id = payload.get("prev_chunk_id")
                next_id = payload.get("next_chunk_id")
                if prev_id and int(prev_id) not in seen_ids:
                    neighbor_ids.add(int(prev_id))
                if next_id and int(next_id) not in seen_ids:
                    neighbor_ids.add(int(next_id))

        if not neighbor_ids:
            return hits

        client = _qdrant()
        neighbor_points = client.retrieve(
            collection_name=collection_name or f"{tenant_id}_docs",
            ids=list(neighbor_ids),
            with_payload=True,
        )
        neighbors = [
            {"id": str(p.id), "text": p.payload.get("text", ""), "score": 0.0, "payload": p.payload, "is_neighbor": True}
            for p in neighbor_points
        ]
        seen_ids.update(neighbor_ids)
        all_hits = hits + neighbors

        # Sort by chunk_index within same doc to preserve reading order
        all_hits.sort(key=lambda h: (
            h.get("payload", {}).get("doc_id", 0),
            h.get("payload", {}).get("chunk_index", 0),
        ))
        log.debug("retrieval.context_expanded", original=len(hits), neighbors=len(neighbors))
        return all_hits
    except Exception as exc:
        log.warning("retrieval.expand_context_failed", error=str(exc))
        return hits


def _dense_search(
    query: str,
    tenant_id: str,
    embedder: Embedder,
    qdrant_filter: Filter | None,
    top_k: int,
    collection_name: str | None = None,
) -> list[dict]:
    try:
        vector = embedder.encode_one(query)
        hits = _qdrant().search(
            collection_name=collection_name or f"{tenant_id}_docs",
            query_vector=NamedVector(name="dense", vector=vector),
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {"id": str(h.id), "text": h.payload.get("text", ""), "score": h.score, "payload": h.payload}
            for h in hits
        ]
    except Exception as exc:
        log.error("retrieval.dense_failed", tenant_id=tenant_id, error=str(exc))
        return []


def _sparse_search(
    query: str,
    tenant_id: str,
    qdrant_filter: Filter | None,
    top_k: int,
    dense_hits: list[dict],
    collection_name: str | None = None,
) -> list[dict]:
    """SPLADE primary, BM25 fallback if SPLADE returns empty."""
    try:
        indices, values = splade_enc.encode(query)

        if indices:
            try:
                hits = _qdrant().search(
                    collection_name=collection_name or f"{tenant_id}_docs",
                    query_vector=NamedSparseVector(
                        name="sparse",
                        vector=SparseVector(indices=indices, values=values),
                    ),
                    limit=top_k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                results = [
                    {"id": str(h.id), "text": h.payload.get("text", ""), "score": h.score, "payload": h.payload}
                    for h in hits
                ]
                if results:
                    log.debug("sparse.splade", hits=len(results))
                    return results
            except Exception as exc:
                log.warning("sparse.splade_failed", error=str(exc))

        log.debug("sparse.bm25_fallback", tenant_id=tenant_id)
        return splade_enc.bm25_search(query, dense_hits)
    except Exception as exc:
        log.error("retrieval.sparse_search_failed", tenant_id=tenant_id, error=str(exc))
        return dense_hits
    finally:
        log.debug("retrieval.sparse_search_exit", tenant_id=tenant_id)


def _build_filter(payload_filter: dict) -> Filter:
    try:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in payload_filter.items()
        ]
        return Filter(must=conditions)
    except Exception as exc:
        log.error("retrieval.build_filter_failed", error=str(exc))
        raise
    finally:
        log.debug("retrieval.build_filter_exit")
