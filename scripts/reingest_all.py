"""Re-ingest all existing ready docs to pick up new metadata (section_num, clause, etc).

Runs synchronously -- no Celery worker required.
Usage:
    uv run python scripts/reingest_all.py
    uv run python scripts/reingest_all.py --tenant legal-rags   # one tenant only
    uv run python scripts/reingest_all.py --dry-run             # show what would run
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from config.logging import configure_logging, get_logger
from config.settings import settings

sys.path.insert(0, str(Path(__file__).parent.parent))


configure_logging(settings.log_level, "console")
log = get_logger()


async def _reingest_doc(tenant_id: str, doc_id: int, file_path: str) -> int:
    """Full ingestion pipeline -- returns new chunk count."""
    import asyncio as _aio

    from qdrant_client import QdrantClient
    from qdrant_client.models import PointIdsList, PointStruct, SparseVector

    from db.postgres import execute, fetch_one
    from ingestion.chunker import chunk_text
    from ingestion.embedder import Embedder
    from ingestion.extractor import extract_metadata
    from ingestion.loader import load
    from retrieval import splade as splade_enc

    row = await fetch_one("SELECT domain FROM tenants WHERE id=$1", tenant_id)
    domain = row["domain"] if row else "general"

    log.info("reingest.start", tenant_id=tenant_id, doc_id=doc_id, domain=domain)

    text = load(file_path)
    chunks = chunk_text(text, domain=domain)
    base_meta = extract_metadata(text, file_path, domain=domain)

    embedder = Embedder(domain)
    texts = [c["text"] for c in chunks]
    dense_vectors = await _aio.to_thread(embedder.encode, texts)
    sparse_vectors = await _aio.to_thread(splade_enc.encode_batch, texts)

    kwargs: dict = {"host": settings.qdrant_host, "port": settings.qdrant_port, "timeout": 120}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
        kwargs["https"] = True
    client = QdrantClient(**kwargs)

    points = []
    for i, (chunk, dense, (sp_idx, sp_val)) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
        meta = {**base_meta, **chunk["metadata"], "doc_id": doc_id, "tenant_id": tenant_id, "text": chunk["text"]}
        points.append(PointStruct(
            id=doc_id * 100_000 + i,
            vector={"dense": dense, "sparse": SparseVector(indices=sp_idx, values=sp_val)},
            payload=meta,
        ))

    for i, point in enumerate(points):
        point.payload["prev_chunk_id"] = points[i - 1].id if i > 0 else None
        point.payload["next_chunk_id"] = points[i + 1].id if i < len(points) - 1 else None

    # Delete old points (scan up to old chunk_count + buffer)
    old_ids = list(range(doc_id * 100_000, doc_id * 100_000 + 2000))
    try:
        client.delete(collection_name=f"{tenant_id}_docs", points_selector=PointIdsList(points=old_ids))
    except Exception:
        pass

    batch_size = 16
    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=f"{tenant_id}_docs", points=points[i:i + batch_size])

    await execute("UPDATE documents SET status='ready', chunk_count=$1 WHERE id=$2", len(chunks), doc_id)

    log.info("reingest.done", tenant_id=tenant_id, doc_id=doc_id, chunks=len(chunks))
    return len(chunks)


async def main(tenant_filter: str | None, dry_run: bool) -> None:
    from db.postgres import fetch_all, init_pool
    await init_pool()

    if tenant_filter:
        rows = await fetch_all(
            "SELECT id, tenant_id, filename, path, chunk_count FROM documents "
            "WHERE status='ready' AND tenant_id=$1 ORDER BY id",
            tenant_filter,
        )
    else:
        rows = await fetch_all(
            "SELECT id, tenant_id, filename, path, chunk_count FROM documents "
            "WHERE status='ready' ORDER BY tenant_id, id",
        )
    docs = [dict(r) for r in rows]

    if not docs:
        print("No ready documents found.")
        return

    print(f"\n{'Tenant':20} {'DocID':6} {'Chunks':7} Filename")
    print("-" * 70)
    for d in docs:
        print(f"{d['tenant_id']:20} {d['id']:6} {str(d['chunk_count']):7} {d['filename']}")
    print(f"\nTotal: {len(docs)} documents\n")

    if dry_run:
        print("Dry-run -- no changes made.")
        return

    for d in docs:
        if not d["path"] or not Path(d["path"]).exists():
            print(f"  SKIP doc {d['id']} -- file not found: {d['path']}")
            continue
        try:
            new_count = await _reingest_doc(d["tenant_id"], d["id"], d["path"])
            print(f"  OK   doc {d['id']} ({d['tenant_id']}) -> {new_count} chunks")
        except Exception as exc:
            print(f"  ERR  doc {d['id']} ({d['tenant_id']}) -> {exc}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", help="Re-ingest only this tenant")
    parser.add_argument("--dry-run", action="store_true", help="List docs without ingesting")
    args = parser.parse_args()
    asyncio.run(main(args.tenant, args.dry_run))
