"""Redis client — semantic cache + tenant-prefixed keys."""
from __future__ import annotations

import hashlib

import redis

from config.logging import get_logger
from config.settings import settings

log = get_logger()

_client: redis.Redis | None = None
_CACHE_TTL = 3600  # 1 hour for query cache


def get_client() -> redis.Redis:
    global _client
    try:
        if _client is None:
            _client = redis.from_url(settings.redis_url, decode_responses=True)
        return _client
    except Exception as exc:
        log.error("redis.connect_failed", url=settings.redis_url, error=str(exc))
        raise
    finally:
        log.debug("redis.get_client_exit")


def _key(tenant_id: str, *parts: str) -> str:
    return f"{tenant_id}:" + ":".join(parts)


def cache_get(tenant_id: str, query: str) -> str | None:
    try:
        h = hashlib.sha256(query.encode()).hexdigest()[:16]
        value = get_client().get(_key(tenant_id, "query", h))
        if value:
            log.info("cache.hit", tenant_id=tenant_id)
        return value
    except Exception as exc:
        log.warning("cache.get_failed", tenant_id=tenant_id, error=str(exc))
        return None


def cache_set(tenant_id: str, query: str, answer: str) -> None:
    try:
        h = hashlib.sha256(query.encode()).hexdigest()[:16]
        get_client().setex(_key(tenant_id, "query", h), _CACHE_TTL, answer)
        log.info("cache.set", tenant_id=tenant_id)
    except Exception as exc:
        log.warning("cache.set_failed", tenant_id=tenant_id, error=str(exc))


def set_doc_status(tenant_id: str, doc_id: int, status: str) -> None:
    try:
        get_client().setex(_key(tenant_id, "doc", str(doc_id), "status"), 86400, status)
    except Exception as exc:
        log.warning("cache.set_doc_status_failed", tenant_id=tenant_id, doc_id=doc_id, error=str(exc))


def get_doc_status(tenant_id: str, doc_id: int) -> str | None:
    try:
        return get_client().get(_key(tenant_id, "doc", str(doc_id), "status"))
    except Exception as exc:
        log.warning("cache.get_doc_status_failed", tenant_id=tenant_id, doc_id=doc_id, error=str(exc))
        return None


def flush_tenant(tenant_id: str) -> int:
    try:
        pattern = _key(tenant_id, "*")
        keys = get_client().keys(pattern)
        if keys:
            get_client().delete(*keys)
        log.info("cache.flushed", tenant_id=tenant_id, keys=len(keys))
        return len(keys)
    except Exception as exc:
        log.warning("cache.flush_failed", tenant_id=tenant_id, error=str(exc))
        return 0
