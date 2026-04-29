"""Verify all docker-compose services start healthy."""
import time

import psycopg2
import redis
from qdrant_client import QdrantClient

POSTGRES_DSN = "postgresql://raguser:ragpass@localhost:5432/ragdb"
REDIS_URL = "redis://localhost:6379/0"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def _wait(fn, timeout: int = 30, interval: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            fn()
            return True
        except Exception:
            time.sleep(interval)
    return False


class TestPostgres:
    def test_connection(self) -> None:
        def connect() -> None:
            conn = psycopg2.connect(POSTGRES_DSN)
            conn.close()

        assert _wait(connect), "Postgres did not become reachable within 30s"

    def test_version(self) -> None:
        conn = psycopg2.connect(POSTGRES_DSN)
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            row = cur.fetchone()
        conn.close()
        assert row is not None
        assert "PostgreSQL 16" in row[0]


class TestRedis:
    def test_ping(self) -> None:
        def ping() -> None:
            r = redis.from_url(REDIS_URL)
            r.ping()

        assert _wait(ping), "Redis did not become reachable within 30s"

    def test_set_get(self) -> None:
        r = redis.from_url(REDIS_URL)
        r.set("__healthcheck__", "ok", ex=5)
        assert r.get("__healthcheck__") == b"ok"


class TestQdrant:
    def test_reachable(self) -> None:
        def check() -> None:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            client.get_collections()

        assert _wait(check), "Qdrant did not become reachable within 30s"

    def test_create_and_delete_collection(self) -> None:
        from qdrant_client.models import Distance, VectorParams

        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        name = "__test_collection__"
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        names = [c.name for c in client.get_collections().collections]
        assert name in names
        client.delete_collection(name)
        names = [c.name for c in client.get_collections().collections]
        assert name not in names
