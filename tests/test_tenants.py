"""Integration tests for POST /signup — requires live Postgres + Qdrant."""
import os

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from api.main import app
from db.postgres import close_pool, execute, init_pool

os.environ.setdefault("DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")


TENANT = "__test_signup__"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture(autouse=True)
async def cleanup():
    await init_pool()
    await execute("DELETE FROM tenants WHERE id = $1", TENANT)
    yield
    await execute("DELETE FROM tenants WHERE id = $1", TENANT)
    await close_pool()


class TestSignup:
    def test_signup_success(self, client: TestClient) -> None:
        resp = client.post("/tenants/signup", json={
            "tenant_id": TENANT,
            "plan": "free",
            "domain": "financial",
        })
        assert resp.status_code == 201
        body = resp.json()
        assert body["tenant_id"] == TENANT
        assert "token" in body
        assert body["embed_model"] == "yiyanghkust/finbert-tone"
        assert body["embed_dim"] == 768

    def test_signup_general_domain_1024(self, client: TestClient) -> None:
        resp = client.post("/tenants/signup", json={
            "tenant_id": TENANT,
            "plan": "pro",
            "domain": "general",
        })
        assert resp.status_code == 201
        assert resp.json()["embed_dim"] == 1024

    def test_signup_duplicate_returns_409(self, client: TestClient) -> None:
        client.post("/tenants/signup", json={"tenant_id": TENANT, "domain": "general"})
        resp = client.post("/tenants/signup", json={"tenant_id": TENANT, "domain": "general"})
        assert resp.status_code == 409

    def test_signup_invalid_domain_returns_422(self, client: TestClient) -> None:
        resp = client.post("/tenants/signup", json={
            "tenant_id": TENANT,
            "domain": "not-a-domain",
        })
        assert resp.status_code == 422

    def test_signup_invalid_plan_returns_422(self, client: TestClient) -> None:
        resp = client.post("/tenants/signup", json={
            "tenant_id": TENANT,
            "plan": "ultra",
        })
        assert resp.status_code == 422

    def test_signup_invalid_tenant_id_returns_422(self, client: TestClient) -> None:
        resp = client.post("/tenants/signup", json={
            "tenant_id": "bad tenant id!",
        })
        assert resp.status_code == 422

    def test_token_is_valid_jwt(self, client: TestClient) -> None:
        import jwt as pyjwt
        resp = client.post("/tenants/signup", json={"tenant_id": TENANT})
        token = resp.json()["token"]
        payload = pyjwt.decode(token, "test-secret", algorithms=["HS256"])
        assert payload["sub"] == TENANT
