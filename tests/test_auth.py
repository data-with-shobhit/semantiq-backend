"""Unit tests for api/auth.py — no DB, no network required."""
import os
import time

import jwt
import pytest
from fastapi import HTTPException

from api.auth import _decode_token, get_tenant, issue_token

os.environ["JWT_SECRET"] = "test-secret-key"



class TestIssueToken:
    def test_returns_string(self) -> None:
        token = issue_token("acme-corp")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_sub_claim(self) -> None:
        token = issue_token("acme-corp")
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == "acme-corp"

    def test_expiry_in_payload(self) -> None:
        token = issue_token("acme-corp", expiry_days=7)
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert "exp" in payload
        assert "iat" in payload
        assert payload["exp"] > payload["iat"]

    def test_different_tenants_different_tokens(self) -> None:
        t1 = issue_token("acme-corp")
        t2 = issue_token("contoso")
        assert t1 != t2


class TestDecodeToken:
    def test_valid_token(self) -> None:
        token = issue_token("acme-corp")
        payload = _decode_token(token)
        assert payload["sub"] == "acme-corp"

    def test_expired_token_raises_401(self) -> None:
        payload = {
            "sub": "acme-corp",
            "exp": int(time.time()) - 1,
            "iat": int(time.time()) - 10,
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_wrong_secret_raises_401(self) -> None:
        token = jwt.encode({"sub": "acme-corp"}, "wrong-secret", algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            _decode_token(token)
        assert exc_info.value.status_code == 401

    def test_garbage_token_raises_401(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _decode_token("not.a.token")
        assert exc_info.value.status_code == 401


class TestGetTenant:
    @pytest.mark.asyncio
    async def test_returns_tenant_id(self) -> None:
        token = issue_token("acme-corp")
        tenant_id = await get_tenant(token)
        assert tenant_id == "acme-corp"

    @pytest.mark.asyncio
    async def test_expired_raises_401(self) -> None:
        payload = {
            "sub": "acme-corp",
            "exp": int(time.time()) - 1,
        }
        token = jwt.encode(payload, "test-secret-key", algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            await get_tenant(token)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_sub_raises_401(self) -> None:
        token = jwt.encode({"foo": "bar"}, "test-secret-key", algorithm="HS256")
        with pytest.raises(HTTPException) as exc_info:
            await get_tenant(token)
        assert exc_info.value.status_code == 401
        assert "sub" in exc_info.value.detail
