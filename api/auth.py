"""JWT signing, verification, and the get_tenant() FastAPI dependency."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from config.logging import get_logger
from config.settings import settings

log = get_logger()

_oauth2 = OAuth2PasswordBearer(tokenUrl="token")


def issue_token(tenant_id: str, expiry_days: int | None = None) -> str:
    try:
        days = expiry_days if expiry_days is not None else settings.jwt_expiry_days
        payload = {
            "sub": tenant_id,
            "exp": datetime.now(timezone.utc) + timedelta(days=days),
            "iat": datetime.now(timezone.utc),
        }
        token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
        log.info("jwt.issued", tenant_id=tenant_id, expiry_days=days)
        return token
    except Exception as exc:
        log.error("jwt.issue_failed", tenant_id=tenant_id, error=str(exc))
        raise
    finally:
        log.debug("jwt.issue_exit", tenant_id=tenant_id)


def _decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        log.warning("jwt.expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        log.warning("jwt.invalid", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        log.error("jwt.decode_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token decode failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
    finally:
        log.debug("jwt.decode_exit")


async def get_tenant(token: str = Depends(_oauth2)) -> str:
    """FastAPI dependency — returns tenant_id from JWT sub claim.

    This is the ONLY source of tenant_id in the entire codebase.
    Never accept tenant_id from request body or URL params.
    """
    try:
        payload = _decode_token(token)
        tenant_id: str | None = payload.get("sub")
        if not tenant_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing sub claim",
            )
        log.debug("jwt.verified", tenant_id=tenant_id)
        return tenant_id
    except HTTPException:
        raise
    except Exception as exc:
        log.error("jwt.get_tenant_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
        )
    finally:
        log.debug("jwt.get_tenant_exit")
