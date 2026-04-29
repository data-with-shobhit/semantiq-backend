"""StorageBackend protocol — swap local ↔ S3 with one config change."""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    def save(self, tenant_id: str, filename: str, data: bytes) -> str:
        """Persist file and return the storage path."""
        ...

    def load(self, path: str) -> bytes:
        """Read file by its storage path."""
        ...

    def delete(self, path: str) -> None:
        """Delete file by its storage path."""
        ...

    def delete_tenant(self, tenant_id: str) -> None:
        """Delete all files for a tenant."""
        ...
