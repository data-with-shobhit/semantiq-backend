"""Local filesystem StorageBackend implementation."""
from __future__ import annotations

import os
import shutil

from config.logging import get_logger
from config.settings import settings

log = get_logger()


class LocalStorage:
    def __init__(self, root: str | None = None) -> None:
        try:
            self.root = root or settings.storage_root
        except Exception as exc:
            log.error("storage.init_failed", error=str(exc))
            raise
        finally:
            log.debug("storage.init_exit", root=self.root)

    def _tenant_dir(self, tenant_id: str) -> str:
        try:
            return os.path.join(self.root, tenant_id)
        except Exception as exc:
            log.error("storage.tenant_dir_failed", tenant_id=tenant_id, error=str(exc))
            raise

    def _full_path(self, tenant_id: str, filename: str) -> str:
        try:
            return os.path.join(self._tenant_dir(tenant_id), filename)
        except Exception as exc:
            log.error("storage.full_path_failed", tenant_id=tenant_id, error=str(exc))
            raise

    def save(self, tenant_id: str, filename: str, data: bytes) -> str:
        try:
            tenant_dir = self._tenant_dir(tenant_id)
            os.makedirs(tenant_dir, exist_ok=True)
            path = self._full_path(tenant_id, filename)
            with open(path, "wb") as f:
                f.write(data)
            log.info("storage.saved", tenant_id=tenant_id, path=path, size=len(data))
            return path
        except Exception as exc:
            log.error("storage.save_failed", tenant_id=tenant_id, filename=filename, error=str(exc))
            raise
        finally:
            log.debug("storage.save_exit", tenant_id=tenant_id, filename=filename)

    def load(self, path: str) -> bytes:
        try:
            with open(path, "rb") as f:
                data = f.read()
            log.debug("storage.loaded", path=path, size=len(data))
            return data
        except FileNotFoundError as exc:
            log.error("storage.load_not_found", path=path, error=str(exc))
            raise
        except Exception as exc:
            log.error("storage.load_failed", path=path, error=str(exc))
            raise
        finally:
            log.debug("storage.load_exit", path=path)

    def delete(self, path: str) -> None:
        try:
            if os.path.exists(path):
                os.remove(path)
                log.info("storage.deleted", path=path)
        except Exception as exc:
            log.error("storage.delete_failed", path=path, error=str(exc))
            raise
        finally:
            log.debug("storage.delete_exit", path=path)

    def delete_tenant(self, tenant_id: str) -> None:
        try:
            tenant_dir = self._tenant_dir(tenant_id)
            if os.path.exists(tenant_dir):
                shutil.rmtree(tenant_dir)
                log.info("storage.tenant_deleted", tenant_id=tenant_id, dir=tenant_dir)
            os.makedirs(tenant_dir, exist_ok=True)
        except Exception as exc:
            log.error("storage.delete_tenant_failed", tenant_id=tenant_id, error=str(exc))
            raise
        finally:
            log.debug("storage.delete_tenant_exit", tenant_id=tenant_id)
