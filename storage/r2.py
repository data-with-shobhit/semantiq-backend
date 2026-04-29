"""Cloudflare R2 StorageBackend — S3-compatible, drop-in replacement for LocalStorage."""
from __future__ import annotations

from config.logging import get_logger
from config.settings import settings

log = get_logger()


def _client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",
    )


class R2Storage:
    def __init__(self) -> None:
        self.bucket = settings.r2_bucket_name
        log.info("storage.r2_init", bucket=self.bucket)

    def _key(self, tenant_id: str, filename: str) -> str:
        return f"{tenant_id}/{filename}"

    def save(self, tenant_id: str, filename: str, data: bytes) -> str:
        try:
            key = self._key(tenant_id, filename)
            _client().put_object(Bucket=self.bucket, Key=key, Body=data)
            path = f"r2://{self.bucket}/{key}"
            log.info("storage.r2_saved", tenant_id=tenant_id, key=key, size=len(data))
            return path
        except Exception as exc:
            log.error("storage.r2_save_failed", tenant_id=tenant_id, filename=filename, error=str(exc))
            raise

    def load(self, path: str) -> bytes:
        try:
            # path format: r2://bucket/tenant_id/filename  OR  tenant_id/filename
            key = path.replace(f"r2://{self.bucket}/", "")
            resp = _client().get_object(Bucket=self.bucket, Key=key)
            data = resp["Body"].read()
            log.debug("storage.r2_loaded", key=key, size=len(data))
            return data
        except Exception as exc:
            log.error("storage.r2_load_failed", path=path, error=str(exc))
            raise

    def delete(self, path: str) -> None:
        try:
            key = path.replace(f"r2://{self.bucket}/", "")
            _client().delete_object(Bucket=self.bucket, Key=key)
            log.info("storage.r2_deleted", key=key)
        except Exception as exc:
            log.error("storage.r2_delete_failed", path=path, error=str(exc))
            raise

    def delete_tenant(self, tenant_id: str) -> None:
        try:
            client = _client()
            paginator = client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=f"{tenant_id}/"):
                objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
                if objects:
                    client.delete_objects(Bucket=self.bucket, Delete={"Objects": objects})
            log.info("storage.r2_tenant_deleted", tenant_id=tenant_id)
        except Exception as exc:
            log.error("storage.r2_delete_tenant_failed", tenant_id=tenant_id, error=str(exc))
            raise
