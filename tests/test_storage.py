"""Unit tests for LocalStorage — no external services needed."""
import os

import pytest

from storage.local import LocalStorage


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(root=str(tmp_path))


class TestLocalStorage:
    def test_save_and_load(self, storage: LocalStorage) -> None:
        path = storage.save("acme-corp", "test.pdf", b"hello bytes")
        assert storage.load(path) == b"hello bytes"

    def test_save_creates_tenant_dir(self, storage: LocalStorage, tmp_path) -> None:
        storage.save("acme-corp", "doc.pdf", b"data")
        assert (tmp_path / "acme-corp").is_dir()

    def test_save_returns_correct_path(self, storage: LocalStorage, tmp_path) -> None:
        path = storage.save("acme-corp", "doc.pdf", b"data")
        assert os.path.exists(path)
        assert "acme-corp" in path
        assert "doc.pdf" in path

    def test_tenant_isolation(self, storage: LocalStorage, tmp_path) -> None:
        storage.save("acme-corp", "file.pdf", b"acme data")
        storage.save("contoso", "file.pdf", b"contoso data")
        acme_path = storage._full_path("acme-corp", "file.pdf")
        contoso_path = storage._full_path("contoso", "file.pdf")
        assert storage.load(acme_path) == b"acme data"
        assert storage.load(contoso_path) == b"contoso data"

    def test_delete_file(self, storage: LocalStorage) -> None:
        path = storage.save("acme-corp", "del.pdf", b"bye")
        assert os.path.exists(path)
        storage.delete(path)
        assert not os.path.exists(path)

    def test_delete_nonexistent_is_safe(self, storage: LocalStorage) -> None:
        storage.delete("/tmp/does_not_exist_xyz.pdf")

    def test_delete_tenant(self, storage: LocalStorage, tmp_path) -> None:
        storage.save("acme-corp", "a.pdf", b"a")
        storage.save("acme-corp", "b.pdf", b"b")
        storage.delete_tenant("acme-corp")
        tenant_dir = tmp_path / "acme-corp"
        assert tenant_dir.exists()
        assert list(tenant_dir.iterdir()) == []

    def test_delete_tenant_does_not_affect_others(self, storage: LocalStorage) -> None:
        storage.save("acme-corp", "a.pdf", b"a")
        contoso_path = storage.save("contoso", "b.pdf", b"b")
        storage.delete_tenant("acme-corp")
        assert storage.load(contoso_path) == b"b"

    def test_satisfies_protocol(self, storage: LocalStorage) -> None:
        from storage.backend import StorageBackend
        assert isinstance(storage, StorageBackend)
