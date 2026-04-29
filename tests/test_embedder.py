"""Tests for model_registry and Embedder.

Model-loading tests are marked slow — they download ~400 MB on first run.
Run fast tests only:  pytest tests/test_embedder.py -m "not slow"
Run all:              pytest tests/test_embedder.py
"""
import pytest

from ingestion.model_registry import DOMAINS, MODEL_REGISTRY, get_model_spec


class TestModelRegistry:
    def test_all_domains_present(self) -> None:
        expected = {"general", "financial", "medical", "clinical", "legal", "scientific", "technical"}
        assert DOMAINS == expected

    def test_general_is_1024(self) -> None:
        assert MODEL_REGISTRY["general"]["dim"] == 1024

    def test_specialist_domains_are_768(self) -> None:
        for domain in DOMAINS - {"general"}:
            assert MODEL_REGISTRY[domain]["dim"] == 768, f"{domain} should be 768-dim"

    def test_get_model_spec_valid(self) -> None:
        spec = get_model_spec("financial")
        assert spec["hf_id"] == "yiyanghkust/finbert-tone"
        assert spec["dim"] == 768

    def test_get_model_spec_invalid(self) -> None:
        with pytest.raises(ValueError, match="Unknown domain"):
            get_model_spec("unknown-domain")

    def test_all_specs_have_required_keys(self) -> None:
        for domain, spec in MODEL_REGISTRY.items():
            assert "hf_id" in spec, f"{domain} missing hf_id"
            assert "dim" in spec, f"{domain} missing dim"
            assert isinstance(spec["hf_id"], str) and spec["hf_id"]
            assert spec["dim"] in (768, 1024)


class TestEmbedder:
    @pytest.mark.slow
    def test_financial_embed_shape(self) -> None:
        from ingestion.embedder import Embedder
        emb = Embedder("financial")
        vecs = emb.encode(["What was Tesla revenue in 2023?"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 768

    @pytest.mark.slow
    def test_general_embed_shape(self) -> None:
        from ingestion.embedder import Embedder
        emb = Embedder("general")
        vecs = emb.encode(["hello world"])
        assert len(vecs) == 1
        assert len(vecs[0]) == 1024

    @pytest.mark.slow
    def test_empty_input_returns_empty(self) -> None:
        from ingestion.embedder import Embedder
        emb = Embedder("financial")
        assert emb.encode([]) == []

    @pytest.mark.slow
    def test_encode_one(self) -> None:
        from ingestion.embedder import Embedder
        emb = Embedder("financial")
        vec = emb.encode_one("test")
        assert len(vec) == 768

    @pytest.mark.slow
    def test_lru_cache_reuse(self) -> None:
        from ingestion.embedder import Embedder, _load_model
        Embedder("financial")
        Embedder("financial")
        info = _load_model.cache_info()
        assert info.hits >= 1

    @pytest.mark.slow
    def test_normalized_vectors(self) -> None:
        import math

        from ingestion.embedder import Embedder
        emb = Embedder("financial")
        vec = emb.encode_one("normalize test")
        magnitude = math.sqrt(sum(v ** 2 for v in vec))
        assert abs(magnitude - 1.0) < 1e-5
