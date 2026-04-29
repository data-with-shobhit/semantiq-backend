"""Tests for chunker and extractor."""

from ingestion.chunker import chunk_text
from ingestion.extractor import extract_metadata


class TestChunker:
    def test_general_returns_chunks(self) -> None:
        text = "Hello world. " * 200
        chunks = chunk_text(text, domain="general")
        assert len(chunks) >= 1
        assert all("text" in c for c in chunks)
        assert all("metadata" in c for c in chunks)

    def test_financial_domain_tagged(self) -> None:
        text = "Revenue grew 20%. " * 100
        chunks = chunk_text(text, domain="financial")
        assert all(c["metadata"]["domain"] == "financial" for c in chunks)

    def test_technical_ast_splits_functions(self) -> None:
        source = '''
def foo(x: int) -> int:
    """Add one."""
    return x + 1

def bar(y: str) -> str:
    return y.upper()

class MyClass:
    def method(self):
        pass
'''
        chunks = chunk_text(source, domain="technical")
        assert len(chunks) == 3
        names = [c["metadata"]["function_name"] for c in chunks]
        assert "foo" in names
        assert "bar" in names
        assert "MyClass" in names

    def test_technical_has_docstring_flag(self) -> None:
        source = '''
def foo():
    """Has docstring."""
    pass

def bar():
    pass
'''
        chunks = chunk_text(source, domain="technical")
        foo = next(c for c in chunks if c["metadata"]["function_name"] == "foo")
        bar = next(c for c in chunks if c["metadata"]["function_name"] == "bar")
        assert foo["metadata"]["has_docstring"] is True
        assert bar["metadata"]["has_docstring"] is False

    def test_technical_no_top_level_falls_back(self) -> None:
        source = "x = 1\ny = 2\n"
        chunks = chunk_text(source, domain="technical")
        assert len(chunks) >= 1

    def test_chunk_index_sequential(self) -> None:
        text = "sentence. " * 300
        chunks = chunk_text(text, domain="general")
        indices = [c["metadata"]["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestExtractor:
    def test_filename_extracted(self) -> None:
        meta = extract_metadata("some text", "/storage/docs/acme/report.pdf")
        assert meta["filename"] == "report.pdf"

    def test_financial_year_extracted(self) -> None:
        text = "This annual report covers fiscal year 2023 performance..."
        meta = extract_metadata(text, "report.pdf", domain="financial")
        assert meta.get("fiscal_year") == "2023"

    def test_financial_section_detected(self) -> None:
        text = "Total revenue and net income for the period..." * 10
        meta = extract_metadata(text, "report.pdf", domain="financial")
        assert meta.get("section") == "income_statement"

    def test_financial_has_numbers(self) -> None:
        meta = extract_metadata("Revenue was $1,234,567", "r.pdf", domain="financial")
        assert meta["has_numbers"] is True

    def test_general_domain_no_financial_fields(self) -> None:
        meta = extract_metadata("some general text", "doc.txt", domain="general")
        assert "fiscal_year" not in meta
        assert "section" not in meta
