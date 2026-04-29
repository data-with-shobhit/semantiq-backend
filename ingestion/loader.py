"""Document loaders — pymupdf4llm (markdown-aware) for PDF, plain text fallback."""
from __future__ import annotations

import re

from config.logging import get_logger

log = get_logger()


def _is_toc_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    dot_count = stripped.count(".")
    if dot_count >= 5 and dot_count / max(len(stripped), 1) > 0.3:
        return True
    if re.search(r"\.\s*\.\s*\.\s*\.\s*\.\s*\d+\s*$", stripped):
        return True
    return False


def _clean_markdown(text: str) -> str:
    """Clean pymupdf4llm markdown output — remove TOC lines, normalize whitespace."""
    lines = []
    for line in text.splitlines():
        if _is_toc_line(line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def load_pdf(path: str) -> str:
    try:
        import pymupdf4llm
        md = pymupdf4llm.to_markdown(path, show_progress=False)
        result = _clean_markdown(md)
        log.info("loader.pdf_markdown", path=path, chars=len(result))
        return result
    except Exception as exc:
        log.warning("loader.pdf_pymupdf_failed", path=path, error=str(exc))
        return _pypdf_fallback(path)
    finally:
        log.debug("loader.pdf_exit", path=path)


def _pypdf_fallback(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = _clean_markdown("\n\n".join(p.strip() for p in pages if p.strip()))
        log.info("loader.pypdf_fallback", path=path, pages=len(reader.pages), chars=len(text))
        return text
    except Exception as exc:
        log.error("loader.pypdf_fallback_failed", path=path, error=str(exc))
        raise
    finally:
        log.debug("loader.pypdf_fallback_exit", path=path)


def load_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        log.info("loader.text", path=path, chars=len(text))
        return text
    except Exception as exc:
        log.error("loader.text_failed", path=path, error=str(exc))
        raise
    finally:
        log.debug("loader.text_exit", path=path)


def load_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        parts: list[str] = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            # Preserve heading structure as markdown-style headers
            style = para.style.name if para.style else ""
            if "Heading 1" in style:
                parts.append(f"# {text}")
            elif "Heading 2" in style:
                parts.append(f"## {text}")
            elif "Heading 3" in style:
                parts.append(f"### {text}")
            else:
                parts.append(text)
        result = "\n\n".join(parts)
        log.info("loader.docx", path=path, chars=len(result), paragraphs=len(parts))
        return result
    except Exception as exc:
        log.error("loader.docx_failed", path=path, error=str(exc))
        raise
    finally:
        log.debug("loader.docx_exit", path=path)


def load(path: str) -> str:
    try:
        lower = path.lower()
        if lower.endswith(".pdf"):
            return load_pdf(path)
        if lower.endswith(".docx"):
            return load_docx(path)
        return load_text(path)
    except Exception as exc:
        log.error("loader.load_failed", path=path, error=str(exc))
        raise
    finally:
        log.debug("loader.load_exit", path=path)
