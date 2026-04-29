"""Metadata extraction from document text and file path."""
from __future__ import annotations

import os
import re
import time

from config.logging import get_logger

log = get_logger()

# Legal reference patterns
_SECTION_RE = re.compile(r"\bsection\s+(\d+)\b", re.IGNORECASE)
_CLAUSE_RE  = re.compile(r"\bclause\s+(\d+(?:\.\d+)*)\b", re.IGNORECASE)
_CHAPTER_RE = re.compile(r"\bchapter\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
_ARTICLE_RE = re.compile(r"\barticle\s+(\d+)\b", re.IGNORECASE)


def extract_metadata(text: str, file_path: str, domain: str = "general") -> dict:
    try:
        stat = os.stat(file_path) if os.path.exists(file_path) else None
        meta: dict = {
            "filename": os.path.basename(file_path),
            "source": os.path.basename(file_path),
            "domain": domain,
            "file_size_bytes": stat.st_size if stat else 0,
            "timestamp": int(stat.st_mtime) if stat else int(time.time()),
        }
        if domain == "financial":
            meta.update(_extract_financial(text))
        elif domain in ("legal",):
            meta.update(_extract_legal(text))
        return meta
    except Exception as exc:
        log.error("extractor.failed", file_path=file_path, domain=domain, error=str(exc))
        return {"filename": os.path.basename(file_path), "domain": domain}
    finally:
        log.debug("extractor.exit", file_path=file_path, domain=domain)


def extract_chunk_metadata(chunk_text: str, section: str, domain: str) -> dict:
    """Per-chunk metadata — section_num, clause extracted from chunk + section title (all domains)."""
    try:
        return _extract_section_metadata(chunk_text, section)
    except Exception as exc:
        log.warning("extractor.chunk_metadata_failed", domain=domain, error=str(exc))
        return {}


def _extract_legal(text: str) -> dict:
    """Document-level legal metadata from first 3000 chars."""
    try:
        sample = text[:3000]
        meta: dict = {}
        m = _SECTION_RE.search(sample)
        if m:
            meta["section_num"] = int(m.group(1))
        m = _CHAPTER_RE.search(sample)
        if m:
            meta["chapter"] = m.group(1).upper()
        return meta
    except Exception as exc:
        log.warning("extractor.legal_failed", error=str(exc))
        return {}


def _extract_section_metadata(text: str, section_title: str) -> dict:
    """Extract section_num and clause from section title + chunk text — all domains."""
    try:
        meta: dict = {}
        # Prefer section title — it's authoritative; fall back to first 200 chars of text
        # Search title only first to avoid false positives from body text
        m = _SECTION_RE.search(section_title)
        if m:
            meta["section_num"] = int(m.group(1))
        else:
            m = _ARTICLE_RE.search(section_title)
            if m:
                meta["section_num"] = int(m.group(1))
        m = _CLAUSE_RE.search(section_title)
        if m:
            meta["clause"] = m.group(1)
        m = _CHAPTER_RE.search(section_title)
        if m:
            meta["chapter"] = m.group(1).upper()
        return meta
    except Exception as exc:
        log.warning("extractor.section_metadata_failed", error=str(exc))
        return {}


def _extract_financial(text: str) -> dict:
    try:
        meta: dict = {}

        year_match = re.search(r"\b(20\d{2})\b", text[:2000])
        if year_match:
            meta["fiscal_year"] = year_match.group(1)

        section_keywords = {
            "income_statement": ["revenue", "net income", "earnings per share", "ebitda"],
            "balance_sheet": ["total assets", "total liabilities", "shareholders equity"],
            "cash_flow": ["cash flow", "operating activities", "capital expenditure"],
            "risk_factors": ["risk factor", "material risk", "uncertainty"],
            "management_discussion": ["management", "discussion", "analysis", "md&a"],
        }
        sample = text[:3000].lower()
        for section, keywords in section_keywords.items():
            if any(kw in sample for kw in keywords):
                meta["section"] = section
                break

        meta["has_numbers"] = bool(re.search(r"\$[\d,]+|\d+%|\d+\.\d+", text[:2000]))
        return meta
    except Exception as exc:
        log.warning("extractor.financial_failed", error=str(exc))
        return {}
    finally:
        log.debug("extractor.financial_exit")
