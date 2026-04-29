"""Document-aware chunker: header-based for all domains + AST for technical code blocks."""
from __future__ import annotations

import re
import tree_sitter_python as tspython
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from tree_sitter import Language, Parser

from config.logging import get_logger
from ingestion.extractor import extract_chunk_metadata

log = get_logger()

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64
_FINANCIAL_CHUNK_SIZE = 1024
_FINANCIAL_CHUNK_OVERLAP = 128

_PY_LANG = Language(tspython.language())


def chunk_text(text: str, domain: str = "general", strategy=None) -> list[dict]:
    """Split text into chunks with metadata. Returns list of {text, metadata}.

    If strategy (IngestionStrategy) is provided, it overrides domain defaults.
    """
    try:
        if strategy is not None:
            return _chunk_with_strategy(text, domain, strategy)
        return _chunk_by_headers(text, domain)
    except Exception as exc:
        log.error("chunker.chunk_text_failed", domain=domain, error=str(exc))
        return [{"text": text, "metadata": {"chunk_index": 0, "domain": domain}}]
    finally:
        log.debug("chunker.chunk_text_exit", domain=domain)


def _chunk_with_strategy(text: str, domain: str, strategy) -> list[dict]:
    """Dispatch to the right chunker based on IngestionStrategy."""
    chunker = strategy.chunker
    chunk_size = strategy.chunk_size
    overlap = strategy.overlap

    if chunker == "legal_sections":
        chunks = _chunk_legal_sections(text)
        if chunks:
            return [{"text": c["text"], "metadata": {**c, "domain": domain}} for c in chunks]
        # fallback to sentence splitter if legal sectioning failed
        return _sentence_split(text, domain, chunk_size, overlap)

    if chunker == "ast_code":
        return _chunk_by_headers(text, "technical")  # reuses existing AST path

    if chunker == "header_split":
        return _chunk_by_headers(text, domain)

    # sentence_splitter / fixed_window / default
    return _sentence_split(text, domain, chunk_size, overlap)


def _sentence_split(text: str, domain: str, chunk_size: int, overlap: int) -> list[dict]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    nodes = splitter.get_nodes_from_documents([Document(text=text)])
    chunks = []
    for i, node in enumerate(nodes):
        chunk_text_val = node.get_content()
        meta = extract_chunk_metadata(chunk_text_val, "", domain)
        meta.update({"chunk_index": i, "section": "", "section_level": 1, "domain": domain})
        chunks.append({"text": chunk_text_val, "metadata": meta})
    return chunks


def _parse_header(header: str) -> tuple[str, int]:
    """Return (section_title, section_level) from a markdown header string."""

    m = re.match(r"^(#{1,3})\s+(.+)$", header.strip())
    if m:
        return m.group(2).strip(), len(m.group(1))
    return header.strip(), 1


def _split_section_body(header: str, body: str, domain: str) -> list[dict]:
    """Split one section body. Returns list of {text, section, section_level}."""

    section_title, section_level = _parse_header(header)
    results: list[dict] = []

    if domain == "technical":
        code_fence_re = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)
        parts = code_fence_re.split(body)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("```") and part.endswith("```"):
                code = re.sub(r"^```\w*\n?", "", part).rstrip("`").strip()
                try:
                    ast_chunks = _ast_chunk(code)
                    for c in ast_chunks:
                        results.append({
                            "text": c["text"],
                            "section": section_title,
                            "section_level": section_level,
                            **{k: v for k, v in c.get("metadata", {}).items() if k not in ("chunk_index",)},
                        })
                except Exception:
                    results.append({"text": part, "section": section_title, "section_level": section_level})
            else:
                splitter = SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
                nodes = splitter.get_nodes_from_documents([Document(text=part)])
                for node in nodes:
                    results.append({"text": node.text, "section": section_title, "section_level": section_level})
    else:
        chunk_size = _FINANCIAL_CHUNK_SIZE if domain == "financial" else _CHUNK_SIZE
        chunk_overlap = _FINANCIAL_CHUNK_OVERLAP if domain == "financial" else _CHUNK_OVERLAP
        full = body
        if len(full) <= chunk_size * 4:
            results.append({"text": full, "section": section_title, "section_level": section_level})
        else:
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = splitter.get_nodes_from_documents([Document(text=body)])
            for node in nodes:
                results.append({"text": node.text, "section": section_title, "section_level": section_level})

    return results or [{"text": body, "section": section_title, "section_level": section_level}]


def _chunk_legal_sections(text: str) -> list[dict]:
    """Split Indian legal docs on bold section numbers: '**104.**' anywhere in line."""


    # BNS format: marginal note then **N.** body — match **N.** anywhere
    sec_re = re.compile(r"\*\*(\d{1,4})\.\*\*\s+")
    matches = list(sec_re.finditer(text))
    if len(matches) < 3:
        # Fallback: plain numbered lines "104. Body..."
        sec_re = re.compile(r"(?:^|\n)(\d{1,4})\.\s+[A-Z]", re.MULTILINE)
        matches = list(sec_re.finditer(text))
    if len(matches) < 3:
        return []
    sections: list[dict] = []
    for i, m in enumerate(matches):
        sec_num = int(m.group(1))
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < 20:
            continue
        sections.append({"text": body, "section": f"Section {sec_num}", "section_level": 1, "section_num": sec_num})
    return sections


def _chunk_by_headers(text: str, domain: str) -> list[dict]:
    """Split markdown on headers for all domains. Returns chunks with section metadata."""

    try:
        # Legal domain: try numbered section pattern first (Indian legal doc format)
        if domain == "legal":
            legal_sections = _chunk_legal_sections(text)
            if legal_sections:
                chunks = []
                for i, s in enumerate(legal_sections):
                    chunks.append({
                        "text": s["text"],
                        "metadata": {
                            "chunk_index": i,
                            "domain": domain,
                            "section": s["section"],
                            "section_level": 1,
                            "section_num": s["section_num"],
                            "prev_chunk_id": None,
                            "next_chunk_id": None,
                        },
                    })
                log.debug("chunker.legal_sections", n=len(chunks))
                return chunks

        header_re = re.compile(r"^(#{1,3} .+)$", re.MULTILINE)
        positions = [(m.start(), m.group(0)) for m in header_re.finditer(text)]
        raw_sections: list[dict] = []

        for i, (pos, header) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
            body = text[pos + len(header):end].strip()
            if not body:
                continue
            raw_sections.extend(_split_section_body(header, body, domain))

        # Fallback: no headers found → sentence split with no section
        if not raw_sections:
            chunk_size = _FINANCIAL_CHUNK_SIZE if domain == "financial" else _CHUNK_SIZE
            chunk_overlap = _FINANCIAL_CHUNK_OVERLAP if domain == "financial" else _CHUNK_OVERLAP
            splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            nodes = splitter.get_nodes_from_documents([Document(text=text)])
            raw_sections = [{"text": n.text, "section": "", "section_level": 0} for n in nodes]

        chunks = []
        for i, s in enumerate(raw_sections):
            section = s.get("section", "")
            chunk_meta = extract_chunk_metadata(s["text"], section, domain)
            chunks.append({
                "text": s["text"],
                "metadata": {
                    "chunk_index": i,
                    "domain": domain,
                    "section": section,
                    "section_level": s.get("section_level", 0),
                    "prev_chunk_id": None,
                    "next_chunk_id": None,
                    **chunk_meta,
                    **{k: v for k, v in s.items() if k not in ("text", "section", "section_level")},
                },
            })
        log.debug("chunker.header_split", domain=domain, n=len(chunks))
        return chunks
    except Exception as exc:
        log.error("chunker.header_split_failed", domain=domain, error=str(exc))
        return [{"text": text, "metadata": {"chunk_index": 0, "domain": domain, "section": "", "section_level": 0, "prev_chunk_id": None, "next_chunk_id": None}}]


def _chunk_sentences(text: str, domain: str) -> list[dict]:
    try:
        if domain == "financial":
            return _chunk_by_headers(text, domain)
        splitter = SentenceSplitter(chunk_size=_CHUNK_SIZE, chunk_overlap=_CHUNK_OVERLAP)
        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        chunks = [
            {"text": node.text, "metadata": {"chunk_index": i, "domain": domain}}
            for i, node in enumerate(nodes)
        ]
        log.debug("chunker.sentences", domain=domain, n=len(chunks))
        return chunks
    except Exception as exc:
        log.error("chunker.sentences_failed", domain=domain, error=str(exc))
        return [{"text": text, "metadata": {"chunk_index": 0, "domain": domain}}]


def _chunk_code(source: str) -> list[dict]:
    """Split technical docs into code blocks (AST) + prose blocks (sentence splitter)."""
    try:
    

        # Split on code fence markers or indented blocks (4+ spaces / tabs)
        code_pattern = re.compile(r'((?:(?:^|\n)(?:    |\t)[^\n]+)+)', re.MULTILINE)
        chunks: list[dict] = []
        last_end = 0

        for match in code_pattern.finditer(source):
            # Prose before this code block
            prose = source[last_end:match.start()].strip()
            if prose:
                chunks.extend(_chunk_sentences(prose, "technical"))

            # Code block — try AST, fallback to sentence split
            code_block = match.group(0).strip()
            if code_block and any(kw in code_block for kw in ("def ", "class ", "import ", "async def ")):
                try:
                    ast_chunks = _ast_chunk(code_block)
                    chunks.extend(ast_chunks)
                except Exception:
                    chunks.extend(_chunk_sentences(code_block, "technical"))
            elif code_block:
                chunks.extend(_chunk_sentences(code_block, "technical"))

            last_end = match.end()

        # Remaining prose after last code block
        remaining = source[last_end:].strip()
        if remaining:
            chunks.extend(_chunk_sentences(remaining, "technical"))

        if not chunks:
            return _chunk_sentences(source, "technical")

        # Re-index chunk_index across all chunks
        for i, c in enumerate(chunks):
            c["metadata"]["chunk_index"] = i

        log.debug("chunker.code_hybrid", total=len(chunks))
        return chunks

    except Exception as exc:
        log.warning("chunker.code_hybrid_failed", error=str(exc))
        return _chunk_sentences(source, "technical")


def _ast_chunk(source: str) -> list[dict]:
    try:
        parser = Parser(_PY_LANG)
        tree = parser.parse(source.encode())
        root = tree.root_node

        top_level = [
            n for n in root.children
            if n.type in ("function_definition", "class_definition", "decorated_definition")
        ]

        if not top_level:
            return _chunk_sentences(source, "technical")

        chunks = []
        for node in top_level:
            chunk_text_val = source[node.start_byte:node.end_byte]
            name = _extract_name(node)
            chunks.append({
                "text": chunk_text_val,
                "metadata": {
                    "node_type": node.type.replace("_definition", ""),
                    "function_name": name,
                    "language": "python",
                    "has_docstring": '"""' in chunk_text_val or "'''" in chunk_text_val,
                    "domain": "technical",
                },
            })

        log.debug("chunker.ast", nodes=len(chunks))
        return chunks
    except Exception as exc:
        log.error("chunker.ast_chunk_failed", error=str(exc))
        raise
    finally:
        log.debug("chunker.ast_chunk_exit")


def _extract_name(node) -> str:
    try:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode()
            if child.type == "decorated_definition":
                return _extract_name(child)
        return "unknown"
    except Exception as exc:
        log.warning("chunker.extract_name_failed", error=str(exc))
        return "unknown"
    finally:
        log.debug("chunker.extract_name_exit")
