"""Document structure analyzer — decides chunking strategy before ingestion.

Pass 1: heuristic scoring (free, instant).
Pass 2: Gemini 2.0 Flash LLM (only when heuristic confidence < 0.60 or re-ingesting from RAGAS).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from config.logging import get_logger
from config.settings import settings

log = get_logger()

_SAMPLE_CHARS = 6000  # first N chars sent to LLM


@dataclass
class IngestionStrategy:
    chunker: str                    # sentence_splitter | legal_sections | ast_code | header_split | fixed_window
    chunk_size: int                 # tokens
    overlap: int                    # tokens
    reasoning: str
    confidence: float               # 0-1, heuristic confidence
    source: str = "heuristic"       # heuristic | llm | llm_ragas
    extra_params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunker": self.chunker,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "source": self.source,
            "extra_params": self.extra_params,
        }


# ── Heuristic scorer ──────────────────────────────────────────────────────────

_LEGAL_SIGNALS = [
    re.compile(r"\*\*\d{1,4}\.\*\*"),           # **104.** BNS format
    re.compile(r"(?:^|\n)\d{1,4}\.\s+[A-Z]", re.MULTILINE),  # 104. Punishment
    re.compile(r"\b(?:section|article|clause|chapter)\s+\d+", re.IGNORECASE),
    re.compile(r"\bwhereas\b|\bhereinafter\b|\bnotwithstanding\b", re.IGNORECASE),
]
_CODE_SIGNALS = [
    re.compile(r"\bdef\s+\w+\s*\("),
    re.compile(r"\bclass\s+\w+[\s:(]"),
    re.compile(r"^\s*(import|from)\s+\w+", re.MULTILINE),
    re.compile(r"```[\w]*\n"),
    re.compile(r"\bfunction\s+\w+\s*\(|\bconst\s+\w+\s*=\s*(?:async\s+)?\("),
]
_ACADEMIC_SIGNALS = [
    re.compile(r"\bAbstract\b|\bIntroduction\b|\bMethodology\b|\bConclusion\b", re.IGNORECASE),
    re.compile(r"\bFig(?:ure)?\.?\s*\d+|\bTable\s+\d+", re.IGNORECASE),
    re.compile(r"\[\d+\]|\(\w+,\s*\d{4}\)"),   # citations
    re.compile(r"\bDOI:\s*10\.\d+", re.IGNORECASE),
]
_FINANCIAL_SIGNALS = [
    re.compile(r"\$[\d,]+|\b(?:USD|EUR|GBP)\b"),
    re.compile(r"\b(?:fiscal year|FY\d{2,4}|Q[1-4]\s+\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(?:revenue|EBITDA|net income|balance sheet|cash flow)\b", re.IGNORECASE),
    re.compile(r"(?:^\s*[\|\+][-\|\+]+)", re.MULTILINE),  # table rows
]


def _score_signals(text: str, patterns: list[re.Pattern]) -> float:
    hits = sum(1 for p in patterns if p.search(text))
    return hits / len(patterns)


def _heuristic_analyze(text: str, domain: str, filename: str) -> IngestionStrategy:
    sample = text[:_SAMPLE_CHARS]
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    scores = {
        "legal_sections":    _score_signals(sample, _LEGAL_SIGNALS),
        "ast_code":          _score_signals(sample, _CODE_SIGNALS),
        "academic_headers":  _score_signals(sample, _ACADEMIC_SIGNALS),
        "financial_tables":  _score_signals(sample, _FINANCIAL_SIGNALS),
    }

    # Domain boosts
    domain_boost = {
        "legal":      "legal_sections",
        "technical":  "ast_code",
        "scientific": "academic_headers",
        "medical":    "academic_headers",
        "clinical":   "academic_headers",
        "financial":  "financial_tables",
    }
    if domain in domain_boost:
        scores[domain_boost[domain]] = min(1.0, scores[domain_boost[domain]] + 0.25)

    # Extension boosts
    if ext in ("py", "js", "ts", "java", "go", "rs", "cpp", "c"):
        scores["ast_code"] = min(1.0, scores["ast_code"] + 0.4)
    elif ext == "md":
        scores["academic_headers"] = min(1.0, scores["academic_headers"] + 0.2)

    best = max(scores, key=lambda k: scores[k])
    confidence = scores[best]

    strategies = {
        "legal_sections":   IngestionStrategy("legal_sections",   512, 0,   f"Legal structure detected (score={confidence:.2f})", confidence),
        "ast_code":         IngestionStrategy("ast_code",          256, 32,  f"Code structure detected (score={confidence:.2f})", confidence),
        "academic_headers": IngestionStrategy("header_split",      384, 64,  f"Academic/structured doc detected (score={confidence:.2f})", confidence),
        "financial_tables": IngestionStrategy("sentence_splitter", 256, 32,  f"Financial doc with tables detected (score={confidence:.2f})", confidence),
    }

    if confidence < 0.25:
        return IngestionStrategy("sentence_splitter", 384, 64, "No strong structure detected — using sentence splitter", confidence)

    return strategies[best]


# ── LLM analyzer (Gemini 2.0 Flash) ──────────────────────────────────────────

_STRATEGY_SCHEMA = {
    "chunker": ["sentence_splitter", "legal_sections", "ast_code", "header_split", "fixed_window"],
    "chunk_size": "integer 128-1024",
    "overlap": "integer 0-256, must be < chunk_size/2",
    "reasoning": "string — why this strategy fits this document",
    "confidence": "float 0.0-1.0",
}

_ANALYZER_PROMPT = """You are a document ingestion expert. Analyze the document sample below and return a JSON chunking strategy.

DOCUMENT METADATA:
- Filename: {filename}
- Domain: {domain}
- File type: {ext}

DOCUMENT SAMPLE (first {n} chars):
---
{sample}
---

{feedback_section}

Return ONLY valid JSON matching this schema exactly:
{schema}

Rules:
- legal_sections: numbered sections (Section N, **N.**, Article N) — set overlap=0
- ast_code: source code files — chunk at function/class boundaries, chunk_size=256
- header_split: markdown/academic with ## headers — chunk_size=384, overlap=64
- sentence_splitter: default for prose — chunk_size=384, overlap=64
- fixed_window: last resort when no structure — chunk_size=512, overlap=128
- Never set overlap >= chunk_size/2
"""


def _call_gemini(prompt: str) -> str:
    from google import genai
    client = genai.Client(api_key=settings.gemini_api_key)
    resp = client.models.generate_content(model=settings.gemini_model, contents=prompt)
    return resp.text


def _call_groq(prompt: str) -> str:
    from groq import Groq
    client = Groq(api_key=settings.groq_api_key)
    resp = client.chat.completions.create(
        model=settings.groq_generation_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def _llm_analyze(
    text: str,
    domain: str,
    filename: str,
    ragas_feedback: str | None = None,
) -> IngestionStrategy:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "unknown"
    sample = text[:_SAMPLE_CHARS]
    feedback_section = ""
    if ragas_feedback:
        feedback_section = f"PREVIOUS RAGAS EVALUATION FEEDBACK:\n{ragas_feedback}\nUse this to improve the strategy.\n"

    prompt = _ANALYZER_PROMPT.format(
        filename=filename,
        domain=domain,
        ext=ext,
        n=len(sample),
        sample=sample,
        feedback_section=feedback_section,
        schema=json.dumps(_STRATEGY_SCHEMA, indent=2),
    )

    raw: str | None = None
    source_llm = "llm"

    # Gemini primary → Groq fallback on any error
    try:
        raw = _call_gemini(prompt)
    except Exception as gemini_exc:
        log.warning("analyzer.gemini_failed", filename=filename, error=str(gemini_exc))
        try:
            raw = _call_groq(prompt)
            source_llm = "llm_groq"
        except Exception as groq_exc:
            log.error("analyzer.groq_failed", filename=filename, error=str(groq_exc))

    if raw is None:
        return IngestionStrategy(
            "sentence_splitter", 384, 64,
            "Strategy auto-selected (LLM unavailable — using defaults)",
            0.5, "heuristic",
        )

    try:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)

        chunker = data.get("chunker", "sentence_splitter")
        if chunker not in _STRATEGY_SCHEMA["chunker"]:
            chunker = "sentence_splitter"
        chunk_size = max(128, min(1024, int(data.get("chunk_size", 384))))
        overlap = max(0, min(chunk_size // 2 - 1, int(data.get("overlap", 64))))

        if ragas_feedback:
            source_llm = "llm_ragas"
        return IngestionStrategy(
            chunker=chunker,
            chunk_size=chunk_size,
            overlap=overlap,
            reasoning=str(data.get("reasoning", "LLM-determined strategy")),
            confidence=float(data.get("confidence", 0.8)),
            source=source_llm,
        )
    except Exception as exc:
        log.error("analyzer.parse_failed", filename=filename, error=str(exc))
        return IngestionStrategy(
            "sentence_splitter", 384, 64,
            "Strategy auto-selected (could not parse LLM response — using defaults)",
            0.5, "heuristic",
        )


# ── Public entry point ────────────────────────────────────────────────────────

def analyze(
    text: str,
    domain: str,
    filename: str,
    ragas_feedback: str | None = None,
    force_llm: bool = False,
) -> IngestionStrategy:
    """Decide chunking strategy for a document.

    Uses heuristics first; escalates to Gemini when confidence is low
    or when RAGAS feedback is provided for re-ingestion.
    """
    if ragas_feedback or force_llm:
        log.info("analyzer.llm_path", filename=filename, ragas_feedback=bool(ragas_feedback))
        return _llm_analyze(text, domain, filename, ragas_feedback)

    strategy = _heuristic_analyze(text, domain, filename)
    log.info(
        "analyzer.heuristic",
        filename=filename,
        chunker=strategy.chunker,
        confidence=strategy.confidence,
    )

    if strategy.confidence < 0.60:
        log.info("analyzer.escalating_to_llm", filename=filename, confidence=strategy.confidence)
        return _llm_analyze(text, domain, filename)

    return strategy
