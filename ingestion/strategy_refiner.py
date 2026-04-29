"""RAGAS-driven strategy refinement — feeds eval scores back into Gemini to improve chunking."""
from __future__ import annotations

import json
import re

from config.logging import get_logger
from ingestion.analyzer import (_SAMPLE_CHARS, _STRATEGY_SCHEMA,
                                IngestionStrategy, _call_gemini)
from ingestion.loader import load as _load_local

log = get_logger()

_REFINE_PROMPT = """You are a document chunking expert improving a RAG system's ingestion strategy.

DOCUMENT METADATA:
- Filename: {filename}
- Domain: {domain}

CURRENT STRATEGY (version {version}):
{current_strategy}

RAGAS EVALUATION SCORES (higher is better, max 1.0):
- Faithfulness:        {faithfulness:.3f}  (are answers grounded in retrieved chunks?)
- Context Precision:   {context_precision:.3f}  (are retrieved chunks relevant?)
- Average:             {avg:.3f}

{feedback_section}

DOCUMENT SAMPLE:
---
{sample}
---

The current strategy scored {avg:.3f} average. Analyze why it may be underperforming and suggest improvements.

Low faithfulness → chunks may be too large (LLM hallucinates to fill gaps) or too small (missing context).
Low context precision → wrong chunker type for this document structure, or chunk boundaries split key info.

Return ONLY valid JSON:
{schema}
"""


async def refine_strategy(
    doc_id: int,
    tenant_id: str,
    file_path: str,
    filename: str,
    domain: str,
    user_feedback: str | None = None,
) -> IngestionStrategy | None:
    """Pull latest RAGAS eval + active strategy, ask Gemini for improvement, save new strategy."""
    from db.postgres import (get_active_strategy, get_latest_ragas_eval,
                             save_strategy)

    active = await get_active_strategy(doc_id, tenant_id)
    if not active:
        log.warning("refiner.no_active_strategy", doc_id=doc_id, tenant_id=tenant_id)
        return None

    latest_eval = await get_latest_ragas_eval(doc_id, tenant_id)
    if not latest_eval:
        log.warning("refiner.no_ragas_eval", doc_id=doc_id, tenant_id=tenant_id)
        return None

    faithfulness = float(latest_eval["faithfulness"] or 0)
    context_precision = float(latest_eval["context_precision"] or 0)
    avg = round((faithfulness + context_precision) / 2, 4)

    feedback_section = ""
    if user_feedback:
        feedback_section = f"USER FEEDBACK:\n{user_feedback}\n"

    try:
        text = _load_local(file_path)
        sample = text[:_SAMPLE_CHARS]
    except Exception as exc:
        log.warning("refiner.load_failed", doc_id=doc_id, error=str(exc))
        sample = "(file not accessible)"

    current = {
        "chunker": active["chunker"],
        "chunk_size": active["chunk_size"],
        "overlap": active["overlap"],
        "reasoning": active["reasoning"],
    }

    prompt = _REFINE_PROMPT.format(
        filename=filename,
        domain=domain,
        version=active["version"],
        current_strategy=json.dumps(current, indent=2),
        faithfulness=faithfulness,
        context_precision=context_precision,
        avg=avg,
        feedback_section=feedback_section,
        sample=sample,
        schema=json.dumps(_STRATEGY_SCHEMA, indent=2),
    )

    try:
        raw = _call_gemini(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)

        chunker = data.get("chunker", active["chunker"])
        if chunker not in _STRATEGY_SCHEMA["chunker"]:
            chunker = active["chunker"]
        chunk_size = max(128, min(1024, int(data.get("chunk_size", active["chunk_size"]))))
        overlap = max(0, min(chunk_size // 2 - 1, int(data.get("overlap", active["overlap"]))))
        reasoning = str(data.get("reasoning", "Refined from RAGAS feedback"))
        confidence = float(data.get("confidence", 0.8))

        strategy = IngestionStrategy(
            chunker=chunker,
            chunk_size=chunk_size,
            overlap=overlap,
            reasoning=reasoning,
            confidence=confidence,
            source="llm_ragas",
        )

        strategy_id = await save_strategy(
            doc_id, tenant_id,
            strategy.chunker, strategy.chunk_size, strategy.overlap,
            strategy.reasoning, strategy.source, strategy.extra_params,
        )
        if user_feedback:
            from db.postgres import execute
            await execute(
                "UPDATE ingestion_strategies SET user_feedback=$1 WHERE id=$2",
                user_feedback, strategy_id,
            )

        log.info(
            "refiner.new_strategy",
            doc_id=doc_id, tenant_id=tenant_id,
            chunker=strategy.chunker, chunk_size=strategy.chunk_size,
            prev_avg=avg, strategy_id=strategy_id,
        )
        return strategy

    except Exception as exc:
        log.error("refiner.gemini_failed", doc_id=doc_id, error=str(exc))
        return None
