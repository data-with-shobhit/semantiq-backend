"""LangGraph Agentic RAG — decompose → rewrite → retrieve → generate → grade → loop."""
from __future__ import annotations

import asyncio
import re
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from config.logging import get_logger
from config.settings import settings
from ingestion.embedder import Embedder
from llm.backend import LLMBackend
from retrieval.hybrid import retrieve
from retrieval.reranker import rerank
from retrieval.rewriter import hyde_rewrite

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def _count_tokens(text: str) -> int:
        return len(_enc.encode(text, disallowed_special=()))
except Exception:
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return len(text) // 4

log = get_logger()

_MAX_HISTORY = 4
_MAX_CHUNKS = settings.max_chunks
_MAX_ITERATIONS = 3

_SYSTEM_PROMPT = """You are a precise document assistant. Your answers must be strictly grounded in the provided sources.

STRICT GROUNDING RULES — follow every rule exactly:
1. Every claim, fact, number, date, and name in your answer MUST be explicitly stated in one of the provided sources. Do not infer, extrapolate, or add context from your training knowledge — even if you believe it to be true.
2. If sources partially cover the question, answer ONLY the covered part. Do not fill any gap with assumed or background knowledge.
3. Write answers in plain, natural language. Do NOT mention source labels, source numbers, or tags (e.g. do not write "DOC Source 1", "WEB Source 2", "According to Source 3", or any similar reference). The user does not see the sources — just give the answer directly.
4. Match answer depth to the question: simple factual questions get 1-2 sentences; complex or multi-part questions get a structured explanation with bullets or steps.
5. If the answer contains code, reproduce it EXACTLY in a fenced code block (```python ... ```). Never paraphrase or summarize code.
6. If the sources do not contain enough information to answer: say exactly "I don't have that information in your documents." Do not guess or speculate.
7. Prefer DOC sources over WEB sources. For WEB sources: verify the source discusses the exact entity being asked — if it is about a different company or topic, ignore it."""

_ABSTENTION_PHRASES = [
    "don't have that information",
    "don't have enough",
    "i don't have",
    "not in your documents",
    "no information",
    "cannot find",
    "not available",
]

# Cached compiled graphs per embedder domain — fix #4
_graph_cache: dict[str, Any] = {}


class AgentState(TypedDict):
    tenant_id: str
    domain: str
    workspace_id: int | None
    collection_name: str | None
    use_hyde: bool
    question: str
    history: list[dict]
    hyde_query: str
    expanded_query: str
    chunks: list[dict]
    all_chunks: list[dict]
    answer: str
    trace: list[str]
    iteration: int
    sub_questions: list[str]
    refined_query: str
    llm_calls: int
    tokens_used: int
    search_context: str
    web_search_enabled: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compress_chunk(text: str, query: str, max_chars: int = 500) -> str:
    query_terms = set(re.findall(r"\b[a-z0-9]+\b", query.lower()))
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"\s*\|\s*", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    cleaned_text = " ".join(lines)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
    scored = []
    for s in sentences:
        terms = set(re.findall(r"\b[a-z0-9]+\b", s.lower()))
        score = len(terms & query_terms)
        if re.search(r"\d", s):
            score += 2
        scored.append((score, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return " ".join(s for _, s in scored if s)[:max_chars]


def _format_chunks(chunks: list[dict], query: str = "") -> str:
    """Fix #2: clearly tag WEB vs DOC sources."""
    try:
        parts = []
        for i, c in enumerate(chunks[:_MAX_CHUNKS], 1):
            score = c.get("rerank_score", c.get("score", 0.0))
            is_web = c.get("web", False)
            is_neighbor = c.get("is_neighbor", False)
            source_tag = "WEB" if is_web else "DOC"
            url = f" | {c['source']}" if is_web and c.get("source") else ""
            section = c.get("payload", {}).get("section", "") or c.get("section", "")
            section_str = f" | §{section}" if section else ""
            neighbor_str = " | context" if is_neighbor else ""
            text = _compress_chunk(c["text"], query) if query else c["text"][:500]
            parts.append(f"[{source_tag} Source {i} | score: {score:.2f}{section_str}{neighbor_str}{url}]\n{text}")
        return "\n\n".join(parts)
    except Exception as exc:
        log.error("agent.format_chunks_failed", error=str(exc))
        return ""


def _trim_history(history: list[dict]) -> list[dict]:
    return history[-(_MAX_HISTORY * 2):]


def _is_complex_question(question: str) -> bool:
    patterns = [
        r"\band\b.{10,}\?",
        r"\bcompare\b",
        r"\bdifference between\b",
        r"\bvs\.?\b",
        r"\balso\b.{5,}\?",
        r"\bwhat about\b",
        r"\bboth\b",
    ]
    return any(re.search(p, question.lower()) for p in patterns)


def _heuristic_refine(question: str, answer: str, sub_questions: list[str], iteration: int) -> str:
    """Fix #1: free heuristic query refinement — no LLM call."""
    if iteration + 1 < len(sub_questions):
        return sub_questions[iteration + 1]
    stopwords = {"what", "is", "are", "the", "a", "an", "how", "does", "do", "in", "of", "for"}
    tokens = [w for w in re.findall(r"\b[a-z0-9]+\b", question.lower()) if w not in stopwords]
    return " ".join(tokens[:8])


def _chunk_key(c: dict) -> str:
    """Stable dedup key — prefer payload IDs over raw text comparison."""
    payload = c.get("payload", {})
    doc_id = payload.get("doc_id", "")
    idx = payload.get("chunk_index", "")
    if doc_id and idx != "":
        return f"{doc_id}:{idx}"
    return c.get("text", "")[:120]


async def _judge_hallucination(
    question: str, answer: str, chunks: list[dict], llm: LLMBackend
) -> bool:
    """LLM-as-judge: returns True when answer contains claims not found in sources."""
    sources = "\n\n".join(
        f"[Source {i + 1}]: {c['text'][:400]}" for i, c in enumerate(chunks[:3])
    )
    prompt = (
        f"SOURCES:\n{sources}\n\n"
        f"ANSWER: {answer[:400]}\n\n"
        "Does the answer contain specific facts, numbers, or claims NOT found in the sources above? "
        "Reply with ONLY 'yes' or 'no'."
    )
    try:
        verdict = await llm.generate(
            prompt, system="You are a factual accuracy checker. Reply only yes or no."
        )
        return verdict.strip().lower().startswith("y")
    except Exception:
        return False


# ---------------------------------------------------------------------------
# LangSmith token tracking
# ---------------------------------------------------------------------------

def post_langsmith_metrics(
    tenant_id: str,
    run_id: str | None,
    answer: str,
    chunks: list[dict],
    tokens: int,
    llm_calls: int,
    used_web: bool,
) -> None:
    """Post per-query metrics as LangSmith feedback on the traced run.

    Called from query.py after ainvoke completes, with the run_id that was
    pre-generated and injected via LangGraph config — no get_current_run_tree()
    guessing needed.
    """
    if not settings.langchain_api_key or not run_id:
        return
    try:
        from langsmith import Client
        client = Client(api_key=settings.langchain_api_key)
        top_score = max((c.get("rerank_score", c.get("score", 0.0)) for c in chunks), default=0.0)
        abstention = any(p in answer.lower() for p in _ABSTENTION_PHRASES)
        feedback = {
            "tokens_used": float(tokens),
            "llm_calls": float(llm_calls),
            "n_chunks": float(len(chunks)),
            "top_rerank_score": round(top_score, 4),
            "used_web_search": 1.0 if used_web else 0.0,
            "abstention": 1.0 if abstention else 0.0,
        }
        for key, value in feedback.items():
            client.create_feedback(run_id=run_id, key=key, score=value)
        log.info("agent.langsmith_metrics_posted", tenant_id=tenant_id, run_id=str(run_id), **feedback)
    except Exception as exc:
        log.warning("agent.langsmith_metrics_failed", tenant_id=tenant_id, run_id=str(run_id), error=str(exc))


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def _decompose_node(state: AgentState, llm: LLMBackend) -> dict:
    try:
        if not _is_complex_question(state["question"]):
            return {
                "sub_questions": [state["question"]],
                "refined_query": state["question"],
                "all_chunks": [],
                "iteration": 0,
                "llm_calls": state.get("llm_calls", 0) ,
                "trace": state.get("trace", []) + ["decompose:simple"],
            }
        prompt = (
            "Break this question into 2-3 specific sub-questions for document retrieval. "
            "Return ONLY a numbered list, one per line, nothing else.\n\n"
            f"Question: {state['question']}"
        )
        raw = await llm.generate(prompt, system="You are a query decomposition assistant.")
        sub_qs = [
            re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            for line in raw.strip().splitlines()
            if line.strip() and re.match(r"^\d", line.strip())
        ] or [state["question"]]

        log.info("agent.decompose", tenant_id=state["tenant_id"], n_sub=len(sub_qs))
        return {
            "sub_questions": sub_qs,
            "refined_query": sub_qs[0],
            "all_chunks": [],
            "iteration": 0,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "trace": state.get("trace", []) + [f"decompose:{len(sub_qs)}"],
        }
    except Exception as exc:
        log.error("agent.decompose_failed", tenant_id=state["tenant_id"], error=str(exc))
        return {
            "sub_questions": [state["question"]],
            "refined_query": state["question"],
            "all_chunks": [],
            "iteration": 0,
            "llm_calls": state.get("llm_calls", 0),
            "trace": state.get("trace", []) + ["decompose:error"],
        }


async def _rewrite_node(state: AgentState, llm: LLMBackend) -> dict:
    try:
        query = state.get("refined_query") or state["question"]
        # Skip HyDE for long queries — they already carry enough signal for dense retrieval
        use_hyde = state.get("use_hyde", settings.use_hyde) and len(query.split()) <= 10
        if use_hyde:
            hypothesis = await hyde_rewrite(query, llm)
            log.info("agent.rewrite_hyde", tenant_id=state["tenant_id"])
            return {
                "hyde_query": hypothesis,
                "expanded_query": hypothesis,
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": state.get("trace", []) + ["rewrite:hyde"],
            }
        return {"hyde_query": query, "expanded_query": query, "trace": state.get("trace", []) + ["rewrite:skipped"]}
    except Exception as exc:
        log.error("agent.rewrite_node_failed", tenant_id=state["tenant_id"], error=str(exc))
        return {"hyde_query": state["question"], "expanded_query": state["question"], "trace": state.get("trace", []) + ["rewrite:fallback"]}


async def _retrieve_node(state: AgentState, embedder: Embedder) -> dict:
    try:
        iteration = state.get("iteration", 0)
        sub_qs = state.get("sub_questions", [])

        # First iteration with multiple sub-questions: retrieve each separately and merge.
        # Uses HyDE query for sub_qs[0], raw text for the rest (no extra LLM calls).
        if iteration == 0 and len(sub_qs) > 1:
            candidates: list[dict] = []
            for i, q in enumerate(sub_qs):
                query = state["hyde_query"] if i == 0 else q
                candidates.extend(retrieve(
                    query=query,
                    tenant_id=state["tenant_id"],
                    embedder=embedder,
                    domain=state.get("domain", "general"),
                    collection_name=state.get("collection_name"),
                ))
        else:
            candidates = retrieve(
                query=state["hyde_query"],
                tenant_id=state["tenant_id"],
                embedder=embedder,
                domain=state.get("domain", "general"),
                collection_name=state.get("collection_name"),
            )

        reranked = rerank(state["question"], candidates)
        existing = state.get("all_chunks", [])
        seen = {_chunk_key(c) for c in existing}
        new_chunks = [c for c in reranked if _chunk_key(c) not in seen]
        all_chunks = sorted(existing + new_chunks, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
        log.info("agent.retrieve", tenant_id=state["tenant_id"], new=len(new_chunks),
                 total=len(all_chunks), iteration=iteration)
        return {
            "chunks": all_chunks[:_MAX_CHUNKS],
            "all_chunks": all_chunks,
            "trace": state.get("trace", []) + [f"retrieve:iter{iteration}"],
        }
    except Exception as exc:
        log.error("agent.retrieve_node_failed", tenant_id=state["tenant_id"], error=str(exc))
        return {"chunks": [], "all_chunks": state.get("all_chunks", []), "trace": state.get("trace", []) + ["retrieve:error"]}


async def _generate_node(state: AgentState, llm: LLMBackend) -> dict:
    try:
        if not state["chunks"]:
            return {"answer": "I don't have that information in your documents.", "trace": state.get("trace", []) + ["generate:no_chunks"]}

        sources = _format_chunks(state["chunks"], query=state["question"])
        history_text = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in _trim_history(state.get("history", []))
        )
        top_score = max((c.get("rerank_score", 0.0) for c in state["chunks"]), default=0.0)
        has_web = any(c.get("web") for c in state["chunks"])

        confidence_note = (
            "NOTE: Sources have LOW relevance scores. Be extra critical.\n\n"
            if top_score < 0.40 else ""
        )
        web_note = (
            "NOTE: Some sources are from the web [WEB]. Prefer document sources [DOC] where available.\n\n"
            if has_web else ""
        )
        sub_qs = state.get("sub_questions", [])
        sub_context = (
            "This question has multiple parts. Address each:\n" +
            "\n".join(f"- {q}" for q in sub_qs) + "\n\n"
            if len(sub_qs) > 1 else ""
        )

        prompt = (
            f"{confidence_note}{web_note}{sub_context}"
            f"SOURCES:\n{sources}\n\n"
            + (f"CONVERSATION HISTORY:\n{history_text}\n\n" if history_text else "")
            + f"QUESTION: {state['question']}\n\n"
            + "Answer using ONLY what is explicitly stated in the sources above. "
            + "Every fact and claim must come from the sources — do not add, infer, or extrapolate. "
            + "Do NOT mention source labels or numbers (no 'DOC Source 1', no 'According to Source 2'). Write the answer in plain natural language as if you already knew this. "
            + "If a piece of information is not in the sources, omit it entirely. "
            + "Match depth to complexity — brief for simple questions, thorough for complex ones. "
            + "If sources contain code, reproduce it EXACTLY in a fenced code block."
        )

        answer = await llm.generate(prompt, system=_SYSTEM_PROMPT)
        prompt_tokens = _count_tokens(prompt)
        answer_tokens = _count_tokens(answer)
        total_tokens = state.get("tokens_used", 0) + prompt_tokens + answer_tokens
        log.info("agent.generate", tenant_id=state["tenant_id"], chars=len(answer),
                 prompt_tokens=prompt_tokens, answer_tokens=answer_tokens,
                 llm_calls=state.get("llm_calls", 0) + 1)
        return {
            "answer": answer,
            "llm_calls": state.get("llm_calls", 0) + 1,
            "tokens_used": total_tokens,
            "trace": state.get("trace", []) + ["generate"],
        }
    except Exception as exc:
        log.error("agent.generate_node_failed", tenant_id=state["tenant_id"], error=str(exc))
        return {"answer": "An error occurred while generating the answer.", "trace": state.get("trace", []) + ["generate:error"]}


async def _grade_node(state: AgentState, llm: LLMBackend) -> dict:
    """Heuristic grading + LLM-as-judge hallucination check on confident-looking answers."""
    try:
        answer = state.get("answer", "")
        iteration = state.get("iteration", 0)
        llm_calls = state.get("llm_calls", 0)
        top_score = max((c.get("rerank_score", 0.0) for c in state.get("chunks", [])), default=0.0)

        is_abstention = any(p in answer.lower() for p in _ABSTENTION_PHRASES)
        is_error = "error occurred" in answer.lower()
        low_confidence = top_score < 0.40
        needs_retry = is_abstention or is_error or low_confidence

        # LLM-as-judge: catches hallucination when heuristics say the answer looks fine.
        # Only on iteration 0 — already retrying answers don't need another grading pass.
        if not needs_retry and iteration == 0 and llm_calls < 4:
            doc_chunks = [c for c in state.get("chunks", []) if not c.get("web")]
            if doc_chunks:
                hallucinating = await _judge_hallucination(state["question"], answer, doc_chunks, llm)
                llm_calls += 1
                if hallucinating:
                    log.info("agent.grade_hallucination_detected", tenant_id=state["tenant_id"])
                    needs_retry = True

        if not needs_retry:
            log.info("agent.grade_pass", tenant_id=state["tenant_id"], iteration=iteration, top_score=top_score)
            return {
                "llm_calls": llm_calls,
                "trace": state.get("trace", []) + [f"grade:pass:iter{iteration}"],
            }

        # Exhausted doc retries — signal web search
        if iteration >= _MAX_ITERATIONS - 1:
            log.info("agent.grade_web_fallback", tenant_id=state["tenant_id"], iteration=iteration)
            return {
                "iteration": iteration + 1,
                "llm_calls": llm_calls,
                "trace": state.get("trace", []) + [f"grade:retry:iter{iteration}"],
            }

        # Heuristic refine first (free), LLM refine only on iteration 0
        refined = _heuristic_refine(state["question"], answer, state.get("sub_questions", []), iteration)
        if iteration == 0 and llm_calls < 5:
            try:
                prompt = (
                    f"Original question: {state['question']}\n"
                    f"Insufficient answer: {answer[:150]}\n\n"
                    "Write ONE alternative search query. Return ONLY the query."
                )
                refined = (await llm.generate(prompt, system="You are a search query optimizer.")).strip().strip('"\'')
                llm_calls += 1
            except Exception:
                pass

        log.info("agent.grade_retry", tenant_id=state["tenant_id"], iteration=iteration, refined=refined[:60])
        return {
            "refined_query": refined,
            "iteration": iteration + 1,
            "llm_calls": llm_calls,
            "trace": state.get("trace", []) + [f"grade:retry:iter{iteration}"],
        }
    except Exception as exc:
        log.error("agent.grade_failed", tenant_id=state["tenant_id"], error=str(exc))
        return {"trace": state.get("trace", []) + ["grade:error"]}


async def _web_search_node(state: AgentState, llm: LLMBackend) -> dict:
    """Fix #3: circuit breaker with timeout; fix #2: web chunks tagged."""
    try:
        from tavily import TavilyClient

        def _search():
            client = TavilyClient(api_key=settings.tavily_api_key)
            base_query = state.get("refined_query") or state["question"]
            prefix = state.get("search_context", "").strip()
            query = f"{prefix} {base_query}".strip() if prefix else base_query
            log.info("agent.web_search_query", tenant_id=state["tenant_id"], query=query)
            return client.search(query=query, search_depth="basic", max_results=5)

        # Circuit breaker: 8s timeout
        loop = asyncio.get_event_loop()
        try:
            results = await asyncio.wait_for(loop.run_in_executor(None, _search), timeout=8.0)
        except asyncio.TimeoutError:
            log.warning("agent.web_search_timeout", tenant_id=state["tenant_id"])
            return {
                "answer": "I don't have that information in your documents, and web search timed out.",
                "trace": state.get("trace", []) + ["web_search:timeout"],
            }

        web_chunks = [
            {
                "text": r.get("content", "")[:800],
                "rerank_score": round(r.get("score", 0.5), 4),
                "source": r.get("url", ""),
                "title": r.get("title", ""),
                "web": True,  # fix #2: explicit tag
            }
            for r in results.get("results", [])
            if r.get("content")
        ]

        log.info("agent.web_search", tenant_id=state["tenant_id"], n=len(web_chunks),
                 llm_calls=state.get("llm_calls", 0))
        return {
            "chunks": web_chunks[:_MAX_CHUNKS],
            "all_chunks": state.get("all_chunks", []) + web_chunks,
            "trace": state.get("trace", []) + ["web_search"],
        }
    except Exception as exc:
        log.error("agent.web_search_failed", tenant_id=state["tenant_id"], error=str(exc))
        return {
            "answer": "I don't have that information in your documents.",
            "trace": state.get("trace", []) + ["web_search:error"],
        }


def _route_after_grade(state: AgentState) -> str:
    trace = state.get("trace", [])
    last = trace[-1] if trace else ""
    iteration = state.get("iteration", 0)

    if "retry" not in last:
        return END
    # Exhausted doc retries → web search only if tenant has opted in
    if iteration >= _MAX_ITERATIONS:
        if state.get("web_search_enabled") and settings.tavily_api_key:
            return "web_search"
        return END  # small/private companies: just say "don't have that info"
    return "rewrite"


# ---------------------------------------------------------------------------
# Graph builder — fix #4: cache per domain
# ---------------------------------------------------------------------------

def get_graph(llm: LLMBackend, embedder: Embedder):
    """Return cached compiled graph for this embedder domain."""
    key = embedder.domain if hasattr(embedder, "domain") else embedder.model_id
    if key not in _graph_cache:
        _graph_cache[key] = _build_graph(llm, embedder)
        log.info("agent.graph_cached", domain=key)
    return _graph_cache[key]


def _build_graph(llm: LLMBackend, embedder: Embedder):
    try:
        async def decompose(state: AgentState) -> dict:
            return await _decompose_node(state, llm)

        async def rewrite(state: AgentState) -> dict:
            return await _rewrite_node(state, llm)

        async def retrieve_node(state: AgentState) -> dict:
            return await _retrieve_node(state, embedder)

        async def generate(state: AgentState) -> dict:
            return await _generate_node(state, llm)

        async def grade(state: AgentState) -> dict:
            return await _grade_node(state, llm)

        async def web_search(state: AgentState) -> dict:
            return await _web_search_node(state, llm)

        graph = StateGraph(AgentState)
        graph.add_node("decompose", decompose)
        graph.add_node("rewrite", rewrite)
        graph.add_node("retrieve", retrieve_node)
        graph.add_node("generate", generate)
        graph.add_node("grade", grade)
        graph.add_node("web_search", web_search)

        graph.set_entry_point("decompose")
        graph.add_edge("decompose", "rewrite")
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "grade")
        graph.add_edge("web_search", "generate")
        graph.add_conditional_edges(
            "grade", _route_after_grade,
            {"rewrite": "rewrite", "web_search": "web_search", END: END},
        )

        compiled = graph.compile()
        log.info("agent.graph_built", mode="agentic")
        return compiled
    except Exception as exc:
        log.error("agent.build_graph_failed", error=str(exc))
        raise


# Keep backward compat
def build_graph(llm: LLMBackend, embedder: Embedder):
    return get_graph(llm, embedder)
