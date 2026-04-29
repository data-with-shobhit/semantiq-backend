"""Streamlit test UI for the RAG platform."""
from __future__ import annotations

import time

import requests
import streamlit as st

API = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Platform",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state defaults ────────────────────────────────────────────────────
for k, v in {
    "token": "",
    "workspace_id": None,
    "workspace_name": "",
    "messages": [],
    "web_search": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Auto-capture token from Google OAuth redirect (?token=...)
_params = st.query_params
if "token" in _params and not st.session_state.token:
    st.session_state.token = _params["token"]
    st.query_params.clear()
    st.rerun()
if "error" in _params:
    st.error(f"Login failed: {_params['error']}")
    st.query_params.clear()


def _headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def _api(method: str, path: str, **kwargs):
    try:
        r = getattr(requests, method)(f"{API}{path}", headers=_headers(), timeout=60, **kwargs)
        return r
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach API — is `uvicorn api.main:app --reload` running?")
        return None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 RAG Platform")

    # ── Auth ─────────────────────────────────────────────────────────────────
    st.subheader("Authentication")
    token_input = st.text_input(
        "JWT Token",
        value=st.session_state.token,
        type="password",
        placeholder="Paste your token or login with Google",
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save Token", use_container_width=True):
            st.session_state.token = token_input.strip()
            st.session_state.messages = []
            st.session_state.workspace_id = None
            st.rerun()
    with col2:
        st.link_button("Google Login", f"{API}/auth/google", use_container_width=True)

    if st.session_state.token:
        r = _api("get", "/auth/me")
        if r and r.status_code == 200:
            me = r.json()
            st.success(f"✓ {me['email']}")
            st.caption(f"Plan: {me['plan']} · Domain: {me['domain']}")
        else:
            st.warning("Token invalid or expired")

    st.divider()

    # ── Workspace selector ────────────────────────────────────────────────────
    st.subheader("Workspace")
    if st.session_state.token:
        r = _api("get", "/workspaces")
        if r and r.status_code == 200:
            workspaces = r.json().get("workspaces", [])
            if workspaces:
                ws_options = {f"{w['name']} ({w['domain']})": w for w in workspaces}
                ids = [w["id"] for w in workspaces]
                default_idx = 0
                if st.session_state.workspace_id in ids:
                    default_idx = ids.index(st.session_state.workspace_id)
                selected = st.selectbox("Select workspace", list(ws_options.keys()), index=default_idx)
                ws = ws_options[selected]
                if st.session_state.workspace_id != ws["id"]:
                    st.session_state.workspace_id = ws["id"]
                    st.session_state.workspace_name = ws["name"]
                    st.session_state.messages = []
                used_mb = ws.get("storage_mb", 0)
                st.caption(f"Storage: {used_mb} MB / 200 MB")
                st.progress(min(used_mb / 200, 1.0))
            else:
                st.info("No workspaces yet")

        with st.expander("+ New Workspace"):
            ws_name = st.text_input("Name", key="new_ws_name")
            ws_domain = st.selectbox(
                "Domain",
                ["general", "financial", "legal", "medical", "clinical", "scientific", "technical"],
                key="new_ws_domain",
            )
            if st.button("Create", key="create_ws"):
                r = _api("post", "/workspaces", json={"name": ws_name, "domain": ws_domain})
                if r and r.status_code == 201:
                    st.success(f"Workspace '{ws_name}' created!")
                    st.rerun()
                elif r:
                    st.error(r.json().get("detail", "Failed"))

    st.divider()

    # ── Document upload ───────────────────────────────────────────────────────
    st.subheader("Upload Document")
    if st.session_state.workspace_id:
        uploaded = st.file_uploader(
            "PDF / TXT / MD / DOCX",
            type=["pdf", "txt", "md", "docx", "py", "js", "ts"],
        )
        if uploaded and st.button("Upload & Ingest"):
            with st.spinner("Uploading…"):
                r = requests.post(
                    f"{API}/ingest",
                    headers=_headers(),
                    params={"workspace_id": st.session_state.workspace_id},
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=60,
                )
            if r.status_code == 202:
                j = r.json()
                st.success(f"Ingesting doc #{j['doc_id']} — job `{j['job_id'][:8]}…`")
            elif r.status_code == 413:
                detail = r.json().get("detail", {})
                if isinstance(detail, dict):
                    st.error(f"Storage full! {detail.get('message', '')}")
                else:
                    st.error(str(detail))
            else:
                st.error(r.text)
    else:
        st.caption("Select a workspace first")

    st.divider()

    # ── Documents list ────────────────────────────────────────────────────────
    st.subheader("Documents")
    if st.session_state.workspace_id and st.session_state.token:
        if st.button("Refresh", key="refresh_docs"):
            st.rerun()
        r = _api("get", f"/ingest/list?workspace_id={st.session_state.workspace_id}") if st.session_state.workspace_id else None
        if r and r.status_code == 200:
            docs = r.json().get("documents", [])
            if docs:
                for doc in docs[:10]:
                    status_icon = {"ready": "✅", "processing": "⏳", "failed": "❌"}.get(doc["status"], "❓")
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.caption(f"{status_icon} {doc['filename'][:25]}")
                        if doc.get("chunk_count"):
                            st.caption(f"   {doc['chunk_count']} chunks")
                    with col2:
                        if st.button("🗑", key=f"del_{doc['id']}"):
                            _api("delete", f"/ingest/{doc['id']}")
                            st.rerun()
            else:
                st.caption("No documents yet")

    st.divider()

    # ── Query settings ────────────────────────────────────────────────────────
    st.subheader("Query Settings")
    st.session_state.web_search = st.toggle(
        "Web Search",
        value=st.session_state.web_search,
        help="Supplement document retrieval with live Tavily web search.",
    )
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
if not st.session_state.token:
    st.markdown("## Welcome to RAG Platform")
    st.info("Paste your JWT token in the sidebar to get started, or click **Google Login**.")
    st.stop()

if not st.session_state.workspace_id:
    st.markdown("## Select a workspace from the sidebar to start chatting.")
    st.stop()

tab_chat, tab_docs, tab_ragas, tab_library = st.tabs(["💬 Chat", "📂 Documents & Reingest", "📊 RAGAS Eval", "📚 Strategy Library"])

# ── Tab 2: Documents & Reingest ───────────────────────────────────────────────
with tab_docs:
    st.markdown(f"### Documents in **{st.session_state.workspace_name}**")
    if st.button("Refresh list", key="refresh_docs_main"):
        st.rerun()

    r = _api("get", f"/ingest/list?workspace_id={st.session_state.workspace_id}")
    docs = r.json().get("documents", []) if r and r.status_code == 200 else []

    if not docs:
        st.info("No documents ingested yet. Upload one from the sidebar.")
    else:
        for doc in docs:
            status_icon = {"ready": "✅", "processing": "⏳", "failed": "❌"}.get(doc["status"], "❓")
            with st.expander(f"{status_icon} {doc['filename']}  —  {doc.get('chunk_count') or 0} chunks"):
                col1, col2, col3 = st.columns(3)
                col1.metric("Chunks", doc.get("chunk_count") or 0)
                col2.metric("Status", doc["status"].capitalize())
                col3.metric("Doc ID", doc["id"])

                st.divider()

                # ── Strategy history ──────────────────────────────────────────
                st.markdown("**Strategy History**")
                sr = _api("get", f"/eval/strategy/{doc['id']}")
                strategies = sr.json().get("strategies", []) if sr and sr.status_code == 200 else []

                if strategies:
                    for s in strategies:
                        active_badge = "🟢 active" if s.get("is_active") else "⚪"
                        st.markdown(
                            f"{active_badge} **v{s['version']}** · `{s['chunker']}` · "
                            f"size={s['chunk_size']} overlap={s['overlap']} · "
                            f"source=`{s['source']}`"
                        )
                        if s.get("reasoning"):
                            st.caption(f"Reasoning: {s['reasoning'][:120]}")
                        if s.get("ragas_scores"):
                            scores = s["ragas_scores"]
                            sc1, sc2, sc3, sc4 = st.columns(4)
                            sc1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
                            sc2.metric("Context Precision", f"{scores.get('context_precision', 0):.2f}")
                            sc3.metric("Answer Relevance", f"{scores.get('answer_relevance', 0):.2f}")
                            sc4.metric("Avg Score", f"{scores.get('avg', 0):.2f}")
                        if not s.get("is_active") and doc["status"] == "ready":
                            if st.button(f"↩ Revert to v{s['version']}", key=f"revert_{doc['id']}_{s['version']}"):
                                rv = _api("post", "/eval/revert", json={
                                    "doc_id": doc["id"],
                                    "version": s["version"],
                                    "workspace_id": st.session_state.workspace_id,
                                })
                                if rv and rv.status_code == 200:
                                    st.success(f"Reverted to v{s['version']} and re-ingesting…")
                                    st.rerun()
                                elif rv:
                                    st.error(rv.json().get("detail", "Revert failed"))
                        st.divider()
                else:
                    st.caption("No strategy history yet.")

                # ── Reingest with feedback ────────────────────────────────────
                if doc["status"] == "ready":
                    st.markdown("**Reingest with Refined Strategy**")
                    st.caption("Requires at least one RAGAS eval to have been run first.")
                    feedback = st.text_area(
                        "User feedback (optional)",
                        placeholder="e.g. 'Chunks are too large, queries miss specific clauses'",
                        key=f"feedback_{doc['id']}",
                    )
                    if st.button("🔄 Reingest", key=f"reingest_{doc['id']}"):
                        with st.spinner("Refining strategy and re-ingesting…"):
                            payload: dict = {}
                            if feedback.strip():
                                payload["user_feedback"] = feedback.strip()
                            rr = _api("post", f"/eval/reingest/{doc['id']}", json=payload)
                        if rr and rr.status_code == 202:
                            j = rr.json()
                            st.success(f"Re-ingesting · job `{str(j.get('job_id', ''))[:8]}…`")
                            st.rerun()
                        elif rr:
                            detail = rr.json().get("detail", "Reingest failed")
                            if "No RAGAS eval" in str(detail):
                                st.warning("Run a RAGAS eval first, then reingest.")
                            else:
                                st.error(detail)

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    st.markdown(f"### 💬 {st.session_state.workspace_name} — Chat")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("chunks"):
                with st.expander(f"📄 {len(msg['chunks'])} source chunks"):
                    for i, c in enumerate(msg["chunks"]):
                        score = c.get("score", 0)
                        source = "🌐 Web" if c.get("web") else f"📄 Doc #{c.get('payload', {}).get('doc_id', '?')}"
                        st.markdown(f"**[Source {i+1} | {source} | score: {score:.2f}]**")
                        st.markdown(c.get("text", "")[:500])
                        st.divider()
            if msg.get("meta"):
                m = msg["meta"]
                cols = st.columns(3)
                cols[0].caption(f"⏱ {m.get('latency_ms', '?')} ms")
                cols[1].caption(f"🔁 {m.get('llm_calls', '?')} LLM calls")
                cols[2].caption(f"{'💾 cached' if m.get('cached') else '🔄 fresh'}")

    question = st.chat_input("Ask a question about your documents…")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking…")

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ]

            t0 = time.perf_counter()
            r = _api(
                "post",
                "/query",
                json={
                    "question": question,
                    "workspace_id": st.session_state.workspace_id,
                    "history": history[-8:],
                    "web_search": st.session_state.web_search,
                },
            )
            latency_ms = round((time.perf_counter() - t0) * 1000)

            if r is None:
                placeholder.error("API unreachable")
                st.stop()

            if r.status_code != 200:
                placeholder.error(f"Error {r.status_code}: {r.text}")
                st.stop()

            data = r.json()
            answer = data.get("answer", "No answer returned")
            chunks = data.get("chunks", [])
            cached = data.get("cached", False)
            llm_calls = data.get("llm_calls", 0)

            placeholder.markdown(answer)

            if chunks:
                with st.expander(f"📄 {len(chunks)} source chunks"):
                    for i, c in enumerate(chunks):
                        score = c.get("score", 0)
                        source = "🌐 Web" if c.get("web") else f"📄 Doc #{c.get('payload', {}).get('doc_id', '?')}"
                        st.markdown(f"**[Source {i+1} | {source} | score: {score:.2f}]**")
                        st.markdown(c.get("text", "")[:500])
                        st.divider()

            meta = {"latency_ms": latency_ms, "llm_calls": llm_calls, "cached": cached}
            cols = st.columns(3)
            cols[0].caption(f"⏱ {latency_ms} ms")
            cols[1].caption(f"🔁 {llm_calls} LLM calls")
            cols[2].caption(f"{'💾 cached' if cached else '🔄 fresh'}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "chunks": chunks,
            "meta": meta,
        })

# ── Tab 3: RAGAS Eval ────────────────────────────────────────────────────────
with tab_ragas:
    st.markdown(f"### 📊 RAGAS Evaluation — {st.session_state.workspace_name}")
    st.caption("Select a document, enter up to 10 question + ground truth pairs, then run the eval.")

    # Load docs for this workspace
    _rdocs_r = _api("get", f"/ingest/list?workspace_id={st.session_state.workspace_id}")
    _rdocs = _rdocs_r.json().get("documents", []) if _rdocs_r and _rdocs_r.status_code == 200 else []
    _ready_docs = [d for d in _rdocs if d.get("chunk_count") or d["status"] == "ready"]

    if not _ready_docs:
        st.info("No documents with chunks found. Upload and ingest a document first.")
    else:
        doc_options = {f"{d['filename']} (id={d['id']}, {d.get('chunk_count') or 0} chunks)": d for d in _ready_docs}
        selected_doc_label = st.selectbox("Document to evaluate", list(doc_options.keys()), key="ragas_doc_select")
        eval_doc = doc_options[selected_doc_label]
        eval_doc_id = eval_doc["id"]

        st.divider()

        # ── Current eval result ───────────────────────────────────────────────
        _hr = _api("get", f"/eval/history/{eval_doc_id}")
        _current_eval = _hr.json().get("eval") if _hr and _hr.status_code == 200 else None

        if _current_eval:
            st.markdown("**Current eval result** *(running a new eval will replace this)*")
            st.caption(f"Run: {_current_eval.get('created_at', '')[:19]}  ·  {_current_eval.get('n_questions', 0)} questions")

            st.markdown("*Generation*")
            ec1, ec2, ec3, ec4 = st.columns(4)
            ec1.metric("Faithfulness", f"{_current_eval.get('faithfulness') or 0:.2f}")
            ec2.metric("Context Prec.", f"{_current_eval.get('context_precision') or 0:.2f}")
            ec3.metric("Answer Rel.", f"{_current_eval.get('answer_relevance') or 0:.2f}")
            ec4.metric("Avg", f"{_current_eval.get('avg_score') or 0:.2f}")

            _rm = _current_eval.get("retrieval_metrics") or {}
            if _rm:
                st.markdown("*Retrieval*")
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Hit Rate@10", f"{_rm.get('hit_rate_at_10') or 0:.3f}")
                r2.metric("Precision@10", f"{_rm.get('precision_at_10') or 0:.3f}")
                r3.metric("Recall@10", f"{_rm.get('recall_at_10') or 0:.3f}")
                r4.metric("MRR", f"{_rm.get('mrr') or 0:.3f}")
                r5.metric("NDCG@10", f"{_rm.get('ndcg_at_10') or 0:.3f}")
            st.divider()
        else:
            st.info("No eval result yet — run one below.")

        # ── New eval form — wrapped in st.form so reruns only fire on submit ───
        st.markdown("**Run new evaluation**")
        with st.form("ragas_eval_form"):
            n_pairs = st.number_input("Number of Q+A pairs", min_value=1, max_value=10, value=5)
            delay_s = 3

            ragas_questions: list[str] = []
            ragas_ground_truths: list[str] = []
            for qi in range(int(n_pairs)):
                col_q, col_gt = st.columns(2)
                with col_q:
                    q = st.text_input(f"Question {qi + 1}", key=f"ragas_q_{qi}",
                                      placeholder="e.g. What is the punishment for murder?")
                with col_gt:
                    gt = st.text_area(f"Ground truth {qi + 1}", key=f"ragas_gt_{qi}", height=100,
                                      placeholder="Expected answer the model should produce")
                ragas_questions.append(q)
                ragas_ground_truths.append(gt)

            submitted = st.form_submit_button("▶ Run RAGAS Evaluation", type="primary", use_container_width=True)

        if submitted:
            filled_q = [q for q in ragas_questions if q.strip()]
            filled_gt = [g for g in ragas_ground_truths if g.strip()]
            if len(filled_q) != int(n_pairs) or len(filled_gt) != int(n_pairs):
                st.warning("Fill in all questions and ground truths before running.")
            elif len(filled_q) != len(filled_gt):
                st.warning("Number of questions and ground truths must match.")
            else:
                est_s = len(filled_q) * 18  # ~3s delay + ~15s per query+RAGAS
                with st.spinner(f"Running RAGAS eval on {len(filled_q)} pairs — estimated ~{int(est_s)}s…"):
                    try:
                        er = requests.post(
                            f"{API}/eval/run",
                            headers=_headers(),
                            json={
                                "doc_id": eval_doc_id,
                                "questions": filled_q,
                                "ground_truths": filled_gt,
                                "delay_s": float(delay_s),
                            },
                            timeout=600,
                        )
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach API")
                        er = None
                if er and er.status_code == 202:
                    res = er.json()
                    st.success("Evaluation complete!")
                    st.markdown("*Generation*")
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Faithfulness", f"{res.get('faithfulness', 0):.2f}")
                    r2.metric("Context Precision", f"{res.get('context_precision', 0):.2f}")
                    r3.metric("Answer Relevance", f"{res.get('answer_relevance', 0):.2f}")
                    r4.metric("Avg Score", f"{res.get('avg_score', 0):.2f}")
                    _rr = {k: v for k, v in res.items() if k.startswith("hit_rate") or k in ("mrr", "precision_at_10", "recall_at_10", "ndcg_at_10")}
                    if _rr:
                        st.markdown("*Retrieval*")
                        rr1, rr2, rr3, rr4, rr5 = st.columns(5)
                        rr1.metric("Hit Rate@10", f"{_rr.get('hit_rate_at_10', 0):.3f}")
                        rr2.metric("Precision@10", f"{_rr.get('precision_at_10', 0):.3f}")
                        rr3.metric("Recall@10", f"{_rr.get('recall_at_10', 0):.3f}")
                        rr4.metric("MRR", f"{_rr.get('mrr', 0):.3f}")
                        rr5.metric("NDCG@10", f"{_rr.get('ndcg_at_10', 0):.3f}")
                    st.rerun()
                elif er:
                    st.error(er.json().get("detail", "Eval failed"))


# ── Tab 4: Strategy Library ───────────────────────────────────────────────────
with tab_library:
    st.markdown("### 📚 Strategy Library")
    st.caption("All chunking strategies across your documents. Name them to reuse on future uploads.")

    col1, col2 = st.columns([1, 1])
    named_only = col1.checkbox("Named strategies only", value=False)
    if col2.button("Refresh", key="refresh_library"):
        st.rerun()

    lr = _api("get", f"/eval/strategies?named_only={str(named_only).lower()}")
    strategies = lr.json().get("strategies", []) if lr and lr.status_code == 200 else []

    if not strategies:
        st.info("No strategies found. Ingest a document first.")
    else:
        for s in strategies:
            active_badge = "🟢" if s.get("is_active") else "⚪"
            label = f"{active_badge} **{s['name'] or 'unnamed'}** · `{s['chunker']}` · v{s['version']} · {s['filename']}"
            with st.expander(label):
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Chunk Size", s["chunk_size"])
                sc2.metric("Overlap", s["overlap"])
                sc3.metric("Source", s["source"])

                if s.get("reasoning"):
                    st.caption(f"Reasoning: {s['reasoning'][:200]}")

                if s.get("ragas_scores"):
                    scores = s["ragas_scores"]
                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("Faithfulness", f"{scores.get('faithfulness', 0):.2f}")
                    rc2.metric("Context Prec.", f"{scores.get('context_precision', 0):.2f}")
                    rc3.metric("Answer Rel.", f"{scores.get('answer_relevance', 0):.2f}")
                    rc4.metric("Avg Score", f"{scores.get('avg', 0):.2f}")

                # Name / rename
                st.markdown("**Save as named strategy**")
                new_name = st.text_input(
                    "Strategy name",
                    value=s.get("name") or "",
                    placeholder="e.g. legal-sections-v1",
                    key=f"sname_{s['strategy_id']}",
                )
                if st.button("💾 Save name", key=f"savename_{s['strategy_id']}"):
                    nr = _api("post", f"/eval/strategy/{s['strategy_id']}/name", json={"name": new_name.strip()})
                    if nr and nr.status_code == 200:
                        st.success(f"Saved as '{new_name}'")
                        st.rerun()
                    elif nr:
                        st.error(nr.json().get("detail", "Failed to save name"))
