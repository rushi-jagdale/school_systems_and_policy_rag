#!/usr/bin/env python3
"""
app.py — School Policy RAG (Streamlit UI)
----------------------------------------
A tiny front-end for the FastAPI server.

What you can do:
  • Health check the server
  • Build/Rebuild the index from your PDFs
  • Ask a single question
  • Run a batch of questions from a JSONL file and download results

Expected FastAPI endpoints:
  GET  /health
  POST /build   {"data_dir": "...", "files": ["a.pdf","b.pdf",...]}
  POST /ask     {"id": "q1", "question": "..."}

Run:
  streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import requests
import streamlit as st

# ---------------- Page setup ----------------

st.set_page_config(page_title="School Policy RAG", page_icon="", layout="wide")
st.title(" School Policy RAG — UI")
st.caption("A tiny Streamlit client for the FastAPI backend.")

# ---------------- Helpers ----------------

def _clean_base(url: str) -> str:
    """Remove trailing slash so we can safely append endpoints."""
    url = (url or "").strip()
    while url.endswith("/"):
        url = url[:-1]
    return url

def api_get(session: requests.Session, base: str, path: str, timeout: int = 15) -> requests.Response:
    return session.get(f"{_clean_base(base)}{path}", timeout=timeout)

def api_post(session: requests.Session, base: str, path: str, json_payload: Dict[str, Any], timeout: int = 60) -> requests.Response:
    return session.post(f"{_clean_base(base)}{path}", json=json_payload, timeout=timeout)

def pretty_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)

# Keep a single requests session
if "session" not in st.session_state:
    st.session_state.session = requests.Session()

# ---------------- Sidebar ----------------

with st.sidebar:
    st.header(" Server")
    server_url = st.text_input("Server URL", value=st.session_state.get("server_url", "http://localhost:8000"))
    st.session_state.server_url = server_url

    colA, colB = st.columns(2)
    with colA:
        if st.button("Health check", use_container_width=True):
            try:
                with st.spinner("Checking server health..."):
                    r = api_get(st.session_state.session, server_url, "/health")
                    r.raise_for_status()
                    st.success(r.json())
            except Exception as e:
                st.error(f"Health check failed: {e}")

    with colB:
        if st.button("Show config", use_container_width=True):
            try:
                with st.spinner("Fetching /config..."):
                    r = api_get(st.session_state.session, server_url, "/config")
                    if r.status_code == 404:
                        st.info("This server doesn't expose /config. That's okay — try Health check instead.")
                    else:
                        r.raise_for_status()
                        st.success(r.json())
            except Exception as e:
                st.error(f"Config fetch failed: {e}")

    st.divider()
    st.header(" Build / Rebuild Index")

    data_dir = st.text_input("Data directory", value=st.session_state.get("data_dir", "data/docs"))
    st.session_state.data_dir = data_dir

    default_files_list = [
        "attendance_policy.pdf",
        "code_of_conduct.pdf",
        "grading_policy.pdf",
    ]
    files_text = st.text_area(
        "PDF files (one per line)",
        value=st.session_state.get("files_text", "\n".join(default_files_list)),
        height=100,
        help="Paths are relative to the data directory above."
    )
    st.session_state.files_text = files_text
    files = [f.strip() for f in files_text.splitlines() if f.strip()]

    if st.button("Build / Rebuild", type="primary", use_container_width=True):
        try:
            payload = {"data_dir": data_dir, "files": files}
            with st.spinner("Building index... (first run will download the embedding model)"):
                r = api_post(st.session_state.session, server_url, "/build", json_payload=payload, timeout=180)
            if r.status_code == 200:
                st.success(r.json())
                st.toast("Index built.", icon="")
            else:
                st.error(f"Build failed: {r.status_code}\n{r.text}")
        except Exception as e:
            st.error(f"Build error: {e}")

# ---------------- Ask a single question ----------------

st.subheader(" Ask a question")

c1, c2 = st.columns([1, 6])
with c1:
    qid = st.text_input("Question ID", value=st.session_state.get("qid", "q1"))
    st.session_state.qid = qid
with c2:
    question = st.text_area(
        "Question",
        value=st.session_state.get("question", "After how many unauthorised sessions may a penalty notice be considered?"),
        placeholder="Type your question here...",
    )
    st.session_state.question = question

if st.button("Ask", type="primary"):
    try:
        payload = {"id": qid, "question": question}
        with st.spinner("Asking the API..."):
            r = api_post(st.session_state.session, server_url, "/ask", json_payload=payload, timeout=60)
        if r.status_code == 200:
            res = r.json()
            st.success("Answer received")

            # Raw JSON
            with st.expander("See raw JSON response"):
                st.code(pretty_json(res), language="json")

            # Pretty card
            st.markdown("---")
            st.markdown("#### Answer")
            st.markdown(f"**{res.get('answers','')}**")
            src = res.get("doc_id") or "—"
            st.caption(f"Source: `{src}`")
        else:
            st.error(f"Ask failed: {r.status_code}\n{r.text}")
    except Exception as e:
        st.error(f"Request error: {e}")

# ---------------- Batch (JSONL) ----------------

st.divider()
st.subheader(" Batch (JSONL)")
st.caption("Upload a file where each line is a JSON object like:  {\"id\": \"q1\", \"question\": \"...\"}")

uploaded = st.file_uploader("Upload questions.jsonl", type=["jsonl"])

if uploaded is not None:
    try:
        content = uploaded.read().decode("utf-8", errors="ignore")
        lines = [ln for ln in content.splitlines() if ln.strip()]
        st.info(f"Loaded {len(lines)} lines.")
    except Exception as e:
        lines = []
        st.error(f"Couldn't read file: {e}")

    if lines:
        if st.button("Run batch", type="primary"):
            results: List[Dict[str, Any]] = []
            errors = 0
            progress = st.progress(0, text="Processing...")
            for i, ln in enumerate(lines, start=1):
                try:
                    obj = json.loads(ln)
                    payload = {"id": obj.get("id", f"q{i}"), "question": obj["question"]}
                    r = api_post(st.session_state.session, server_url, "/ask", json_payload=payload, timeout=60)
                    if r.status_code == 200:
                        results.append(r.json())
                    else:
                        results.append({
                            "id": payload["id"],
                            "question": payload["question"],
                            "answers": "I don't know.",
                            "doc_id": None
                        })
                        errors += 1
                except Exception:
                    errors += 1
                finally:
                    progress.progress(min(i / len(lines), 1.0))

            st.success(f"Completed. Errors: {errors}. Results: {len(results)}")

            # Show a compact preview
            with st.expander("Preview first 10 results"):
                preview = results[:10]
                st.code("\n".join(json.dumps(x, ensure_ascii=False) for x in preview), language="json")

            # Offer download
            jsonl = "\n".join(json.dumps(x, ensure_ascii=False) for x in results)
            st.download_button(
                "Download results.jsonl",
                data=jsonl.encode("utf-8"),
                file_name="results.jsonl",
                mime="application/json"
            )
            st.toast("Batch complete.")
