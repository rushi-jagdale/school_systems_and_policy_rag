from __future__ import annotations

"""
server.py — School Policy RAG API
---------------------------------
A small FastAPI wrapper around the RAG engine. It exposes:
  - GET  /health  -> quick status check
  - POST /build   -> (re)build the vector index from your PDFs
  - POST /ask     -> answer a single question

Quickstart
    uvicorn server:app --host 0.0.0.0 --port 8000

Notes
  • The server keeps a single in-memory index (thread-safe via a lock).
  • If you call /ask before /build, it tries a "lazy build" using defaults
    as long as the default data directory exists.
  • The RAG engine itself lives in `school_policy_rag.py` and includes a tiny
    LFU cache for repeated questions.
"""

import os
import threading
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Use the human-friendly, cache-enabled RAG implementation
from school_policy_rag import PolicyIndex, build_index, answer_question

# ---------------- Defaults ----------------

DEFAULT_FILES = [
    "attendance_policy.pdf",
    "code_of_conduct.pdf",
    "grading_policy.pdf",
]
DEFAULT_DATA_DIR = "data/docs"

app = FastAPI(
    title="School Policy RAG API",
    version="1.0",
    description="Answer questions about school policy PDFs (attendance, conduct, grading).",
)

# A single global index guarded by a lock (simple and effective)
INDEX: Optional[PolicyIndex] = None
INDEX_LOCK = threading.Lock()
CURRENT_DATA_DIR = DEFAULT_DATA_DIR
CURRENT_FILES = DEFAULT_FILES.copy()

# ---------------- Schemas ----------------

class BuildRequest(BaseModel):
    data_dir: str = Field(default=DEFAULT_DATA_DIR, description="Folder containing policy PDFs")
    files: List[str] = Field(default_factory=lambda: DEFAULT_FILES, description="PDF filenames to index")

class BuildResponse(BaseModel):
    status: str
    data_dir: str
    files: List[str]

class AskRequest(BaseModel):
    question: str = Field(..., description="A single natural-language question")
    id: str = Field("q1", description="Optional caller-provided ID for the question")

class AskResponse(BaseModel):
    id: str
    question: str
    answers: str
    doc_id: Optional[str]

# ---------------- Helpers ----------------

def _validate_pdfs(data_dir: str, files: List[str]) -> None:
    """Validate that the directory exists and all named PDFs are present."""
    if not os.path.isdir(data_dir):
        raise HTTPException(status_code=400, detail=f"Directory not found: {data_dir}")
    missing = [f for f in files if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing PDFs in {data_dir}: {', '.join(missing)}")

# ---------------- Routes ----------------

@app.get("/health", summary="Health check")
def health():
    """Lightweight liveness probe."""
    return {"status": "ok"}

@app.post("/build", response_model=BuildResponse, summary="(Re)build the vector index")
def build(req: BuildRequest):
    """Validate paths, build the index, and set it as the active index."""
    global INDEX, CURRENT_DATA_DIR, CURRENT_FILES
    with INDEX_LOCK:
        _validate_pdfs(req.data_dir, req.files)
        INDEX = build_index(req.data_dir, req.files)
        CURRENT_DATA_DIR = req.data_dir
        CURRENT_FILES = list(req.files)
    return BuildResponse(status="built", data_dir=CURRENT_DATA_DIR, files=CURRENT_FILES)

@app.post("/ask", response_model=AskResponse, summary="Answer a single question")
def ask(req: AskRequest):
    """
    Answer one question against the current index.
    If no index is built yet, we attempt a 'lazy build' using defaults.
    """
    global INDEX
    with INDEX_LOCK:
        if INDEX is None:
            # Lazy build using defaults (if available)
            if not os.path.isdir(CURRENT_DATA_DIR):
                raise HTTPException(status_code=400, detail="Index not built and default data directory not found.")
            _validate_pdfs(CURRENT_DATA_DIR, CURRENT_FILES)
            INDEX = build_index(CURRENT_DATA_DIR, CURRENT_FILES)

        try:
            result: Dict = answer_question(INDEX, req.question, q_id=req.id, k=6, threshold=0.25)
        except Exception as e:
            # Defensive: bubble up a clean message without leaking internals
            raise HTTPException(status_code=500, detail=f"Failed to answer question: {e!s}")

    # The engine already returns exactly {id, question, answers, doc_id}
    return AskResponse(**result)

# (Optional) Peek at current config without rebuilding
@app.get("/config", summary="Current index configuration (data_dir + files)")
def config():
    return {"data_dir": CURRENT_DATA_DIR, "files": CURRENT_FILES, "ready": INDEX is not None}
