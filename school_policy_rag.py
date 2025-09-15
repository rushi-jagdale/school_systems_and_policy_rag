#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
school_policy_rag.py
--------------------
A small, readable Retrieval-Augmented QA (RAG) utility that answers questions
about school policy PDFs (Attendance, Code of Conduct, Grading).

What it does
- Parses PDFs with PyMuPDF and normalises odd Unicode (soft hyphens, smart quotes).
- Splits text into section-aware, sentence-aware chunks.
- Embeds with SentenceTransformers (all-MiniLM-L6-v2) and searches via FAISS (cosine).
- Extracts concise answers with a few domain-aware guardrails.
- **New:** includes a tiny LFU cache (capacity 10) for repeated questions.

Why the cache?
- If users ask the same question repeatedly, we avoid re-embedding/retrieving.
- The cache stores up to 10 questions by normalised text.
- It evicts the **least-frequently used** entry; ties broken by oldest.

CLI examples
    # One-off question
    python school_policy_rag.py --data_dir data/docs --ask "What time do registers open and close?"

    # Interactive
    python school_policy_rag.py --data_dir data/docs --loop

Integrating elsewhere
    from school_policy_rag import build_index, answer_question
    idx = build_index("data/docs", ["attendance_policy.pdf","code_of_conduct.pdf","grading_policy.pdf"])
    res = answer_question(idx, "After how many unauthorised sessions may a penalty notice be considered?", q_id="q1")
    print(res)

Dependencies
    pymupdf, faiss-cpu, sentence-transformers, numpy
"""

import os
import re
import json
import argparse
import unicodedata
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------- UNICODE NORMALISATION ----------------

_TRANSLATE_MAP = {
    0x00AD: None,   # SOFT HYPHEN
    0x200B: None,   # ZERO WIDTH SPACE
    0x200C: None,   # ZERO WIDTH NON-JOINER
    0x200D: None,   # ZERO WIDTH JOINER
    0xFEFF: None,   # ZERO WIDTH NO-BREAK SPACE
    0x2010: ord('-'), 0x2011: ord('-'), 0x2012: ord('-'),
    0x2013: ord('-'), 0x2014: ord('-'), 0x2212: ord('-'),
    0x2043: ord('-'),  # HYPHEN BULLET
    0x2018: ord("'"), 0x2019: ord("'"),
    0x201C: ord('"'), 0x201D: ord('"'),
    0x2022: ord('*'),
    0x00A0: ord(' '),
}

def normalize_text(s: str) -> str:
    """Normalise to a single line (remove newlines, collapse spaces)."""
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TRANSLATE_MAP)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\s*\n\s*", " ", s)
    return s.strip()

def normalize_text_keep_newlines(s: str) -> str:
    """Normalise but preserve newlines (helps detect headings/bullets)."""
    if not s:
        return s
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TRANSLATE_MAP)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t\f\v]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    s = re.sub(r" *\n *", "\n", s)
    return s.strip()

# ---------------- PDF LOADING ----------------

def load_pdf_text(path: str) -> str:
    """Read text from a PDF and return a normalised string (with newlines)."""
    doc = fitz.open(path)
    pages = []
    for p in doc:
        txt = p.get_text()
        pages.append(normalize_text_keep_newlines(txt))
    return normalize_text_keep_newlines("\n\n".join(pages))

def looks_like_heading(line: str) -> bool:
    """Heuristic to spot section headings."""
    l = line.strip()
    if len(l) < 3 or len(l) > 160:
        return False
    if re.match(r"^\s*\d{1,2}[\.)]\s+\S", l):
        return True
    if l.upper() == l and re.search(r"[A-Z]", l) and " " in l:
        return True
    if re.match(r"^[A-Z][A-Za-z0-9 &/\-]{2,}$", l) and not l.endswith("."):
        return True
    return False

def split_into_sections(full_text: str) -> List[Tuple[str, str]]:
    """Split into (title, body) using the heading heuristic."""
    lines = full_text.split("\n")
    sections: List[Tuple[str, str]] = []
    cur_title = "Document"
    cur_buf: List[str] = []
    for line in lines:
        if looks_like_heading(line):
            if cur_buf:
                body = normalize_text_keep_newlines("\n".join(cur_buf)).strip()
                if body:
                    sections.append((cur_title, body))
                cur_buf = []
            cur_title = normalize_text(line)
        else:
            cur_buf.append(line)
    if cur_buf:
        body = normalize_text_keep_newlines("\n".join(cur_buf)).strip()
        if body:
            sections.append((cur_title, body))
    if not sections:
        return [("Document", full_text.strip())]
    return sections

# ---------------- SENTENCE TOKENISER ----------------

def sent_tokenize(text: str) -> List[str]:
    """Simple sentence splitter tolerant to PDF bulleting quirks."""
    t = normalize_text_keep_newlines(text)
    t = re.sub(r"(?m)^\s*[-*•]\s*", ". ", t)  # treat bullets like sentence breaks
    sents: List[str] = []
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", normalize_text(line))
        for s in parts:
            s = s.strip(" -•*")
            if s:
                sents.append(s)
    return sents

def chunk_section(section_text: str, max_chars: int = 1000, overlap: int = 120) -> List[str]:
    """Make ~1000‑char chunks with sentence boundaries and overlap."""
    sents = sent_tokenize(section_text)
    chunks = []
    cur = ""
    for s in sents:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s
        else:
            chunks.append(cur)
            if overlap > 0 and len(cur) > overlap:
                cur = cur[-overlap:] + " " + s
            else:
                cur = s
    if cur:
        chunks.append(cur)
    if not chunks and section_text:
        t = normalize_text_keep_newlines(section_text)
        i = 0
        while i < len(t):
            chunks.append(t[i:i+max_chars])
            i_next = i + max_chars - overlap
            i = i_next if i_next > i else len(t)
    return [normalize_text(c) for c in chunks]

# ---------------- INDEX BUILDING ----------------

class PolicyIndex:
    """FAISS index + metadata with a simple search method."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.meta: List[Dict] = []

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / n

    def add_documents(self, docs: List[Dict]):
        """Add docs (text, doc_id, section_title)."""
        if not docs:
            return
        texts = [normalize_text(d["text"]) for d in docs]
        embs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        embs = self._normalize(embs).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(embs.shape[1])  # cosine on normalised vectors
        self.index.add(embs)
        self.meta.extend([
            {"doc_id": d["doc_id"], "section_title": normalize_text(d.get("section_title", "")), "text": t}
            for d, t in zip(docs, texts)
        ])

    def search(self, query: str, k: int = 6) -> List[Tuple[float, Dict]]:
        """Return top‑k (score, metadata) for a query."""
        if self.index is None or not self.meta:
            return []
        q = normalize_text(query)
        q = self.model.encode([q], convert_to_numpy=True, show_progress_bar=False)
        q = self._normalize(q).astype("float32")
        D, I = self.index.search(q, k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx == -1:
                continue
            out.append((float(score), self.meta[idx]))
        return out

def build_index(data_dir: str, files: List[str]) -> PolicyIndex:
    """Build an index from PDFs in a directory."""
    pi = PolicyIndex()
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing PDF: {path}")
        full_text = load_pdf_text(path)
        sections = split_into_sections(full_text)
        docs = []
        for title, body in sections:
            chunks = chunk_section(body, max_chars=1000, overlap=120)
            for ch in chunks:
                docs.append({"text": ch, "doc_id": f, "section_title": title})
        pi.add_documents(docs)
    return pi

# ---------------- ANSWER EXTRACTION ----------------

STOP_WORDS = set('a an the and or but if while with without to from in on at for by of as is are was were be been being have has had do does did this that these those it its i you he she they we us our their your'.split())

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")
DURATION_RE = re.compile(r"\b(within|after|by)\s+\d+\s+(minutes?|hours?|days?|weeks?)\b", re.I)
DATE_WORD_RE = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|term|week|day|date)\b", re.I)

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
UNIT_NEARBY = re.compile(r"\b(sessions?|absences?|days?|weeks?|lessons?|periods?)\b", re.I)

NUMBER_WORDS = {
    "zero":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,
}

SYNONYMS = {
    "guardian": ["parent", "carer", "caregiver"],
    "unexcused": ["unauthorised", "unauthorized"],
    "absences": ["absence", "attendance", "missing"],
    "detention": ["sanction", "consequence"],
}

CONTACT_RE = re.compile(r"\b(contact|notify|call|phone|inform|report)\b", re.I)
PARENT_RE  = re.compile(r"\b(parent|guardian|carer|caregiver)\b", re.I)
REGISTER_RE = re.compile(r"\bregisters?\b", re.I)

FINANCE_Q_TERMS = {"fee","fees","cost","charge","charges","price","loan"}

def sanitize_question(q: str) -> str:
    """Tidy the question; drop very short/obviously incomplete ones."""
    q = normalize_text(q)
    q = q.strip(' "”‘’“')
    if len(q.split()) < 5:
        return ""
    if q.endswith(tuple(["will","the","school","after","how","many","contact","notify"])):
        return ""
    return q

def expand_synonyms(q: str) -> str:
    """Add a few synonyms to help retrieval land in the right chunk."""
    ql = q.lower()
    extra = []
    for k, vals in SYNONYMS.items():
        if k in ql or any(v in ql for v in vals):
            extra.extend(vals + [k])
    if extra:
        return q + " " + " ".join(sorted(set(extra)))
    return q

def keyword_set(q: str) -> set:
    """Lowercased keywords minus stop words."""
    q = normalize_text(q)
    tokens = re.findall(r"[a-zA-Z]+", q.lower())
    return {t for t in tokens if t not in STOP_WORDS and len(t) > 2}

def sentence_score(sent: str, q_keywords: set) -> float:
    """Simple lexical‑overlap score between a sentence and the query."""
    sent = normalize_text(sent)
    stoks = re.findall(r"[a-zA-Z]+", sent.lower())
    sset = set(stoks)
    overlap = len(q_keywords & sset)
    return overlap / (len(q_keywords) + 1e-6)

def expects_number(q: str) -> bool:
    """Detect questions that expect a numeric answer."""
    ql = q.lower().strip()
    return (ql.startswith(("how many", "how much", "after how many", "how long"))
            or re.search(r"\bhow\s+often\b", ql) is not None
            or re.search(r"\b(number|count|times?)\b", ql) is not None)

def expects_time(q: str) -> bool:
    """Detect questions that expect a time/duration/day answer."""
    ql = q.lower()
    return ql.startswith("when") or "what time" in ql or "by when" in ql or "deadline" in ql or ("open" in ql and "close" in ql and "register" in ql)

def number_from_words_or_digits(s: str) -> Optional[str]:
    """Return a digit string if a number is found (word or digit)."""
    m = re.search(r"\b(\d+)\b", s)
    if m:
        return m.group(1)
    for w, n in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(w)}\b", s, flags=re.I):
            return str(n)
    return None

def expand_phrase_around_number(s: str, num: str) -> str:
    """Return a compact phrase around that number, including nearby unit words."""
    toks = s.split()
    idxs = [i for i, t in enumerate(toks) if re.sub(r"\D", "", t) == num]
    if not idxs:
        return normalize_text(s)
    idx = idxs[0]
    left = max(0, idx - 4)
    right = idx + 1
    while right < len(toks):
        if re.search(r"[.;:]", toks[right]):
            break
        if UNIT_NEARBY.search(toks[right]):
            right = min(len(toks), right + 3)
            break
        right += 1
    phrase = " ".join(toks[left:right])
    tail = " ".join(toks[idx: min(len(toks), idx + 24)])
    m = re.search(r"(within|rolling).+?(weeks?|period|school\-week)\b", tail, re.I)
    if m:
        phrase = " ".join(toks[left: idx + m.end()])
    return normalize_text(phrase)

def valid_numeric_for_count(ans: str) -> bool:
    """Require a number and a nearby unit (e.g., sessions/weeks)."""
    ans_wo_years = YEAR_RE.sub("", ans)
    num = re.search(r"\b\d+(\.\d+)?\b", ans_wo_years)
    return bool(num and UNIT_NEARBY.search(ans))

def extract_concise_answer_from_sentence(s: str, question: str) -> str:
    """Clean a sentence; compress to a crisp phrase for number/time questions."""
    s = re.sub(r"^\s*[-•*]\s*", "", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    if expects_number(question):
        num = number_from_words_or_digits(s)
        if num:
            return expand_phrase_around_number(s, num)
    if expects_time(question):
        m = TIME_RE.search(s) or DURATION_RE.search(s)
        if m or DATE_WORD_RE.search(s):
            return normalize_text(s)
    return normalize_text(s)

def extract_register_times(passages, question):
    """Pull 'open at HH:MM' and 'close at HH:MM' (same/adjacent sentences)."""
    if not ("open" in question.lower() and "close" in question.lower() and "register" in question.lower()):
        return None
    for _, meta in passages:
        text = meta["text"]
        sents = sent_tokenize(text)
        for i, s in enumerate(sents):
            if REGISTER_RE.search(s):
                m = re.search(r"open[s]?\s+(?:at\s+)?(\d{1,2}:\d{2}).+?close[s]?\s+(?:at\s+)?(\d{1,2}:\d{2})", s, re.I)
                if m:
                    t1, t2 = m.group(1), m.group(2)
                    return (f"Registers open at {t1} and close at {t2}.", meta["doc_id"])
                t_open = re.search(r"open[s]?\s+(?:at\s+)?(\d{1,2}:\d{2})", s, re.I)
                t_close = re.search(r"close[s]?\s+(?:at\s+)?(\d{1,2}:\d{2})", s, re.I)
                if t_open and not t_close and i+1 < len(sents):
                    t_close2 = re.search(r"close[s]?\s+(?:at\s+)?(\d{1,2}:\d{2})", sents[i+1], re.I)
                    if t_close2:
                        return (f"Registers open at {t_open.group(1)} and close at {t_close2.group(1)}.", meta["doc_id"])
                if t_close and not t_open and i>0:
                    t_open2 = re.search(r"open[s]?\s+(?:at\s+)?(\d{1,2}:\d{2})", sents[i-1], re.I)
                    if t_open2:
                        return (f"Registers open at {t_open2.group(1)} and close at {t_close.group(1)}.", meta["doc_id"])
                times = re.findall(r"\b(\d{1,2}:\d{2})\b", " ".join(sents[max(0,i-1):min(len(sents), i+2)]))
                if len(times) >= 2:
                    t1, t2 = times[0], times[1]
                    return (f"Registers open at {t1} and close at {t2}.", meta["doc_id"])
    return None

def extract_ks3_bands(passages, question):
    """Return the KS3 band names when asked."""
    ql = question.lower()
    if "ks3" not in ql or "band" not in ql:
        return None
    labset = []
    for _, meta in passages:
        text = meta["text"]
        labels = re.findall(r"\b(Emerging|Developing|Secure|Exceeding)\b", text, flags=re.I)
        ordered = []
        for lab in labels:
            L = lab.capitalize()
            if L not in ordered:
                ordered.append(L)
        for L in ordered:
            if L not in labset:
                labset.append(L)
        if len(labset) >= 3:
            return (", ".join(labset), meta["doc_id"])
    return None

def extract_concise_answer(passages: List[Tuple[float, Dict]], question: str, top_sents: int = 4) -> Tuple[str, Optional[str], float]:
    """Score sentences in retrieved chunks and return the best candidate answer."""
    qk = keyword_set(expand_synonyms(question))
    best = ("", None, -1.0)
    for retr_score, meta in passages:
        sents = sent_tokenize(meta["text"])
        if not sents:
            continue
        ranked = []
        for s in sents:
            base = sentence_score(s, qk)
            if re.search(r"\b(must|should|required|will|may|shall)\b", s, re.I):
                base += 0.1
            if expects_number(question) and re.search(r"\b\d+(\.\d+)?\b", s):
                base += 0.2
            if expects_time(question) and (TIME_RE.search(s) or DURATION_RE.search(s) or DATE_WORD_RE.search(s)):
                base += 0.2
            ranked.append((base, s))
        ranked.sort(reverse=True)
        for score_local, s in ranked[:top_sents]:
            cand = extract_concise_answer_from_sentence(s, question)
            total = retr_score + 0.5 * score_local
            if total > best[2] and cand:
                best = (cand, meta["doc_id"], total)
    return best

def prefer_contact_sentence(passages, question):
    """If parents/guardians are involved, prefer a clear 'how/when to report' sentence."""
    ql = question.lower()
    if not (PARENT_RE.search(ql) and (("contact" in ql) or ("notify" in ql) or ("inform" in ql) or ("report" in ql))):
        return None
    for _, meta in passages:
        for s in sent_tokenize(meta["text"]):
            if CONTACT_RE.search(s) and (re.search(r"\b(first day|same day|by \d{1,2}:\d{2}|within \d+ (school )?days?|attendance line|absence line|phone)\b", s, re.I)):
                return (normalize_text(s), meta["doc_id"])
    return None

def doc_prior(question: str, doc_id: str) -> float:
    """Small doc‑type priors to help retrieval pick the right PDF."""
    q = question.lower()
    if doc_id == "attendance_policy.pdf" and any(w in q for w in ["attendance","register","absence","unauthorised","unexcused","parent","guardian","phone","contact","notify","report"]):
        return 0.08
    if doc_id == "code_of_conduct.pdf" and any(w in q for w in ["conduct","mobile","device","banned","detention","sanction","values"]):
        return 0.08
    if doc_id == "grading_policy.pdf" and any(w in q for w in ["ks3","gcse","dcp","mock","grading","assessment","band"]):
        return 0.08
    return 0.0

def is_privacy_question(q: str) -> bool:
    """Refuse questions like 'what is my name?'."""
    return bool(re.search(r"\b(my|your)\s+name\b", q.lower()))

# ---------------- LFU CACHE ----------------

class QuestionCache:
    """
    Tiny LFU cache for Q -> (answers, doc_id).
    - Capacity 10 by default.
    - get(key): returns (answers, doc_id) or None; increments freq on hit.
    - put(key, val): inserts/updates; if full, evicts least‑frequent, oldest tie‑break.
    """
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.store: Dict[str, Dict] = {}
        self.clock = 0

    def _tick(self) -> int:
        self.clock += 1
        return self.clock

    def get(self, key: str) -> Optional[Tuple[str, Optional[str]]]:
        if key in self.store:
            entry = self.store[key]
            entry["freq"] += 1
            entry["ts"] = self._tick()
            return entry["val"]
        return None

    def put(self, key: str, val: Tuple[str, Optional[str]]):
        if key in self.store:
            entry = self.store[key]
            entry["val"] = val
            entry["freq"] += 1  # treat update as a use
            entry["ts"] = self._tick()
            return
        if len(self.store) >= self.capacity:
            evict_key = min(self.store.items(), key=lambda kv: (kv[1]["freq"], kv[1]["ts"]))[0]
            del self.store[evict_key]
        self.store[key] = {"val": val, "freq": 1, "ts": self._tick()}

# Global cache instance
_CACHE = QuestionCache(capacity=10)

def cache_key_from_question(q: str) -> str:
    """Normalised cache key: lowercase + collapse whitespace."""
    return re.sub(r"\s+", " ", normalize_text(q)).lower().strip()

# ---------------- CORE ANSWERING (no cache) ----------------

def _answer_question_impl(pi: PolicyIndex, question: str, q_id: str = "q1",
                          k: int = 6, threshold: float = 0.25) -> Dict:
    """
    Core QA flow (retrieval + extraction + guardrails). Separate from cache wrapper
    so we can reuse it easily and keep the cache logic clean.
    """
    raw_q = question
    question2 = sanitize_question(question)
    if not question2 or is_privacy_question(question2):
        return {"id": q_id, "question": raw_q, "answers": "I don't know.", "doc_id": None}

    # Retrieve and apply small doc priors
    results = pi.search(expand_synonyms(question2), k=k)
    if not results:
        return {"id": q_id, "question": question2, "answers": "I don't know.", "doc_id": None}
    reranked = sorted(
        [(score + doc_prior(question2, meta["doc_id"]), meta) for score, meta in results],
        key=lambda x: -x[0]
    )

    # Special‑cases
    reg = extract_register_times(reranked, question2)
    if reg:
        ans, doc_id = reg
        return {"id": q_id, "question": raw_q, "answers": ans, "doc_id": doc_id}

    ks3 = extract_ks3_bands(reranked, question2)
    if ks3:
        ans, doc_id = ks3
        return {"id": q_id, "question": raw_q, "answers": ans, "doc_id": doc_id}

    pref = prefer_contact_sentence(reranked, question2)
    if pref:
        ans, doc_id = pref
        return {"id": q_id, "question": raw_q, "answers": ans, "doc_id": doc_id}

    # Generic extraction
    ans, doc_id, _ = extract_concise_answer(reranked, question2, top_sents=4)
    if not ans or doc_id is None:
        return {"id": q_id, "question": raw_q, "answers": "I don't know.", "doc_id": None}

    # Guardrails
    q_terms = set(re.findall(r"[a-zA-Z]+", question2.lower()))
    if expects_number(question2) and not valid_numeric_for_count(ans):
        return {"id": q_id, "question": raw_q, "answers": "I don't know.", "doc_id": None}
    if expects_time(question2) and not (TIME_RE.search(ans) or DURATION_RE.search(ans) or DATE_WORD_RE.search(ans)):
        return {"id": q_id, "question": raw_q, "answers": "I don't know.", "doc_id": None}
    if q_terms & FINANCE_Q_TERMS:
        if not (set(re.findall(r"[a-zA-Z]+", ans.lower())) & FINANCE_Q_TERMS):
            return {"id": q_id, "question": raw_q, "answers": "I don't know.", "doc_id": None}

    ans = re.sub(r"\s{2,}", " ", ans).strip()
    return {"id": q_id, "question": raw_q, "answers": ans if ans else "I don't know.", "doc_id": doc_id}

# ---------------- CACHE WRAPPER (public API) ----------------

def answer_question(pi: PolicyIndex, question: str, q_id: str = "q1",
                    k: int = 6, threshold: float = 0.25) -> Dict:
    """
    Cache-first wrapper:
    1) Look up the normalised question in an LFU cache (capacity 10).
    2) If found, serve cached (answers, doc_id) but keep the incoming q_id/question.
    3) Otherwise, compute via core RAG and store the result.
    """
    key = cache_key_from_question(question)
    hit = _CACHE.get(key)
    if hit is not None:
        answers, doc_id = hit
        return {"id": q_id, "question": question, "answers": answers, "doc_id": doc_id}

    res = _answer_question_impl(pi, question, q_id=q_id, k=k, threshold=threshold)
    _CACHE.put(key, (res.get("answers", "I don't know."), res.get("doc_id", None)))
    return res

# ---------------- CLI ----------------

def main():
    parser = argparse.ArgumentParser(description="School Policy RAG (with 10‑entry LFU cache)")
    parser.add_argument("--data_dir", default="data/docs", help="Folder containing the PDFs")
    parser.add_argument("--files", nargs="+", default=[
        "attendance_policy.pdf",
        "code_of_conduct.pdf",
        "grading_policy.pdf"
    ], help="PDF filenames to index")
    parser.add_argument("--ask", default=None, help="Ask a single question and exit")
    parser.add_argument("--loop", action="store_true", help="Interactive loop")
    parser.add_argument("--k", type=int, default=6, help="Top‑k chunks to retrieve")
    parser.add_argument("--threshold", type=float, default=0.25, help="Similarity threshold (kept for compatibility)")
    args = parser.parse_args()

    print(" Building index...")
    pi = build_index(args.data_dir, args.files)
    print("Ready.")

    if args.ask:
        result = answer_question(pi, args.ask, q_id="q1", k=args.k, threshold=args.threshold)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # Interactive
    if args.loop or not args.ask:
        i = 1
        print("\n Type your question (or 'exit' to quit):\n")
        while True:
            try:
                q = input(f"Q{i}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n Bye")
                break
            if q.lower() in {"exit", "quit"}:
                print(" Bye")
                break
            result = answer_question(pi, q, q_id=f"q{i}", k=args.k, threshold=args.threshold)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            i += 1

if __name__ == "__main__":
    main()
