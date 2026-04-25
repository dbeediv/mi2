"""
retriever.py  (v4 — Groq API, secure key via st.secrets / env)
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import requests

try:
    import streamlit as st
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

from ingestion import (
    EmbeddingModel,
    FAISSIndexManager,
    BM25IndexManager,
    MetadataStore,
    SIMILARITY_THRESHOLD,
    EMBED_MODEL_NAME,
)
from prompts import (
    build_rag_prompt,
    build_summarization_prompt,
    ANSWER_NOT_FOUND,
    NO_CITATION_FALLBACK,
    CITATION_PATTERN,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TOP_K: int             = 5
FAISS_FETCH_K: int     = 12
BM25_FETCH_K: int      = 12
RRF_K: int             = 60
MAX_CONTEXT_CHARS: int = 8_000
GROQ_MODEL: str        = "llama-3.3-70b-versatile"

def _get_groq_key() -> str:
    """Read key from st.secrets first, then environment variable. Never hardcoded."""
    if _STREAMLIT_AVAILABLE:
        try:
            return st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    return os.environ.get("GROQ_API_KEY", "")

SUMMARIZE_TRIGGERS = {
    "summarize", "summary", "abstract", "explain", "overview",
    "what is this", "what does this paper", "describe",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    section: str
    similarity_score: float
    is_table: bool = False
    is_priority: bool = False


@dataclass
class RetrievalResult:
    answer: str
    sources: List[RetrievedChunk] = field(default_factory=list)
    is_grounded: bool = False
    answer_found: bool = True
    raw_llm_output: str = ""


# ---------------------------------------------------------------------------
# Groq client
# ---------------------------------------------------------------------------
class GroqClient:

    def __init__(self, model: str = GROQ_MODEL, timeout: int = 60) -> None:
        self.model   = model
        self.timeout = timeout
        self.url     = "https://api.groq.com/openai/v1/chat/completions"

    @property
    def api_key(self) -> str:
        return _get_groq_key()

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        key = self.api_key
        if not key:
            raise RuntimeError("GROQ_API_KEY not set. Add it to Streamlit secrets or environment variables.")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        }
        r = requests.post(
            self.url,
            json=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("Groq returned no choices.")
        return choices[0].get("message", {}).get("content", "").strip()

    def health_check(self) -> bool:
        try:
            return bool(self.generate("Hi", temperature=0.0))
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Hybrid retriever (FAISS + BM25 → RRF)
# ---------------------------------------------------------------------------
class HybridRetriever:

    def __init__(self, faiss_mgr: FAISSIndexManager, bm25_mgr: BM25IndexManager, embedder: EmbeddingModel, rrf_k: int = RRF_K) -> None:
        self.faiss_mgr = faiss_mgr
        self.bm25_mgr  = bm25_mgr
        self.embedder  = embedder
        self.rrf_k     = rrf_k

    def retrieve(self, query: str, session_id: str, top_k: int = TOP_K, faiss_k: int = FAISS_FETCH_K, bm25_k: int = BM25_FETCH_K) -> List[tuple]:
        query_vec  = self.embedder.embed_query(query)
        faiss_hits = self.faiss_mgr.search(session_id, query_vec, top_k=faiss_k)
        bm25_hits  = self.bm25_mgr.search(session_id, query, top_k=bm25_k)

        faiss_ranks: Dict[str, int] = {cid: rank for rank, (cid, _) in enumerate(faiss_hits)}
        bm25_ranks:  Dict[str, int] = {cid: rank for rank, (cid, _) in enumerate(bm25_hits)}
        all_ids = set(faiss_ranks) | set(bm25_ranks)

        rrf_scores: Dict[str, float] = {}
        for cid in all_ids:
            score = 0.0
            if cid in faiss_ranks:
                score += 1.0 / (self.rrf_k + faiss_ranks[cid])
            if cid in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[cid])
            rrf_scores[cid] = score

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[: top_k * 3]


# ---------------------------------------------------------------------------
# Low-value chunk filter
# ---------------------------------------------------------------------------
class ChunkFilter:
    REJECT_SECTIONS = {"references", "bibliography", "acknowledgement", "acknowledgments"}

    @classmethod
    def is_low_value(cls, chunk: RetrievedChunk) -> bool:
        section = (chunk.section or "").lower()
        if any(k in section for k in cls.REJECT_SECTIONS):
            return True
        citation_hits = re.findall(r"\[\d+\]", chunk.text)
        total_words   = max(len(chunk.text.split()), 1)
        if len(citation_hits) / total_words > 0.15:
            return True
        if re.search(r"(\[\d+\].*\n){3,}", chunk.text):
            return True
        return False


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------
class ContextAssembler:

    def assemble(self, chunks: List[RetrievedChunk]) -> str:
        parts: List[str] = []
        chars_per_chunk  = MAX_CONTEXT_CHARS // max(len(chunks), 1)
        for i, chunk in enumerate(chunks, start=1):
            label = (
                f"--- Source {i} | File: {chunk.source_file} | "
                f"Page: {chunk.page_number or 'N/A'} | "
                f"Section: {chunk.section or 'General'} "
                f"({'TABLE' if chunk.is_table else 'TEXT'}, score={chunk.similarity_score:.4f}) ---"
            )
            parts.append(f"{label}\n{chunk.text[:chars_per_chunk]}")
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Response validator
# ---------------------------------------------------------------------------
class ResponseValidator:

    def validate(self, raw: str, chunks: List[RetrievedChunk]) -> tuple:
        stripped = raw.strip()
        if ANSWER_NOT_FOUND.lower() in stripped.lower():
            return ANSWER_NOT_FOUND, False
        if re.findall(CITATION_PATTERN, stripped, re.IGNORECASE):
            return stripped, True
        logger.warning("LLM answer had no citations — showing answer without grounding mark.")
        return stripped, False


# ---------------------------------------------------------------------------
# Main retrieval pipeline
# ---------------------------------------------------------------------------
class RetrievalPipeline:

    def __init__(self, groq_model: str = GROQ_MODEL, top_k: int = TOP_K) -> None:
        self.top_k          = top_k
        self.embedder       = EmbeddingModel(EMBED_MODEL_NAME)
        self.faiss_mgr      = FAISSIndexManager()
        self.bm25_mgr       = BM25IndexManager()
        self.metadata_store = MetadataStore()
        self.retriever      = HybridRetriever(self.faiss_mgr, self.bm25_mgr, self.embedder)
        self.llm            = GroqClient(model=groq_model)
        self.assembler      = ContextAssembler()
        self.validator      = ResponseValidator()
        self.filter         = ChunkFilter()

    @staticmethod
    def _is_summarize_query(question: str) -> bool:
        q = question.lower()
        return any(t in q for t in SUMMARIZE_TRIGGERS)

    def _to_retrieved_chunk(self, row: dict, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id         = row["chunk_id"],
            text             = row["text"],
            source_file      = row["source_file"],
            page_number      = row["page_number"],
            section          = row.get("section") or "",
            similarity_score = score,
            is_table         = bool(row.get("is_table", 0)),
            is_priority      = bool(row.get("is_priority", 0)),
        )

    def query(self, question: str, session_id: str) -> RetrievalResult:
        logger.info("Query | session=%s | q=%s", session_id, question[:80])
        is_summary = self._is_summarize_query(question)

        # Step 1: Hybrid retrieval
        rrf_hits = self.retriever.retrieve(query=question, session_id=session_id, top_k=self.top_k, faiss_k=FAISS_FETCH_K, bm25_k=BM25_FETCH_K)
        if not rrf_hits:
            return RetrievalResult(answer=ANSWER_NOT_FOUND, sources=[], is_grounded=False, answer_found=False)

        # Step 2: Fetch metadata
        score_map = dict(rrf_hits)
        rows      = self.metadata_store.fetch_chunks_by_ids(list(score_map.keys()))
        chunks    = [self._to_retrieved_chunk(r, score_map[r["chunk_id"]]) for r in rows]

        # Step 3: Priority chunks for summary queries
        if is_summary:
            priority_rows = self.metadata_store.fetch_priority_chunks(session_id)
            if priority_rows:
                existing_ids    = {r["chunk_id"] for r in rows}
                priority_chunks = [self._to_retrieved_chunk(pr, 1.5) for pr in priority_rows if pr["chunk_id"] not in existing_ids]
                chunks = priority_chunks + chunks

        # Step 4: Filter low-value chunks
        clean = [c for c in chunks if not self.filter.is_low_value(c)]
        if not clean:
            clean = chunks

        # Step 5: Sort and select top-K
        clean.sort(key=lambda c: (c.is_priority, c.similarity_score), reverse=True)
        final_chunks = clean[: self.top_k]

        if not final_chunks:
            return RetrievalResult(answer=ANSWER_NOT_FOUND, sources=[], is_grounded=False, answer_found=False)

        # Step 6: Assemble context
        context = self.assembler.assemble(final_chunks)

        # Step 7: Build prompt
        prompt = build_summarization_prompt(question, context) if is_summary else build_rag_prompt(question, context)

        # Step 8: Call Groq
        try:
            raw_output = self.llm.generate(prompt, temperature=0.0)
            logger.info("Groq response: %d chars", len(raw_output))
        except RuntimeError as exc:
            logger.error("Groq error: %s", exc)
            return RetrievalResult(
                answer=f"⚠️ Groq API error: {exc}",
                sources=final_chunks, is_grounded=False, answer_found=True, raw_llm_output=str(exc),
            )

        # Step 9: Validate
        final_answer, is_grounded = self.validator.validate(raw_output, final_chunks)

        return RetrievalResult(
            answer=final_answer, sources=final_chunks,
            is_grounded=is_grounded, answer_found=True, raw_llm_output=raw_output,
        )

    def check_session_ready(self, session_id: str) -> bool:
        return self.faiss_mgr.session_exists(session_id)

    def get_groq_status(self) -> dict:
        ok = self.llm.health_check()
        return {"online": ok, "model": self.llm.model}