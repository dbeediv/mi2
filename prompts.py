"""
prompts.py
==========
Prompt templates and fallback constants for the Market Intelligence RAG system.

Design Principles
-----------------
1.  CONTEXT ISOLATION  — The LLM is explicitly forbidden from using any
    knowledge outside the provided context passages.
2.  CITATION MANDATE   — Every factual claim must carry a [Source: ...] tag.
    The ResponseValidator in retriever.py enforces this post-generation.
3.  GRACEFUL FALLBACK  — If the answer is not in the context, the LLM is
    instructed to return an exact sentinel string (ANSWER_NOT_FOUND) so the
    system can detect and surface a clean "not found" message to the user.
4.  ZERO FABRICATION   — Statistics, figures, dates, and names must only
    appear if they exist verbatim in the context.

Author: Senior AI Architect
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Sentinel strings
# ---------------------------------------------------------------------------

# Exact string the LLM must return when the answer is not in the context.
# Keep this short and unambiguous — the ResponseValidator matches on it.
ANSWER_NOT_FOUND: str = (
    "I cannot find this information in the uploaded documents."
)

# Shown to the user when the LLM produced an answer but omitted citations,
# indicating the answer could not be grounded in retrieved context.
NO_CITATION_FALLBACK: str = (
    "⚠️ The system could not verify this answer against the uploaded documents. "
    "Please rephrase your question or check that the relevant documents have been uploaded."
)

# Regex pattern to detect [Source: <filename>, Page N] citations in LLM output.
# Matches patterns like:
#   [Source: annual_report.pdf, Page 12]
#   [Source: competitor_analysis.docx, Page N/A]
#   [Source: market_report.pdf]
CITATION_PATTERN: str = r"\[Source:\s*[^\]]+\]"

# ---------------------------------------------------------------------------
# System instructions (injected into every prompt)
# ---------------------------------------------------------------------------

_SYSTEM_INSTRUCTIONS: str = """\
You are a Market Intelligence Assistant for a financial research platform.

STRICT OPERATING RULES — READ CAREFULLY:

1. CONTEXT ONLY: You MUST answer exclusively using the context passages
   provided below. Do NOT use any knowledge from your training data,
   prior conversations, or external sources.

2. CITATION REQUIRED: Every factual statement in your answer MUST end with
   a source citation in exactly this format:
       [Source: <filename>, Page <number>]
   If the page is unknown, write:
       [Source: <filename>, Page N/A]

3. NO FABRICATION: Do NOT invent, extrapolate, or estimate any statistics,
   figures, dates, names, or claims beyond what is explicitly stated in the
   context passages.

4. NOT FOUND RESPONSE: If the provided context passages do not contain
   sufficient information to answer the question, respond with EXACTLY
   this sentence and nothing else:
       I cannot find this information in the uploaded documents.

4a. EXPLANATORY QUERIES: For requests like "explain", "summarize", or
    "what is this paper about", if relevant context passages are present,
    you MUST provide a grounded summary from those passages (with citations)
    and MUST NOT return the not-found sentence.

5. PARTIAL ANSWERS: If the context only partially answers the question,
   answer only what can be supported, cite those sources, and state clearly
   which part of the question could not be answered from the available documents.

6. NO OPINION: Do not express opinions, recommendations, or speculative
   analysis beyond what is directly supported by the context.

7. FORMAT: Respond in clear, professional prose. Use bullet points only
   when the source material is itself structured as a list.

8. CONCISE AND DIRECT: Answer the question directly in 2–4 sentences maximum.
   Do NOT add background context, preamble, or explanation beyond what the
   question asks. Do NOT restate the question. Lead immediately with the answer.
   If the answer requires a list, use at most 4 bullet points.
"""

# ---------------------------------------------------------------------------
# Main RAG prompt template
# ---------------------------------------------------------------------------

_RAG_PROMPT_TEMPLATE: str = """\
{system_instructions}

=== CONTEXT PASSAGES ===
{context}
=== END OF CONTEXT ===

QUESTION:
{question}

ANSWER (cite every claim with [Source: filename, Page N]):
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_rag_prompt(question: str, context: str) -> str:
    """
    Assemble the final prompt string to send to the LLM.

    Args:
        question : The user's natural-language question.
        context  : Pre-assembled context string from ContextAssembler,
                   containing source-labelled chunk passages.

    Returns:
        A fully-formatted prompt string ready for Ollama.

    Raises:
        ValueError: If question or context are empty.
    """
    if not question or not question.strip():
        raise ValueError("Question must not be empty.")
    if not context or not context.strip():
        raise ValueError(
            "Context is empty — retrieval must have failed before prompt build."
        )

    prompt = _RAG_PROMPT_TEMPLATE.format(
        system_instructions=_SYSTEM_INSTRUCTIONS.strip(),
        context=context.strip(),
        question=question.strip(),
    )
    return prompt


# ---------------------------------------------------------------------------
# Specialised prompt variants
# ---------------------------------------------------------------------------

def build_comparison_prompt(question: str, context: str) -> str:
    """
    Variant for comparative questions (e.g. 'Compare our revenue vs competitor X').
    Adds a structured comparison instruction block.
    """
    comparison_addon = """
COMPARISON FORMAT INSTRUCTION:
When comparing entities, structure your answer as follows:
  • [Entity A] — <finding> [Source: ..., Page N]
  • [Entity B] — <finding> [Source: ..., Page N]
  • Key Difference: <synthesis> [Source: ..., Page N]
Only include entities and figures explicitly mentioned in the context passages.
"""
    base_prompt = build_rag_prompt(question, context)
    # Insert the addon before the QUESTION line
    return base_prompt.replace(
        "QUESTION:", comparison_addon.strip() + "\n\nQUESTION:", 1
    )


def build_summarization_prompt(question: str, context: str) -> str:
    """
    Variant for document summarization requests.
    Constrains output length and enforces section-level citations.
    """
    summary_addon = """
SUMMARIZATION INSTRUCTION:
- Provide a structured summary with NO MORE THAN 3 bullet points.
- Each bullet must represent one distinct, specific finding from the context.
- Every bullet must end with [Source: filename, Page N].
- Keep each bullet to one sentence only.
- Do NOT repeat information across bullets.
- Do NOT introduce background knowledge or general industry context.
"""
    base_prompt = build_rag_prompt(question, context)
    return base_prompt.replace(
        "QUESTION:", summary_addon.strip() + "\n\nQUESTION:", 1
    )


# ---------------------------------------------------------------------------
# Prompt introspection utilities (for testing / debugging)
# ---------------------------------------------------------------------------

def extract_citations_from_response(response_text: str) -> list[dict]:
    """
    Parse all [Source: ...] citations from a generated response.

    Returns:
        List of dicts: [{"raw": "[Source: report.pdf, Page 3]",
                         "filename": "report.pdf", "page": "3"}, ...]
    """
    full_pattern = r"\[Source:\s*([^,\]]+?)(?:,\s*Page\s*([^\]]+))?\]"
    matches = re.findall(full_pattern, response_text, re.IGNORECASE)
    citations = []
    for filename, page in matches:
        citations.append({
            "raw": f"[Source: {filename.strip()}, Page {page.strip() or 'N/A'}]",
            "filename": filename.strip(),
            "page": page.strip() if page else "N/A",
        })
    return citations


def count_uncited_sentences(response_text: str) -> int:
    """
    Heuristic: count sentences in response that lack a [Source:] tag.
    Used in testing to measure citation coverage.
    """
    sentences = re.split(r"(?<=[.!?])\s+", response_text.strip())
    uncited = sum(
        1 for s in sentences
        if s.strip()
        and not re.search(CITATION_PATTERN, s, re.IGNORECASE)
        and len(s.split()) > 5   # ignore very short transitional phrases
    )
    return uncited


# ---------------------------------------------------------------------------
# Prompt catalogue (for UI display / documentation)
# ---------------------------------------------------------------------------

PROMPT_CATALOGUE: dict[str, str] = {
    "standard_rag": "Default Q&A prompt with strict context-only rules.",
    "comparison": "Structured comparison of two or more entities from documents.",
    "summarization": "Bullet-point summary of uploaded document content.",
}

PROMPT_BUILDERS: dict[str, object] = {
    "standard_rag": build_rag_prompt,
    "comparison": build_comparison_prompt,
    "summarization": build_summarization_prompt,
}


def get_prompt_builder(mode: str = "standard_rag"):
    """
    Factory: return the appropriate prompt builder function by mode name.

    Args:
        mode: One of 'standard_rag', 'comparison', 'summarization'.

    Returns:
        Callable (question, context) -> str
    """
    builder = PROMPT_BUILDERS.get(mode)
    if builder is None:
        raise ValueError(
            f"Unknown prompt mode: '{mode}'. "
            f"Valid options: {list(PROMPT_BUILDERS.keys())}"
        )
    return builder