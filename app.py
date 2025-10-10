import os
import json
from pathlib import Path
from typing import List

import streamlit as st
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv

import nltk

# Ensure necessary tokenizer resources are downloaded
nltk.download('punkt')
nltk.download('punkt_tab')

# Load environment variables from .env if present
load_dotenv()

# Prefer the exact LLM setup used in the notebook: langchain_groq.ChatGroq
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False



BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_PATH = DATA_DIR / "legal_chunks_conceptual.json"
EVAL_PATH = DATA_DIR / "eval_dataset_conceptual.json"


@st.cache_data
def load_chunks() -> List[dict]:
    with open(CHUNKS_PATH, "r") as f:
        data = json.load(f)
    return data


@st.cache_data
def load_eval() -> List[dict]:
    with open(EVAL_PATH, "r") as f:
        data = json.load(f)
    return data


@st.cache_data
def build_bm25(corpus_texts: List[str]):
    tokenized = [word_tokenize(t.lower()) for t in corpus_texts]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


def llm_chain_answer(question: str, retrieved_texts: List[str]) -> str:
    """Compose a short prompt and call the LLM configured in the notebook (ChatGroq) if available.

    Falls back to OpenAI if configured, and finally to a concatenation heuristic.
    """
    # Prepare context for RAG: join retrieved chunks with '\n---\n'
    context_for_llm = "\n---\n".join(retrieved_texts)

    prompt_for_llm = f"""
Based on the following context, please answer the user's question.
If the answer cannot be found in the context, please state that you cannot answer based on the provided information.

Context:
{context_for_llm}

Question:
{question}

Answer:
"""

    # Use ChatGroq (GROQ) only — per notebook setup. If unavailable or the call fails,
    # fall back to concatenating retrieved excerpts.
    if GROQ_AVAILABLE:
        try:
            llm = ChatGroq(model="openai/gpt-oss-20b")
            # Use the invoke pattern from your snippet
            llm_response = llm.invoke(prompt_for_llm)
            # Return the content field if present
            try:
                return llm_response.content
            except Exception:
                return str(llm_response)
        except Exception:
            # If GROQ call fails, we'll fall back to the simple concatenation below
            pass

    # Last-resort fallback: concatenated excerpts
    answer = "".join([t.strip() + "\n\n" for t in retrieved_texts[:3]])
    answer = f"(LLM not configured) Relevant excerpts:\n\n{answer}\nSuggested answer based on excerpts:\n{question}"
    return answer


def retrieve_bm25(bm25, tokenized_corpus, docs_texts, query: str, top_n: int = 5):
    q_tok = word_tokenize(query.lower())
    scores = bm25.get_scores(q_tok)
    import numpy as np

    idxs = np.argsort(scores)[-top_n:][::-1]
    retrieved = [docs_texts[i] for i in idxs]
    return retrieved, idxs, scores[idxs]


def main():
    st.set_page_config(page_title="Legal Research Assistant", layout="wide")

    st.sidebar.title("Legal Research Assistant")
    st.sidebar.markdown("""
    **BM25 retriever + RAG answer composer.**  
    **Domain:** New York State attorney-client fee arbitration.  
    **What it can answer:** Questions about rules, procedures, exceptions, and case law related to fee disputes between attorneys and clients under Parts 136, 137, 1400, etc. Covers topics like notice requirements, arbitration hearings, post-award procedures, and miscellaneous legal principles.  
    **Data source:** Case summaries and legal documents from the New York State Unified Court System Office of ADR Programs.
    """)
    # Environment status placed here, where the title is
    st.sidebar.write({
        "GROQ_CONFIGURED": bool(os.getenv("GROQ_API_KEY") or os.getenv("GROQ_KEY")),
        "GROQ_AVAILABLE": GROQ_AVAILABLE,
    })

    chunks = load_chunks()
    eval_dataset = load_eval()

    docs_texts = [c.get("text") for c in chunks]
    doc_ids = [c.get("chunk_id") for c in chunks]

    bm25, tokenized = build_bm25(docs_texts)

    # prepare suggested questions list
    import re
    suggested_questions = [re.sub(r'^\d+\.\s*', '', it.get('query')) for it in eval_dataset]

    # Init session state for suggestion rotation
    if 'suggestion_idx' not in st.session_state:
        st.session_state.suggestion_idx = 0
    if 'last_rotate_time' not in st.session_state:
        st.session_state.last_rotate_time = 0.0
    if 'auto_rotate' not in st.session_state:
        st.session_state.auto_rotate = True  # Default to auto-rotate
    rotate_interval = 5.0  # seconds between rotations

    # Data viewers in the main area, above the question box
    st.header("Data viewers")
    if st.checkbox("Show raw eval dataset"):
        st.json(eval_dataset)
    if st.checkbox("Show raw legal chunks"):
        st.json(chunks)

    st.title("Legal Research Chatbot")

    # Suggested questions via selectbox
    selected_suggestion = st.selectbox("Suggested questions (optional)", [""] + suggested_questions[:10])  # Show top 10
    if selected_suggestion:
        st.session_state.query = selected_suggestion

    # Typing box
    query = st.text_input("Ask a legal question", value=st.session_state.get('query', ''), key='input')

    top_n = st.slider("Number of retrieved chunks", min_value=1, max_value=10, value=5)

    if st.button("Search & Answer") and query.strip():
        with st.spinner("Retrieving relevant chunks..."):
            retrieved, idxs, scores = retrieve_bm25(bm25, tokenized, docs_texts, query, top_n=top_n)

        # show retrieved snippets
        st.subheader("Retrieved snippets (BM25)")
        for rank, (r, idx, s) in enumerate(zip(retrieved, idxs, scores), start=1):
            cid = doc_ids[idx]
            with st.expander(f"{rank}. Chunk {cid} — score: {float(s):.3f}", expanded=False):
                # Render snippet inside a light box; replace newlines with markdown line breaks
                snippet_html = (
                    "<div style='background:#f8f9fa;border-radius:6px;padding:8px'>"
                    + r[:1500].replace("\n", "  \n")
                    + "</div>"
                )
                st.markdown(snippet_html, unsafe_allow_html=True)

        # Compose answer using LLM or fallback
        with st.spinner("Composing answer with RAG..."):
            answer = llm_chain_answer(query, retrieved)

        st.subheader("Answer (rendered from markdown)")
        st.markdown(answer)


if __name__ == "__main__":
    main()
