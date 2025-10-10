# Legal Research Assistant (Streamlit)

This repo contains a Streamlit-based chatbot that uses a BM25 retriever (via rank_bm25) over a pre-chunked legal dataset and a simple RAG pipeline to compose answers.

## Experimentations in Notebooks

### 1. `notebooks/create_eval_dataset.ipynb`
This notebook focuses on creating an evaluation dataset from the provided legal PDF (`data/legal_summaries.pdf`).

**Steps:**
- **Ollama Setup:** Installs and runs Ollama locally, pulls the Mistral model for question generation.
- **PDF Processing:** Extracts text from the PDF using PyPDF, cleans it, and splits into individual case summaries using regex patterns (e.g., detecting "X v. Y" case names).
- **Chunking:** Groups cases into larger chunks (4 cases each) to create meaningful retrieval units.
- **Question Generation:** For each chunk, uses the Mistral LLM via Ollama to generate 2-3 research-oriented questions focused on legal principles (e.g., rules for fee recovery, arbitration rights) rather than specific cases. Prompts emphasize conceptual questions suitable for a search assistant.
- **Output:** Saves `data/legal_chunks_conceptual.json` (chunked documents) and `data/eval_dataset_conceptual.json` (generated questions with relevant chunk IDs).

**Purpose:** Creates a synthetic evaluation dataset for testing retrieval systems, ensuring questions are grounded in the legal domain but abstract enough for general research.

### 2. `notebooks/retrieval_and_rag.ipynb`
This notebook implements and compares multiple retrieval techniques for the legal research assistant scenario, using the chunks and eval dataset created above.

**Implemented Techniques:**
- **Naive Keyword Search:** Simple string matching on document text, returning snippets with query highlights.
- **BM25 Retrieval:** Probabilistic retrieval using term frequency-inverse document frequency (TF-IDF) weighting. Tokenizes documents and queries with NLTK, ranks chunks by relevance scores.
- **Dense Retrieval:** Semantic search using vector embeddings. Embeds documents and queries with a sentence transformer (e.g., all-MiniLM-L6-v2), stores in Qdrant vector database, performs cosine similarity search.
- **Retrieval-Augmented Generation (RAG):** Combines BM25 retrieval with LLM generation. Retrieves top chunks, joins them with "\n---\n", feeds to ChatGroq (GROQ API) with a strict prompt to answer only based on provided context.
- **Hybrid Retrieval:** Merges BM25 and Dense results using Reciprocal Rank Fusion (RRF), which combines rankings from both methods for improved performance.

**Evaluation:**
- Uses the eval dataset (`eval_dataset_conceptual.json`) to benchmark techniques.
- Metrics: Precision@K, Recall@K, Mean Reciprocal Rank (MRR) at K=5.
- Compares BM25, Dense, and Hybrid retrieval across all queries, reporting average scores.
- Demonstrates how advanced techniques outperform naive search, with hybrid often yielding the best results by leveraging keyword and semantic matching.

**Key Findings:**
- BM25 excels in exact keyword matching.
- Dense retrieval captures semantic similarity, better for paraphrased queries.
- Hybrid (RRF) balances both, often achieving highest precision/recall.
- RAG improves answer quality by synthesizing retrieved info into coherent responses.

These experiments show the progression from basic to advanced retrieval, highlighting trade-offs and the value of combining methods for legal research.

## Files
- `app.py` - Streamlit application. Loads `data/legal_chunks_conceptual.json` and `data/eval_dataset_conceptual.json`, builds BM25, and composes answers using Groq if configured.
- `.env` - example environment file. Add `GROQ_API_KEY` here for LLM answers.
- `requirements.txt` - Python dependencies.
- `notebooks/` - Jupyter notebooks for data preparation and retrieval experiments.

## Run locally

1. Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Add your Groq API KEY to `.env` or export it:

```bash
export GROQ_API_KEY="sk-..."
```

3. Start the Streamlit app:

```bash
streamlit run app.py
```

## Notes
- The app uses NLTK tokenization. If you see missing NLTK data errors, run Python and download punkt:

```python
import nltk
nltk.download('punkt')
```

- If Groq is not configured, the app falls back to returning concatenated retrieved excerpts and a simple placeholder answer.
# Legal_Research_Assistant
