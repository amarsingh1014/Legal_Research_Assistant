# Legal Research Assistant (Streamlit)

This repo contains a Streamlit-based chatbot that uses a BM25 retriever (via rank_bm25) over a pre-chunked legal dataset and a simple RAG pipeline to compose answers.

Files:
- `app.py` - Streamlit application. Loads `data/legal_chunks_conceptual.json` and `data/eval_dataset_conceptual.json`, builds BM25, and composes answers using OpenAI if configured.
- `.env` - example environment file. Add `OPENAI_API_KEY` here if you want LLM answers.
- `requirements.txt` - Python dependencies.

Run locally:

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

Notes:
- The app uses NLTK tokenization. If you see missing NLTK data errors, run Python and download punkt:

```python
import nltk
nltk.download('punkt')
```

- If OpenAI is not configured, the app falls back to returning concatenated retrieved excerpts and a simple placeholder answer.
# Legal_Research_Assistant
