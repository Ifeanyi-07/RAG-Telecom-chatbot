# 📶 Telecom Customer Care Assistant (RAG)

A Retrieval-Augmented Generation (RAG) chatbot for telecom customer support, built with LangChain, Chroma, and Groq. It answers customer questions by pulling from three knowledge sources — an FAQ, a PDF product guide, and a database of resolved support tickets — and cites exactly where each piece of information came from. Available both as a command-line chatbot and a deployed Streamlit web app.

**🔗 Live app:** https://telecom-chatbot-production-0aa2.up.railway.app/

---

## Features

- **Multi-source retrieval** — searches FAQ entries, past resolved tickets, and a chunked PDF guide, then merges the results.
- **Source citations** — every claim in the answer is tagged with where it came from: `[FAQ #12]`, `[Ticket #TK-007]`, or `[Guide, page 4]`.
- **Confidence-based fallback** — if nothing retrieved is similar enough to the question, the LLM is skipped entirely and the customer gets a canned "please call 611" response instead of a guessed answer.
- **Conversation memory** — the chat remembers earlier turns in the same session, both for display and as context for the LLM.
- **Sample questions** — one-click example queries in the sidebar.
- **Evals included** — a retrieval recall test against real resolved tickets, so you can measure retrieval quality after changing chunking, embeddings, or k values.
- API key is read only from the server environment — never shown, editable, or shared in the UI.

---

## Project structure

```
.
├── main.py                       # CLI entry point
├── app2.py      # Streamlit web UI (entry point, chat + sample questions + memory)
├── rag_chain2.py                  # Prompt, citations, confidence/fallback logic, LLM call
├── retriever2.py                  # Merged retrieval across faq/tickets/guides with similarity scores
├── ingest_faq.py                 # One-time script: embeds data/faq.csv into Chroma
├── ingest_pdf.py                 # One-time script: chunks + embeds data/telecom_guide.pdf into Chroma
├── ingest_tickets.py             # One-time script: embeds resolved tickets from data/tickets.db into Chroma
├── requirements.txt  # Python dependencies
├── Procfile                      # Railway start command
├── .gitignore                    # Keeps .env and local files out of git
├── retrieval_eval2.py         # Top-3 recall eval against real resolved tickets
├── data/
│   ├── faq.csv
│   ├── telecom_guide.pdf
│   └── tickets.db
├── chroma_store/                 # Pre-built vector database — committed to the repo (see Deployment)
└── README.md                     # This file
```

---

## Requirements

- Python 3.9+
- A [Groq API key](https://console.groq.com/keys)
- Your source data in place: `data/faq.csv`, `data/telecom_guide.pdf`, `data/tickets.db`

---

## Local setup

1. **Clone the repo and enter the project folder**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. **(Recommended) Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API key**

   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_key_here
   ```

5. **Build the vector store** (run once, and again whenever the source data changes)
   ```bash
   python ingest_faq.py
   python ingest_pdf.py
   python ingest_tickets.py
   ```
   This creates a `chroma_store/` directory with three collections: `faq`, `guides`, and `tickets`.

---

## Running locally

### Command line
```bash
python main.py
```
Type your question and press Enter. Type `quit` to exit.

### Streamlit web app
```bash
streamlit run app2.py
```
Opens in your browser, usually at `http://localhost:8501`. Includes a chat window with full conversation memory, sidebar buttons with sample questions, and a **New conversation** reset button.

---

## Example interaction

```
Customer: My phone shows 'No SIM card detected', what should I do?

Assistant: Try restarting your phone first, as this often resolves temporary
SIM detection issues (FAQ #8). If that doesn't work, this matches a known
case where the SIM lost provisioning after a software update — our team
re-provisioned it remotely, which resolved it within 10 minutes without a
store visit (Ticket #TK-005). If the issue persists, please call 611.

Customer: what's the weather like today

Assistant: I'm sorry, I don't have enough information to answer that
confidently. Please call 611 or use the MyTelecom app for further assistance.
```

(The second example triggers the confidence-based fallback since nothing in the knowledge base is relevant — the LLM is never called for it.)

---

## How the pieces fit together

1. **`retriever2.py`** queries all three Chroma collections with `similarity_search_with_score()` and returns a merged, score-sorted list of `(Document, score)` pairs. Chroma's default distance is L2 — **lower score = more similar**.
2. **`rag_chain2.py`** filters that list against `CONFIDENCE_THRESHOLD`. If nothing clears it, it returns the fallback message and skips the LLM. Otherwise it formats the surviving chunks with citation tags and streams an answer from Groq, with the conversation history included for context.
3. **`main.py`** and **`telecom_streamlit_app.py`** are just two different front ends calling the same `answer_question()` / `build_chain()` logic.

---

## Deployment (Railway)

This app is deployed on [Railway](https://railway.app). One thing about this app makes it different to deploy compared to a stateless chatbot: it depends on a **pre-built vector database** (`chroma_store/`), not just source code.

1. Run the three ingest scripts locally first, so `chroma_store/` exists.
2. **Commit `chroma_store/` to the repo.** Railway builds fresh from git on every deploy — if this folder isn't pushed, the app will start fine but every question will hit the fallback message, since there's nothing to retrieve. Double-check `.gitignore` doesn't have a stray `chroma_store/` line excluding it.
3. Push `app2.py`, `rag_chain2.py`, `retriever2.py`, `main.py`, the ingest scripts, `requirements.txt`, and `Procfile` to GitHub. Keeping `data/` in the repo too is recommended (not required at runtime, but needed if you ever re-run ingestion).
4. On Railway: **New Project → Deploy from GitHub repo** and select the repo.
5. In **Variables**, add `GROQ_API_KEY` with your real key. It's read from the environment only — never entered or displayed in the app itself.
6. Check **Settings → Deploy** for a **Custom Start Command**. If one is set, it overrides the Procfile — clear it, or make sure it matches exactly:
   ```
   web: streamlit run app2.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
   ```
7. Once **Active**, go to **Settings → Networking → Generate Domain** for a public URL.

**Note on first-request latency:** the embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is downloaded from Hugging Face the first time `retriever.py` runs in a fresh container — expect the very first question after a deploy to be noticeably slower than the rest.

---

## Tuning the confidence threshold

`CONFIDENCE_THRESHOLD` in `rag_chain.py` is a starting value (`1.0`) and has **not** been calibrated against your specific embedding model and data. To tune it:
1. Run `eval/retrieval_eval.py` and note the scores printed for both correct and incorrect matches.
2. Try a few real customer-style questions through `retriever.retrieve_with_scores()` directly and print the scores.
3. Adjust the threshold so genuinely relevant chunks pass and irrelevant ones get filtered out.

---

## Running the eval

```bash
python retrieval_eval2.py
```

Measures top-3 recall: for 10 hand-crafted questions (based on real tickets in `data/tickets.db`), was the correct ticket retrieved in the top 3 results? Useful for catching regressions after changing the embedding model, chunk size, or `k` values.

---

## Security notes

- The API key is **never hardcoded** and **never entered through the UI** — it's read exclusively from the `GROQ_API_KEY` environment variable (`.env` locally, a Railway variable in production).
- `.gitignore` excludes `.env` so the key can never be accidentally committed.
- If sharing this repo publicly, include a `.env.example` with a placeholder value instead of a real key, and consider whether `data/tickets.db` or `chroma_store/` contain anything sensitive before making the repo public.

---

