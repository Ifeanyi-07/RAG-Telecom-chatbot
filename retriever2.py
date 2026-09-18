"""
Builds retrieval across all three Chroma collections:
  - faq     : FAQ entries (no chunking — 1 row = 1 doc)
  - tickets : resolved support tickets (no chunking — 1 ticket = 1 doc)
  - guides  : PDF guide chunks (RecursiveCharacterTextSplitter applied at ingest)

Unlike the original version, this exposes similarity_search_with_score()
results (Document, score) instead of a plain retriever, so the caller can
apply confidence-based fallback logic before ever calling the LLM.

Note on scores: with the default Chroma configuration used by the ingest
scripts (no explicit distance metric set at collection creation), Chroma
returns an L2 (squared Euclidean) distance from similarity_search_with_score()
— LOWER means MORE similar, and there's no fixed upper bound. The
CONFIDENCE_THRESHOLD used downstream (in rag_chain.py) has not been tuned for
your specific data — see eval/retrieval_eval.py as a starting point for
calibrating it against real scores from your own collections.
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Lazily initialized so importing this module doesn't immediately load the
# embedding model or open the Chroma stores.
_embeddings = None
_stores = None


def _get_stores() -> dict:
    global _embeddings, _stores
    if _stores is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        _stores = {
            "faq": Chroma(
                collection_name="faq",
                embedding_function=_embeddings,
                persist_directory=CHROMA_DIR,
            ),
            "tickets": Chroma(
                collection_name="tickets",
                embedding_function=_embeddings,
                persist_directory=CHROMA_DIR,
            ),
            "guides": Chroma(
                collection_name="guides",
                embedding_function=_embeddings,
                persist_directory=CHROMA_DIR,
            ),
        }
    return _stores


def retrieve_with_scores(
    query: str,
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
) -> list[tuple[Document, float]]:
    """
    Query all three collections and return a single merged list of
    (Document, score) pairs, sorted by score ascending (lower = more similar,
    per Chroma's default L2 distance — see module docstring).
    """
    stores = _get_stores()

    results: list[tuple[Document, float]] = []
    results += stores["faq"].similarity_search_with_score(query, k=k_faq)
    results += stores["tickets"].similarity_search_with_score(query, k=k_tickets)
    results += stores["guides"].similarity_search_with_score(query, k=k_guides)

    results.sort(key=lambda pair: pair[1])
    return results
