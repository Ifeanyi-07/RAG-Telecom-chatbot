"""
Builds the RAG answering logic:
  retrieve_with_scores → confidence check → prompt (with citations + history)
  → Qwen/gpt-oss-120b on Groq → streamed answer

Exposes `answer_question(question, chat_history=None)`, a generator that
yields response chunks as strings. When no retrieved chunk clears
CONFIDENCE_THRESHOLD, it yields a single canned fallback message and never
calls the LLM at all.

`build_chain()` is kept for backward compatibility with main.py's original
`chain.stream(question)` interface.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_groq import ChatGroq

from retriever2 import retrieve_with_scores

# ---------------------------------------------------------------------------
# Confidence / fallback config
# ---------------------------------------------------------------------------
# See retriever.py's module docstring: this is an L2 distance, LOWER = more
# similar. This starting value has NOT been tuned for your data — adjust it
# based on eval/retrieval_eval.py and by printing scores for real queries.
CONFIDENCE_THRESHOLD = 1.0

FALLBACK_MESSAGE = (
    "I'm sorry, I don't have enough information to answer that confidently. "
    "Please call 611 or use the MyTelecom app for further assistance."
)

SYSTEM_PROMPT = """You are a helpful and professional telecom customer care assistant.
Your job is to help customers resolve technical issues with their mobile service.

Use ONLY the context below to answer the customer's question.
The context comes from three sources, each labeled with a citation tag:
- [FAQ #<id>] — general policy and how-to information
- [Ticket #<id>] — real resolved support cases with step-by-step resolutions
- [Guide, page <n>] — the official telecom guide document

For every piece of information you use in your answer, cite its source
inline using the SAME tag format shown above. For example:
"You can reset your APN settings under Settings > Network (Guide, page 4)."
or "This matches a known issue we've resolved before (Ticket #245)."

If the context does not contain enough information to answer confidently,
say so clearly and suggest the customer call 611 or use the MyTelecom app.

Context:
{context}
"""

# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def _citation_tag(doc: Document) -> str:
    source = doc.metadata.get("source", "unknown")
    if source == "faq":
        return f"[FAQ #{doc.metadata.get('faq_id', '?')}]"
    if source == "ticket":
        return f"[Ticket #{doc.metadata.get('ticket_id', '?')}]"
    if source == "guide":
        page = doc.metadata.get("page")
        page_display = page + 1 if isinstance(page, int) else "?"
        return f"[Guide, page {page_display}]"
    return "[Unknown source]"


def _format_docs(docs: list[Document]) -> str:
    sections = []
    for doc in docs:
        tag = _citation_tag(doc)
        sections.append(f"{tag}\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


# ---------------------------------------------------------------------------
# Prompt / LLM (lazily built so importing this module doesn't require an
# API key to already be set)
# ---------------------------------------------------------------------------

_prompt = None
_llm = None


def _get_prompt_and_llm():
    global _prompt, _llm
    if _prompt is None:
        _prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{question}"),
            ]
        )
        _llm = ChatGroq(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )
    return _prompt, _llm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    chat_history: list | None = None,
    k_faq: int = 3,
    k_tickets: int = 3,
    k_guides: int = 3,
):
    """
    Retrieve context for `question`, apply confidence-based fallback, and
    yield the answer as a stream of string chunks.

    `chat_history`, if given, is a list of langchain_core.messages
    (HumanMessage/AIMessage) representing prior turns — used to give the LLM
    conversational memory. Retrieval itself is still based only on the
    current `question` (no query rewriting from history).

    If no retrieved chunk clears CONFIDENCE_THRESHOLD, this yields a single
    canned fallback message and the LLM is never called.
    """
    scored_docs = retrieve_with_scores(
        question, k_faq=k_faq, k_tickets=k_tickets, k_guides=k_guides)

    confident_docs = [doc for doc,
                      score in scored_docs if score <= CONFIDENCE_THRESHOLD]

    if not confident_docs:
        yield FALLBACK_MESSAGE
        return

    prompt, llm = _get_prompt_and_llm()
    context = _format_docs(confident_docs)

    payload = {"context": context, "question": question}
    if chat_history:
        payload["chat_history"] = chat_history

    chain = prompt | llm
    for chunk in chain.stream(payload):
        if chunk.content:
            yield chunk.content


def build_chain():
    """
    Backward-compatible wrapper matching the original interface main.py
    expects: an object with a .stream(question) method that yields string
    chunks (with no citations/fallback config needed on the caller's side).
    """

    class _Chain:
        def stream(self, question: str):
            return answer_question(question)

    return _Chain()
