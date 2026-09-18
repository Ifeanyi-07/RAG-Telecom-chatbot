"""
Telecom Customer Care Assistant - Streamlit App
------------------------------------------------
A chat UI for the RAG-based telecom support chatbot (Chroma + Groq).

Features:
- Conversation memory: full chat history is both displayed and passed to the
  LLM as context for follow-up questions
- Source citations for every claim (FAQ #, Ticket #, or Guide page number)
- Confidence-based fallback: if nothing relevant enough is retrieved, the app
  skips the LLM entirely and shows a canned "call 611" message
- Sample questions in the sidebar for one-click testing

Run with:
    streamlit run telecom_streamlit_app.py

Requires retriever.py and rag_chain.py in the same folder, a populated
chroma_store/ directory (run ingest_faq.py, ingest_pdf.py, and
ingest_tickets.py first), and a GROQ_API_KEY set as an environment variable
(a local .env file, or a Railway/host environment variable in production).
The key is never shown or editable in the UI.
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

st.set_page_config(page_title="Telecom Support Assistant", page_icon="📶", layout="wide")

# ----------------------------------------------------------------------
# API key check — read purely from the environment, never shown or
# editable in the UI. Checked BEFORE rag_chain is imported, since ChatGroq
# reads the key at construction time.
# ----------------------------------------------------------------------
if not os.getenv("GROQ_API_KEY"):
    st.title("📶 Telecom Support Assistant")
    st.error(
        "GROQ_API_KEY is not set. Add it as an environment variable "
        "(a .env file locally, or a Railway/host variable in production)."
    )
    st.stop()

from rag_chain import answer_question  # noqa: E402
from langchain_core.messages import HumanMessage, AIMessage  # noqa: E402

# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # HumanMessage/AIMessage list sent to the LLM as memory
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []  # [{"role", "content"}] for rendering
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

SAMPLE_QUESTIONS = [
    "How do I check my remaining data balance?",
    "My phone shows 'No SIM card detected', what should I do?",
    "How do I unlock my phone to use another carrier?",
    "I was charged twice for the same month, what happened?",
    "Wifi calling keeps disconnecting, how do I fix it?",
    "How do I set up my APN for mobile data?",
]

# ----------------------------------------------------------------------
# Sidebar: sample questions + reset
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Telecom Support Assistant")
    st.divider()
    st.subheader("💡 Try asking")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"sample_{q}"):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🔄 New conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.display_messages = []
        st.rerun()

# ----------------------------------------------------------------------
# Main chat UI
# ----------------------------------------------------------------------
st.title("📶 Telecom Support Assistant")
st.caption("Ask about billing, SIM issues, data, coverage, or anything from our FAQ and support history.")

for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

typed_question = st.chat_input("Describe your issue...")
question = st.session_state.pending_question or typed_question
st.session_state.pending_question = None  # consume so it doesn't re-fire on the next rerun

if question:
    st.session_state.display_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            for chunk in answer_question(question, chat_history=st.session_state.chat_history):
                full_response += chunk
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Sorry, something went wrong: {e}"
            placeholder.markdown(full_response)

    st.session_state.display_messages.append({"role": "assistant", "content": full_response})
    st.session_state.chat_history.append(HumanMessage(content=question))
    st.session_state.chat_history.append(AIMessage(content=full_response))
