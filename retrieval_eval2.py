"""
Retrieval quality eval for the 'tickets' collection.

Measures top-K recall: for each hand-crafted (question, expected_ticket_id)
pair, was the correct ticket retrieved within the top K results?

TEST_CASES below are drawn from the 19 real resolved tickets in tickets.db
(10 of the 19, spanning connectivity, sim, billing, roaming, voice, and data
categories) — each question is phrased the way a customer might actually ask,
paired with that ticket's real ticket_id.

Run with:
    python eval/retrieval_eval.py
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


CHROMA_DIR = "chroma_store"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

TEST_CASES = [
    {"question": "I lost all mobile internet after my phone switched from 3G to 4G",
        "expected_ticket_id": "TK-001"},
    {"question": "I bought a 5GB data add-on but my balance still shows the old amount",
        "expected_ticket_id": "TK-003"},
    {"question": "I got charged $240 more than expected after a trip to Spain even though I had a roaming bundle",
        "expected_ticket_id": "TK-004"},
    {"question": "My phone says SIM not provisioned after I updated my Android software",
        "expected_ticket_id": "TK-005"},
    {"question": "I was billed twice for my monthly plan on the same day",
        "expected_ticket_id": "TK-006"},
    {"question": "All my incoming calls go straight to voicemail even with full signal",
        "expected_ticket_id": "TK-007"},
    {"question": "The QR code for my eSIM activation keeps failing on my new iPhone",
        "expected_ticket_id": "TK-009"},
    {"question": "I have no signal at all while traveling in Japan even though I enabled roaming",
        "expected_ticket_id": "TK-012"},
    {"question": "I forgot my password and the reset email for the MyTelecom app never arrives",
        "expected_ticket_id": "TK-013"},
    {"question": "I hear an echo of my own voice with a delay on every call I make",
        "expected_ticket_id": "TK-017"},
]


def run_eval() -> None:
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    tickets_store = Chroma(
        collection_name="tickets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    hits = 0
    for i, case in enumerate(TEST_CASES, start=1):
        results = tickets_store.similarity_search_with_score(
            case["question"], k=TOP_K)
        retrieved_ids = [doc.metadata.get("ticket_id") for doc, _ in results]

        hit = case["expected_ticket_id"] in retrieved_ids
        if hit:
            hits += 1

        status = "HIT " if hit else "MISS"
        print(f'[{status}] Test {i}: "{case["question"]}"')
        print(f"    Expected: {case['expected_ticket_id']}")
        print(f"    Retrieved: {retrieved_ids}")
        print()

    recall = hits / len(TEST_CASES)
    print(f"Top-{TOP_K} recall: {hits}/{len(TEST_CASES)} = {recall:.0%}")


if __name__ == "__main__":
    run_eval()
