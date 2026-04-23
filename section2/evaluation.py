"""
section2/evaluation.py
Precision@3 evaluation harness for the RAG pipeline.
Runs retrieval-only (no LLM call needed) to measure whether
the correct chunk appears in top-3 retrieved results.

Usage:
    python -m section2.evaluation
"""
from __future__ import annotations
import json
import logging
from section2.ingestion import PDFIngestionPipeline
from section2.vectorstore import VectorStore
from section2.retrieval import HybridRetriever

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during eval

# 10 manually written Q&A pairs covering all 3 sample documents.
# expected_content: substring that MUST appear in top-3 retrieved chunks.
QA_PAIRS = [
    {
        "id": "q01",
        "question": "What is the notice period for termination in the NDA with Vendor A?",
        "expected_content": "thirty",
        "source_doc": "NDA_Vendor_A.pdf",
    },
    {
        "id": "q02",
        "question": "Who are the parties to the NDA with Vendor A?",
        "expected_content": "articulate technologies",
        "source_doc": "NDA_Vendor_A.pdf",
    },
    {
        "id": "q03",
        "question": "What is the definition of confidential information in the NDA?",
        "expected_content": "confidential information",
        "source_doc": "NDA_Vendor_A.pdf",
    },
    {
        "id": "q04",
        "question": "What is the governing law clause in the NDA with Vendor A?",
        "expected_content": "laws of india",
        "source_doc": "NDA_Vendor_A.pdf",
    },
    {
        "id": "q05",
        "question": "What are the payment terms in the Service Agreement with Vendor B?",
        "expected_content": "thirty (30) days",
        "source_doc": "Service_Agreement_B.pdf",
    },
    {
        "id": "q06",
        "question": "What is the limitation of liability in the Service Agreement?",
        "expected_content": "liability",
        "source_doc": "Service_Agreement_B.pdf",
    },
    {
        "id": "q07",
        "question": "What are the termination conditions in the Service Agreement with Vendor B?",
        "expected_content": "terminat",
        "source_doc": "Service_Agreement_B.pdf",
    },
    {
        "id": "q08",
        "question": "What is the data retention schedule in the privacy policy?",
        "expected_content": "seven (7) years",
        "source_doc": "Policy_Document_C.pdf",
    },
    {
        "id": "q09",
        "question": "What security measures are required under the data privacy policy?",
        "expected_content": "aes-256",
        "source_doc": "Policy_Document_C.pdf",
    },
    {
        "id": "q10",
        "question": "What is the breach notification timeline in the privacy policy?",
        "expected_content": "seventy-two (72) hours",
        "source_doc": "Policy_Document_C.pdf",
    },
]


def run_evaluation(
    pdf_dir: str = "section2/sample_docs",
    persist_dir: str = "./chroma_store",
) -> dict:
    # Setup
    chunks = PDFIngestionPipeline().ingest_directory(pdf_dir)
    vs = VectorStore(persist_dir)
    vs.add_chunks(chunks)

    corpus = [
        {"text": c.text, "document": c.document,
         "page": c.page, "chunk_index": c.chunk_index, "score": 0.0}
        for c in chunks
    ]
    retriever = HybridRetriever(use_reranker=True)
    retriever.build_bm25_index(corpus)

    # Evaluate
    hits = 0
    results = []

    print(f"\nRunning Precision@3 evaluation — {len(QA_PAIRS)} questions\n")
    print(f"{'ID':<6} {'Status':<6} {'Question':<55} {'Expected'}")
    print("-" * 100)

    for pair in QA_PAIRS:
        dense = vs.query(pair["question"], n_results=10)
        top   = retriever.retrieve(pair["question"], dense, top_k=3)
        combined = " ".join([c["text"].lower() for c in top])
        hit = pair["expected_content"].lower() in combined
        hits += hit

        status = "HIT " if hit else "MISS"
        print(f"{pair['id']:<6} {status:<6} {pair['question'][:55]:<55} {pair['expected_content']}")

        results.append({
            "id":       pair["id"],
            "question": pair["question"],
            "hit":      hit,
            "sources":  [{"document": c["document"], "page": c["page"]}
                         for c in top],
        })

    precision_at_3 = hits / len(QA_PAIRS)
    print(f"\nPrecision@3: {hits}/{len(QA_PAIRS)} = {precision_at_3:.0%}\n")

    return {
        "precision_at_3": precision_at_3,
        "hits":           hits,
        "total":          len(QA_PAIRS),
        "results":        results,
    }


if __name__ == "__main__":
    report = run_evaluation()
    with open("section2/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report saved to section2/eval_report.json")
