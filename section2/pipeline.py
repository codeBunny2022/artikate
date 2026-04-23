from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any
from openai import OpenAI
from section2.ingestion import PDFIngestionPipeline
from section2.vectorstore import VectorStore
from section2.retrieval import HybridRetriever

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.30
GEMINI_BASE_URL      = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL         = "models/gemini-2.0-flash-lite"

SYSTEM_PROMPT = """You are a precise legal document analyst.
Answer questions using ONLY the context provided below. Do not use any external knowledge.

Rules:
1. Base every claim on the provided document excerpts only.
2. Always cite the document name and page number for every fact you state.
3. If the context does not contain enough information to answer, respond with exactly:
   "I cannot find this information in the available documents."
4. Be precise with legal terms. Do not paraphrase obligations loosely.

Context:
{context}
"""


class RAGPipeline:
    def __init__(self, persist_dir: str = "./chroma_store",
                 use_reranker: bool = True,
                 model: str = GEMINI_MODEL):
        self.model        = model
        self.vector_store = VectorStore(persist_dir)
        self.retriever    = HybridRetriever(use_reranker=use_reranker)
        self.ingestion    = PDFIngestionPipeline()
        self._corpus: list[dict] = []

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        self.llm = OpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)

    # ingestion

    def ingest_directory(self, directory: str | Path) -> int:
        chunks = self.ingestion.ingest_directory(directory)
        self.vector_store.add_chunks(chunks)
        self._corpus = [
            {"text": c.text, "document": c.document,
             "page": c.page, "chunk_index": c.chunk_index, "score": 0.0}
            for c in chunks
        ]
        self.retriever.build_bm25_index(self._corpus)
        return len(chunks)

    def ingest_pdf(self, pdf_path: str | Path) -> int:
        chunks = self.ingestion.ingest(pdf_path)
        self.vector_store.add_chunks(chunks)
        new = [{"text": c.text, "document": c.document,
                "page": c.page, "chunk_index": c.chunk_index, "score": 0.0}
               for c in chunks]
        self._corpus.extend(new)
        self.retriever.build_bm25_index(self._corpus)
        return len(chunks)

    # query

    def query(self, question: str,
              filter_document: str | None = None) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if self.vector_store.count() == 0:
            raise RuntimeError("No documents ingested. Call ingest_directory() first.")

        # stage 1: dense retrieval — preserving scores here
        dense = self.vector_store.query(question, n_results=20,
                                        filter_document=filter_document)
        if not dense:
            return self._refusal("No relevant documents found.")

        # dense score lookup by (document, chunk_index)
        # these are cosine similarities in [0,1] — reliable confidence signal
        dense_score_lookup: dict[tuple, float] = {
            (c["document"], c["chunk_index"]): c["score"]
            for c in dense
        }

        # stage 2: hybrid retrieval + reranking
        # reranker reorders chunks but its raw scores are NOT calibrated
        # confidence values — we use dense scores for confidence instead
        top_chunks = self.retriever.retrieve(question, dense, top_k=3)

        # stage 3: confidence from dense scores (not rerank scores)
        confidence = self._confidence(top_chunks, dense_score_lookup)
        logger.info("Confidence: %.3f (threshold: %.2f)", confidence, CONFIDENCE_THRESHOLD)

        if confidence < CONFIDENCE_THRESHOLD:
            logger.warning("Refusing — confidence %.3f below threshold", confidence)
            return self._refusal("Insufficient context to answer reliably.",
                                 confidence=confidence)

        # stage 4: LLM generation
        context = self._format_context(top_chunks)
        answer  = self._generate(question, context)

        if "cannot find this information" in answer.lower():
            return self._refusal(answer, confidence=confidence)

        sources = [
            {"document": c["document"], "page": c["page"], "chunk": c["text"]}
            for c in top_chunks
        ]
        return {"answer": answer, "sources": sources,
                "confidence": round(confidence, 4)}

    # helpers

    @staticmethod
    def _confidence(chunks: list[dict],
                    dense_score_lookup: dict[tuple, float]) -> float:
        """
        Use dense cosine similarity scores for confidence.
        Reranker scores are relative ordering logits — not calibrated
        confidence values, so we do not use them here.
        Average across top-k retrieved chunks.
        """
        if not chunks:
            return 0.0
        scores = [
            dense_score_lookup.get((c["document"], c["chunk_index"]), 0.0)
            for c in chunks
        ]
        return float(sum(scores) / len(scores))

    def _generate(self, question: str, context: str) -> str:
        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user",
                     "content": SYSTEM_PROMPT.format(context=context)
                                + "\n\nQuestion: " + question},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            raise RuntimeError(f"LLM generation failed: {e}") from e

    @staticmethod
    def _format_context(chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(
                f"[Source {i}] Document: {c['document']} | Page: {c['page']}\n{c['text']}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _refusal(reason: str = "Insufficient context.",
                 confidence: float = 0.0) -> dict[str, Any]:
        return {
            "answer": "I cannot find this information in the available documents."
                      f" ({reason})",
            "sources": [],
            "confidence": round(confidence, 4),
        }
