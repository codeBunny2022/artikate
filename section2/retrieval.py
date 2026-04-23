from __future__ import annotations
import logging
from typing import Any
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


class HybridRetriever:
    """
    two-stage retrieval:
      Stage 1 -> BM25 + dense vector retrieval fused via Reciprocal Rank Fusion
      Stage 2 -> Cross-encoder reranking on fused candidate pool
    """

    def __init__(self, use_reranker: bool = True):
        self.use_reranker = use_reranker
        self._reranker: CrossEncoder | None = None
        self._corpus: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

        if use_reranker:
            logger.info("Loading reranker: %s", RERANKER_MODEL)
            self._reranker = CrossEncoder(RERANKER_MODEL, max_length=512)

    def build_bm25_index(self, chunks: list[dict[str, Any]]) -> None:
        """build BM25 index from chunk dicts. call after each ingest."""
        self._corpus = chunks
        tokenized = [c["text"].lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built over %d chunks", len(chunks))

    def retrieve(self, query: str, dense_candidates: list[dict[str, Any]],
                 top_k: int = 3) -> list[dict[str, Any]]:
        """
        full hybrid pipeline:
          1. BM25 retrieval over corpus
          2. RRF fusion with dense candidates
          3. Cross-encoder reranking
        returns top_k results.
        """
        bm25_candidates = self._bm25_retrieve(query, top_n=20)
        fused = self._rrf(dense_candidates, bm25_candidates)
        pool = fused[:20]

        if self.use_reranker and self._reranker and len(pool) > 1:
            pool = self._rerank(query, pool)

        return pool[:top_k]

    def _bm25_retrieve(self, query: str, top_n: int) -> list[dict[str, Any]]:
        if not self._bm25 or not self._corpus:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(range(len(scores)),
                             key=lambda i: scores[i], reverse=True)[:top_n]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = dict(self._corpus[idx])
                chunk["bm25_score"] = float(scores[idx])
                results.append(chunk)
        return results

    def _rrf(self, dense: list[dict], bm25: list[dict]) -> list[dict]:
        """reciprocal Rank Fusion -> no hyperparameter tuning needed"""
        rrf_scores: dict[str, float] = {}
        all_chunks: dict[str, dict] = {}

        for rank, chunk in enumerate(dense):
            cid = f"{chunk['document']}::chunk_{chunk['chunk_index']}"
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)
            all_chunks[cid] = chunk

        for rank, chunk in enumerate(bm25):
            cid = f"{chunk['document']}::chunk_{chunk['chunk_index']}"
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (RRF_K + rank + 1)
            all_chunks[cid] = chunk

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)
        fused = []
        for cid in sorted_ids:
            chunk = dict(all_chunks[cid])
            chunk["rrf_score"] = rrf_scores[cid]
            fused.append(chunk)
        return fused

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """cross-encoder sees full (query, chunk) pair -> more accurate than bi-encoder"""
        pairs = [(query, c["text"]) for c in candidates]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        result = []
        for chunk, score in ranked:
            chunk = dict(chunk)
            chunk["rerank_score"] = float(score)
            result.append(chunk)
        return result
