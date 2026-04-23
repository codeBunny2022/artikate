from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from section2.ingestion import TextChunk

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "legal_docs"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class VectorStore:
    """
    persistent ChromaDB store with local bge embeddings.
    no external API calls -> all embeddings computed locally.
    """

    def __init__(self, persist_dir: str = "./chroma_store"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        self._embedder = SentenceTransformer(EMBEDDING_MODEL)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store ready — %d chunks indexed", self._collection.count())

    def add_chunks(self, chunks: list[TextChunk], batch_size: int = 64) -> None:
        if not chunks:
            return

        ids = [f"{c.document}::chunk_{c.chunk_index}" for c in chunks]

        # skip already-indexed chunks (idempotent)
        existing = set(self._collection.get(ids=ids)["ids"])
        new_chunks = [c for c, cid in zip(chunks, ids) if cid not in existing]
        new_ids   = [cid for cid in ids if cid not in existing]

        if not new_chunks:
            logger.info("All chunks already indexed — skipping")
            return

        logger.info("Embedding %d new chunks…", len(new_chunks))
        for i in range(0, len(new_chunks), batch_size):
            batch     = new_chunks[i:i + batch_size]
            batch_ids = new_ids[i:i + batch_size]
            texts     = [c.text for c in batch]

            embeddings = self._embedder.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

            metadatas = [
                {
                    "document":    c.document,
                    "page":        c.page,
                    "chunk_index": c.chunk_index,
                }
                for c in batch
            ]
            self._collection.add(
                ids=batch_ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        logger.info("Indexed %d chunks. Total: %d",
                    len(new_chunks), self._collection.count())

    def query(self, query_text: str, n_results: int = 20,
              filter_document: str | None = None) -> list[dict[str, Any]]:
        # bge models perform better with this query prefix
        query_embedding = self._embedder.encode(
            BGE_QUERY_PREFIX + query_text,
            normalize_embeddings=True,
        ).tolist()

        where = {"document": filter_document} if filter_document else None

        n = min(n_results, self._collection.count())
        if n == 0:
            return []

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            candidates.append({
                "text":        doc,
                "document":    meta["document"],
                "page":        meta["page"],
                "chunk_index": meta["chunk_index"],
                "score":       float(1.0 - dist),  # cosine distance → similarity
            })
        return candidates

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Vector store cleared.")
