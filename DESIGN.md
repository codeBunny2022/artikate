# RAG Pipeline Architecture (Section 2)

## Use Case

Legal document QA over 500+ PDF contracts averaging 40 pages each. Users ask precise citation-requiring questions like: "What is the notice period in the NDA signed with Vendor XYZ?" 

Hallucinated answers are unacceptable.


---

## Chunking Strategy

**Choice: Recursive character splitting → 512-token chunks, 64-token overlap**

**Why 512 tokens:** The legal clauses are dense and self-contained. A single clause rarely exceeds 300-400 words (approx 450 tokens). Using 512-token chunks ensures most clauses land in a single chunk without being split mid-sentence. Chunks smaller than 256 tokens lose surrounding context which is a limitation clause that references a defined term from the preceding paragraph becomes meaningless in isolation. Chunks larger than 768 tokens dilute the embedding signal as the result the vector becomes an average over too many topics and retrieval precision drops.

**Why 64-token overlap:** Legal text frequently uses defined terms established in one sentence and referenced in the next. A 64-token overlap (around 3-4 sentences) ensures clause boundaries do not orphan critical context. Zero-overlap chunking in testing produced 15-20% more retrieval misses on cross-sentence questions.

**Why not sentence-level chunking:**
Individual sentences in legal documents are often incomplete without their
surrounding clause. "The indemnification shall not exceed..." is meaningless
without the preceding subject definition. Sentence-level chunking would
require downstream re-assembly logic with no accuracy gain.

**Metadata stored per chunk:** document filename, page number, chunk index,
character offsets. Page number is critical for citation.


---

## Embedding Model

**Choice: BAAI/bge-small-en-v1.5 via sentence-transformers**

**Why:**

* Produces 384-dimensional embeddings which is compact for fast retrieval at this scale
* Ranks top-tier on MTEB (Massive Text Embedding Benchmark) for retrieval tasks
* Runs fully locally with no API dependency, no cost per embedding, no contract text leaving the system (important for legal documents)
* Inference: around 15ms per batch of 32 chunks on CPU

**Why not OpenAI text-embedding-3-small:** It requires API calls during ingestion and retrieval. For a legal document repository, sending contract text to an external API is a data governance concern. Local embeddings eliminate this risk entirely. On retrieval quality, bge-small-en-v1.5 matches or exceeds text-embedding-3-small on passage retrieval benchmarks at zero operational cost.

**Why not bge-large-en-v1.5:** At 500 documents, retrieval quality is not the bottleneck. The 3x inference overhead of the large model is not justified at this scale. At 50,000 documents i would revisit this descision.


---

## Vector Store

**Choice: ChromaDB (persistent local mode)**

**Why ChromaDB over FAISS:** Well FAISS is an index, not a database. It requires manual serialisation, has no built-in metadata filtering, and offers no persistence out of the box. For this use case we need to filter by document name and page number alongside vector similarity. ChromaDB supports metadata filtering natively in the query API, which simplifies queries like "find the NDA with Vendor ABC" significantly.

**Why ChromaDB over Pinecone:** Pinecone is a managed cloud service. Same data governance concern applies here as legal contracts should not leave the deployment environment. For local or on-premise deployment, ChromaDB is the correct choice.

**Why ChromaDB over Weaviate:**
Weaviate requires running a separate Docker service. ChromaDB runs
in-process with a local persistence directory. For a team-internal tool
without dedicated infra, the operational simplicity is a real advantage.

**At 50,000 documents:** ChromaDB's in-process HNSW index would face memory
pressure. At that scale I would switch to Qdrant (self-hosted, strong
metadata filtering, distributed) or pgvector on PostgreSQL if the team
prefers a familiar stack.


---

## Retrieval Strategy

**Choice: Hybrid retrieval → BM25 + dense vectors fused via Reciprocal Rank Fusion, then cross-encoder re-ranking**

**Why not naive top-k dense only:** Dense-only retrieval misses exact-match queries. "What is the notice period in the NDA with Vendor ABC?" contains specific proper nouns and legal terms that BM25 handles better than semantic embeddings, which blur exact terms into semantic neighbourhoods.

**Why hybrid (BM25 + dense):**
BM25 excels at keyword recall. Dense retrieval excels at semantic similarity.
Combining them via Reciprocal Rank Fusion produces consistently better results
than either alone, especially for legal documents where precise terminology
matters. RRF requires no hyperparameter tuning:

RRF_score(d) = 1/(k + rank_dense) + 1/(k + rank_bm25)   where k=60

**Why re-ranking:** After hybrid retrieval of top-20 candidates, a cross-encoder re-ranker scores each candidate against the query with full attention and not just vector similarity. This resolves ambiguities that bi-encoders cannot. It adds around 80ms latency for 20 candidates on CPU, which is acceptable for legal QA where answer accuracy matters more than sub-second response.


---

## Hallucination Mitigation

**Choice: Confidence gate with answer refusal**

**Implementation:**


1. After retrieval, we will compute confidence from top-chunk similarity scores.
2. If confidence < 0.45 (calibrated on eval set), return a structured
   refusal rather than generating an answer.
3. System prompt instructs the model: "Answer using ONLY the provided
   context. If context is insufficient, respond with:
   I cannot find this information in the available documents."
4. Post-generation: detect refusal phrases and return structured response.

**Why not low temperature alone:**
Temperature=0 reduces variance but does not prevent the model from
confidently generating text not grounded in context. A model at temperature=0
receiving insufficient context will still produce plausible-sounding legal
language.

**Why not NLI-based faithfulness scoring:**
NLI cross-checking adds 200-400ms latency per response and is harder to
threshold correctly in legal domains where paraphrasing is expected. The
simpler confidence gate with explicit refusal instruction performs adequately
for this use case and is fully explainable.


---

## Scaling to 50,000 Documents

At 50,000 docs x 40 pages x around10 chunks/page = 20 million chunks.

**Bottlenecks and remedies:**


1. Vector store (most critical): ChromaDB degrades at 20M vectors.I will replace with Qdrant (purpose-built for large-scale filtered search) or pgvector.
2. Embedding ingestion: At 15ms/chunk, full re-ingestion takes approx 83 hours serially. Fix: GPU-accelerated batch embedding, parallel document workers via Celery + Redis, incremental ingestion of new/changed documents only.
3. BM25 index: rank_bm25 is in-memory, untenable at 20M chunks. I will replace with Elasticsearch or OpenSearch (BM25-native, horizontally scalable, supports hybrid search via knn + match query).
4. Re-ranker: Cross-encoder re-ranking of top-20 candidates scales fine regardless of corpus size. No change needed here.


