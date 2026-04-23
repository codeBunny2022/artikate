# artikate


---

## What's inside

| Section | Task | Status |
|----|----|----|
| 1 | Diagnose a failing LLM pipeline | Written — `ANSWERS.md` |
| 2 | Production-grade RAG pipeline | Code + eval — `section2/` |
| 3 | Fine-tuned ticket classifier | Code + eval — `section3/` |
| 4 | Systems design (Prompt Injection + On-Prem LLM) | Written — `ANSWERS.md` |


---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/codeBunny2022/artikate.git
cd artikate
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+ required. All dependencies install in one command — no Docker, no external services.

### 2. API key

Section 2 uses Gemini for answer generation via OpenAI-compatible endpoint.
Get a free key at [aistudio.google.com](https://aistudio.google.com) → Get API Key.

```bash
export GEMINI_API_KEY=your-key-here
```

Retrieval, embeddings, and the classifier all run fully locally with no API key needed.


---

## Section 1 — LLM Pipeline Diagnosis

No code to run. Full diagnosis in `ANSWERS.md`:

* **Problem 1** (wrong pricing): root cause, how to distinguish from temperature/retrieval issues, and the live-injection fix
* **Problem 2** (language switching): mechanism explained, exact prompt fix with per-language enumeration
* **Problem 3** (latency degradation): three causes ordered by investigation priority, why context growth is investigated first
* **Post-mortem**: plain-language summary for a non-technical stakeholder


---

## Section 2 — RAG Pipeline

Legal document QA over PDF contracts. Answers cite exact source document and page number.

### Architecture

```
PDF ingestion → chunking → bge-small embeddings → ChromaDB
                                                       ↓
                                    BM25 + dense hybrid via RRF
                                                       ↓
                                    cross-encoder reranking
                                                       ↓
                            confidence gate → Gemini generation
```

Key decisions (full reasoning in `DESIGN.md`):

* **1000-char chunks, 150-char overlap** — preserves legal clause integrity
* **BAAI/bge-small-en-v1.5** — top MTEB retrieval, runs fully local, no data leaves the system
* **ChromaDB** — persistent, metadata filtering, no external service required
* **Hybrid BM25 + dense + RRF** — catches exact legal terms that dense-only misses
* **Confidence gate at 0.30** — refuses to generate when retrieved context is insufficient

### Run it

```bash
# Generate sample legal PDFs
python -m section2.create_sample_docs

# Ask a question
python -c "
from section2.pipeline import RAGPipeline
p = RAGPipeline()
p.ingest_directory('section2/sample_docs')
result = p.query('What is the notice period for termination in the NDA with Vendor A?')
print('Answer:    ', result['answer'])
print('Confidence:', result['confidence'])
print('Sources:   ', [(s['document'], s['page']) for s in result['sources']])
"

# Run precision@3 evaluation (no API key needed)
python -m section2.evaluation
```

### Evaluation results

```
Precision@3: 9/10 = 90%
```

Evaluated on 10 manually written question-answer pairs across 3 legal documents.
The one miss (notice period) is a chunking boundary edge case where the relevant
clause lands just outside the top-3 retrieved window.

### Required interface

```python
result = pipeline.query("What is the notice period in the NDA with Vendor X?")

# result = {
#   "answer":     str,
#   "sources":    [{"document": str, "page": int, "chunk": str}],
#   "confidence": float   # 0.0 – 1.0
# }
```


---

## Section 3 — Ticket Classifier

Classifies support tickets into: `billing`, `technical_issue`, `feature_request`, `complaint`, `other`.

### Why DistilBERT, not a few-shot LLM

| Approach | CPU Latency | Fits 500ms SLA? |
|----|----|----|
| DistilBERT (this implementation) | \~45ms | ✅ Yes |
| GPT-4o few-shot via API | 1,500–3,500ms | ❌ No |
| Local Llama 3 8B on CPU | 4,000–15,000ms | ❌ No |

At 2,880 tickets/day, DistilBERT handles the full load on a single CPU thread
with 600× throughput headroom. The LLM API alternative fails the latency SLA
on every single request and adds unnecessary cost.

### Run it

```bash
# Generate training data
python -m section3.data_generator

# Train (downloads DistilBERT ~250MB on first run, trains in ~3 mins on CPU)
python -m section3.classifier train

# Evaluate
python -m section3.evaluate

# Latency assertion test (20 tickets, asserts valid label + under 500ms each)
python -m section3.test_latency
```

### Results

```
Accuracy  : 93%
Macro F1  : 0.9297
Latency   : 45ms mean | 50ms p95
SLA (500ms): ✅ All 20 test tickets pass
```

Per-class F1:

| Class | F1 |
|----|----|
| billing | 0.9474 |
| technical_issue | 0.8649 |
| feature_request | 1.0000 |
| complaint | 0.8837 |
| other | 0.9524 |

**Most confused pair:** `technical_issue` → `complaint`
Frustrated users describing a technical problem use emotionally charged language
that overlaps with complaint vocabulary. A ticket like "your app keeps crashing
and this is completely unacceptable" carries both a technical signal and a
complaint signal. Additional features that would help: presence of error codes,
feature names, or HTTP status codes reliably indicates `technical_issue`
regardless of emotional tone.


---

## Section 4 — Systems Design

Full written answers in `ANSWERS.md`. Questions answered:

**Question A — Prompt Injection & LLM Security**
Five distinct techniques covered: direct instruction override, persona injection,
RAG poisoning via retrieved content, token smuggling through encoding, and
multi-turn context poisoning. Each paired with a specific application-layer
mitigation — not just general advice.

**Question C — On-Premise LLM Deployment**
Full VRAM calculations for 70B model deployment on 2×A100 80GB GPUs.
Recommendation: Llama 3.1 70B in INT8 via AWQ quantisation, served with vLLM
tensor parallelism. Expected throughput: 2.1–2.8s for 500-token input, within
the 3-second SLA. Honest acknowledgement of the \~0.5–2% accuracy trade-off
from INT8 quantisation.


---

## Project structure

```
artikate/
├── README.md
├── ANSWERS.md          ← written answers: Sections 1, 3, 4
├── DESIGN.md           ← RAG architecture decisions
├── requirements.txt
│
├── section2/
│   ├── ingestion.py         ← PDF extraction + chunking
│   ├── vectorstore.py       ← ChromaDB + bge embeddings
│   ├── retrieval.py         ← BM25 + RRF + cross-encoder reranking
│   ├── pipeline.py          ← main query interface
│   ├── evaluation.py        ← precision@3 harness
│   ├── create_sample_docs.py
│   └── sample_docs/         ← 3 sample legal PDFs
│
└── section3/
    ├── classifier.py        ← DistilBERT trainer + inference
    ├── data_generator.py    ← synthetic training data
    ├── evaluate.py          ← accuracy, F1, confusion matrix
    ├── test_latency.py      ← 20-ticket SLA assertion test
    └── data/                ← generated training data (gitignored)
```


---

## Notes

* No API keys, credentials, or `.env` files are committed to this repository
* Training data for Section 3 is template-generated and documented in `data_generator.py`
* Evaluation sets are manually written in both sections — not LLM-generated
* All model downloads happen automatically on first run via HuggingFace
* Tested on Python 3.12, Ubuntu 22.04, CPU-only


