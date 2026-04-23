
---

## Section 3 — Model Selection Justification

### Choice: Fine-tuned DistilBERT

**Latency calculation:**

DistilBERT has 66M parameters. On CPU, single-sequence inference runs at
80–120ms in PyTorch. Our actual measured latency: **45ms mean, 50ms p95**.
Both comfortably within the 500ms SLA.

| Approach | Latency (CPU) | Fits 500ms SLA? |
|---|---|---|
| DistilBERT (PyTorch, this implementation) | ~45ms | Yes |
| GPT-4o few-shot via API | 1,500–3,500ms | No |
| Llama 3 8B local CPU | 4,000–15,000ms | No |

**Throughput calculation:**

2,880 tickets/day = 1 ticket every 30 seconds.
DistilBERT at 45ms handles ~22 requests/second = ~1.9M requests/day.
We have 660× headroom on a single CPU thread with no batching needed.

**Why not few-shot LLM:**
OpenAI API p95 latency is 1.5–3.5s for a classification prompt. This fails
the 500ms SLA on every single request. Additionally, API costs at 2,880
calls/day × ~300 tokens/call add up unnecessarily when a fine-tuned local
model achieves higher accuracy on a fixed label set at zero inference cost.

**Why not BERT-base or RoBERTa:**
BERT-base (110M params) runs at ~200ms on CPU — still within SLA but with
4× less headroom than DistilBERT. DistilBERT retains 97% of BERT's
performance at 40% the size. For 5-class classification with 1,000 training
examples, DistilBERT is the right size.

### Evaluation Results

- **Accuracy: 93%** on 100 manually written evaluation examples
- **Macro F1: 0.9297**
- **Mean latency: 45ms | P95: 50ms**

Per-class F1:
- billing: 0.9474
- technical_issue: 0.8649
- feature_request: 1.0000
- complaint: 0.8837
- other: 0.9524

### Most Confused Classes: technical_issue → complaint (4 misclassifications)

These two classes are difficult to separate because frustrated users
describing a technical problem use emotionally charged language that
overlaps with complaint vocabulary. A ticket like "your app keeps crashing
and this is completely unacceptable" contains both a clear technical signal
(crashing) and strong complaint signal (unacceptable). The model sees similar
token distributions without reliable surface features to distinguish intent.

Additional signals that would improve separation:
1. Presence of specific error messages, status codes, or feature names
   strongly indicates technical_issue regardless of emotional tone.
2. Absence of any technical reference with pure frustration language
   indicates complaint.
3. More training examples of ambiguous edge cases would help the model
   learn this boundary more precisely.

