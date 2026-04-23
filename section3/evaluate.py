"""
section3/evaluate.py
Evaluation: accuracy, per-class F1, confusion matrix.
Evaluation set is MANUALLY WRITTEN — not LLM generated.

Usage:
    python -m section3.evaluate
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
import numpy as np
from section3.classifier import TicketClassifier, LABELS, LABEL2ID

logging.basicConfig(level=logging.WARNING)

# 100 manually written examples — 20 per class
# These are NOT from the training templates
EVAL_SET = [
    # --- BILLING (20) ---
    {"text": "I was charged twice for the same subscription in March.", "label": "billing"},
    {"text": "My invoice shows an incorrect amount.", "label": "billing"},
    {"text": "I need a refund for the charge on my account last week.", "label": "billing"},
    {"text": "Why was I billed for a plan I never signed up for?", "label": "billing"},
    {"text": "The March bill is wrong — I should be on the free tier.", "label": "billing"},
    {"text": "I cancelled but still got charged this month.", "label": "billing"},
    {"text": "Please remove the unauthorised charge from my account.", "label": "billing"},
    {"text": "I see a charge I did not authorise on my credit card.", "label": "billing"},
    {"text": "My subscription was supposed to be cheaper than what you charged.", "label": "billing"},
    {"text": "I would like to dispute the charge from last Tuesday.", "label": "billing"},
    {"text": "You charged me in the wrong currency.", "label": "billing"},
    {"text": "I need an invoice for my company's accounting team.", "label": "billing"},
    {"text": "The discount code I applied did not reduce my bill.", "label": "billing"},
    {"text": "I was charged for an annual plan when I selected monthly.", "label": "billing"},
    {"text": "Please send me a receipt for my last payment.", "label": "billing"},
    {"text": "My card was declined but you still took the money.", "label": "billing"},
    {"text": "I switched plans midway — can I get a prorated refund?", "label": "billing"},
    {"text": "There is a tax charge on my bill that should not be there.", "label": "billing"},
    {"text": "I never received confirmation of my payment.", "label": "billing"},
    {"text": "The early cancellation fee you charged is not in the contract.", "label": "billing"},

    # --- TECHNICAL ISSUE (20) ---
    {"text": "The export button does nothing when I click it.", "label": "technical_issue"},
    {"text": "I keep getting a server error when I try to log in.", "label": "technical_issue"},
    {"text": "The mobile app crashes every time I open it.", "label": "technical_issue"},
    {"text": "My data is not syncing across devices.", "label": "technical_issue"},
    {"text": "The video call feature is not connecting.", "label": "technical_issue"},
    {"text": "I cannot upload files — it stops at 80% every time.", "label": "technical_issue"},
    {"text": "The email notifications have stopped working.", "label": "technical_issue"},
    {"text": "I am locked out of my account after resetting my password.", "label": "technical_issue"},
    {"text": "The search bar returns no results even for exact terms.", "label": "technical_issue"},
    {"text": "Your API is returning 503 errors intermittently.", "label": "technical_issue"},
    {"text": "The integration with our CRM stopped working yesterday.", "label": "technical_issue"},
    {"text": "I cannot see my team members in the workspace.", "label": "technical_issue"},
    {"text": "The report generation is stuck at 0% for 30 minutes.", "label": "technical_issue"},
    {"text": "Two factor authentication is not sending me the code.", "label": "technical_issue"},
    {"text": "The browser extension stopped working after the update.", "label": "technical_issue"},
    {"text": "Images are not loading on the dashboard.", "label": "technical_issue"},
    {"text": "I cannot delete records — the delete button is greyed out.", "label": "technical_issue"},
    {"text": "The import from CSV is failing with a format error.", "label": "technical_issue"},
    {"text": "My calendar is showing events on the wrong dates.", "label": "technical_issue"},
    {"text": "The audio is not working during screen share.", "label": "technical_issue"},

    # --- FEATURE REQUEST (20) ---
    {"text": "It would be helpful to have a dark mode option.", "label": "feature_request"},
    {"text": "Can you add support for importing data from Excel?", "label": "feature_request"},
    {"text": "I would love to have a mobile app for this product.", "label": "feature_request"},
    {"text": "Please add the ability to schedule reports weekly.", "label": "feature_request"},
    {"text": "It would be great to have keyboard shortcuts.", "label": "feature_request"},
    {"text": "Can you support multiple languages in the interface?", "label": "feature_request"},
    {"text": "Please add a bulk edit feature for records.", "label": "feature_request"},
    {"text": "I would like an undo button after deleting items.", "label": "feature_request"},
    {"text": "Can you add a print-friendly view for reports?", "label": "feature_request"},
    {"text": "Please add the option to pin important items to the top.", "label": "feature_request"},
    {"text": "It would be useful to filter by date range on the dashboard.", "label": "feature_request"},
    {"text": "Can you integrate with Microsoft Teams?", "label": "feature_request"},
    {"text": "I would like to be able to export to PDF directly.", "label": "feature_request"},
    {"text": "Please add an audit log so we can track changes.", "label": "feature_request"},
    {"text": "Can you add a drag and drop interface for organising items?", "label": "feature_request"},
    {"text": "It would be nice to have colour coded labels for tasks.", "label": "feature_request"},
    {"text": "Please allow custom fields on the user profile page.", "label": "feature_request"},
    {"text": "I would like two way sync with Google Sheets.", "label": "feature_request"},
    {"text": "Can you add a progress bar for long running operations?", "label": "feature_request"},
    {"text": "Please add the ability to set reminders for tasks.", "label": "feature_request"},

    # --- COMPLAINT (20) ---
    {"text": "This is the third time I have raised this issue with no resolution.", "label": "complaint"},
    {"text": "Your support team is completely unresponsive.", "label": "complaint"},
    {"text": "I am very disappointed with the quality of this product.", "label": "complaint"},
    {"text": "This is not what was promised during the demo.", "label": "complaint"},
    {"text": "I have been waiting two weeks and no one has helped me.", "label": "complaint"},
    {"text": "Your service has degraded significantly in the past month.", "label": "complaint"},
    {"text": "I am seriously considering switching to a competitor.", "label": "complaint"},
    {"text": "The response I got from support was completely unhelpful.", "label": "complaint"},
    {"text": "I feel like paying customers are being ignored.", "label": "complaint"},
    {"text": "This level of service is unacceptable for what I am paying.", "label": "complaint"},
    {"text": "Your team promised a fix two weeks ago and nothing has happened.", "label": "complaint"},
    {"text": "I am extremely frustrated and need this escalated immediately.", "label": "complaint"},
    {"text": "Every update you release seems to break something new.", "label": "complaint"},
    {"text": "The product is nowhere near as reliable as advertised.", "label": "complaint"},
    {"text": "I have never experienced such poor after-sales support.", "label": "complaint"},
    {"text": "Your onboarding process was chaotic and poorly managed.", "label": "complaint"},
    {"text": "I was misled about the features included in my plan.", "label": "complaint"},
    {"text": "The lack of communication from your team is unacceptable.", "label": "complaint"},
    {"text": "I am going to leave a public review about this experience.", "label": "complaint"},
    {"text": "This has been the worst software experience of my career.", "label": "complaint"},

    # --- OTHER (20) ---
    {"text": "How do I add a new user to my account?", "label": "other"},
    {"text": "What are your customer support hours?", "label": "other"},
    {"text": "Where can I find the documentation for the API?", "label": "other"},
    {"text": "Can I transfer my account to a different email address?", "label": "other"},
    {"text": "How many projects can I create on the free plan?", "label": "other"},
    {"text": "How do I download my data before closing my account?", "label": "other"},
    {"text": "Do you offer discounts for annual subscriptions?", "label": "other"},
    {"text": "How do I connect my custom domain to your platform?", "label": "other"},
    {"text": "Is your platform compliant with GDPR?", "label": "other"},
    {"text": "Where can I read your terms and conditions?", "label": "other"},
    {"text": "Can I have a demo of the Enterprise features?", "label": "other"},
    {"text": "How do I change the language settings in my account?", "label": "other"},
    {"text": "What file formats do you support for import?", "label": "other"},
    {"text": "How long does it take to set up a new workspace?", "label": "other"},
    {"text": "Do you have a referral programme?", "label": "other"},
    {"text": "Can I use your product offline?", "label": "other"},
    {"text": "What is the difference between the Pro and Enterprise plans?", "label": "other"},
    {"text": "How do I enable single sign-on for my organisation?", "label": "other"},
    {"text": "Can I white label your product for my clients?", "label": "other"},
    {"text": "Where do I submit a request to delete my account?", "label": "other"},
]


def run_evaluation(model_dir: str = "./section3/model") -> dict:
    clf = TicketClassifier(model_dir=model_dir)

    texts       = [ex["text"] for ex in EVAL_SET]
    true_labels = [ex["label"] for ex in EVAL_SET]

    results     = clf.predict_batch(texts)
    pred_labels = [r["label"] for r in results]
    latencies   = [r["latency_ms"] for r in results]

    true_ids = np.array([LABEL2ID[l] for l in true_labels])
    pred_ids = np.array([LABEL2ID[l] for l in pred_labels])

    accuracy = float(np.mean(true_ids == pred_ids))

    # Confusion matrix
    n  = len(LABELS)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_ids, pred_ids):
        cm[t][p] += 1

    # Per-class F1
    f1_per_class = {}
    for i, label in enumerate(LABELS):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class[label] = {"precision": round(prec, 4),
                                "recall": round(rec, 4),
                                "f1": round(f1, 4)}

    macro_f1 = float(np.mean([v["f1"] for v in f1_per_class.values()]))

    # Most confused pair
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    idx = np.unravel_index(cm_off.argmax(), cm_off.shape)
    most_confused = (LABELS[idx[0]], LABELS[idx[1]], int(cm_off[idx]))

    # Print report
    print("\n" + "=" * 60)
    print("TICKET CLASSIFIER EVALUATION")
    print("=" * 60)
    print(f"Eval examples : {len(EVAL_SET)} (manually written)")
    print(f"Accuracy      : {accuracy:.1%}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print()
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 55)
    for label, m in f1_per_class.items():
        print(f"{label:<20} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")
    print()
    print("Confusion Matrix:")
    header = f"{'':20}" + "".join(f"{l[:8]:>10}" for l in LABELS)
    print(header)
    for i, label in enumerate(LABELS):
        row = f"{label:<20}" + "".join(f"{v:>10}" for v in cm[i])
        print(row)
    print()
    print(f"Most confused : '{most_confused[0]}' → '{most_confused[1]}' ({most_confused[2]} times)")
    print(f"Latency mean  : {np.mean(latencies):.1f}ms | p95: {np.percentile(latencies, 95):.1f}ms")
    print("=" * 60)

    report = {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": f1_per_class,
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
        "most_confused": most_confused,
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
    }

    out = Path("section3/eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {out}")
    return report


if __name__ == "__main__":
    run_evaluation()
