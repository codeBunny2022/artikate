from __future__ import annotations
import time
import sys
import logging
logging.basicConfig(level=logging.WARNING)

VALID_LABELS = {"billing", "technical_issue", "feature_request", "complaint", "other"}
SLA_MS       = 500.0
MODEL_DIR    = "./section3/model"

TEST_TICKETS = [
    "I was charged twice for the same subscription in March.",
    "The export to CSV button does nothing when I click it.",
    "It would be great if you could add Slack integration.",
    "Your customer service is terrible and I am very frustrated.",
    "How do I reset my password?",
    "Please refund the charge from last week.",
    "Login fails with a 500 error every time I try.",
    "Can you add a dark mode to the dashboard?",
    "I have been waiting 2 weeks for a response — this is unacceptable.",
    "Where can I find the API documentation?",
    "My invoice amount does not match the plan I am on.",
    "The app crashes immediately after opening on Android.",
    "I would love a Gantt chart view for projects.",
    "Nobody is responding to my escalation tickets.",
    "Can I change my account primary email address?",
    "???",
    "URGENT please help me NOW",
    "Hi",
    "I need help with billing and also the app is broken",
    "Billing issue and I am very angry about it",
]


def run_test() -> bool:
    from section3.classifier import TicketClassifier
    clf = TicketClassifier(model_dir=MODEL_DIR)

    print(f"\nLatency assertion test — {len(TEST_TICKETS)} tickets (SLA: {SLA_MS}ms)\n")
    print(f"{'#':<4} {'Label':<20} {'Conf':>6} {'ms':>8}  {'OK':>5}")
    print("-" * 50)

    failures = []
    for i, ticket in enumerate(TEST_TICKETS, 1):
        text = ticket.strip() or "."
        start = time.perf_counter()
        result = clf.predict(text)
        elapsed = (time.perf_counter() - start) * 1000

        label = result["label"]
        ok    = label in VALID_LABELS and elapsed < SLA_MS
        mark  = "✓" if ok else "✗"

        print(f"{i:<4} {label:<20} {result['confidence']:>6.1%} {elapsed:>7.1f}ms  {mark:>5}")

        if label not in VALID_LABELS:
            failures.append(f"Ticket {i}: invalid label '{label}'")
        if elapsed >= SLA_MS:
            failures.append(f"Ticket {i}: {elapsed:.1f}ms exceeds {SLA_MS}ms SLA")

    print("-" * 50)
    if failures:
        print(f"\n❌ {len(failures)} failure(s):")
        for f in failures:
            print(f"   {f}")
        return False
    else:
        print(f"\n✅ All {len(TEST_TICKETS)} predictions: valid labels ✓  under {SLA_MS}ms ✓")
        return True


# pytest-compatible functions
def test_all_labels_valid():
    from section3.classifier import TicketClassifier
    clf = TicketClassifier(model_dir=MODEL_DIR)
    for ticket in TEST_TICKETS:
        result = clf.predict(ticket.strip() or ".")
        assert result["label"] in VALID_LABELS, \
            f"Invalid label '{result['label']}' for: '{ticket[:50]}'"


def test_all_within_500ms():
    from section3.classifier import TicketClassifier
    clf = TicketClassifier(model_dir=MODEL_DIR)
    for ticket in TEST_TICKETS:
        start = time.perf_counter()
        clf.predict(ticket.strip() or ".")
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < SLA_MS, \
            f"Latency {elapsed:.1f}ms exceeds {SLA_MS}ms for: '{ticket[:50]}'"


if __name__ == "__main__":
    passed = run_test()
    sys.exit(0 if passed else 1)
