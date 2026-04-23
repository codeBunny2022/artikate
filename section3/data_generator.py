from __future__ import annotations
import json
import random
from pathlib import Path

LABELS = ["billing", "technical_issue", "feature_request", "complaint", "other"]

TEMPLATES: dict[str, list[str]] = {
    "billing": [
        "I was charged {amount} twice for the same subscription.",
        "My invoice shows {amount} but I'm on the {plan} plan.",
        "Please refund the duplicate payment of {amount} from {month}.",
        "I cancelled my plan but still got charged {amount}.",
        "There's an unexpected charge of {amount} on my account.",
        "The promo code didn't apply to my invoice of {amount}.",
        "Why am I being charged {amount} when I only have {n} users?",
        "I need a GST invoice for the {month} payment of {amount}.",
        "The payment failed but {amount} was still deducted.",
        "You charged me in USD but I set up billing in INR.",
        "I upgraded my plan but was charged for both old and new.",
        "My bank shows a charge from you that I don't recognise.",
        "I did not authorise the renewal payment last week.",
        "The amount charged does not match the price on your website.",
        "Can I get an itemised breakdown of my last invoice?",
    ],
    "technical_issue": [
        "The {feature} button does nothing when I click it.",
        "I'm getting a {error} error on every API call.",
        "App crashes immediately after login on {device}.",
        "The dashboard is showing a blank white screen.",
        "File uploads fail at 95% every time.",
        "Notifications are not arriving despite being enabled.",
        "The search function returns no results for exact matches.",
        "Two-factor authentication code is not being accepted.",
        "The report takes over 10 minutes and then times out.",
        "Login with Google OAuth fails and redirects back to login.",
        "Sync between mobile and desktop is broken since the update.",
        "The bulk delete option disappeared from the UI.",
        "My password reset email never arrives.",
        "Webhook events are being delivered with a 2-hour delay.",
        "Charts are rendering incorrectly in {browser}.",
    ],
    "feature_request": [
        "It would be great to have a {integration} integration.",
        "Can you add dark mode to the dashboard?",
        "I'd love the ability to bulk-assign tickets to team members.",
        "Please add a calendar view for task management.",
        "Would be useful to export reports in {format} format.",
        "Can you add keyboard shortcuts for navigation?",
        "A mobile app for Android would be very helpful.",
        "Please support SSO with Azure Active Directory.",
        "It would be nice to have custom email templates.",
        "Can you add a recycle bin so deleted items can be recovered?",
        "I'd like to see a Gantt chart view for projects.",
        "Please add two-way sync with Google Calendar.",
        "Can I configure different user roles with granular permissions?",
        "Please make the sidebar collapsible.",
        "Can you add time tracking to tasks?",
    ],
    "complaint": [
        "This is unacceptable — the same bug has been reported {n} times.",
        "Your customer service is terrible. I've been waiting {duration} for a response.",
        "I'm extremely frustrated with how this issue has been handled.",
        "Your product quality has significantly declined since the new update.",
        "I'm considering cancelling my account if this isn't resolved.",
        "Nobody on your team seems to know what they're doing.",
        "This is not what I was promised during the sales call.",
        "I've never experienced such poor service from any SaaS product.",
        "The support team keeps closing my tickets without resolving anything.",
        "I've been escalating this issue for {duration} with no progress.",
        "Your status page showed green when the service was clearly down.",
        "Every time you deploy an update something else breaks.",
        "I'm very disappointed with how long basic features take to ship.",
        "Your chatbot is completely useless and a waste of time.",
        "I'm writing a negative review unless this is resolved today.",
    ],
    "other": [
        "How do I add a new team member to my workspace?",
        "What are your support hours?",
        "Where can I find the API documentation?",
        "Can I change my account email address?",
        "Is there a limit on how many projects I can create?",
        "How do I export all my data before cancelling?",
        "What is your data residency policy?",
        "Do you offer a non-profit discount?",
        "Is HIPAA compliance available on the Enterprise plan?",
        "Where is your privacy policy?",
        "Can you send us a security questionnaire?",
        "I need to change the primary admin on our account.",
        "What happens to my data if I downgrade my plan?",
        "How long does onboarding take for an Enterprise team?",
        "What is the maximum file size I can upload?",
    ],
}

FILL: dict[str, list[str]] = {
    "{amount}":      ["₹999", "₹1,499", "$29.99", "$99", "₹4,999", "₹2,999"],
    "{plan}":        ["Pro", "Starter", "Enterprise", "Basic"],
    "{month}":       ["January", "February", "March", "April"],
    "{n}":           ["2", "3", "5", "three", "four"],
    "{feature}":     ["export", "dashboard", "analytics", "search", "import"],
    "{error}":       ["404", "500", "403", "timeout"],
    "{device}":      ["iOS 17", "Android 14", "iPhone 15"],
    "{browser}":     ["Chrome", "Firefox", "Safari", "Edge"],
    "{integration}": ["Slack", "Salesforce", "HubSpot", "Zapier"],
    "{format}":      ["CSV", "PDF", "Excel", "JSON"],
    "{duration}":    ["3 days", "a week", "two weeks", "over a month"],
}


def fill(template: str, rng: random.Random) -> str:
    for placeholder, values in FILL.items():
        if placeholder in template:
            template = template.replace(placeholder, rng.choice(values), 1)
    return template


def generate(n_per_class: int = 200, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    examples = []
    for label in LABELS:
        templates = TEMPLATES[label]
        for i in range(n_per_class):
            tmpl = templates[i % len(templates)]
            text = fill(tmpl, rng)
            examples.append({"text": text, "label": label})
    rng.shuffle(examples)
    return examples


if __name__ == "__main__":
    out = Path("section3/data/train.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    examples = generate(n_per_class=200)
    with open(out, "w") as f:
        json.dump(examples, f, indent=2)

    from collections import Counter
    dist = Counter(e["label"] for e in examples)
    print(f"Generated {len(examples)} examples:")
    for label, count in sorted(dist.items()):
        print(f"  {label}: {count}")
    print(f"Saved to {out}")
