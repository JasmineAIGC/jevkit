# -*- coding: utf-8 -*-
"""Shared fixtures for the examples: the triage question set, hand-crafted
mock probabilities (so tier boundaries actually trigger), and the demo policy.

The three fixture rows reproduce the classic shapes from the study notes:
  invoice → high-confidence billing          (AUTO path)
  shoes   → low-confidence multi-topic       (HUMAN path, high escalation)
  vague   → flat distribution                (HUMAN path, borderline escalate)
"""

from jevkit import Choice, Gate, Noul, Policy, Score, Signal, Tier

TRIAGE_QUESTIONS = {
    "department": Choice("Which team should handle this ticket?", {
        "returns": "Exchanges, refunds, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems",
    }),
    "escalate": Noul("Does this need urgent human attention?"),
    "frustration": Score("How frustrated is the customer?",
                         ["Calm", "Frustrated", "Very angry"]),
}

TRIAGE_FIXTURES = {
    "The invoice for order #4411 was charged twice. Please refund one of them.": {
        "department": {"type": "choice", "choice": "billing", "confidence": 0.88,
                       "probabilities": {"returns": 0.04, "shipping": 0.08,
                                         "billing": 0.88}},
        "escalate": {"type": "noul", "noul": 0.72},
        "frustration": {"type": "score", "score": 0.61, "confidence": 0.55,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0.45, "1": 0.49, "2": 0.06}},
    },
    "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.": {
        "department": {"type": "choice", "choice": "returns", "confidence": 0.21,
                       "probabilities": {"returns": 0.47, "shipping": 0.28,
                                         "billing": 0.25}},
        "escalate": {"type": "noul", "noul": 0.93},
        "frustration": {"type": "score", "score": 1.44, "confidence": 0.78,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0.00, "1": 0.56, "2": 0.44}},
    },
    "Hi, about my thing... it's not right. Please check?": {
        "department": {"type": "choice", "choice": "shipping", "confidence": 0.18,
                       "probabilities": {"returns": 0.30, "shipping": 0.36,
                                         "billing": 0.34}},
        "escalate": {"type": "noul", "noul": 0.51},
        "frustration": {"type": "score", "score": 0.98, "confidence": 0.31,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0.35, "1": 0.32, "2": 0.33}},
    },
}

TRIAGE_SAMPLES = [
    ("invoice", "The invoice for order #4411 was charged twice. Please refund one of them."),
    ("shoes", "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."),
    ("vague", "Hi, about my thing... it's not right. Please check?"),
]


def triage_policy() -> Policy:
    """Two axes, two cost structures: routing a ticket wrong is reversible
    (loose 0.70); a false escalation pages someone at 3am (strict 0.90)."""
    return Policy(version="triage-v1", gates=(
        Gate("department", Signal.CONFIDENCE, (
            Tier("AUTO", 0.70), Tier("DEFER", 0.50), Tier("HUMAN", 0.0))),
        Gate("escalate", Signal.PROBABILITY, (
            Tier("ALERT", 0.90), Tier("DEFER", 0.60), Tier("NORMAL", 0.0))),
    ))
