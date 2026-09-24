#!/usr/bin/env python3
"""Example 01 — The three primitives and the backends (jevkit.core).

System One's entire API surface: one state + a few atomic questions,
typed probabilistic answers back. This example shows how to define the
three question types, ask any backend once, and read the typed answers.

Run:  python examples/01_ask_and_answer.py            # mock backend, no service needed
      JEVKIT_BACKEND=http://127.0.0.1:8009 python …   # your local kev.serve
"""

import os
import sys

from jevkit import Choice, Noul, Score, make_backend

# ---------------------------------------------------------------- 1. Questions
# One question = one concern. Questions cannot see each other; if you need
# dependencies, ask twice or compose in code (see example 02).
#
# The kev/TypeSafe contract allows any JSON content for `state`,
# `instructions`, and criteria descriptions — strings are just the common case.
QUESTIONS = {
    # Choice: pick one of 1–255 named candidates. Answer: chosen key +
    # full probability distribution + confidence.
    "department": Choice(
        instructions={
            "ask": "Which team handles this ticket?",
            "context": ["customer support triage"],
        },  # JSON, not just str
        criteria={
            "returns": "Exchanges, refunds, wrong or damaged items",
            "shipping": "Delivery status, delays, lost packages",
            "billing": None,  # None = the option name is self-explanatory
        },
    ),
    # Noul: a yes/no judgment. Answer: a single 0..1 probability —
    # there is deliberately NO separate confidence field.
    "escalate": Noul("Does this need urgent human attention?"),
    # Score: an ordered scale. Answer: expected level + per-level
    # probabilities + legend. Keep levels interpretable, not finely graded.
    "frustration": Score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
}

STATE = {
    "ticket_id": 4411,
    "subject": "double charge",
    "message": "The invoice was charged twice. Please refund one of them.",
    "tier": "standard",
}

# ---------------------------------------------------------------- 2. Backend
# One line switches the entire runtime:
#   "mock"                        deterministic, no service (tests / demos)
#   "http://127.0.0.1:8009"       a local kev.serve (open-source model)
#   "typesafe://jev-1.13.0"       hosted TypeSafe (pin the version in prod!)
spec = os.environ.get("JEVKIT_BACKEND", "mock")
model = os.environ.get("JEVKIT_MODEL", "kev-latest")
backend = make_backend(spec)

# ---------------------------------------------------------------- 3. Ask once
answers = backend.ask(STATE, QUESTIONS, model=model)

print(
    "backend=%s  model=%s  latency=%sms  request_id=%s"
    % (backend.name, answers.model, answers.latency_ms, answers.request_id)
)
print()

dep = answers.answers["department"]
print("department (choice)")
print("  chosen       :", dep.choice)
print("  confidence   : %.4f" % dep.confidence)  # (p_max − 1/K)/(1 − 1/K)
print("  distribution : %s" % dep.probabilities)  # full, not just the top pick
print("  p_max        : %.4f" % dep.p_max)

esc = answers.answers["escalate"]
print("escalate (noul)")
print("  probability  : %.4f  (this IS the confidence — there is no other)" % esc.noul)

fr = answers.answers["frustration"]
print("frustration (score)")
print("  expected lvl : %.2f  → %r" % (fr.score, fr.legend.get(int(round(fr.score)))))
print("  distribution : %s" % fr.probabilities)

# Two facts worth internalizing:
# - confidence is a summary of the distribution, NOT an accuracy promise
#   (0.88 confidence does not mean "88% likely correct");
# - `answers.raw` keeps the untouched payload — that is what decision logs store.
print()
print("raw payload (what the ledger records):")
for k, v in answers.raw.items():
    print("  %-12s %s" % (k, v))

sys.exit(0)
