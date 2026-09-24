#!/usr/bin/env python3
"""Example 02 — The policy layer (jevkit.policy).

The model gives you probabilities; it never tells you what to DO.
Policy = data + a pure function, not a DSL. Thresholds come from YOUR
false-positive/false-negative costs — and every decision is logged with
the full distribution, so "was it 0.91 vs 0.05 or 0.36 vs 0.34?" is
always answerable.

Run:  python examples/02_policy_and_ledger.py
"""

import sys
from pathlib import Path

from common import TRIAGE_FIXTURES, TRIAGE_QUESTIONS, TRIAGE_SAMPLES, triage_policy

from jevkit import (
    Answers,
    Gate,
    JsonlLedger,
    Policy,
    PolicyError,
    Signal,
    Tier,
    decide,
    make_backend,
)

# ---------------------------------------------------------------- 1. Policy
POLICY = triage_policy()  # version="triage-v1"; thresholds from cost structure

# ---------------------------------------------------------------- 2. Guard rail
# The framework refuses to evaluate a policy that encodes a known failure
# mode: noul answers carry no independent confidence, so a CONFIDENCE gate
# on a noul question fails loudly instead of silently misreading.
bad = Policy(
    version="bad-v1", gates=(Gate("escalate", Signal.CONFIDENCE, (Tier("A", 0.9), Tier("B", 0.0))),)
)
answers = make_backend("mock", fixtures=TRIAGE_FIXTURES).ask(TRIAGE_SAMPLES[0][1], TRIAGE_QUESTIONS)
try:
    decide(answers, bad, state_ref="x")
except PolicyError as e:
    print("blocked by design: %s" % e)
print()

# ---------------------------------------------------------------- 3. Decide + ledger
backend = make_backend("mock", fixtures=TRIAGE_FIXTURES)
ledger = JsonlLedger(str(Path(__file__).parent / "data" / "02-decisions.jsonl"))

for ref, state in TRIAGE_SAMPLES:
    answers = backend.ask(state, TRIAGE_QUESTIONS, model="mock-1.0")
    record = decide(
        answers,
        POLICY,
        backend_name=backend.name,
        state_ref=ref,
        state=state,
        question_set_version="triage-questions-v1",
        question_set=TRIAGE_QUESTIONS,
    )
    ledger.append(record)

    print("[%s] %s" % (ref, state))
    for question, act in record.actions.items():
        print("  %-12s %-7s %s" % (question, act["action"], act["detail"]))
    print()

# ---------------------------------------------------------------- 4. Replay from the log
# The ledger is the audit trail: replay any line and get the same actions.
first = list(ledger)[0]
print("replayed %r from the ledger:" % first.state_ref)
replayed_answers = Answers.from_response({"model": first.model, "answers": first.raw})
replayed = decide(replayed_answers, Policy.from_json(first.policy), state_ref=first.state_ref)
print("  actions identical:", replayed.actions == first.actions)

print()
print("ledger written: examples/data/02-decisions.jsonl")
print("fields per line:", ", ".join(first.to_json().keys()))

sys.exit(0)
