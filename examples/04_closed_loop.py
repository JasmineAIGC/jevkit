#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example 04 — The closed loop (compile → run → relabel → check).

This is where the two layers meet:

    labeled data ──compile──▶ policy.lock.json   (thresholds + evidence)
                                  │ PolicyLock.load(...).policy at runtime
                                  ▼
                            decide() → decisions.jsonl
                                  │ labels flow back
                                  ▼
                          check_drift vs the lock's evidence

compile_policy replaces each gate's TOP (auto) tier with
τ* = argmax coverage s.t. error ≤ budget, computed on your labeled data,
and stores the evidence (n / coverage / accuracy / data fingerprint) in
the lock. Switching models invalidates the lock — kev's calibration
trails hosted Jev, so thresholds never transfer across models.

Run:  python examples/make_example_data.py   # first, if data/ is missing
      python examples/04_closed_loop.py
"""

import copy
import random
import sys
from pathlib import Path

from jevkit import (
    Gate, JsonlLedger, Policy, PolicyLock, PolicyError, Signal, Tier,
    check_drift, compile_policy, decide, make_backend, read_examples,
    sha256_file,
)

DATA_DIR = Path(__file__).parent / "data"
DATA = DATA_DIR / "triage_labeled.jsonl"
if not DATA.exists():
    sys.exit("run `python examples/make_example_data.py` first")

# ---------------------------------------------------------------- 1. Template
# Hand-write the tier STRUCTURE with placeholder auto thresholds; compile
# fills in the numbers from data. Actions and structure stay yours.
TEMPLATE = Policy(version="triage-v1", gates=(
    Gate("department", Signal.CONFIDENCE, (
        Tier("AUTO", 0.99), Tier("DEFER", 0.50), Tier("HUMAN", 0.0))),
    Gate("escalate", Signal.PROBABILITY, (
        Tier("ALERT", 0.99), Tier("DEFER", 0.60), Tier("NORMAL", 0.0))),
))

# ---------------------------------------------------------------- 2. Compile
examples = read_examples(str(DATA))
BUDGET = 0.15
lock = compile_policy(examples, TEMPLATE, budget=BUDGET,
                      data_sha256=sha256_file(str(DATA)))
lock_path = DATA_DIR / "04-policy.lock.json"
lock.save(str(lock_path))

print("compiled %s  (budget %.0f%%, data %s, model %s)"
      % (lock.policy.version, BUDGET * 100,
         lock.evidence["data_sha256"], ", ".join(lock.evidence["models"])))
for ev in lock.evidence["gates"]:
    acc = "%.1f%%" % (ev["accuracy"] * 100) if isinstance(ev["accuracy"], float) \
        or isinstance(ev["accuracy"], int) else "—"
    cov = "%.1f%%" % (ev["coverage"] * 100) if ev["coverage"] else "0%"
    print("  %-12s n=%-4d auto≥%.2f  cov %s  acc %s  %s"
          % (ev["question"], ev["n"], ev["threshold"], cov, acc,
             ev["note"] or "ok"))
print()

# ---------------------------------------------------------------- 3. Run with the lock
runtime_policy = PolicyLock.load(str(lock_path)).policy
backend = make_backend("mock")
ledger = JsonlLedger(str(DATA_DIR / "04-decisions.jsonl"))
for ex in examples[:5]:
    answers = backend.ask(ex.state, ex.questions, model="mock-1.0")
    record = decide(answers, runtime_policy, backend_name=backend.name,
                    state_ref=ex.state_ref, state=ex.state,
                    question_set_version="triage-questions-v1")
    ledger.append(record)
print("ran 5 tickets under the compiled policy → examples/data/04-decisions.jsonl")
for rec in list(ledger)[:2]:
    print("  [%s] department→%s escalate→%s"
          % (rec.state_ref, rec.actions["department"]["action"],
             rec.actions["escalate"]["action"]))
print()

# ---------------------------------------------------------------- 4. Drift check
# Same distribution → clean.
report = check_drift(examples, lock)
print("check on the same data:      %s"
      % ("✅ no drift" if report.ok else "⚠️ drift"))

# Simulate degradation: flip 30% of the department labels (a model upgrade
# or a question rewrite gone wrong would look like this).
rng = random.Random(7)
degraded = copy.deepcopy(examples)
for ex in degraded:
    if rng.random() < 0.3:
        dep = ex.answers.answers.get("department")
        if dep is not None:
            others = [o for o in dep.probabilities if o != dep.choice]
            if others:
                ex.labels["department"] = others[0]

report2 = check_drift(degraded, lock)
print("check on degraded data:      %s" % ("✅ no drift" if report2.ok
                                           else "⚠️ DRIFT DETECTED"))
for row in report2.failures():
    print("  %-12s %-14s lock=%.4f new=%.4f"
          % (row.question, row.metric, row.lock_value, row.new_value))
print()
print("In CI, `jevkit check` turns this into a non-zero exit code; the")
print("response is always: recompile (and re-review the policy), never ride it out.")

sys.exit(0)
