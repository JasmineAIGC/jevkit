#!/usr/bin/env python3
"""Example 03 — The evaluation layer (jevkit.eval).

Before trusting a threshold you need three answers, all PER QUESTION TYPE
(calibration error differs by an order of magnitude across types):
  1. calibrate — when the model says 0.9, is it right 90% of the time?
  2. coverage — at threshold τ, how much traffic can run unattended,
                and at what error rate?
  3. permute  — does the answer survive a reshuffle of the option order?

Data: examples/data/triage_labeled.jsonl (kev.train format: request +
inline labels + recorded answers — the same file feeds kev fine-tuning).

Run:  python examples/make_example_data.py   # first, if data/ is missing
      python examples/03_evaluation.py
"""

import sys
from pathlib import Path

from jevkit import (
    Gate,
    Policy,
    Signal,
    Tier,
    best_threshold,
    calibrate_report,
    make_backend,
    pairs_from_examples,
    permute_state,
    read_examples,
)

DATA = Path(__file__).parent / "data" / "triage_labeled.jsonl"
if not DATA.exists():
    sys.exit("run `python examples/make_example_data.py` first")

examples = read_examples(str(DATA))
labeled = [(ex.answers, ex.labels) for ex in examples if ex.answers is not None]
questions = examples[0].questions
print("loaded %d labeled rows; questions: %s\n" % (len(examples), list(questions)))

# ---------------------------------------------------------------- 1. Calibration
print("══ calibrate (per question type, split-half fit/test)")
for name in questions:
    pairs = pairs_from_examples(labeled, name)
    if len(pairs) < 20:
        print("  %-12s skipped (n=%d < 20)\n" % (name, len(pairs)))
        continue
    kind = "binary" if name == "escalate" else "confidence"
    report = calibrate_report(pairs, kind=kind)
    print(
        "  %-12s T=%.3f   ECE %.4f → %.4f   logloss %.4f → %.4f"
        % (
            name,
            report.temperature,
            report.before["ece"],
            report.after["ece"],
            report.before["logloss"],
            report.after["logloss"],
        )
    )
    print(
        "               accuracy unchanged: %.1f%% (temperature never flips an answer)"
        % (report.before["accuracy"] * 100)
    )
print()

# ---------------------------------------------------------------- 2. Coverage
print("══ coverage–accuracy under an error budget (budget = 0.15)")
BUDGET = 0.15
for name in questions:
    pairs = pairs_from_examples(labeled, name)
    if len(pairs) < 20:
        continue
    tau, cov, acc = best_threshold(pairs, budget=BUDGET)
    if cov > 0:
        print(
            "  %-12s τ*=%.2f → automatable %.1f%% @ accuracy %.1f%%"
            % (name, tau, cov * 100, acc * 100)
        )
    else:
        print("  %-12s no feasible threshold under this budget" % name)
print()
# Three instructive shapes show up here:
# - τ* = 0.00 with full coverage: the WHOLE population already fits the
#   budget — a threshold is unnecessary (department on this mock data);
# - a mid-range τ* trading coverage for accuracy (escalate);
# - "no feasible threshold": even τ = 1.0 exceeds the budget — the answer
#   is better data or a better question, never a threshold from a blog post
#   (frustration).

# ---------------------------------------------------------------- 3. Permutation
print("══ option-order stability (does the answer survive a reshuffle?)")
policy = Policy(
    version="triage-demo",
    gates=(
        Gate("department", Signal.CONFIDENCE, (Tier("AUTO", 0.40), Tier("HUMAN", 0.0))),
        Gate("escalate", Signal.PROBABILITY, (Tier("ALERT", 0.90), Tier("NORMAL", 0.0))),
    ),
)
results = permute_state(
    make_backend("mock"), examples[0].state, questions, policy=policy, n_perm=5, model="mock-1.0"
)
for name, r in results.items():
    print("  %-12s %-7s %s" % (name, r.qtype, r.verdict))
    if r.qtype == "choice":
        print(
            "               drift=%.4f  KL=%.4f  argmax-flip=%.0f%%  "
            "policy-flip=%.0f%%  (mode=%s)"
            % (r.drift_max, r.kl_mean, r.argmax_flip_rate * 100, r.action_flip_rate * 100, r.mode)
        )
print()
print("policy-flip > 0 means your threshold sits inside the drift band —")
print("that question must not run fully automated under this policy.")

sys.exit(0)
