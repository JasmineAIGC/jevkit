#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ticket triage router — the "putting it together" example.

Everything the tutorial scripts show piece by piece, in the shape of a
real service: shared fixtures and policy from common.py, a one-line
backend switch, decide() + ledger, done.

切后端只改一行：
    backend = make_backend("mock", fixtures=TRIAGE_FIXTURES)   # 测试 / demo
    backend = make_backend("http://127.0.0.1:8009")            # 本地 kev serve
    backend = make_backend("typesafe://jev-1.13.0")            # 官方托管（锁定版本！）
"""

import sys
from pathlib import Path

from jevkit import JsonlLedger, decide, make_backend

from common import TRIAGE_FIXTURES, TRIAGE_QUESTIONS, TRIAGE_SAMPLES, triage_policy

backend = make_backend("mock", fixtures=TRIAGE_FIXTURES)
# backend = make_backend("http://127.0.0.1:8009")            # kev: python -m kev.serve ...
# backend = make_backend("typesafe://jev-1.13.0")            # 生产：锁版本 + TYPESAFE_API_KEY

policy = triage_policy()


def main() -> int:
    ledger = JsonlLedger(str(Path(__file__).parent / "data" / "triage-decisions.jsonl"))
    for ref, state in TRIAGE_SAMPLES:
        answers = backend.ask(state, TRIAGE_QUESTIONS, model="mock-1.0")
        record = decide(answers, policy, backend_name=backend.name,
                        state_ref=ref, state=state,
                        question_set_version="triage-questions-v1",
                        question_set=TRIAGE_QUESTIONS)
        ledger.append(record)
        print("─" * 72)
        print("[%s] %s" % (ref, state))
        for q, act in record.actions.items():
            print("  %-12s %-6s %s" % (q, act["action"], act["detail"]))
    print("─" * 72)
    print("decision log → examples/data/triage-decisions.jsonl "
          "(full distributions + policy snapshot, auditable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
