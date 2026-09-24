#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""工单三段式路由 —— jevkit 库 API 的标准用法（原 jev_router_demo.py 的框架化重写）。

对比原脚本（约 210 行、每个后端手写）：这里政策是声明式数据，
后端一行切换，日志 schema 由框架保证，整段"业务代码"只剩题集和阈值定义。

切后端只改一行：
    backend = make_backend("mock", fixtures=FIXTURES)          # 测试 / demo
    backend = make_backend("http://127.0.0.1:8009")            # 本地 kev serve
    backend = make_backend("typesafe://jev-1.13.0")            # 官方托管（锁定版本！）
"""

import sys

from jevkit import (
    Choice, Gate, JsonlLedger, Noul, Policy, Score, Signal, Tier,
    decide, make_backend,
)

# ---------------------------------------------------------------- 1. 题集（一题一事）
QUESTIONS = {
    "department": Choice("Which team should handle this ticket?", {
        "returns": "Exchanges, refunds, wrong or damaged items",
        "shipping": "Delivery status, delays, lost packages",
        "billing": "Charges, invoices, payment problems",
    }),
    "escalate": Noul("Does this need urgent human attention?"),
    "frustration": Score("How frustrated is the customer?",
                         ["Calm", "Frustrated", "Very angry"]),
}

# ---------------------------------------------------------------- 2. 政策（数据 + 纯函数）
# 阈值来自误报/漏报成本的业务判断：路由分错可撤回（宽松 0.70），
# 误报升级会吵醒值班（严格 0.90）。上线前用 `jevkit coverage` 在自己数据上重校，
# 或直接 `jevkit compile` 按错误预算产出 lock。
POLICY = Policy(
    version="triage-v1",
    gates=(
        Gate("department", Signal.CONFIDENCE, (
            Tier("AUTO", 0.70), Tier("DEFER", 0.50), Tier("HUMAN", 0.0))),
        Gate("escalate", Signal.PROBABILITY, (
            Tier("ALERT", 0.90), Tier("DEFER", 0.60), Tier("NORMAL", 0.0))),
    ),
)

# ---------------------------------------------------------------- 3. 后端
FIXTURES = {  # mock 夹具：三条典型工单的概率形态（复刻笔记 3.4 节示例）
    "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.": {
        "department": {"type": "choice", "choice": "returns", "confidence": 0.21,
                       "probabilities": {"returns": 0.47, "shipping": 0.28, "billing": 0.25}},
        "escalate": {"type": "noul", "noul": 0.93},
        "frustration": {"type": "score", "score": 1.44, "confidence": 0.78,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0.00, "1": 0.56, "2": 0.44}}},
    "The invoice for order #4411 was charged twice. Please refund one of them.": {
        "department": {"type": "choice", "choice": "billing", "confidence": 0.88,
                       "probabilities": {"returns": 0.04, "shipping": 0.08, "billing": 0.88}},
        "escalate": {"type": "noul", "noul": 0.72},
        "frustration": {"type": "score", "score": 0.61, "confidence": 0.55,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0.45, "1": 0.49, "2": 0.06}}},
    "Hi, about my thing... it's not right. Please check?": {
        "department": {"type": "choice", "choice": "shipping", "confidence": 0.18,
                       "probabilities": {"returns": 0.30, "shipping": 0.36, "billing": 0.34}},
        "escalate": {"type": "noul", "noul": 0.51},
        "frustration": {"type": "score", "score": 0.98, "confidence": 0.31,
                        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                        "probabilities": {"0": 0.35, "1": 0.32, "2": 0.33}}},
}

backend = make_backend("mock", fixtures=FIXTURES)
# backend = make_backend("http://127.0.0.1:8009")            # kev: python -m kev.serve ...
# backend = make_backend("typesafe://jev-1.13.0")            # 生产：锁版本 + TYPESAFE_API_KEY

SAMPLES = [
    ("shoes", "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."),
    ("invoice", "The invoice for order #4411 was charged twice. Please refund one of them."),
    ("vague", "Hi, about my thing... it's not right. Please check?"),
]


def main() -> int:
    ledger = JsonlLedger("decisions.jsonl")
    for ref, state in SAMPLES:
        answers = backend.ask(state, QUESTIONS, model="mock-1.0")
        record = decide(answers, POLICY, backend_name=backend.name,
                        state_ref=ref, state=state,
                        question_set_version="triage-questions-v1",
                        question_set=QUESTIONS)
        ledger.append(record)
        print("─" * 72)
        print("[%s] %s" % (ref, state))
        for q, act in record.actions.items():
            print("  %-12s %-6s %s" % (q, act["action"], act["detail"]))
    print("─" * 72)
    print("决策日志 → decisions.jsonl（完整概率分布 + 政策快照，随时可审计）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
