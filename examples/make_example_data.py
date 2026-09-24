#!/usr/bin/env python3
"""生成示例数据（在项目根目录运行：python examples/make_example_data.py）。

产出三个文件，演示完整闭环的每一段：

    examples/data/triage_labeled.jsonl        标注数据（kev train 格式 + 内嵌 answers）
    examples/data/triage_states.jsonl         状态集（permute 用，无需标签）
    examples/data/triage-policy-template.json 政策模板（compile 用）

数据形态刻意做成"过度自信但总体 ~85% 对"：confidence 偏高、真实一致率 0.85——
正好能看出 coverage 曲线与温度缩放在干什么。
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevkit import Choice, Noul, Score, make_backend  # noqa: E402

QUESTIONS = {
    "department": Choice(
        "Which team should handle this ticket?",
        {
            "returns": "Exchanges, refunds, wrong or damaged items",
            "shipping": "Delivery status, delays, lost packages",
            "billing": "Charges, invoices, payment problems",
        },
    ),
    "escalate": Noul("Does this need urgent human attention?"),
    "frustration": Score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
}

TEMPLATES = [
    "My order #{n} arrived {issue}. I want to {want}.",
    "Order #{n}: {issue}. This is the {k}th time I write about this.",
    "Hi, about order #{n} — {issue}. Please {want}.",
    "Checking on order #{n}. {issue_cap} Can someone {want}?",
]
ISSUES = {
    "returns": [
        "damaged on arrival",
        "the wrong size",
        "a different color than ordered",
        "broken packaging and missing parts",
    ],
    "shipping": [
        "not delivered yet",
        "stuck in transit for two weeks",
        "marked delivered but never arrived",
        "no tracking updates for days",
    ],
    "billing": [
        "charged twice on my card",
        "an extra fee I never approved",
        "charged the wrong amount",
        "still billed after cancellation",
    ],
}
WANTS = {
    "returns": ["exchange it", "return it for a refund", "get a replacement"],
    "shipping": ["tell me where it is", "escalate to the carrier", "reship it"],
    "billing": ["refund the difference", "fix the invoice", "reverse the charge"],
}

POLICY_TEMPLATE = {
    "version": "triage-v1",
    "gates": [
        {
            "question": "department",
            "signal": "confidence",
            "tiers": [
                {"action": "AUTO", "min_signal": 0.95},
                {"action": "DEFER", "min_signal": 0.50},
                {"action": "HUMAN", "min_signal": 0.0},
            ],
        },
        {
            "question": "escalate",
            "signal": "probability",
            "tiers": [
                {"action": "ALERT", "min_signal": 0.95},
                {"action": "DEFER", "min_signal": 0.60},
                {"action": "NORMAL", "min_signal": 0.0},
            ],
        },
    ],
}


def make_state(rng: random.Random) -> str:
    topic = rng.choice(["returns", "shipping", "billing"])
    tpl = rng.choice(TEMPLATES)
    return tpl.format(
        n=rng.randint(1000, 9999),
        k=rng.randint(2, 4),
        issue=rng.choice(ISSUES[topic]),
        issue_cap=rng.choice(ISSUES[topic]).capitalize(),
        want=rng.choice(WANTS[topic]),
    )


def main() -> int:
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(exist_ok=True)
    rng = random.Random(20260924)
    backend = make_backend("mock")

    q_req = {k: q.to_request() for k, q in QUESTIONS.items()}
    options = list(QUESTIONS["department"].criteria)

    with open(out / "triage_labeled.jsonl", "w", encoding="utf-8") as f:
        for i in range(400):
            state = make_state(rng)
            answers = backend.ask(state, QUESTIONS, model="mock-1.0")
            dep = answers.answers["department"]
            esc = answers.answers["escalate"]
            fr = answers.answers["frustration"]

            # 标注员与模型 ~85% / ~90% / ~80% 一致——真实世界的形态
            if rng.random() < 0.85:
                dep_label = dep.choice
            else:
                dep_label = rng.choice([o for o in options if o != dep.choice])
            esc_label = (esc.noul >= 0.5) if rng.random() < 0.90 else (esc.noul < 0.5)
            fr_level = int(round(fr.score))
            fr_label = (
                fr_level
                if rng.random() < 0.80
                else max(
                    0,
                    min(len(QUESTIONS["frustration"].criteria) - 1, fr_level + rng.choice([-1, 1])),
                )
            )

            row = {
                "state": state,
                "state_ref": "syn-%04d" % i,
                "model": answers.model,
                # label 内联在每道题上——kev.data.load_records 的原生格式，
                # 这份文件可以直接喂 kev.train；answers 是 jevkit 扩展字段
                # （离线评测用），kev/TypeSafe 的请求解析会忽略多余字段
                "questions": {
                    "department": {**q_req["department"], "label": dep_label},
                    "escalate": {**q_req["escalate"], "label": esc_label},
                    "frustration": {**q_req["frustration"], "label": fr_label},
                },
                "answers": answers.raw,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out / "triage_states.jsonl", "w", encoding="utf-8") as f:
        for i in range(30):
            f.write(
                json.dumps(
                    {
                        "state": make_state(rng),
                        "state_ref": "perm-%02d" % i,
                        "questions": dict(q_req),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(out / "triage-policy-template.json", "w", encoding="utf-8") as f:
        json.dump(POLICY_TEMPLATE, f, ensure_ascii=False, indent=2)

    print(
        "已生成：triage_labeled.jsonl（400 行，含 answers+labels）、"
        "triage_states.jsonl（30 行）、triage-policy-template.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
