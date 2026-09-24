"""共享测试夹具。"""

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevkit import (  # noqa: E402
    Choice,
    ChoiceAnswer,
    Gate,
    LabeledExample,
    Noul,
    NoulAnswer,
    Policy,
    Score,
    Signal,
    Tier,
)

TRIAGE_QUESTIONS = {
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


def triage_policy() -> Policy:
    return Policy(
        version="triage-v1",
        gates=(
            Gate(
                "department",
                Signal.CONFIDENCE,
                (Tier("AUTO", 0.70), Tier("DEFER", 0.50), Tier("HUMAN", 0.0)),
            ),
            Gate(
                "escalate",
                Signal.PROBABILITY,
                (Tier("ALERT", 0.90), Tier("DEFER", 0.60), Tier("NORMAL", 0.0)),
            ),
        ),
    )


def make_calibrated_examples(n=300, seed=42, *, question="department", lo=0.35, hi=0.99):
    """构造「校准良好」的 choice 标注样本：confidence 即真实正确率。

    返回 list[LabeledExample]（内嵌 Answers），供 metrics/compile 测试
    做数值断言——数据生成规律已知，阈值选择的结果可推导。
    """
    rng = random.Random(seed)
    options = list(TRIAGE_QUESTIONS[question].criteria)
    examples = []
    for i in range(n):
        conf = rng.uniform(lo, hi)
        correct = rng.random() < conf
        chosen = options[0] if correct else options[1]
        rest = 1.0 - conf
        probs = {o: round(rest / (len(options) - 1), 6) for o in options if o != chosen}
        probs[chosen] = round(conf, 6)
        from jevkit import Answers

        answers = Answers(
            model="synth-1.0",
            answers={question: ChoiceAnswer(choice=chosen, confidence=conf, probabilities=probs)},
            raw={
                question: {
                    "type": "choice",
                    "choice": chosen,
                    "confidence": conf,
                    "probabilities": probs,
                }
            },
        )
        examples.append(
            LabeledExample(
                state="synthetic state %d" % i,
                questions={question: TRIAGE_QUESTIONS[question]},
                labels={question: options[0]},  # 真值恒为 options[0]；对错看 chosen
                answers=answers,
                state_ref="syn-%04d" % i,
            )
        )
    return examples


def make_noul_examples(n=300, seed=7, *, acc=0.9):
    """构造 noul 标注样本：概率高→大概率真，且带受控噪声。"""
    rng = random.Random(seed)
    examples = []
    for i in range(n):
        p = rng.uniform(0.05, 0.98)
        y = rng.random() < p * acc if p > 0.5 else rng.random() < p
        from jevkit import Answers

        answers = Answers(
            model="synth-1.0",
            answers={"escalate": NoulAnswer(noul=p)},
            raw={"escalate": {"type": "noul", "noul": p}},
        )
        examples.append(
            LabeledExample(
                state="urgent state %d" % i,
                questions={"escalate": TRIAGE_QUESTIONS["escalate"]},
                labels={"escalate": bool(y)},
                answers=answers,
                state_ref="noul-%04d" % i,
            )
        )
    return examples


@pytest.fixture
def triage_questions():
    return dict(TRIAGE_QUESTIONS)


@pytest.fixture
def policy():
    return triage_policy()
