# -*- coding: utf-8 -*-
"""政策层：把概率翻译成动作。

核心立场（笔记 5.4）：模型负责不确定性，代码负责政策。
政策在这里是**数据 + 纯函数**，不是 DSL——

    Tier(action="AUTO",  min_signal=0.70)   # 动作名是业务字符串，框架不解释
    Tier(action="DEFER", min_signal=0.50)
    Tier(action="HUMAN", min_signal=0.0)    # 兜底

    Gate(question="department", signal=CONFIDENCE, tiers=(...))
    Policy(version="triage-v1", gates=(...))

阈值从哪来是业务的事（误报/漏报成本、错误预算），框架负责：
结构校验、确定性执行、以及把"当时为什么触发"完整写进日志。

两条硬守护（encode 自踩坑清单）：
- noul 题没有独立 confidence，禁止对它用 CONFIDENCE 信号
- 三种题型校准误差差一个数量级，阈值分 gate 各自设置，不设全局
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ..core.types import (
    Answers, AnyAnswer, NoulAnswer, PolicyError, state_digest,
)
from .record import DecisionRecord

__all__ = ["Signal", "Tier", "Gate", "Policy", "decide", "signal_value"]


class Signal(Enum):
    """门读哪种信号。

    CONFIDENCE  choice/score 的 confidence（分布摘要，(p_max−1/K)/(1−1/K)）
    PROBABILITY noul 的概率本身；choice/score 则等价 p_max
    """

    CONFIDENCE = "confidence"
    PROBABILITY = "probability"


def signal_value(answer: AnyAnswer, signal: Signal) -> float:
    """从单个答案取出信号值。"""
    if signal is Signal.CONFIDENCE:
        if isinstance(answer, NoulAnswer):
            raise PolicyError(
                "noul 题没有独立 confidence（概率本身就是置信度）。"
                "对 noul 题请用 signal=PROBABILITY —— 拿 confidence 当准确率是踩坑第 2 条")
        return float(answer.confidence)
    # PROBABILITY
    if isinstance(answer, NoulAnswer):
        return float(answer.noul)
    return float(answer.p_max)


@dataclass(frozen=True)
class Tier:
    """一段动作区间：signal ≥ min_signal 时命中本档。tiers 按 min_signal 降序排列。"""

    action: str
    min_signal: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_signal <= 1.0:
            raise PolicyError("Tier.min_signal 必须在 [0,1]，当前 %.3f" % self.min_signal)


@dataclass(frozen=True)
class Gate:
    """一道题的政策：读哪种信号、分几段、各段什么动作。"""

    question: str
    signal: Signal
    tiers: tuple[Tier, ...]

    def __post_init__(self) -> None:
        tiers = tuple(self.tiers)
        if not tiers:
            raise PolicyError("Gate %r 至少要有一档 Tier" % self.question)
        thresholds = [t.min_signal for t in tiers]
        if any(a <= b for a, b in zip(thresholds, thresholds[1:])):
            raise PolicyError(
                "Gate %r 的 tiers 必须按 min_signal 严格降序：%s"
                % (self.question, thresholds))
        object.__setattr__(self, "tiers", tiers)

    def evaluate(self, answers: Answers) -> tuple[Tier, float]:
        """命中第一个满足 signal ≥ min_signal 的档；返回 (tier, 信号值)。"""
        value = signal_value(answers.get(self.question), self.signal)
        for tier in self.tiers:
            if value >= tier.min_signal:
                return tier, value
        raise PolicyError(  # 理论到不了这里：最后一档 min_signal 应为 0
            "Gate %r 没有兜底档（signal=%.3f 落在所有档位之下）" % (self.question, value))


@dataclass(frozen=True)
class Policy:
    """一组 gate + 版本号。版本必填：它是决策日志可审计的前提。"""

    version: str
    gates: tuple[Gate, ...]

    def __post_init__(self) -> None:
        gates = tuple(self.gates)
        if not self.version or not str(self.version).strip():
            raise PolicyError("Policy.version 必填（日志审计依赖它）")
        if not gates:
            raise PolicyError("Policy 至少要有一个 Gate")
        seen = [g.question for g in gates]
        if len(set(seen)) != len(seen):
            raise PolicyError("Policy 内的 gate 题目名重复：%s" % seen)
        object.__setattr__(self, "gates", gates)

    # ---- JSON 往返（compile 产物与手写政策完全同构） ----
    def to_json(self) -> dict:
        return {
            "version": self.version,
            "gates": [
                {"question": g.question, "signal": g.signal.value,
                 "tiers": [{"action": t.action, "min_signal": t.min_signal}
                           for t in g.tiers]}
                for g in self.gates
            ],
        }

    @staticmethod
    def from_json(data: Mapping[str, Any]) -> "Policy":
        gates = tuple(
            Gate(question=g["question"],
                 signal=Signal(g["signal"]),
                 tiers=tuple(Tier(t["action"], float(t["min_signal"]))
                             for t in g["tiers"]))
            for g in data["gates"]
        )
        return Policy(version=data["version"], gates=gates)

    def decide(self, answers: Answers, **context: Any) -> DecisionRecord:
        return decide(answers, self, **context)


def decide(answers: Answers, policy: Policy, *,
           backend_name: str = "",
           state_ref: str = "",
           state: Any = None,
           question_set_version: str = "questions-untitled",
           question_set: Mapping[str, Any] | None = None,
           ) -> DecisionRecord:
    """纯函数：类型化答案 + 政策 → 决策记录（含完整审计上下文）。

    state 与 state_ref 至少给一个（日志要能回到现场）；
    question_set 不给时用 answers.raw 的键名做最小记录。
    """
    if state is None and not state_ref:
        raise PolicyError("decide 需要 state 或 state_ref 之一（决策日志必须可回溯）")

    actions: dict[str, dict[str, Any]] = {}
    for gate in policy.gates:
        tier, value = gate.evaluate(answers)
        actions[gate.question] = {
            "signal_kind": gate.signal.value,
            "signal_value": round(value, 6),
            "threshold": tier.min_signal,
            "action": tier.action,
            "detail": "%s=%.4f ≥ %.2f → %s" % (gate.signal.value, value,
                                               tier.min_signal, tier.action),
        }

    sha = state_digest(state) if state is not None else ""
    questions_snapshot = (
        {k: (q.to_request() if hasattr(q, "to_request") else dict(q))
         for k, q in question_set.items()}
        if question_set is not None else dict(answers.raw)
    )

    return DecisionRecord(
        ts=DecisionRecord.now(),
        backend=backend_name,
        model=answers.model,
        request_id=answers.request_id,
        question_set_version=question_set_version,
        state_ref=state_ref,
        state_sha256=sha,
        questions=questions_snapshot,
        raw=answers.raw,
        policy=policy.to_json(),
        actions=actions,
        latency_ms=answers.latency_ms,
        usage={"input_tokens": answers.input_tokens,
               "output_tokens": answers.output_tokens},
    )
