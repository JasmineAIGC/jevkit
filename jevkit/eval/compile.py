"""阈值编译与漂移检查 —— 政策层与评测层的闭环咬合点。

    标注数据 ──compile_policy──▶ policy.lock.json（阈值 + 证据）
                                       │
                              运行时 Policy.from_json 加载
                                       │
    decisions.jsonl ──回流标注──▶ check_drift 对比 lock 证据，超界报警

编译语义：对模板政策里每个 gate，在其题型的标注数据上求
    τ* = argmax_τ coverage(τ)  s.t.  error(τ) ≤ budget
然后把**最高一档（auto 档）**的阈值替换为 τ*，其余档位保持模板原样。
阈值不是抄博客的数字，是"这份数据、这个错误预算下"的最优点，
且证据（N / 覆盖率 / 准确率 / 数据指纹 / 模型版本）随 lock 一起保存。

注意：Jev 与 Kev 校准水平不同（kev 自认落后），lock 不可跨模型迁移——
换模型 = 换数据 = 重新编译。版本升级同理（踩坑第 1 条）。
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .. import __version__
from ..core.types import ChoiceAnswer, NoulAnswer
from ..policy.policy import Gate, Policy, Signal, Tier
from .data import LabeledExample
from .metrics import best_threshold, ece

__all__ = ["GateEvidence", "PolicyLock", "compile_policy", "check_drift", "DriftReport"]


def _gate_pairs(examples: Sequence[LabeledExample], gate: Gate) -> list[tuple[float, bool]]:
    """按 gate 的信号语义构造 (信号值, 是否正确) 对。"""
    out: list[tuple[float, bool]] = []
    for ex in examples:
        if ex.answers is None or gate.question not in ex.labels:
            continue
        a = ex.answers.answers.get(gate.question)
        label = ex.labels[gate.question]
        if a is None:
            continue
        if isinstance(a, NoulAnswer):
            out.append((float(a.noul), bool(label)))
        else:
            value = float(a.p_max) if gate.signal is Signal.PROBABILITY else float(a.confidence)
            if isinstance(a, ChoiceAnswer):
                correct = a.choice == label
            else:  # ScoreAnswer
                correct = int(round(a.score)) == int(label)
            out.append((value, correct))
    return out


@dataclass
class GateEvidence:
    question: str
    signal: str
    n: int
    budget: float
    threshold: float
    coverage: float
    accuracy: float
    base_accuracy: float  # 全量准确率（不设阈值的底线）
    ece: float
    clamped: bool = False  # τ* 低于次高档阈值被抬升
    note: str = ""

    def to_json(self) -> dict[str, Any]:
        import math

        def num(v: float) -> float | None:
            return None if isinstance(v, float) and not math.isfinite(v) else v

        return {
            "question": self.question,
            "signal": self.signal,
            "n": self.n,
            "budget": self.budget,
            "threshold": self.threshold,
            "coverage": num(self.coverage),
            "accuracy": num(self.accuracy),
            "base_accuracy": num(self.base_accuracy),
            "ece": num(self.ece),
            "clamped": self.clamped,
            "note": self.note,
        }


@dataclass
class PolicyLock:
    policy: Policy
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"policy": self.policy.to_json(), "evidence": self.evidence}

    @staticmethod
    def from_json(d: dict[str, Any]) -> PolicyLock:
        return PolicyLock(
            policy=Policy.from_json(d["policy"]), evidence=dict(d.get("evidence", {}))
        )

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(path: str) -> PolicyLock:
        with open(path, encoding="utf-8") as f:
            return PolicyLock.from_json(json.load(f))


def compile_policy(
    examples: Sequence[LabeledExample],
    template: Policy,
    *,
    budget: float = 0.05,
    min_coverage: float = 0.05,
    data_sha256: str = "",
) -> PolicyLock:
    """按错误预算为模板政策的每个 gate 编译 auto 档阈值。"""
    models = sorted({ex.answers.model for ex in examples if ex.answers is not None})
    new_gates: list[Gate] = []
    evidences: list[dict[str, Any]] = []

    for gate in template.gates:
        pairs = _gate_pairs(examples, gate)
        if len(pairs) < 20:
            new_gates.append(gate)
            evidences.append(
                GateEvidence(
                    question=gate.question,
                    signal=gate.signal.value,
                    n=len(pairs),
                    budget=budget,
                    threshold=gate.tiers[0].min_signal,
                    coverage=0.0,
                    accuracy=float("nan"),
                    base_accuracy=float("nan"),
                    ece=float("nan"),
                    note="样本不足（%d < 20），保留模板阈值" % len(pairs),
                ).to_json()
            )
            continue

        tau, cov, acc = best_threshold(pairs, budget=budget, min_coverage=min_coverage)
        base_acc = sum(1 for _, y in pairs if y) / len(pairs)
        calibration = ece(pairs)[0]

        clamped = False
        if len(gate.tiers) > 1 and tau <= gate.tiers[1].min_signal:
            tau = min(gate.tiers[1].min_signal + 0.01, 1.0)
            clamped = True
        # 证据必须与最终（可能被钳制）的阈值一致，否则 check_drift 会误报
        final_tau = round(tau, 2)
        sub = [p for p in pairs if p[0] >= final_tau]
        if sub:
            cov = len(sub) / len(pairs)
            acc = sum(1 for _, y in sub if y) / len(sub)
        else:
            cov, acc = 0.0, float("nan")
        tau = final_tau

        tiers = (Tier(gate.tiers[0].action, tau),) + tuple(gate.tiers[1:])
        new_gates.append(Gate(gate.question, gate.signal, tiers))

        if clamped:
            note = (
                "τ* 低于次高档阈值 %.2f，已抬升——预算与档位结构冲突，"
                "考虑放宽预算或下调 DEFER 档" % gate.tiers[1].min_signal
            )
        elif cov == 0.0:
            note = "该预算下无可自动化区间：即使阈值 1.0 错误率也超预算"
        else:
            note = ""
        evidences.append(
            GateEvidence(
                question=gate.question,
                signal=gate.signal.value,
                n=len(pairs),
                budget=budget,
                threshold=round(tau, 2),
                coverage=cov,
                accuracy=acc,
                base_accuracy=base_acc,
                ece=calibration,
                clamped=clamped,
                note=note,
            ).to_json()
        )

    compiled = Policy(version="%s@budget%.2f" % (template.version, budget), gates=tuple(new_gates))
    evidence = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "jevkit_version": __version__,
        "budget": budget,
        "data_sha256": data_sha256,
        "models": models,
        "n_examples": len(examples),
        "gates": evidences,
    }
    return PolicyLock(policy=compiled, evidence=evidence)


@dataclass
class DriftRow:
    question: str
    metric: str
    lock_value: float
    new_value: float
    ok: bool


@dataclass
class DriftReport:
    rows: list[DriftRow]
    n_examples: int

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.rows)

    def failures(self) -> list[DriftRow]:
        return [r for r in self.rows if not r.ok]


def check_drift(
    examples: Sequence[LabeledExample],
    lock: PolicyLock,
    *,
    tol_accuracy: float = 0.05,
    tol_coverage: float = 0.10,
    tol_ece: float = 0.05,
) -> DriftReport:
    """在新数据上重算每个 gate 的指标，对比 lock 里的证据。

    三个观测轴：阈值处准确率、覆盖率、校准误差（ECE）。
    任一超容差即判漂移——通常意味着：模型版本变了 / 题集改了 / 分布移了，
    三者都要求重新编译而不是硬扛。
    """
    rows: list[DriftRow] = []
    gate_by_q = {g.question: g for g in lock.policy.gates}
    for ev in lock.evidence.get("gates", []):
        q = ev["question"]
        gate = gate_by_q.get(q)
        if gate is None:
            continue
        if not ev.get("n") or ev["n"] < 10:
            continue  # lock 里这个 gate 从未编译过——没有基线，谈不上漂移
        thr = float(ev["threshold"])
        pairs = _gate_pairs(examples, gate)
        if len(pairs) < 10:
            rows.append(DriftRow(q, "n", float(ev["n"]), len(pairs), len(pairs) >= 10))
            continue
        sub = [p for p in pairs if p[0] >= thr]
        acc = (sum(1 for _, y in sub if y) / len(sub)) if sub else float("nan")
        cov = len(sub) / len(pairs)
        new_ece = ece(pairs)[0]

        if ev.get("accuracy") is not None:
            rows.append(
                DriftRow(
                    q,
                    "accuracy@thr",
                    float(ev["accuracy"]),
                    acc,
                    abs(acc - float(ev["accuracy"])) <= tol_accuracy,
                )
            )
        if ev.get("coverage") is not None:
            rows.append(
                DriftRow(
                    q,
                    "coverage@thr",
                    float(ev["coverage"]),
                    cov,
                    (float(ev["coverage"]) - cov) <= tol_coverage,
                )
            )
        if ev.get("ece") is not None:
            rows.append(
                DriftRow(
                    q, "ece", float(ev["ece"]), new_ece, abs(new_ece - float(ev["ece"])) <= tol_ece
                )
            )
    return DriftReport(rows=rows, n_examples=len(examples))
