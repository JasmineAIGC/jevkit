"""评测度量：ECE / Brier / log loss / 覆盖率—准确率 / 错误预算选阈值。

全部泛化到三种题型：调用方先用 pairs_from_examples() 把带标签数据按题型
折算成 (预测概率, 是否正确) 二值对，再喂给这里的函数。

参考基线（笔记附录 A）：二分类瞎猜 log loss = ln2 ≈ 0.693；
分题型校准误差 Noul 0.012 / Choice 0.086 / Score 0.254——
一个总阈值覆盖所有题型是不可能的，所以所有报告都分题型出。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, NamedTuple

from ..core.types import Answers, ChoiceAnswer, NoulAnswer, ScoreAnswer

__all__ = [
    "ece",
    "brier",
    "logloss",
    "accuracy",
    "ReliabilityRow",
    "reliability_table",
    "coverage_table",
    "best_threshold",
    "pairs_from_examples",
    "score_mae",
    "choice_logloss_mc",
]

Pair = tuple[float, bool]


# ---------------------------------------------------------------- 二值视角度量


def ece(pairs: Sequence[Pair], bins: int = 10) -> tuple[float, list[ReliabilityRow]]:
    """期望校准误差：按置信度分箱，加权 |准确率 − 平均置信|。"""
    buckets: list[list[Pair]] = [[] for _ in range(bins)]
    for p, y in pairs:
        buckets[min(int(p * bins), bins - 1)].append((p, y))
    n = len(pairs)
    if n == 0:
        return 0.0, []
    total, rows = 0.0, []
    for i, b in enumerate(buckets):
        if not b:
            continue
        conf = sum(p for p, _ in b) / len(b)
        acc = sum(1 for _, y in b if y) / len(b)
        total += len(b) / n * abs(acc - conf)
        rows.append(
            ReliabilityRow(lo=i / bins, hi=(i + 1) / bins, n=len(b), mean_conf=conf, accuracy=acc)
        )
    return total, rows


def brier(pairs: Sequence[Pair]) -> float:
    """Brier score：概率的均方误差（越低越好）。"""
    return sum((p - (1.0 if y else 0.0)) ** 2 for p, y in pairs) / max(len(pairs), 1)


def logloss(pairs: Sequence[Pair], eps: float = 1e-12) -> float:
    """负对数似然：对"自信地猜错"惩罚最重。接近 ln2 说明概率几乎没有信息量。"""
    if not pairs:
        return float("nan")
    s = 0.0
    for p, y in pairs:
        p = min(max(p, eps), 1 - eps)
        s += -math.log(p if y else 1 - p)
    return s / len(pairs)


def accuracy(pairs: Sequence[Pair]) -> float:
    if not pairs:
        return float("nan")
    return sum(1 for p, y in pairs if (p >= 0.5) == bool(y)) / len(pairs)


class ReliabilityRow(NamedTuple):
    lo: float
    hi: float
    n: int
    mean_conf: float
    accuracy: float


def reliability_table(pairs: Sequence[Pair], bins: int = 10) -> list[ReliabilityRow]:
    return ece(pairs, bins=bins)[1]


# ---------------------------------------------------------------- 覆盖率—准确率


def coverage_table(
    pairs: Sequence[Pair],
    thresholds: Sequence[float] = (0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.0),
) -> list[tuple[float, float, float]]:
    """按置信度从高到低累计：[(阈值, 覆盖率, 准确率), ...]。"""
    n = len(pairs)
    if n == 0:
        return []
    out = []
    for thr in thresholds:
        sub = [x for x in pairs if x[0] >= thr]
        if not sub:
            continue
        acc = sum(1 for _, y in sub if y) / len(sub)
        out.append((float(thr), len(sub) / n, acc))
    return out


def best_threshold(
    pairs: Sequence[Pair], budget: float = 0.05, min_coverage: float = 0.05
) -> tuple[float, float, float]:
    """错误预算下选阈值：τ* = argmax coverage s.t. error(τ) ≤ budget。

    返回 (τ*, coverage, accuracy)。找不到满足预算的阈值时返回 (1.0, 0.0, nan)——
    这个信号本身就是结论：当前题型/模型撑不起这个自动化预算。
    """
    n = len(pairs)
    if n == 0:
        return (1.0, 0.0, float("nan"))
    best = (1.0, 0.0, float("nan"))
    for i in range(0, 100):
        thr = i / 100
        sub = [x for x in pairs if x[0] >= thr]
        if len(sub) < n * min_coverage:
            continue
        acc = sum(1 for _, y in sub if y) / len(sub)
        if 1 - acc <= budget and len(sub) / n > best[1]:
            best = (thr, len(sub) / n, acc)
    return best


# ---------------------------------------------------------------- 从数据到度量对


def pairs_from_examples(
    examples: Iterable[tuple[Answers, Mapping[str, Any]]], question: str
) -> list[Pair]:
    """把 (Answers, labels) 序列折算成二值校准对。

    - noul   → (noul 概率, 标签为真)
    - choice → (confidence, 选中项 == 标签)
    - score  → (confidence, round(期望档位) == 标签档位)

    labels 是 {题目名: 标签} 映射（kev train 格式）。
    """
    out: list[Pair] = []
    for answers, labels in examples:
        if question not in answers.answers or question not in labels:
            continue
        label = labels[question]
        a = answers.answers[question]
        if isinstance(a, NoulAnswer):
            out.append((float(a.noul), bool(label)))
        elif isinstance(a, ChoiceAnswer):
            out.append((float(a.confidence), a.choice == label))
        elif isinstance(a, ScoreAnswer):
            out.append((float(a.confidence), int(round(a.score)) == int(label)))
    return out


def score_mae(examples: Iterable[tuple[Answers, Mapping[str, Any]]], question: str) -> float:
    """score 题专用：期望档位对标签档位的平均绝对误差（档数单位）。"""
    errs = []
    for answers, labels in examples:
        if question not in answers.answers or question not in labels:
            continue
        a = answers.answers[question]
        if isinstance(a, ScoreAnswer):
            errs.append(abs(a.score - int(labels[question])))
    return sum(errs) / max(len(errs), 1)


def choice_logloss_mc(
    examples: Iterable[tuple[Answers, Mapping[str, Any]]], question: str, eps: float = 1e-12
) -> float:
    """choice 题专用：多类 NLL = mean(−log p_label)，衡量整个分布的质量。"""
    vals = []
    for answers, labels in examples:
        if question not in answers.answers or question not in labels:
            continue
        a = answers.answers[question]
        if isinstance(a, ChoiceAnswer):
            p = min(max(a.probabilities.get(str(labels[question]), 0.0), eps), 1 - eps)
            vals.append(-math.log(p))
    return sum(vals) / max(len(vals), 1)
