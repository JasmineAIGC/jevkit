# -*- coding: utf-8 -*-
"""温度缩放校准。

过度自信是常态（合成器复现的那种"报 0.9 实际 0.73"）。温度缩放：

    p(T) = sigmoid(logit(p) / T)

只改概率不改排序 → 答案一个都不变，只有概率变。T > 1 摊平过度自信。

纪律（encode 自笔记 2.7）：
- 在一半数据上拟合 T，另一半上检验——只看 in-sample 是自欺
- **分题型分别拟合**：Nimble 用单一温度后评分题 ECE 反而从 0.105 恶化到 0.177，
  "一个总的 confidence 阈值不可能覆盖所有题型"
- Kev 每个 checkpoint 自带 2.1–2.4 的拟合温度；Jev 的温度未公开
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

from .metrics import Pair, accuracy, brier, ece, logloss

__all__ = ["to_logit", "to_prob", "fit_temperature", "apply_temperature",
           "CalibrationReport", "calibrate_report"]


def to_logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def to_prob(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def fit_temperature(pairs: Sequence[Pair], lo: float = 0.5, hi: float = 6.0,
                    step: float = 0.01) -> tuple[float, float]:
    """网格搜索 T ∈ [lo, hi] 最小化 held-out NLL。返回 (T, 最优 NLL)。"""
    zs = [(to_logit(p), y) for p, y in pairs]
    best_t, best_loss = 1.0, float("inf")
    n_steps = int(round((hi - lo) / step)) + 1
    for i in range(n_steps):
        t = lo + i * step
        loss = logloss([(to_prob(z / t), y) for z, y in zs])
        if loss < best_loss:
            best_loss, best_t = loss, round(t, 4)
    return best_t, best_loss


def apply_temperature(pairs: Sequence[Pair], t: float) -> list[Pair]:
    return [(to_prob(to_logit(p) / t), y) for p, y in pairs]


class CalibrationReport(NamedTuple):
    n: int
    n_fit: int
    n_test: int
    temperature: float
    kind: str                   # binary（noul：p 即正类概率）| confidence（choice/score：p 是置信度，y 是对错）
    before: dict[str, float]      # test 集 T=1：accuracy/ece/brier/logloss
    after: dict[str, float]        # test 集 T=拟合值
    reliability_after: list[tuple]  # 分箱明细


def calibrate_report(pairs: Sequence[Pair], *, fit_fraction: float = 0.5,
                     kind: str = "binary") -> CalibrationReport:
    """对半分割：一半拟合温度，另一半出检验指标。样本 < 20 直接抛错（结果无意义）。

    kind 决定"准确率"口径：
    - binary     p 是正类概率（noul）→ accuracy = (p≥0.5)==y 的命中率
    - confidence p 是置信度、y 是对错（choice/score）→ accuracy = 一致率 mean(y)
    """
    if len(pairs) < 20:
        raise ValueError("样本太少（%d < 20），校准结果没有意义" % len(pairs))
    half = int(len(pairs) * fit_fraction)
    fit_set, test_set = list(pairs[:half]), list(pairs[half:])

    t, _ = fit_temperature(fit_set)
    after = apply_temperature(test_set, t)

    def metrics(ps: Sequence[Pair]) -> dict[str, float]:
        if kind == "confidence":
            acc = sum(1 for _, y in ps if y) / len(ps)
        else:
            acc = accuracy(ps)
        return {"accuracy": acc, "ece": ece(ps)[0],
                "brier": brier(ps), "logloss": logloss(ps)}

    rows = [(r.lo, r.hi, r.n, r.mean_conf, r.accuracy)
            for r in ece(after)[1]]
    return CalibrationReport(
        n=len(pairs), n_fit=len(fit_set), n_test=len(test_set),
        temperature=t, kind=kind,
        before=metrics(test_set),
        after=metrics(after),
        reliability_after=rows,
    )
