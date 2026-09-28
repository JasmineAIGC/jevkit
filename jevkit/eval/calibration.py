"""温度缩放校准。

过度自信是常态（合成器复现的那种"报 0.9 实际 0.73"）。温度缩放：

    p(T) = sigmoid(logit(p) / T)              二值视角（noul / confidence 对）
    q(T) = softmax(log(p) / T)                多类视角（choice/score 的完整分布）

只改概率不改排序 → 答案一个都不变，只有概率变。T > 1 摊平过度自信。

纪律（encode 自笔记 2.7）：
- 在一半数据上拟合 T，另一半上检验——只看 in-sample 是自欺
- **分割前确定性洗牌**：标注数据常按时间/来源排序，直接对半切会让
  fit/test 两半与时间漂移混杂（同一份数据行序反转，温度能差 0.15）
- **分题型分别拟合**：Nimble 用单一温度后评分题 ECE 反而从 0.105 恶化到 0.177，
  "一个总的 confidence 阈值不可能覆盖所有题型"
- Kev 每个 checkpoint 自带 2.1–2.4 的拟合温度；Jev 的温度未公开

拟合算法：粗网格（步长 0.2）定位最优区间，再黄金分割细化到 0.01——
评估次数约 55 次而非 551 次的逐点扫描，万级样本从分钟级降到亚秒级。
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from .metrics import Pair, accuracy, brier, ece, logloss

__all__ = [
    "to_logit",
    "mc_nll",
    "to_prob",
    "fit_temperature",
    "apply_temperature",
    "fit_temperature_mc",
    "apply_temperature_mc",
    "CalibrationReport",
    "calibrate_report",
    "SPLIT_SHUFFLE_SEED",
]

SPLIT_SHUFFLE_SEED = 20260924  # 固定种子：同一份数据永远得到同一份分割


def to_logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def to_prob(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def _softplus(x: float) -> float:
    """log(1 + e^x)，数值稳定版。"""
    if x > 0:
        return math.log1p(math.exp(-x)) + x
    return math.log1p(math.exp(x))


def _binary_nll(zs: Sequence[float], ys: Sequence[int], t: float) -> float:
    """NLL(T) 的稳定快速内环：-log σ(y·z/T) = softplus(−y·z/T)。"""
    inv = 1.0 / t
    log1p, exp = math.log1p, math.exp
    total = 0.0
    for z, y in zip(zs, ys, strict=True):
        a = -z * y * inv
        if a > 0:
            total += log1p(exp(-a)) + a
        else:
            total += log1p(exp(a))
    return total / len(zs)


def _minimize_1d(
    fn, lo: float, hi: float, *, coarse: float = 0.2, tol: float = 0.01
) -> tuple[float, float]:
    """粗网格 + 黄金分割求一维最小值。fn 单峰时精确到 tol。"""
    n = int((hi - lo) / coarse) + 1
    pts = [lo + i * coarse for i in range(n)] + [hi]
    best_i, best_v = min(((i, fn(t)) for i, t in enumerate(pts)), key=lambda kv: kv[1])
    left = pts[max(best_i - 1, 0)]
    right = pts[min(best_i + 1, len(pts) - 1)]
    gr = (math.sqrt(5) - 1) / 2  # 0.618…
    a, b = left, right
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = fn(c), fn(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = fn(d)
    t = (a + b) / 2
    return round(t, 4), fn(t)


def fit_temperature(
    pairs: Sequence[Pair], lo: float = 0.5, hi: float = 6.0, step: float = 0.01
) -> tuple[float, float]:
    """在 [lo, hi] 内最小化 NLL 拟合温度 T。返回 (T, 最优 NLL)。

    step 是收敛精度（黄金分割终止宽度），不再是逐点扫描的步长。
    """
    zs = [to_logit(p) for p, _ in pairs]
    ys = [1 if y else -1 for _, y in pairs]
    return _minimize_1d(lambda t: _binary_nll(zs, ys, t), lo, hi, tol=step)


def apply_temperature(pairs: Sequence[Pair], t: float) -> list[Pair]:
    return [(to_prob(to_logit(p) / t), y) for p, y in pairs]


# ---------------------------------------------------------------- 多类温度

MCRow = tuple[Mapping[Any, float], Any]  # (完整概率分布, 正确项的 key)


def _log_softmax(logits: dict[Any, float], t: float) -> dict[Any, float]:
    inv = 1.0 / t
    scaled = {k: v * inv for k, v in logits.items()}
    m = max(scaled.values())
    total = m + math.log(sum(math.exp(v - m) for v in scaled.values()))
    return {k: v - total for k, v in scaled.items()}


def mc_nll(rows: Sequence[MCRow], t: float = 1.0) -> float:
    """多类 NLL(T) = −mean log softmax(log p / T)[label]。t=1 即原始分布。"""
    return _mc_nll(rows, t)


def _mc_nll(rows: Sequence[MCRow], t: float) -> float:
    total, n = 0.0, 0
    for dist, label in rows:
        logits = {k: math.log(max(v, 1e-12)) for k, v in dist.items()}
        ls = _log_softmax(logits, t)
        total -= ls[label]
        n += 1
    return total / max(n, 1)


def fit_temperature_mc(
    rows: Sequence[MCRow], lo: float = 0.5, hi: float = 6.0, tol: float = 0.01
) -> tuple[float, float]:
    """对完整分布拟合多类温度（choice/score 用，信息量比只看 confidence 大）。"""
    return _minimize_1d(lambda t: _mc_nll(rows, t), lo, hi, tol=tol)


def apply_temperature_mc(dist: Mapping[Any, float], t: float) -> dict[Any, float]:
    """softmax(log p / T)：保序（argmax 不变）、归一。"""
    logits = {k: math.log(max(v, 1e-12)) for k, v in dist.items()}
    return {k: math.exp(v) for k, v in _log_softmax(logits, t).items()}


def mc_rows_from_examples(
    examples: Sequence[tuple[Any, Mapping[str, Any]]], question: str
) -> list[MCRow]:
    """从 (Answers, labels) 序列提取 (完整分布, label) 行——choice/score 专用。"""
    from ..core.types import ChoiceAnswer, ScoreAnswer

    rows: list[MCRow] = []
    for answers, labels in examples:
        if question not in labels:
            continue
        a = answers.answers.get(question)
        label = labels[question]
        if isinstance(a, ChoiceAnswer):
            rows.append((a.probabilities, label))
        elif isinstance(a, ScoreAnswer):
            rows.append((a.probabilities, int(label)))
    return rows


# ---------------------------------------------------------------- 报告


class CalibrationReport(NamedTuple):
    n: int
    n_fit: int
    n_test: int
    temperature: float
    kind: str  # binary（noul：p 即正类概率）
    # confidence（choice/score：p 是置信度，y 是对错）
    before: dict[str, float]  # test 集 T=1：accuracy/ece/brier/logloss
    after: dict[str, float]  # test 集 T=拟合值
    reliability_after: list[tuple]  # 分箱明细


def calibrate_report(
    pairs: Sequence[Pair], *, fit_fraction: float = 0.5, kind: str = "binary"
) -> CalibrationReport:
    """确定性洗牌后对半分割：一半拟合温度，另一半出检验指标。

    样本 < 20 直接抛错（结果无意义）。洗牌用固定种子，结果可复现且与行序无关。

    kind 决定"准确率"口径：
    - binary     p 是正类概率（noul）→ accuracy = (p≥0.5)==y 的命中率
    - confidence p 是置信度、y 是对错（choice/score）→ accuracy = 一致率 mean(y)
    """
    if len(pairs) < 20:
        raise ValueError("样本太少（%d < 20），校准结果没有意义" % len(pairs))
    # 先规范排序再洗牌：同一批样本无论行序如何，得到完全相同的分割
    ordered = sorted(pairs)
    random.Random(SPLIT_SHUFFLE_SEED).shuffle(ordered)
    half = int(len(ordered) * fit_fraction)
    fit_set, test_set = ordered[:half], ordered[half:]

    t, _ = fit_temperature(fit_set)
    after = apply_temperature(test_set, t)

    def metrics(ps: Sequence[Pair]) -> dict[str, float]:
        if kind == "confidence":
            acc = sum(1 for _, y in ps if y) / len(ps)
        else:
            acc = accuracy(ps)
        return {"accuracy": acc, "ece": ece(ps)[0], "brier": brier(ps), "logloss": logloss(ps)}

    rows = [(r.lo, r.hi, r.n, r.mean_conf, r.accuracy) for r in ece(after)[1]]
    return CalibrationReport(
        n=len(pairs),
        n_fit=len(fit_set),
        n_test=len(test_set),
        temperature=t,
        kind=kind,
        before=metrics(test_set),
        after=metrics(after),
        reliability_after=rows,
    )
