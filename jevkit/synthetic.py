# -*- coding: utf-8 -*-
"""合成数据：造"过度自信"的概率对，用于演示与测试校准方法本身。

复现笔记里 Nimble 论文 T=1 时的状态：挑中的答案平均报 ~0.9，
但真实正确率只有 ~0.73 → log loss ≈ 0.692 ≈ ln2。
"""

from __future__ import annotations

import random
from typing import Sequence

from .metrics import Pair

__all__ = ["overconfident_pairs", "perfectly_calibrated_pairs"]


def overconfident_pairs(n: int = 600, seed: int = 7, acc: float = 0.73) -> list[Pair]:
    """答对时平均报 ~0.92，答错时也平均报 ~0.78——典型的过度自信。"""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        y = rnd.random() < acc
        p = rnd.uniform(0.86, 0.98) if y else rnd.uniform(0.66, 0.92)
        out.append((round(p, 4), y))
    return out


def perfectly_calibrated_pairs(n: int = 600, seed: int = 11) -> list[Pair]:
    """对照组：p 就是真实正确率（校准误差应接近 0，温度应接近 1）。"""
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        p = round(rnd.uniform(0.5, 0.99), 4)
        out.append((p, rnd.random() < p))
    return out
