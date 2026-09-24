"""排列稳定性评测：打乱选项顺序，答案和概率稳不稳？

笔记 2.3 的证据：反转选项顺序能把概率从 0.84–0.89 推到 0.93–0.96；
加一个无关选项能让两选项 log-odds 从 +0.38 掉到 0.11。
如果阈值恰好卡在漂移带里（0.90 的阈值遇上 0.89↔0.91 的摆动），
真实动作会随顺序悄悄翻转——所以本模块的核心指标不是 argmax 翻转，
而是**政策动作翻转率**：同一 state、同一政策，换个顺序问，动作变不变。

范围说明：只对 choice 题做排列（候选顺序是纯呈现顺序）；
score 的档位顺序是语义本身（Calm→Very angry），打乱等于换题，跳过；
noul 无候选，跳过。

实现：后端是 kev 服务（HttpBackend）时走原生端点 POST /v1/systemone/permute
——一次转发在服务端跑多种顺序，省 token 也省往返；其他后端（官方 TypeSafe、
mock）客户端打乱题集逐次重问。两条路径产出同构的 PermuteResult。
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..core.types import (
    Answers,
    BackendError,
    Choice,
    ChoiceAnswer,
    Noul,
    Question,
    Score,
)
from ..policy.policy import Policy, decide

__all__ = ["PermuteResult", "permute_state", "permute_report", "shuffled_choice_questions"]


@dataclass
class PermuteResult:
    question: str
    qtype: str
    n_perms: int = 0
    drift_max: float = float("nan")  # max |p_k − p'_k|（逐选项、逐次排列）
    kl_mean: float = float("nan")  # mean KL(p ‖ p')
    argmax_flip_rate: float = float("nan")  # 选中项随顺序改变的比例
    action_flip_rate: float = float("nan")  # 政策动作随顺序改变的比例（核心指标）
    picks: dict[str, int] = field(default_factory=dict)
    verdict: str = ""
    mode: str = "client"  # client（客户端打乱）| native（kev 原生端点）

    def to_json(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "qtype": self.qtype,
            "n_perms": self.n_perms,
            "drift_max": None if math.isnan(self.drift_max) else round(self.drift_max, 4),
            "kl_mean": None if math.isnan(self.kl_mean) else round(self.kl_mean, 4),
            "argmax_flip_rate": None
            if math.isnan(self.argmax_flip_rate)
            else round(self.argmax_flip_rate, 4),
            "action_flip_rate": None
            if math.isnan(self.action_flip_rate)
            else round(self.action_flip_rate, 4),
            "picks": self.picks,
            "verdict": self.verdict,
            "mode": self.mode,
        }


def shuffled_choice_questions(
    questions: Mapping[str, Question], rng: random.Random
) -> dict[str, Question]:
    """返回一份只重排了 choice 候选顺序的新题集（score/noul 原样保留）。"""
    out: dict[str, Question] = {}
    for key, q in questions.items():
        if isinstance(q, Choice) and len(q.criteria) > 1:
            order = list(q.criteria.keys())
            rng.shuffle(order)
            out[key] = q.shuffled(order)
        else:
            out[key] = q
    return out


def _with_choice_run(base: Answers, key: str, run: Mapping[str, Any]) -> Answers:
    """把原生 permute 的一次 run（order/probabilities/choice）合成为完整 Answers：
    基准答案里只替换目标题，其余题不动（与"只打乱这一题顺序"的语义一致）。"""
    probs = {str(k): float(v) for k, v in run["probabilities"].items()}
    chosen = str(run["choice"])
    k = len(probs)
    conf = (probs[chosen] - 1 / k) / (1 - 1 / k) if k > 1 and chosen in probs else 1.0
    answers = dict(base.answers)
    answers[key] = ChoiceAnswer(chosen, conf, probs)
    raw = dict(base.raw)
    raw[key] = {"type": "choice", "choice": chosen, "confidence": conf, "probabilities": probs}
    return replace(base, answers=answers, raw=raw)


def permute_state(
    backend,
    state: Any,
    questions: Mapping[str, Question],
    *,
    policy: Policy | None = None,
    n_perm: int = 6,
    model: str = "",
    seed: int = 1,
    state_ref: str = "",
) -> dict[str, PermuteResult]:
    """对一个 state 做 n_perm 次不同顺序的询问，产出每题的稳定性结果。

    后端若提供原生排列（kev 的 POST /v1/systemone/permute，见 HttpBackend.permute）
    则走服务端；否则客户端打乱题集逐次重问。两条路径产出同构的结果。
    """
    choice_keys = [k for k, q in questions.items() if isinstance(q, Choice) and len(q.criteria) > 1]
    if not choice_keys:
        raise BackendError("没有可排列的 choice 题（score/noul 不适用排列测试）")

    base = backend.ask(state, questions, model=model)
    rng = random.Random(seed)

    # 优先原生端点（一次转发跑多种顺序，省 token 也省往返）
    per_question: dict[str, list[Answers]] = {}
    modes: dict[str, str] = {}
    if getattr(backend, "supports_native_permute", False):
        for key in choice_keys:
            try:
                runs = backend.permute(
                    state, questions, question=key, n_perm=n_perm, seed=seed, model=model
                )
            except (BackendError, AttributeError):
                runs = None  # 官方端点没有这个路由（404 会以 BackendError 冒出）
            if runs:
                per_question[key] = [_with_choice_run(base, key, r) for r in runs]
                modes[key] = "native"

    # 客户端回退：对没有原生结果的题，整题集打乱重问
    perms: list[dict[str, Question]] = []
    perm_answers: list[Answers] = []
    missing = [k for k in choice_keys if k not in per_question]
    if missing:
        seen_orders: list[list[str]] = []
        attempts = 0
        while len(perms) < n_perm and attempts < n_perm * 4:
            attempts += 1
            cand = shuffled_choice_questions(questions, rng)
            sig = [
                list(cand[k].criteria.keys()) if isinstance(cand[k], Choice) else []
                for k in sorted(cand)
            ]
            if sig in seen_orders:
                continue
            seen_orders.append(sig)
            perms.append(cand)
        perm_answers = [backend.ask(state, pq, model=model) for pq in perms]
        for key in missing:
            modes[key] = "client"

    results: dict[str, PermuteResult] = {}
    for key, q in questions.items():
        if not (isinstance(q, Choice) and len(q.criteria) > 1):
            results[key] = PermuteResult(
                question=key,
                qtype=q.type,
                verdict="skipped（score 档序是语义、noul 无候选）"
                if isinstance(q, (Score, Noul))
                else "skipped",
            )
            continue

        runs = per_question.get(key, perm_answers)
        b = base.answers[key]
        drift_max, kl_sum, flips = 0.0, 0.0, 0
        picks: dict[str, int] = {}
        for pa in runs:
            a = pa.answers[key]
            for opt in b.probabilities:
                drift_max = max(
                    drift_max, abs(b.probabilities[opt] - a.probabilities.get(opt, 0.0))
                )
            kl = sum(
                p * math.log(p / max(a.probabilities.get(k, 1e-9), 1e-9))
                for k, p in b.probabilities.items()
                if p > 0
            )
            kl_sum += kl
            if a.choice != b.choice:
                flips += 1
            picks[a.choice] = picks.get(a.choice, 0) + 1

        action_flips = float("nan")
        if policy is not None:
            base_action = decide(
                base,
                policy,
                state_ref=state_ref or "permute-base",
                state=state,
                question_set=questions,
                question_set_version="permute",
            ).actions[key]["action"]
            changed = 0
            for pa in runs:
                rec = decide(
                    pa,
                    policy,
                    state_ref=state_ref or "permute-perm",
                    state=state,
                    question_set=questions,
                    question_set_version="permute",
                )
                if rec.actions[key]["action"] != base_action:
                    changed += 1
            action_flips = changed / max(len(runs), 1)

        r = PermuteResult(
            question=key,
            qtype="choice",
            n_perms=len(runs),
            drift_max=drift_max,
            kl_mean=kl_sum / max(len(runs), 1),
            argmax_flip_rate=flips / max(len(runs), 1),
            action_flip_rate=action_flips,
            picks=picks,
            mode=modes.get(key, "client"),
        )
        if not math.isnan(action_flips) and action_flips > 0:
            r.verdict = "⚠️ 动作随顺序翻转 %.0f%%——阈值卡在漂移带内，该题不适合此政策下的全自动" % (
                action_flips * 100
            )
        elif flips > 0:
            r.verdict = "⚠️ argmax 随顺序改变，但未跨过政策阈值（留意余量）"
        else:
            r.verdict = "✅ 全部顺序下答案一致"
        results[key] = r
    return results


def permute_report(
    examples: Sequence,
    backend,
    *,
    policy: Policy | None = None,
    n_perm: int = 6,
    model: str = "",
    seed: int = 1,
) -> list[tuple[str, dict[str, PermuteResult]]]:
    """对一批数据行逐行排列测试。返回 [(state_ref, {题目: 结果}), ...]。"""
    out = []
    for i, ex in enumerate(examples):
        ref = ex.state_ref or "row-%d" % (i + 1)
        out.append(
            (
                ref,
                permute_state(
                    backend,
                    ex.state,
                    ex.questions,
                    policy=policy,
                    n_perm=n_perm,
                    model=model,
                    seed=seed,
                    state_ref=ref,
                ),
            )
        )
    return out
