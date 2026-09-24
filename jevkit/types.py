# -*- coding: utf-8 -*-
"""三原语与类型化答案。

Jev / System One 的全部接口面：一份 state + 若干原子题（choice / score / noul），
返回类型化概率。本模块把请求侧的三种题和响应侧的三种答案都变成带校验的类型，
让下游（政策层、评测层）拿到的每个字段都能直接写进 if 语句。

硬约束（以 kev 源码 kev/api.py 为准，TypeSafe 官方文档同构）：
- state / instructions / criteria 描述都接受任意 JSON 内容（str/dict/list/数字/bool/None）
- choice 候选数 1–255；criteria 为 {选项名: 描述或 None}
- score 档数 1–255（kev 上限；TypeSafe 托管端点建议 2–10 档）
- noul 二值，仅返回 0~1 概率，没有独立 confidence
- confidence = (p_max − 1/K) / (1 − 1/K)，是分布摘要，不是准确率
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Union

__all__ = [
    "JevkitError", "ValidationError", "BackendError", "PolicyError",
    "JSONContent", "Choice", "Score", "Noul", "Question",
    "ChoiceAnswer", "ScoreAnswer", "NoulAnswer", "Answers", "state_digest",
]

# 与 kev.api.JSONContent 对齐：请求里的自由文本字段都允许任意 JSON 内容
JSONContent = Union[str, dict, list, int, float, bool, None]


class JevkitError(Exception):
    """jevkit 所有异常的基类。"""


class ValidationError(JevkitError):
    """题目定义或响应结构不合法。"""


class BackendError(JevkitError):
    """后端调用失败（连接、鉴权、过载重试耗尽等）。"""


class PolicyError(JevkitError):
    """政策与答案不匹配（缺题、信号不可用等）。"""


# ---------------------------------------------------------------- 请求侧：三种题

@dataclass(frozen=True)
class Choice:
    """从候选里选一个。返回 choice + 每候选概率 + confidence。"""

    instructions: JSONContent
    criteria: Mapping[str, JSONContent]

    def __post_init__(self) -> None:
        criteria = dict(self.criteria)
        if not 1 <= len(criteria) <= 255:
            raise ValidationError(
                "choice 题候选数必须在 1–255 之间，当前 %d" % len(criteria))
        object.__setattr__(self, "criteria", criteria)

    @property
    def type(self) -> str:
        return "choice"

    def to_request(self) -> dict:
        return {"type": "choice", "instructions": self.instructions,
                "criteria": dict(self.criteria)}

    def shuffled(self, order: list[str]) -> "Choice":
        """按给定顺序重排候选（排列测试用）。order 必须是候选的一个排列。"""
        if sorted(order) != sorted(self.criteria):
            raise ValidationError("order 不是 criteria 的排列")
        return Choice(self.instructions, {k: self.criteria[k] for k in order})


@dataclass(frozen=True)
class Score:
    """有序尺度评分。返回 score（期望档位）+ 各档概率 + legend + confidence。"""

    instructions: JSONContent
    criteria: list  # 每档描述，任意 JSON 内容

    def __post_init__(self) -> None:
        if not 1 <= len(self.criteria) <= 255:
            raise ValidationError(
                "score 题档数必须在 1–255 之间，当前 %d（TypeSafe 托管端点建议 2–10 档）"
                % len(self.criteria))

    @property
    def type(self) -> str:
        return "score"

    def to_request(self) -> dict:
        return {"type": "score", "instructions": self.instructions,
                "criteria": list(self.criteria)}

    def legend(self) -> dict[int, Any]:
        return {i: text for i, text in enumerate(self.criteria)}


@dataclass(frozen=True)
class Noul:
    """判断是否成立。只返回 0~1 概率——概率本身就是置信度，没有独立 confidence。"""

    instructions: JSONContent
    criteria: Union[Mapping[str, JSONContent], None] = None  # 可选：{"true": …, "false": …}

    @property
    def type(self) -> str:
        return "noul"

    def to_request(self) -> dict:
        req = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            req["criteria"] = dict(self.criteria)
        return req


Question = Union[Choice, Score, Noul]


def question_from_request(payload: Mapping[str, Any]) -> Question:
    """从请求 JSON 反序列化题目（读数据文件、还原日志用）。"""
    qtype = payload.get("type")
    if qtype == "choice":
        return Choice(payload["instructions"], payload["criteria"])
    if qtype == "score":
        return Score(payload["instructions"], payload["criteria"])
    if qtype == "noul":
        return Noul(payload["instructions"], payload.get("criteria"))
    raise ValidationError("未知题目类型：%r" % (qtype,))


# ---------------------------------------------------------------- 响应侧：三种答案

@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    confidence: float
    probabilities: dict[str, float]

    @property
    def type(self) -> str:
        return "choice"

    @property
    def p_max(self) -> float:
        return max(self.probabilities.values())


@dataclass(frozen=True)
class ScoreAnswer:
    score: float            # 期望档位 Σ i·p_i
    confidence: float
    legend: dict[int, str]
    probabilities: dict[int, float]

    @property
    def type(self) -> str:
        return "score"

    @property
    def p_max(self) -> float:
        return max(self.probabilities.values())


@dataclass(frozen=True)
class NoulAnswer:
    noul: float

    @property
    def type(self) -> str:
        return "noul"


AnyAnswer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


def _parse_choice_answer(a: Mapping[str, Any]) -> ChoiceAnswer:
    try:
        return ChoiceAnswer(
            choice=str(a["choice"]),
            confidence=float(a["confidence"]),
            probabilities={str(k): float(v) for k, v in a["probabilities"].items()},
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValidationError("choice 答案结构不合法：%r（%s）" % (dict(a), e))


def _parse_score_answer(a: Mapping[str, Any]) -> ScoreAnswer:
    try:
        return ScoreAnswer(
            score=float(a["score"]),
            confidence=float(a["confidence"]),
            legend={int(k): str(v) for k, v in a["legend"].items()},
            probabilities={int(k): float(v) for k, v in a["probabilities"].items()},
        )
    except (KeyError, TypeError, ValueError) as e:
        raise ValidationError("score 答案结构不合法：%r（%s）" % (dict(a), e))


def parse_answer(a: Mapping[str, Any]) -> AnyAnswer:
    """解析单个答案 dict。

    兼容两种响应形状：官方 HTTP 原生（无 type 字段，靠键判别）与
    官方 SDK 风格（带 type 判别字段）。
    """
    if "type" in a:
        t = a["type"]
        if t == "noul":
            return NoulAnswer(float(a["noul"]))
        if t == "choice":
            return _parse_choice_answer(a)
        if t == "score":
            return _parse_score_answer(a)
        raise ValidationError("未知答案类型：%r" % (t,))
    if "noul" in a:
        return NoulAnswer(float(a["noul"]))
    if "choice" in a:
        return _parse_choice_answer(a)
    if "score" in a:
        return _parse_score_answer(a)
    raise ValidationError("无法判别答案类型：%r" % (dict(a),))


@dataclass(frozen=True)
class Answers:
    """一次 /v1/systemone 调用的类型化结果。

    raw 保留原始 answers 映射（完整概率分布的审计原文），
    决策日志的 raw 字段直接取它，保证"当时是 0.91 对 0.05"永远可回答。
    """

    model: str
    answers: dict[str, AnyAnswer]
    raw: dict[str, Any] = field(default_factory=dict)
    request_id: Union[str, None] = None
    input_tokens: Union[int, None] = None
    output_tokens: Union[int, None] = None
    latency_ms: Union[float, None] = None

    @staticmethod
    def from_response(payload: Mapping[str, Any],
                      latency_ms: Union[float, None] = None) -> "Answers":
        if not isinstance(payload, Mapping) or "answers" not in payload:
            raise ValidationError("响应缺少 answers 字段：%r" % (payload,))
        answers = {k: parse_answer(v) for k, v in payload["answers"].items()}
        usage = payload.get("usage") or {}
        return Answers(
            model=str(payload.get("model", "unknown")),
            answers=answers,
            raw=dict(payload["answers"]),
            request_id=payload.get("request_id"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            latency_ms=latency_ms if latency_ms is not None else payload.get("latency_ms"),
        )

    def get(self, question: str) -> AnyAnswer:
        try:
            return self.answers[question]
        except KeyError:
            raise PolicyError("答案里没有题目 %r；现有题目：%s"
                              % (question, sorted(self.answers))) from None


def state_digest(state: Any) -> str:
    """state 的短哈希（日志引用用）。str 直接哈希，其余先做规范化 JSON。"""
    if isinstance(state, str):
        data = state.encode("utf-8")
    else:
        data = json.dumps(state, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]
