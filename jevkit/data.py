# -*- coding: utf-8 -*-
"""标注数据 IO —— 采纳 kev train JSONL 格式。

每行 = 一次完整请求（state + questions）+ 每题 label（可选内嵌 answers）：

    {"state": "...",
     "questions": {
       "department": {"type": "choice", "instructions": "...",
                      "criteria": {...}, "label": "billing"},
       "escalate":   {"type": "noul", "instructions": "...", "label": true},
       "frustration":{"type": "score", "instructions": "...",
                      "criteria": [...], "label": 1}},
     "answers": {"department": {"choice": "billing", ...}}   ← 可选：记录的预测

label 语义与 kev.train 一致：choice → 选项名；noul → true/false；
score → 零基档位索引。也接受顶层 "labels" 映射的变体。

设计动机：**同一份语料既喂 kev.train 微调，也喂 jevkit 评测**——
训练与评测共享分布，才谈得上闭环。answers 可选：没有时用 --backend 现场
取预测（花 token），有记录时离线直接算（免费）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Union

from .types import Answers, Question, ValidationError, question_from_request

__all__ = ["LabeledExample", "read_examples", "iter_examples", "sha256_file",
           "with_predictions", "split_half"]


@dataclass
class LabeledExample:
    state: Any
    questions: dict[str, Question]
    labels: dict[str, Any] = field(default_factory=dict)
    answers: Union[Answers, None] = None       # 内嵌预测（可选）
    state_ref: str = ""                        # 行内可选 "state_ref" 字段

    @property
    def labeled_pairs(self) -> Iterator[tuple[Answers, dict[str, Any]]]:
        """给 metrics.pairs_from_examples 直接消费的形状。"""
        if self.answers is not None:
            yield self.answers, self.labels


def _extract_labels(row: Mapping[str, Any], questions: dict[str, Question]) -> dict[str, Any]:
    labels: dict[str, Any] = {}
    top = row.get("labels")
    if isinstance(top, Mapping):
        labels.update(top)
    q = row.get("questions") or {}
    for name, qp in q.items():
        if isinstance(qp, Mapping) and "label" in qp:
            labels[name] = qp["label"]
    return {k: v for k, v in labels.items() if k in questions}


def parse_row(row: Mapping[str, Any]) -> LabeledExample:
    if "state" not in row or "questions" not in row:
        raise ValidationError(
            "数据行必须包含 state 与 questions 字段：%r" % (sorted(row.keys()),))
    questions = {k: question_from_request(v)
                 for k, v in row["questions"].items()}
    ex = LabeledExample(
        state=row["state"],
        questions=questions,
        labels=_extract_labels(row, questions),
        answers=Answers.from_response(
            {"model": row.get("model", "recorded"),
             "answers": row["answers"]}) if row.get("answers") else None,
        state_ref=str(row.get("state_ref", "")),
    )
    return ex


def iter_examples(path: str) -> Iterator[LabeledExample]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield parse_row(json.loads(line))
            except (ValidationError, KeyError, ValueError) as e:
                raise ValidationError("%s 第 %d 行解析失败：%s" % (path, lineno, e)) from e


def read_examples(path: str) -> list[LabeledExample]:
    return list(iter_examples(path))


def sha256_file(path: str) -> str:
    """数据文件指纹，进 lock 证据——阈值是"哪份数据上定的"的可审计答案。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def with_predictions(examples: list[LabeledExample], backend, *,
                     model: str = "") -> list[LabeledExample]:
    """对缺 answers 的行现场取预测（返回新列表，原行不动）。"""
    out = []
    for ex in examples:
        if ex.answers is None:
            ex.answers = backend.ask(ex.state, ex.questions, model=model)
        out.append(ex)
    return out


def split_half(seq: list) -> tuple[list, list]:
    half = len(seq) // 2
    return seq[:half], seq[half:]
