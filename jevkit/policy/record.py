# -*- coding: utf-8 -*-
"""决策日志。

生产日志必须能回答："当时是 0.91 对 0.05，还是 0.36 对 0.34？"（笔记 5.4）
因此记录里永远存**完整概率分布**（raw）而非最终标签，且模型版本 / 题集版本 /
政策快照 / 实际动作一个不缺。本 schema 是 decisions.jsonl 的超集，
新增 backend / request_id / usage 三组审计字段。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping, Union

__all__ = ["DecisionRecord", "JsonlLedger"]


@dataclass(frozen=True)
class DecisionRecord:
    """一条决策的完整审计记录。字段集对应笔记 5.4③ 的必存清单。"""

    ts: str                                        # ISO 8601（毫秒，UTC）
    backend: str                                   # typesafe-sdk / http / mock
    model: str                                     # 如 jev-1.13.0（锁版本的证据）
    request_id: Union[str, None]                   # 官方端点回传，追责用
    question_set_version: str                      # 如 triage-v1
    state_ref: str                                 # 业务侧引用（工单号等）
    state_sha256: str                              # state 短哈希
    questions: dict[str, Any]                      # 当次的完整题集
    raw: dict[str, Any]                            # 完整原始答案（含全部概率）
    policy: dict[str, Any]                         # 政策快照（版本+阈值）
    actions: dict[str, Any]                        # 每 gate 的动作与命中详情
    latency_ms: Union[float, None] = None
    usage: Union[dict[str, Any], None] = None      # input/output tokens

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    def to_line(self) -> str:
        return json.dumps(self.to_json(), ensure_ascii=False)

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> "DecisionRecord":
        return DecisionRecord(
            ts=d["ts"],
            backend=d.get("backend", ""),
            model=d["model"],
            request_id=d.get("request_id"),
            question_set_version=d.get("question_set_version", ""),
            state_ref=d.get("state_ref", ""),
            state_sha256=d.get("state_sha256", ""),
            questions=dict(d.get("questions", {})),
            raw=dict(d.get("raw", {})),
            policy=dict(d.get("policy", {})),
            actions=dict(d.get("actions", {})),
            latency_ms=d.get("latency_ms"),
            usage=d.get("usage"),
        )

    @staticmethod
    def from_line(line: str) -> "DecisionRecord":
        return DecisionRecord.from_json(json.loads(line))


class JsonlLedger:
    """决策日志的追加写 / 迭代读。"""

    def __init__(self, path: str):
        self.path = path

    def append(self, record: DecisionRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(record.to_line() + "\n")

    def extend(self, records: Iterator[DecisionRecord]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(r.to_line() + "\n")

    def __iter__(self) -> Iterator[DecisionRecord]:
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield DecisionRecord.from_line(line)

    def read_all(self) -> list[DecisionRecord]:
        return list(self)
