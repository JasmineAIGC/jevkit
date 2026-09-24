"""后端：把"问一次 System One"抽象成一个可替换的实现。

    make_backend("typesafe://jev-1.13.0")   官方 SDK（typesafe-sdk，可选依赖）
    make_backend("http://127.0.0.1:8009")   直连本地 kev serve 或任何兼容端点
    make_backend("mock")                    确定性假后端（测试 / demo）

Kev 官方声明其 POST /v1/systemone 与 TypeSafe 完全兼容，官方 Python SDK 可直连
本地 kev 服务——因此 typesafe:// 后端把 base_url 指向 http://127.0.0.1:8009、
model 填 kev-latest 即可切换到开源模型（注意：kev 校准落后于 Jev，阈值不可直接迁移）。
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

from .types import (
    Answers,
    BackendError,
    Choice,
    Noul,
    Question,
    Score,
    ValidationError,
)

__all__ = ["Backend", "HttpBackend", "TypesafeSdkBackend", "MockBackend", "make_backend"]

_RETRY_STATUS = frozenset({429, 529})


@runtime_checkable
class Backend(Protocol):
    """所有后端的最小协议：问一次，拿类型化答案。"""

    name: str

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str = "",
        timeout: float = 30.0,
    ) -> Answers: ...


# ---------------------------------------------------------------- HTTP 直连


class HttpBackend:
    """stdlib urllib 直连 POST /v1/systemone。

    无第三方依赖即可对接 kev serve（或任何遵循 TypeSafe OpenAPI 规范的端点）。
    429（限流）/ 529（过载）按指数退避自动重试；其余 HTTP 错误原样抛出。
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.name = "http"

    def _payload(self, state: Any, questions: Mapping[str, Question], model: str) -> bytes:
        return json.dumps(
            {
                "state": state,
                "model": model,
                "questions": {k: q.to_request() for k, q in questions.items()},
            },
            ensure_ascii=False,
        ).encode("utf-8")

    def _post(self, path: str, body: bytes, timeout: float) -> tuple[dict, dict]:
        """POST 并返回 (json 响应, 响应头)。429/529 指数退避重试。"""
        url = self.base_url + path
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read().decode("utf-8")), dict(resp.headers)
            except urllib.error.HTTPError as e:
                try:
                    detail = e.read().decode("utf-8", "ignore")[:500]
                except OSError:
                    detail = ""
                if e.code in _RETRY_STATUS and attempt < self.max_retries:
                    last_err = e
                    time.sleep(self.backoff_base * (2**attempt))
                    continue
                raise BackendError("HTTP %s（%s）：%s" % (e.code, url, detail)) from e
            except urllib.error.URLError as e:
                if attempt < self.max_retries:
                    last_err = e
                    time.sleep(self.backoff_base * (2**attempt))
                    continue
                raise BackendError(
                    "连不上 %s（%s）。本地 kev 请先跑："
                    "uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009"
                    % (url, e.reason)
                ) from e
        raise BackendError("重试耗尽：%r" % (last_err,))

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str = "",
        timeout: float = 30.0,
    ) -> Answers:
        if not model:
            raise BackendError("HttpBackend 需要显式 model（生产请锁定版本，如 jev-1.13.0）")
        start = time.monotonic()
        payload, resp_headers = self._post(
            "/v1/systemone", self._payload(state, questions, model), timeout
        )
        answers = Answers.from_response(payload, latency_ms=(time.monotonic() - start) * 1000)
        if answers.request_id is None:
            # kev 与 TypeSafe 都把请求 ID 放在 x-typesafe-request-id 响应头
            answers = replace(answers, request_id=resp_headers.get("x-typesafe-request-id"))
        return answers

    # kev 独有扩展：POST /v1/systemone/permute（一次转发跑多种选项顺序）
    supports_native_permute = True

    def permute(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        question: str,
        n_perm: int = 6,
        seed: int = 0,
        model: str = "",
        timeout: float = 60.0,
    ) -> list[dict]:
        """调用 kev 的原生排列端点，返回每次顺序的
        [{"order": [...], "probabilities": {...}, "choice": str}, ...]。

        官方 TypeSafe 端点没有这个路由——调用方应把 AttributeError 当作
        "该后端不支持原生排列" 的信号，回退到客户端打乱（见 permute.py）。
        """
        body = json.dumps(
            {
                "request": {
                    "state": state,
                    "model": model,
                    "questions": {k: q.to_request() for k, q in questions.items()},
                },
                "question": question,
                "n_perm": n_perm,
                "seed": seed,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        payload, _ = self._post("/v1/systemone/permute", body, timeout)
        return payload.get("runs", [])


# ---------------------------------------------------------------- 官方 SDK


class TypesafeSdkBackend:
    """包装官方 typesafe-sdk（pip install 'jevkit[typesafe]'）。

    复用官方客户端的鉴权（TYPESAFE_API_KEY）、重试（429/529 自动退避）与
    环境配置（TYPESAFE_BASE_URL / TYPESAFE_DEFAULT_MODEL）。
    client 参数可注入测试替身。
    """

    def __init__(
        self,
        *,
        model: str = "jev-1.13.0",
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client = client
        self.name = "typesafe-sdk"

    def _sdk(self):
        if self._client is None:
            try:
                import typesafe_sdk  # 懒导入：核心保持零依赖
            except ImportError as e:
                raise BackendError(
                    "官方 SDK 未安装。安装：pip install 'jevkit[typesafe]'"
                    "（或 pip install typesafe-sdk）"
                ) from e
            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = typesafe_sdk.TypeSafeClient(**kwargs)
        return self._client

    @staticmethod
    def _convert_questions(questions: Mapping[str, Question]) -> dict[str, Any]:
        try:
            import typesafe_sdk as ts
        except ImportError:
            # 注入测试替身且环境未装 SDK：退回原始 dict 题（官方 SDK 同样支持）
            return {k: q.to_request() for k, q in questions.items()}
        conv = {}
        for key, q in questions.items():
            if isinstance(q, Noul):
                conv[key] = ts.Noul(
                    instructions=q.instructions, criteria=dict(q.criteria) if q.criteria else None
                )
            elif isinstance(q, Choice):
                conv[key] = ts.Choice(instructions=q.instructions, criteria=dict(q.criteria))
            elif isinstance(q, Score):
                conv[key] = ts.Score(instructions=q.instructions, criteria=list(q.criteria))
            else:
                raise ValidationError("未知题目类型：%r" % type(q).__name__)
        return conv

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str = "",
        timeout: float = 30.0,
    ) -> Answers:
        client = self._sdk()
        start = time.monotonic()
        resp = client.system_one(
            state, self._convert_questions(questions), model=model or self.model
        )
        raw = {}
        if resp.raw_http_response is not None:
            try:
                raw = resp.raw_http_response.json()["answers"]
            except Exception:
                raw = {}
        # 统一转成 jevkit 类型（SDK 的 pydantic 模型字段名与原始响应一致）
        from .types import ChoiceAnswer, NoulAnswer, ScoreAnswer

        typed = {}
        for key, a in (resp.answers or {}).items():
            if hasattr(a, "noul"):
                typed[key] = NoulAnswer(float(a.noul))
                raw.setdefault(key, {"type": "noul", "noul": float(a.noul)})
            elif hasattr(a, "choice"):
                probs = {str(k): float(v) for k, v in a.probabilities.items()}
                typed[key] = ChoiceAnswer(
                    choice=str(a.choice), confidence=float(a.confidence), probabilities=probs
                )
                raw.setdefault(
                    key,
                    {
                        "type": "choice",
                        "choice": str(a.choice),
                        "confidence": float(a.confidence),
                        "probabilities": probs,
                    },
                )
            elif hasattr(a, "score"):
                legend = {int(k): str(v) for k, v in a.legend.items()}
                probs = {int(k): float(v) for k, v in a.probabilities.items()}
                typed[key] = ScoreAnswer(
                    score=float(a.score),
                    confidence=float(a.confidence),
                    legend=legend,
                    probabilities=probs,
                )
                raw.setdefault(
                    key,
                    {
                        "type": "score",
                        "score": float(a.score),
                        "confidence": float(a.confidence),
                        "legend": legend,
                        "probabilities": probs,
                    },
                )
            else:
                continue  # 未知 answer 类型：SDK 记 warning 跳过，这里保持一致
        usage = getattr(resp, "usage", None)
        return Answers(
            model=str(getattr(resp, "model", model or self.model)),
            answers=typed,
            raw=raw,
            request_id=getattr(resp, "request_id", None),
            input_tokens=getattr(usage, "input_tokens", None) if usage else None,
            output_tokens=getattr(usage, "output_tokens", None) if usage else None,
            latency_ms=(time.monotonic() - start) * 1000,
        )


# ---------------------------------------------------------------- Mock


class MockBackend:
    """确定性假后端。

    fixtures: {state 精确匹配: answers payload}，命中直接返回；
    未命中时用 sha256(state) 播种的随机数生成与题集对应的分布——
    同一 state + 同一题集永远得到同一组概率，测试可断言。
    bias 模拟"越靠前的选项吃位置红利"（排列测试演示用），默认 0。
    """

    def __init__(
        self,
        fixtures: Mapping[str, Mapping[str, Any]] | None = None,
        bias: float = 0.0,
        model_name: str = "mock-1.0",
    ):
        self.fixtures = dict(fixtures or {})
        self.bias = bias
        self.model_name = model_name
        self.name = "mock"

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Question],
        *,
        model: str = "",
        timeout: float = 30.0,
    ) -> Answers:
        if isinstance(state, str) and state in self.fixtures:
            payload = {"model": self.model_name, "answers": self.fixtures[state]}
            return Answers.from_response(payload, latency_ms=3.0)

        rng = random.Random("jevkit|" + _stable_state_key(state))
        answers: dict[str, Any] = {}
        for key, q in questions.items():
            if isinstance(q, Noul):
                answers[key] = {"type": "noul", "noul": round(rng.uniform(0.05, 0.97), 4)}
            elif isinstance(q, Choice):
                options = list(q.criteria.keys())
                probs = self._choice_probs(_stable_state_key(state), options)
                top = max(options, key=lambda o: probs[o])
                k = len(options)
                conf = (probs[top] - 1 / k) / (1 - 1 / k) if k > 1 else 1.0
                answers[key] = {
                    "type": "choice",
                    "choice": top,
                    "confidence": round(conf, 4),
                    "probabilities": probs,
                }
            elif isinstance(q, Score):
                n = len(q.criteria)
                raw = [rng.uniform(0.3, 1.0) for _ in range(n)]
                if self.bias:
                    raw = [v * (1 + self.bias * (n - i)) for i, v in enumerate(raw)]
                total = sum(raw)
                probs = {str(i): round(v / total, 4) for i, v in enumerate(raw)}
                expected = sum(i * (v / total) for i, v in enumerate(raw))
                legend = {str(i): text for i, text in enumerate(q.criteria)}
                p_max = max(probs.values())
                conf = (p_max - 1 / n) / (1 - 1 / n) if n > 1 else 1.0
                answers[key] = {
                    "type": "score",
                    "score": round(expected, 4),
                    "confidence": round(conf, 4),
                    "legend": legend,
                    "probabilities": probs,
                }
        return Answers.from_response({"model": self.model_name, "answers": answers}, latency_ms=3.0)

    def _choice_probs(self, state_key: str, options: list[str]) -> dict[str, float]:
        # 每个"选项"的底分由选项名播种（与位置无关）→ 无偏置时答案不随顺序变；
        # bias 再按位置加乘数 → 有偏置时顺序改变概率（排列测试的物理基础）
        raw = [
            random.Random("jevkit|opt|" + state_key + "|" + o).uniform(0.3, 1.0) for o in options
        ]
        if self.bias:
            raw = [v * (1 + self.bias * (len(options) - i)) for i, v in enumerate(raw)]
        total = sum(raw)
        probs = {o: round(v / total, 4) for o, v in zip(options, raw, strict=True)}
        # 归一化舍入残差，保证 Σp = 1
        drift = round(1.0 - sum(probs.values()), 4)
        if drift:
            last = options[-1]
            probs[last] = round(probs[last] + drift, 4)
        return probs


def _stable_state_key(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------- 工厂


def make_backend(spec: str, **kwargs: Any) -> Backend:
    """按 URL 风格的 spec 构造后端。

    - "mock" / "mock://fixtures.json"     → MockBackend（fixtures 路径可选）
    - "http://…" / "https://…"            → HttpBackend（直连 kev serve 等）
    - "typesafe://jev-1.13.0" / "typesafe"→ TypesafeSdkBackend（模型名取自 host 部分）
    """
    if spec == "mock" or spec.startswith("mock://"):
        fixtures = kwargs.pop("fixtures", None)
        if fixtures is None and spec.startswith("mock://") and len(spec) > 7:
            path = spec[len("mock://") :]
            with open(path, encoding="utf-8") as f:
                fixtures = json.load(f)
        return MockBackend(fixtures=fixtures, **kwargs)
    if spec.startswith(("http://", "https://")):
        return HttpBackend(base_url=spec, **kwargs)
    if spec == "typesafe" or spec.startswith("typesafe://"):
        model = (
            spec[len("typesafe://") :]
            if spec.startswith("typesafe://")
            else kwargs.pop("model", "jev-1.13.0")
        )
        return TypesafeSdkBackend(model=model, **kwargs)
    raise BackendError("无法识别的后端 spec：%r（可用：mock / http://… / typesafe://model）" % spec)
