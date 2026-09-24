"""三种后端：mock 确定性、http 重试、官方 SDK 适配（fake client 注入）。"""

import json
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jevkit import (
    BackendError,
    Choice,
    HttpBackend,
    MockBackend,
    Noul,
    TypesafeSdkBackend,
    make_backend,
)

QUESTIONS = {
    "department": Choice("Which team?", {"returns": "R", "shipping": "S", "billing": "B"}),
    "escalate": Noul("Urgent?"),
}


class TestMockBackend:
    def test_deterministic_fallback(self):
        b = MockBackend()
        a1 = b.ask("some ticket text", QUESTIONS)
        a2 = b.ask("some ticket text", QUESTIONS)
        assert a1.raw == a2.raw
        assert set(a1.answers) == {"department", "escalate"}
        dep = a1.answers["department"]
        # 选中项必须是概率 argmax；confidence 与 p_max 同源
        assert dep.choice == max(dep.probabilities, key=dep.probabilities.get)

    def test_score_only_question_set(self):
        # 回归：score 分支曾误用 choice 分支的遗留变量（仅 score 题时 NameError）
        from jevkit import Score

        answers = MockBackend().ask(
            "any state", {"frustration": Score("How upset?", ["calm", "angry"])}
        )
        a = answers.answers["frustration"]
        assert 0.0 <= a.score <= 1.0
        assert sum(a.probabilities.values()) == pytest.approx(1.0, abs=1e-3)

    def test_fixture_hit(self):
        fixtures = {"exact state": {"escalate": {"type": "noul", "noul": 0.93}}}
        a = MockBackend(fixtures=fixtures).ask("exact state", QUESTIONS)
        assert a.answers["escalate"].noul == 0.93

    def test_probabilities_sum_to_one(self):
        b = MockBackend()
        a = b.ask("another ticket", QUESTIONS)
        assert sum(a.answers["department"].probabilities.values()) == pytest.approx(1.0, abs=1e-3)

    def test_bias_changes_order_sensitivity(self):
        # 有偏置时，同一 state、不同候选顺序会得到不同分布（排列测试的物理基础）
        state = "biased ticket"
        b_bias, b_flat = MockBackend(bias=0.3), MockBackend(bias=0.0)
        q1 = QUESTIONS
        q2 = {
            "department": q1["department"].shuffled(
                list(reversed(list(q1["department"].criteria)))
            ),
            "escalate": q1["escalate"],
        }
        p_flat_1 = b_flat.ask(state, q1).answers["department"].probabilities
        p_flat_2 = b_flat.ask(state, q2).answers["department"].probabilities
        assert p_flat_1 == p_flat_2  # 无偏置 → 顺序无关
        p_bias_1 = b_bias.ask(state, q1).answers["department"].probabilities
        p_bias_2 = b_bias.ask(state, q2).answers["department"].probabilities
        assert p_bias_1 != p_bias_2  # 有偏置 → 顺序改变概率


class _CannedHandler(BaseHTTPRequestHandler):
    attempts = 0

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _CannedHandler.attempts += 1
        if _CannedHandler.attempts <= 2:
            self.send_response(429)
            self.end_headers()
            return
        resp = {
            "model": "kev-latest",
            "usage": {"input_tokens": 50, "output_tokens": 5},
            "answers": {
                "department": {
                    "type": "choice",
                    "choice": "billing",
                    "confidence": 0.9,
                    "probabilities": {"returns": 0.05, "shipping": 0.05, "billing": 0.9},
                }
            },
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class TestHttpBackend:
    def _server(self):
        _CannedHandler.attempts = 0
        server = HTTPServer(("127.0.0.1", 0), _CannedHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    def test_ask_with_retry_on_429(self):
        server = self._server()
        try:
            backend = HttpBackend("http://127.0.0.1:%d" % server.server_port, backoff_base=0.01)
            answers = backend.ask("ticket", QUESTIONS, model="kev-latest")
            assert _CannedHandler.attempts == 3  # 两次 429 + 一次成功
            assert answers.answers["department"].choice == "billing"
            assert answers.model == "kev-latest"
        finally:
            server.shutdown()

    def test_model_required(self):
        backend = HttpBackend("http://127.0.0.1:1")
        with pytest.raises(BackendError, match="model"):
            backend.ask("x", QUESTIONS)

    def test_hard_error_no_retry(self):
        class Handler(_CannedHandler):
            def do_POST(self):
                data = b"validation failed"
                self.send_response(422)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            backend = HttpBackend("http://127.0.0.1:%d" % server.server_port, max_retries=0)
            with pytest.raises(BackendError, match="422"):
                backend.ask("x", QUESTIONS, model="m")
        finally:
            server.shutdown()


class TestTypesafeSdkBackend:
    def test_import_error_message(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "typesafe_sdk", None)
        backend = TypesafeSdkBackend()
        with pytest.raises(BackendError, match=r"jevkit\[typesafe\]"):
            backend.ask("x", QUESTIONS)

    def test_fake_client_happy_path(self):
        fake_answer = types.SimpleNamespace(
            choice="billing",
            confidence=0.9,
            probabilities={"returns": 0.05, "shipping": 0.05, "billing": 0.9},
        )
        fake_noul = types.SimpleNamespace(noul=0.31)
        fake_resp = types.SimpleNamespace(
            model="jev-1.13.0",
            answers={"department": fake_answer, "escalate": fake_noul},
            usage=types.SimpleNamespace(input_tokens=120, output_tokens=8),
            request_id="req_abc",
            raw_http_response=None,
        )

        seen = {}

        def fake_system_one(state, questions, model=None, **kw):
            seen["state"] = state
            seen["questions"] = questions
            seen["model"] = model
            return fake_resp

        client = types.SimpleNamespace(system_one=fake_system_one)
        backend = TypesafeSdkBackend(client=client, model="jev-1.13.0")
        answers = backend.ask("ticket text", QUESTIONS)
        assert seen["model"] == "jev-1.13.0"
        assert answers.request_id == "req_abc"
        assert answers.answers["department"].choice == "billing"
        assert answers.answers["escalate"].noul == 0.31
        assert answers.input_tokens == 120
        assert answers.raw["escalate"]["noul"] == 0.31

    def test_make_backend_typesafe_scheme(self):
        b = make_backend("typesafe://jev-1.13.0")
        assert isinstance(b, TypesafeSdkBackend) and b.model == "jev-1.13.0"
