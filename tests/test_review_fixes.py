"""本轮深度体检修复的回归测试。

覆盖：温度拟合提速与等价性、分割行序无关性、多类温度、
数据驱动阈值、后端错误包装、Policy schema 字段。
"""

import time

import pytest
from conftest import TRIAGE_QUESTIONS, make_calibrated_examples

from jevkit import (
    BackendError,
    Policy,
    apply_temperature_mc,
    best_threshold,
    calibrate_report,
    fit_temperature,
    fit_temperature_mc,
    make_backend,
    mc_nll,
    mc_rows_from_examples,
    overconfident_pairs,
)
from jevkit.eval.calibration import _binary_nll, to_logit


class TestFitTemperatureSpeed:
    def test_five_thousand_pairs_under_five_seconds(self):
        pairs = overconfident_pairs(5000, seed=3)
        start = time.perf_counter()
        t, loss = fit_temperature(pairs)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, "黄金分割应把万级样本拟合压到亚秒级，实测 %.1fs" % elapsed
        assert 1.0 < t < 3.5 and loss > 0

    def test_matches_exhaustive_grid(self):
        # 粗网格+黄金分割的最优值不劣于逐点扫描（差距 < 0.01 NLL）
        pairs = overconfident_pairs(400, seed=11)
        t_fast, loss_fast = fit_temperature(pairs)
        lo = 0.5
        zs = [to_logit(p) for p, _ in pairs]
        ys = [1 if y else -1 for _, y in pairs]
        best = min(
            (_binary_nll(zs, ys, lo + i * 0.01), lo + i * 0.01)
            for i in range(int((6.0 - 0.5) / 0.01) + 1)
        )
        assert loss_fast <= best[0] + 0.01


class TestOrderInsensitiveSplit:
    def test_reversing_row_order_same_temperature(self):
        pairs = overconfident_pairs(600, seed=7)
        forward = calibrate_report(pairs)
        backward = calibrate_report(list(reversed(pairs)))
        assert forward.temperature == backward.temperature
        assert forward.after == backward.after


class TestMultinomialTemperature:
    def _rows(self, n=400, seed=5, p_top=0.9, acc=0.75):
        """真值恒为 "a"；模型 75% 时候选 a 且报 0.9——典型的过度自信分布。"""
        import random

        rng = random.Random(seed)
        rows = []
        for _ in range(n):
            correct = rng.random() < acc
            picked = "a" if correct else "b"
            dist = {picked: p_top, ("b" if picked == "a" else "a"): round(1 - p_top, 4)}
            rows.append((dist, "a"))
        return rows

    def test_overconfident_distribution_gets_t_above_one(self):
        rows = self._rows()
        t, _ = fit_temperature_mc(rows)
        assert t > 1.2
        # held-out NLL 应改善
        half = len(rows) // 2
        assert mc_nll(rows[half:], t) < mc_nll(rows[half:], 1.0)

    def test_apply_preserves_argmax_and_normalizes(self):
        dist = {"a": 0.7, "b": 0.2, "c": 0.1}
        scaled = apply_temperature_mc(dist, 2.5)
        assert max(scaled, key=scaled.get) == "a"
        assert sum(scaled.values()) == pytest.approx(1.0, abs=1e-9)

    def test_rows_from_examples(self):
        examples = make_calibrated_examples(30, seed=2)
        rows = mc_rows_from_examples([(e.answers, e.labels) for e in examples], "department")
        assert len(rows) == 30
        dist, label = rows[0]
        assert set(dist) == {"returns", "shipping", "billing"} and label in dist


class TestDataDrivenThreshold:
    def test_exact_boundary_not_snapped_to_grid(self):
        # 观测值 0.975 处的断点：旧整数网格只能取 0.97/0.98，现在精确取 0.975
        pairs = [(0.975, True)] * 19 + [(0.5, False)] * 80 + [(0.5, True)] * 1
        tau, cov, acc = best_threshold(pairs, budget=0.05, min_coverage=0.05)
        assert tau == pytest.approx(0.975)
        assert cov == pytest.approx(19 / 100)

    def test_no_feasible_still_signaled(self):
        assert best_threshold([(0.99, False)] * 50, budget=0.05) == (
            1.0,
            0.0,
            pytest.approx(float("nan"), nan_ok=True),
        )


class TestBackendErrorWrapping:
    def test_mock_fixture_missing_file(self):
        with pytest.raises(BackendError, match="夹具文件读取失败"):
            make_backend("mock:///nonexistent/fixtures.json")

    def test_mock_fixture_invalid_json(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        with pytest.raises(BackendError, match="不是合法 JSON"):
            make_backend("mock://%s" % bad)


class TestPolicySchema:
    def test_json_carries_schema(self):
        import sys

        sys.path.insert(0, "tests")
        from conftest import triage_policy

        j = triage_policy().to_json()
        assert j["schema"] == 1
        assert Policy.from_json(j).to_json() == j

    def test_future_schema_rejected(self):
        with pytest.raises(Exception, match="升级 jevkit"):
            Policy.from_json({"schema": 99, "version": "x", "gates": []})


class TestNativePermute:
    def test_kev_native_endpoint_used_when_available(self):
        """本地起一个实现 /v1/systemone/permute 的假 kev 路由，验证原生路径。"""
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from jevkit import HttpBackend, permute_state

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/v1/systemone":
                    # 与 permute 路由同一分布 → 基准与各顺序无漂移
                    probs = {"returns": 0.5, "shipping": 0.3, "billing": 0.2}
                    resp = {
                        "model": "kev-latest",
                        "answers": {
                            "department": {
                                "type": "choice",
                                "choice": "returns",
                                "confidence": 0.25,
                                "probabilities": probs,
                            },
                            "escalate": {"type": "noul", "noul": 0.5},
                        },
                    }
                else:
                    opts = list(body["request"]["questions"]["department"]["criteria"])
                    runs = [
                        {
                            "order": [opts[0], opts[1], opts[2]]
                            if i == 0
                            else [opts[i % 3], opts[(i + 1) % 3], opts[(i + 2) % 3]],
                            "probabilities": {"returns": 0.5, "shipping": 0.3, "billing": 0.2},
                            "choice": "returns",
                        }
                        for i in range(body["n_perm"])
                    ]
                    resp = {"runs": runs, "argmax_stable": True, "spread": {}}
                data = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            backend = HttpBackend("http://127.0.0.1:%d" % server.server_port)
            results = permute_state(
                backend, "any state", TRIAGE_QUESTIONS, n_perm=4, model="kev-latest"
            )
            r = results["department"]
            assert r.mode == "native"
            assert r.n_perms == 4
            assert r.drift_max == 0.0  # 假服务端每次返回同一分布 → 完全稳定
            assert "✅" in r.verdict
        finally:
            server.shutdown()
