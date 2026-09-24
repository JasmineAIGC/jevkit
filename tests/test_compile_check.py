# -*- coding: utf-8 -*-
"""阈值编译与漂移检查：闭环两端的数值断言。"""

import pytest

from jevkit import Gate, Policy, PolicyLock, Signal, Tier, check_drift, compile_policy
from conftest import make_calibrated_examples, make_noul_examples, triage_policy


def template():
    return Policy(version="triage-tpl", gates=(
        Gate("department", Signal.CONFIDENCE, (
            Tier("AUTO", 0.95), Tier("DEFER", 0.50), Tier("HUMAN", 0.0))),
        Gate("escalate", Signal.PROBABILITY, (
            Tier("ALERT", 0.95), Tier("DEFER", 0.60), Tier("NORMAL", 0.0))),
    ))


class TestCompile:
    def test_threshold_respects_budget(self):
        examples = make_calibrated_examples(400, seed=11)
        lock = compile_policy(examples, template(), budget=0.10)
        ev = {e["question"]: e for e in lock.evidence["gates"]}["department"]
        # 校准良好的数据：高置信段的错误率应 ≤ 预算
        assert ev["coverage"] > 0
        assert 1 - ev["accuracy"] <= 0.10 + 1e-9
        gate = {g.question: g for g in lock.policy.gates}["department"]
        assert gate.tiers[0].min_signal == ev["threshold"]
        assert lock.policy.version == "triage-tpl@budget0.10"

    def test_no_data_keeps_template(self):
        lock = compile_policy([], template(), budget=0.05)
        ev = {e["question"]: e for e in lock.evidence["gates"]}["department"]
        assert ev["note"].startswith("样本不足")
        assert ev["threshold"] == 0.95

    def test_clamp_when_tau_below_defer(self):
        # 全部 confidence=0.42、错误率 45%：预算 0.45 下 τ*=0（全覆盖即达标）
        # → τ* 低于 DEFER 档 0.50，必须被抬升并标记
        from jevkit import Answers, ChoiceAnswer, LabeledExample
        from conftest import TRIAGE_QUESTIONS
        examples = []
        for i in range(100):
            correct = i < 55          # 精确 55% 正确 → 错误率恰 45% = 预算
            chosen = "returns" if correct else "shipping"
            label = "returns"
            probs = {"returns": 0.42, "shipping": 0.38, "billing": 0.20}
            answers = Answers(model="synth", answers={
                "department": ChoiceAnswer(chosen, 0.42, probs)})
            examples.append(LabeledExample(
                state="s%d" % i,
                questions={"department": TRIAGE_QUESTIONS["department"]},
                labels={"department": label}, answers=answers))
        lock = compile_policy(examples, template(), budget=0.45)
        ev = {e["question"]: e for e in lock.evidence["gates"]}["department"]
        assert ev["clamped"] is True
        gate = {g.question: g for g in lock.policy.gates}["department"]
        assert gate.tiers[0].min_signal > 0.50
        assert "冲突" in ev["note"]

    def test_noul_gate_compiles(self):
        examples = make_noul_examples(300, seed=5)
        lock = compile_policy(examples, template(), budget=0.15)
        ev = {e["question"]: e for e in lock.evidence["gates"]}["escalate"]
        assert ev["n"] == 300
        assert ev["signal"] == "probability"

    def test_lock_roundtrip(self, tmp_path):
        examples = make_calibrated_examples(200, seed=13)
        lock = compile_policy(examples, template(), budget=0.08,
                              data_sha256="deadbeefcafe0123")
        path = tmp_path / "policy.lock.json"
        lock.save(str(path))
        loaded = PolicyLock.load(str(path))
        assert loaded.to_json() == lock.to_json()
        assert loaded.evidence["data_sha256"] == "deadbeefcafe0123"


class TestCheckDrift:
    def _lock(self):
        examples = make_calibrated_examples(400, seed=21)
        return compile_policy(examples, template(), budget=0.10), examples

    def test_same_distribution_ok(self):
        lock, _ = self._lock()
        fresh = make_calibrated_examples(400, seed=99)   # 同分布不同种子
        report = check_drift(fresh, lock)
        assert report.ok, [r for r in report.failures()]

    def test_broken_distribution_flags_drift(self):
        lock, _ = self._lock()
        import random
        rng = random.Random(0)
        broken = make_calibrated_examples(400, seed=21)
        # 模拟分布崩坏：高置信段全错
        for ex in broken:
            a = ex.answers.answers["department"]
            if a.confidence > 0.8:
                ex.labels["department"] = "shipping" if a.choice == "returns" else "returns"
        report = check_drift(broken, lock)
        assert not report.ok
        assert any(r.metric == "accuracy@thr" and not r.ok for r in report.rows)

    def test_policy_usable_after_compile(self):
        from jevkit import Answers, ChoiceAnswer, NoulAnswer, decide
        lock, _ = self._lock()
        policy = lock.policy
        a = Answers(model="synth", answers={
            "department": ChoiceAnswer("returns", 0.97,
                                       {"returns": 0.97, "shipping": 0.02,
                                        "billing": 0.01}),
            "escalate": NoulAnswer(0.05)})
        rec = decide(a, policy, state_ref="t", state="s")
        assert rec.actions["department"]["action"] == "AUTO"
        assert rec.actions["escalate"]["action"] in ("NORMAL", "DEFER")
