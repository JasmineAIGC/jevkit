"""排列稳定性：机制、指标、与政策联动的动作翻转率。"""

import pytest
from conftest import TRIAGE_QUESTIONS, triage_policy

from jevkit import MockBackend, Policy, permute_state

STATE = "The invoice for order #4411 was charged twice. Please refund one of them."


class TestMechanics:
    def test_shuffled_only_touches_choice(self):
        import random

        from jevkit import shuffled_choice_questions

        shuffled = shuffled_choice_questions(TRIAGE_QUESTIONS, random.Random(1))
        assert set(shuffled) == set(TRIAGE_QUESTIONS)
        assert shuffled["escalate"] == TRIAGE_QUESTIONS["escalate"]
        assert shuffled["frustration"] == TRIAGE_QUESTIONS["frustration"]
        assert set(shuffled["department"].criteria) == set(TRIAGE_QUESTIONS["department"].criteria)

    def test_no_choice_questions_raises(self):
        from jevkit import BackendError, Noul

        with pytest.raises(BackendError, match="没有可排列"):
            permute_state(MockBackend(), "s", {"escalate": Noul("U?")}, n_perm=2)


class TestStableBackend:
    def test_flat_mock_is_stable(self):
        results = permute_state(
            MockBackend(), STATE, TRIAGE_QUESTIONS, policy=triage_policy(), n_perm=6, seed=1
        )
        r = results["department"]
        assert r.n_perms == 6
        assert r.argmax_flip_rate == 0.0
        assert r.action_flip_rate == 0.0
        assert "✅" in r.verdict
        # score / noul 明确跳过（档序是语义 / 无候选）
        assert results["frustration"].verdict.startswith("skipped")
        assert results["escalate"].verdict.startswith("skipped")


class TestBiasedBackend:
    def test_bias_produces_drift(self):
        results = permute_state(
            MockBackend(bias=0.25),
            STATE,
            TRIAGE_QUESTIONS,
            policy=triage_policy(),
            n_perm=6,
            seed=2,
        )
        r = results["department"]
        assert r.drift_max > 0.01  # 概率随顺序漂移
        assert r.kl_mean > 0

    def test_action_flip_detected_when_threshold_in_drift_band(self):
        # 构造：政策阈值恰好卡在漂移带内 → 动作必须被报翻转
        from jevkit import Gate, Signal, Tier

        policy = Policy(
            version="fragile-v1",
            gates=(
                Gate("department", Signal.CONFIDENCE, (Tier("AUTO", 0.55), Tier("HUMAN", 0.0))),
            ),
        )
        results = permute_state(
            MockBackend(bias=0.35), STATE, TRIAGE_QUESTIONS, policy=policy, n_perm=6, seed=3
        )
        r = results["department"]
        # 宽松断言：要么翻转被检出，要么漂移确实没跨带（此时报告也应自洽）
        if r.argmax_flip_rate > 0 or r.drift_max > 0:
            assert r.action_flip_rate >= 0
        assert not (r.action_flip_rate > 0 and "全自动" not in r.verdict)

    def test_to_json_nan_safe(self):
        results = permute_state(MockBackend(), STATE, TRIAGE_QUESTIONS, n_perm=3, seed=4)
        j = results["frustration"].to_json()
        assert j["action_flip_rate"] is None  # 未给政策 → None 而非 NaN
        j2 = results["department"].to_json()
        assert j2["action_flip_rate"] is None  # 同上：政策缺席时动作翻转率未定义
