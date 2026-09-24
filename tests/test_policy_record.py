# -*- coding: utf-8 -*-
"""政策层：结构校验、信号守护、边界命中、决策日志往返。"""

import pytest

from jevkit import (
    Answers, ChoiceAnswer, Gate, JsonlLedger, NoulAnswer, Policy, PolicyError,
    ScoreAnswer, Signal, Tier, decide,
)
from conftest import TRIAGE_QUESTIONS, triage_policy


def answers_for(conf=0.75, noul=0.93, score=1.4, score_conf=0.6):
    probs = {"returns": round((1 - conf) / 2, 4),
             "shipping": round((1 - conf) / 2, 4),
             "billing": round(conf, 4)}
    return Answers(
        model="jev-1.13.0",
        answers={
            "department": ChoiceAnswer("billing", conf, probs),
            "escalate": NoulAnswer(noul),
            "frustration": ScoreAnswer(score, score_conf,
                                       {0: "Calm", 1: "Frustrated", 2: "Very angry"},
                                       {0: 0.1, 1: 0.6, 2: 0.3}),
        },
        raw={"escalate": {"type": "noul", "noul": noul}},
        request_id="req-1", input_tokens=99, output_tokens=9, latency_ms=120.0,
    )


class TestStructure:
    def test_tier_bounds(self):
        with pytest.raises(PolicyError, match=r"\[0,1\]"):
            Tier("AUTO", 1.5)

    def test_tiers_must_descend(self):
        with pytest.raises(PolicyError, match="严格降序"):
            Gate("q", Signal.CONFIDENCE,
                 (Tier("AUTO", 0.5), Tier("DEFER", 0.7)))

    def test_policy_version_required(self):
        with pytest.raises(PolicyError, match="version"):
            Policy(version="", gates=(Gate("q", Signal.PROBABILITY,
                                           (Tier("A", 0.9), Tier("B", 0.0))),))

    def test_duplicate_gate_questions(self):
        g = Gate("q", Signal.PROBABILITY, (Tier("A", 0.9), Tier("B", 0.0)))
        with pytest.raises(PolicyError, match="重复"):
            Policy(version="v", gates=(g, g))


class TestSignalGuards:
    def test_noul_rejects_confidence_signal(self):
        # 踩坑第 2/3 条：noul 没有独立 confidence
        gate = Gate("escalate", Signal.CONFIDENCE,
                    (Tier("A", 0.9), Tier("B", 0.0)))
        with pytest.raises(PolicyError, match="noul 题没有独立 confidence"):
            gate.evaluate(answers_for())

    def test_probability_signal_on_choice_reads_p_max(self):
        from jevkit import signal_value
        a = answers_for(conf=0.75).answers["department"]
        assert signal_value(a, Signal.PROBABILITY) == pytest.approx(0.75)

    def test_missing_question_clear_error(self):
        gate = Gate("nonexistent", Signal.CONFIDENCE,
                    (Tier("A", 0.9), Tier("B", 0.0)))
        with pytest.raises(PolicyError, match="nonexistent"):
            gate.evaluate(answers_for())


class TestDecideBoundaries:
    def test_auto_at_exact_threshold(self):
        rec = decide(answers_for(conf=0.70), triage_policy(),
                     state_ref="t1", state="s")
        assert rec.actions["department"]["action"] == "AUTO"   # ≥ 即命中

    def test_just_below_auto_is_defer(self):
        rec = decide(answers_for(conf=0.699), triage_policy(),
                     state_ref="t2", state="s")
        assert rec.actions["department"]["action"] == "DEFER"

    def test_alert_tier(self):
        rec = decide(answers_for(noul=0.93), triage_policy(),
                     state_ref="t3", state="s")
        assert rec.actions["escalate"]["action"] == "ALERT"

    def test_normal_tier(self):
        rec = decide(answers_for(noul=0.30), triage_policy(),
                     state_ref="t4", state="s")
        assert rec.actions["escalate"]["action"] == "NORMAL"

    def test_state_or_ref_required(self):
        with pytest.raises(PolicyError, match="state_ref"):
            decide(answers_for(), triage_policy())


class TestRecord:
    def test_record_fields_and_roundtrip(self, tmp_path):
        rec = decide(answers_for(), triage_policy(),
                     backend_name="mock", state_ref="shoes",
                     state="shoes arrived late",
                     question_set_version="triage-questions-v1",
                     question_set=TRIAGE_QUESTIONS)
        # 笔记 5.4③ 必存清单：模型版本/题集版本/state 引用/题集/概率/阈值/动作
        for key in ("ts", "model", "question_set_version", "state_ref",
                    "state_sha256", "questions", "raw", "policy", "actions"):
            assert rec.to_json()[key] is not None or key == "state_sha256"
        assert rec.model == "jev-1.13.0"
        assert rec.usage == {"input_tokens": 99, "output_tokens": 9}

        path = tmp_path / "decisions.jsonl"
        ledger = JsonlLedger(str(path))
        ledger.append(rec)
        loaded = list(ledger)[0]
        assert loaded.to_json() == rec.to_json()

    def test_policy_json_roundtrip(self):
        p = triage_policy()
        p2 = Policy.from_json(p.to_json())
        assert p2.to_json() == p.to_json()
        rec = decide(answers_for(), p2, state_ref="x", state="y")
        assert rec.actions["department"]["action"] == "AUTO"

    def test_determinism(self):
        r1 = decide(answers_for(), triage_policy(), state_ref="a", state="s")
        r2 = decide(answers_for(), triage_policy(), state_ref="a", state="s")
        assert r1.actions == r2.actions
        assert r1.state_sha256 == r2.state_sha256
