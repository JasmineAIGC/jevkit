"""度量与校准：数值断言 + 温度缩放性质。"""

import math

import pytest
from conftest import make_calibrated_examples, make_noul_examples

from jevkit import (
    MockBackend,
    accuracy,
    apply_temperature,
    best_threshold,
    brier,
    calibrate_report,
    choice_logloss_mc,
    coverage_table,
    ece,
    fit_temperature,
    logloss,
    pairs_from_examples,
    score_mae,
)
from jevkit.eval.synthetic import overconfident_pairs, perfectly_calibrated_pairs


class TestBasicMetrics:
    def test_perfect_prediction(self):
        pairs = [(0.9, True)] * 10 + [(0.1, False)] * 10
        assert accuracy(pairs) == 1.0
        assert brier(pairs) == pytest.approx(0.01 * 2 / 2, abs=0.02)

    def test_logloss_of_random_is_ln2(self):
        pairs = [(0.5, True), (0.5, False)] * 20
        assert logloss(pairs) == pytest.approx(math.log(2), abs=1e-9)

    def test_ece_zero_when_calibrated_in_expectation(self):
        # 每箱内平均置信 ≈ 实际正确率（构造：p=0.95 全对，p=0.65 的 65% 对）
        import random

        rng = random.Random(3)
        pairs = [(0.95, True)] * 400
        pairs += [(0.65, rng.random() < 0.65) for _ in range(4000)]
        assert ece(pairs)[0] < 0.03

    def test_coverage_table_monotone(self):
        pairs = overconfident_pairs(600)
        table = coverage_table(pairs)
        covs = [c for _, c, _ in table]
        assert covs == sorted(covs)  # 表按阈值从高到低 → 覆盖率单调不降


class TestBestThreshold:
    def test_budget_respected(self):
        pairs = [(0.9, True)] * 90 + [(0.9, False)] * 10 + [(0.6, True)] * 40 + [(0.6, False)] * 60
        tau, cov, acc = best_threshold(pairs, budget=0.10, min_coverage=0.05)
        sub = [y for p, y in pairs if p >= tau]
        assert len(sub) / len(pairs) == pytest.approx(cov)
        assert 1 - sum(sub) / len(sub) <= 0.10 + 1e-9
        # 只有 0.9 段满足预算；(0.60, 0.90] 内任意阈值给出同一个子集
        assert 0.60 < tau <= 0.90
        assert cov == pytest.approx(0.5)

    def test_no_feasible_threshold(self):
        pairs = [(0.99, False)] * 100
        tau, cov, _ = best_threshold(pairs, budget=0.05)
        assert cov == 0.0  # 信号本身就是"撑不起预算"


class TestTemperature:
    def test_improves_logloss_keeps_accuracy(self):
        pairs = overconfident_pairs(800, seed=5)
        t, _ = fit_temperature(pairs[:400])
        before = logloss(pairs[400:])
        after = logloss(apply_temperature(pairs[400:], t))
        assert t > 1.2  # 过度自信 → 摊平
        assert after < before  # NLL 改善
        assert accuracy(pairs[400:]) == accuracy(apply_temperature(pairs[400:], t))

    def test_calibrated_data_keeps_t_near_one(self):
        pairs = perfectly_calibrated_pairs(2000, seed=2)
        t, _ = fit_temperature(pairs)
        assert 0.7 < t < 1.3

    def test_report_rejects_small_samples(self):
        with pytest.raises(ValueError, match="20"):
            calibrate_report(overconfident_pairs(10))

    def test_report_shape(self):
        r = calibrate_report(overconfident_pairs(600))
        assert r.n_fit + r.n_test == r.n
        assert set(r.before) == {"accuracy", "ece", "brier", "logloss"}
        assert r.reliability_after


class TestPairsFromExamples:
    def test_choice_pairs(self):
        examples = make_calibrated_examples(200, seed=1)
        pairs = pairs_from_examples([(e.answers, e.labels) for e in examples], "department")
        assert len(pairs) == 200
        acc = sum(1 for _, y in pairs if y) / len(pairs)
        assert 0.5 < acc < 1.0  # 构造的目标区间

    def test_noul_pairs(self):
        examples = make_noul_examples(100, seed=3)
        pairs = pairs_from_examples([(e.answers, e.labels) for e in examples], "escalate")
        assert all(isinstance(y, bool) for _, y in pairs)
        assert len(pairs) == 100

    def test_missing_labels_skipped(self):
        examples = make_calibrated_examples(50, seed=4)
        examples[0].labels = {}
        pairs = pairs_from_examples([(e.answers, e.labels) for e in examples], "department")
        assert len(pairs) == 49


class TestWithPredictions:
    def test_original_examples_untouched(self):
        from conftest import TRIAGE_QUESTIONS

        from jevkit import LabeledExample, with_predictions

        raw = [
            LabeledExample(state="s%d" % i, questions={"escalate": TRIAGE_QUESTIONS["escalate"]})
            for i in range(3)
        ]
        out = with_predictions(raw, MockBackend(), model="mock-1.0")
        assert all(ex.answers is not None for ex in out)
        assert all(ex.answers is None for ex in raw)  # 原对象不被修改


class TestSupplementaryMetrics:
    def test_score_mae(self):
        from conftest import TRIAGE_QUESTIONS

        from jevkit import Answers, LabeledExample, ScoreAnswer

        exs = []
        for i, (score, label) in enumerate([(1.4, 1), (0.4, 0), (2.0, 2)]):
            a = Answers(
                model="m",
                answers={
                    "frustration": ScoreAnswer(
                        score, 0.5, {0: "a", 1: "b", 2: "c"}, {0: 0.2, 1: 0.5, 2: 0.3}
                    )
                },
            )
            exs.append(
                LabeledExample(
                    state="s%d" % i,
                    questions={"frustration": TRIAGE_QUESTIONS["frustration"]},
                    labels={"frustration": label},
                    answers=a,
                )
            )
        assert score_mae([(e.answers, e.labels) for e in exs], "frustration") == pytest.approx(
            (0.4 + 0.4 + 0.0) / 3, abs=0.01
        )

    def test_choice_logloss_mc(self):
        examples = make_calibrated_examples(100, seed=8)
        v = choice_logloss_mc([(e.answers, e.labels) for e in examples], "department")
        assert v > 0
