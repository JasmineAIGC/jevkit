"""三原语与答案解析的校验规则。"""

import pytest

from jevkit import (
    Answers,
    BackendError,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    ValidationError,
    make_backend,
    parse_answer,
    state_digest,
)


class TestChoice:
    def test_valid(self):
        c = Choice("Which team?", {"a": "desc", "b": None})
        assert c.to_request() == {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"a": "desc", "b": None},
        }

    def test_rich_json_content_allowed(self):
        # kev 契约：instructions 与 criteria 描述接受任意 JSON 内容
        c = Choice({"ask": "Which?", "ctx": ["a"]}, {"a": {"d": 1}, "b": ["l"], "c": None, "d": 3})
        req = c.to_request()
        assert req["instructions"]["ask"] == "Which?"
        assert req["criteria"]["c"] is None

    def test_too_many_candidates(self):
        with pytest.raises(ValidationError, match="1–255"):
            Choice("q", {str(i): "" for i in range(256)})

    def test_255_is_allowed(self):
        c = Choice("q", {str(i): "" for i in range(255)})
        assert len(c.criteria) == 255

    def test_shuffled_roundtrip(self):
        c = Choice("q", {"a": "1", "b": "2", "c": "3"})
        s = c.shuffled(["c", "a", "b"])
        assert list(s.criteria) == ["c", "a", "b"]
        assert s.criteria["a"] == "1"
        with pytest.raises(ValidationError, match="排列"):
            c.shuffled(["a", "b"])


class TestScore:
    def test_valid(self):
        s = Score("How bad?", ["low", "high"])
        assert s.legend() == {0: "low", 1: "high"}

    @pytest.mark.parametrize("levels", [["only-one"], list("abcdefghijk")])
    def test_level_counts_kev_allows(self, levels):
        # kev 契约：score 档数 1–255（单档与 11 档都合法）
        Score("q", levels)

    def test_level_count_over_255_rejected(self):
        with pytest.raises(ValidationError, match="1–255"):
            Score("q", ["l%d" % i for i in range(256)])


class TestNoul:
    def test_request_minimal(self):
        n = Noul("Is it urgent?")
        assert n.to_request() == {"type": "noul", "instructions": "Is it urgent?"}


class TestParseAnswer:
    def test_native_shape_without_type(self):
        a = parse_answer(
            {
                "choice": "billing",
                "confidence": 0.88,
                "probabilities": {"returns": 0.04, "shipping": 0.08, "billing": 0.88},
            }
        )
        assert isinstance(a, ChoiceAnswer)
        assert a.p_max == 0.88

    def test_sdk_shape_with_type(self):
        a = parse_answer({"type": "noul", "noul": 0.93})
        assert isinstance(a, NoulAnswer) and a.noul == 0.93

    def test_score_string_keys_become_int(self):
        a = parse_answer(
            {
                "type": "score",
                "score": 1.44,
                "confidence": 0.78,
                "legend": {"0": "Calm", "1": "Frustrated", "2": "Angry"},
                "probabilities": {"0": 0.0, "1": 0.56, "2": 0.44},
            }
        )
        assert isinstance(a, ScoreAnswer)
        assert set(a.probabilities) == {0, 1, 2}
        assert a.score == pytest.approx(1.44)

    def test_unknown_shape_raises(self):
        with pytest.raises(ValidationError, match="无法判别"):
            parse_answer({"something": "else"})


class TestAnswersFromResponse:
    def test_full_payload(self):
        payload = {
            "model": "jev-1.13.0",
            "usage": {"input_tokens": 210, "output_tokens": 12},
            "answers": {"escalate": {"type": "noul", "noul": 0.4}},
        }
        a = Answers.from_response(payload, latency_ms=88.0)
        assert a.model == "jev-1.13.0"
        assert a.input_tokens == 210 and a.output_tokens == 12
        assert a.latency_ms == 88.0
        assert a.raw == {"escalate": {"type": "noul", "noul": 0.4}}

    def test_missing_answers_raises(self):
        with pytest.raises(ValidationError, match="answers"):
            Answers.from_response({"model": "x"})


class TestStateDigest:
    def test_stable_and_short(self):
        assert state_digest("abc") == state_digest("abc")
        assert len(state_digest("abc")) == 16
        assert state_digest({"b": 1, "a": 2}) == state_digest({"a": 2, "b": 1})


class TestMakeBackend:
    def test_unknown_spec(self):
        with pytest.raises(BackendError, match="无法识别"):
            make_backend("ftp://nope")
