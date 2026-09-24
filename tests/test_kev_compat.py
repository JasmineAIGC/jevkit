"""与 jaredpalmer/kev 真实源码的兼容性验证。

这些测试需要一份 kev 检出（默认浅克隆）：

    KEV_SRC=/path/to/kev pytest tests/test_kev_compat.py

请求契约测试只用 kev.api（依赖 pydantic）；数据格式测试用 kev.data.load_records
（需要 datasets）。有运行中的 kev 服务时可加 JEVKIT_KEV_BASE_URL 做在线验证：

    KEV_SRC=… JEVKIT_KEV_BASE_URL=http://127.0.0.1:8009 pytest tests/test_kev_compat.py -k live
"""

import json
import os
import sys
from pathlib import Path

import pytest

KEV_SRC = os.environ.get("KEV_SRC", "")

pytestmark = pytest.mark.skipif(
    not KEV_SRC or not Path(KEV_SRC, "kev", "api.py").exists(),
    reason="需要 KEV_SRC 指向 kev 仓库检出（git clone https://github.com/jaredpalmer/kev）",
)


@pytest.fixture(scope="module")
def kev_api():
    sys.path.insert(0, str(Path(KEV_SRC).resolve()))
    import kev.api as api

    yield api
    sys.path.remove(str(Path(KEV_SRC).resolve()))


@pytest.fixture(scope="module")
def kev_data():
    pytest.importorskip("datasets", reason="kev.data 需要 datasets 库")
    sys.path.insert(0, str(Path(KEV_SRC).resolve()))
    import kev.data as data

    yield data
    sys.path.remove(str(Path(KEV_SRC).resolve()))


class TestRequestContract:
    """jevkit 的题目序列化必须能被 kev 的 SystemOneRequest（pydantic）原样接受。"""

    def _roundtrip(self, kev_api, state, questions):
        payload = {
            "state": state,
            "model": "kev-latest",
            "questions": {k: q.to_request() for k, q in questions.items()},
        }
        req = kev_api.SystemOneRequest(**json.loads(json.dumps(payload)))
        rec, meta = kev_api.to_record(req)  # 编码层也要接受
        assert rec["state"] and len(rec["questions"]) == len(questions)
        return req

    def test_triage_question_set(self, kev_api, triage_questions):
        self._roundtrip(kev_api, "Order #1 charged twice, please refund", triage_questions)

    def test_rich_json_content(self, kev_api):
        # kev 契约：state/instructions/criteria 描述都允许任意 JSON
        from jevkit import Choice, Noul, Score

        questions = {
            "mixed": Choice(
                instructions={"ask": "Which team?", "context": ["ticket", "escalated"]},
                criteria={"a": {"desc": "full object"}, "b": ["list", "form"], "c": None, "d": 3},
            ),
            "flag": Noul(instructions=None, criteria={"true": "clearly urgent", "false": None}),
            "levels": Score(instructions=42, criteria=["low", "mid", "high"]),
        }
        self._roundtrip(kev_api, {"ticket": 4411, "tags": ["billing"], "vip": True}, questions)

    def test_boundary_option_counts(self, kev_api):
        from jevkit import Choice, Score

        self._roundtrip(kev_api, "s", {"c255": Choice("q", {str(i): None for i in range(255)})})
        self._roundtrip(kev_api, "s", {"s1": Score("q", ["only level"])})

    def test_kev_boundaries_rejected_by_jevkit_too(self, kev_api):
        from jevkit import Choice, ValidationError

        with pytest.raises(ValidationError):
            Choice("q", {str(i): None for i in range(256)})


class TestResponseContract:
    """kev 的真实答案序列化器（to_answers）输出必须能被 jevkit 解析。"""

    def test_to_answers_roundtrip(self, kev_api, triage_questions):
        meta = [
            {
                "id": "department",
                "type": "choice",
                "keys": list(triage_questions["department"].criteria),
            },
            {"id": "escalate", "type": "noul", "keys": ["false", "true"]},
            {
                "id": "frustration",
                "type": "score",
                "keys": ["0", "1", "2"],
                "legend": {
                    str(i): t for i, t in enumerate(triage_questions["frustration"].criteria)
                },
            },
        ]
        probs = [[0.05, 0.08, 0.87], [0.4, 0.6], [0.1, 0.6, 0.3]]
        answers = kev_api.to_answers(probs, meta)
        from jevkit import Answers

        parsed = Answers.from_response({"model": "kev-latest", "answers": answers})
        assert parsed.answers["department"].choice == "billing"
        assert parsed.answers["department"].probabilities["billing"] == pytest.approx(0.87)
        assert parsed.answers["escalate"].noul == pytest.approx(0.6)
        assert parsed.answers["frustration"].score == pytest.approx(1.2, abs=0.01)


class TestDataFormat:
    """jevkit 生成的标注数据必须能被 kev.data.load_records 直接读取（可直接喂 kev.train）。"""

    def test_example_data_loads_in_kev(self, kev_data):
        path = Path(__file__).resolve().parents[1] / "examples" / "data" / "triage_labeled.jsonl"
        if not path.exists():
            pytest.skip("示例数据未生成（python examples/make_example_data.py）")
        records = kev_data.load_records(str(path))
        assert len(records) == 400
        first = records[0]
        # load_records 会补 _meta/src 并校验每题都有 label
        assert first["state"]
        assert {qid: q["label"] for qid, q in first["questions"].items()} == {
            qid: q["label"] for qid, q in first["questions"].items()
        }


class TestLiveServer:
    """在线验证（可选）：JEVKIT_KEV_BASE_URL 指向运行中的 kev.serve。"""

    BASE = os.environ.get("JEVKIT_KEV_BASE_URL", "")

    def test_ask_and_native_permute(self, kev_api, triage_questions):
        if not self.BASE:
            pytest.skip("未设置 JEVKIT_KEV_BASE_URL")
        from jevkit import Answers, HttpBackend, permute_state

        backend = HttpBackend(self.BASE)
        answers = backend.ask(
            "Order #99 was charged twice, please refund.", triage_questions, model="kev-latest"
        )
        assert isinstance(answers, Answers)
        assert answers.request_id  # x-typesafe-request-id 响应头
        assert set(answers.answers) >= {"department", "escalate", "frustration"}
        results = permute_state(
            backend,
            "Refund my damaged shoes please",
            triage_questions,
            n_perm=4,
            model="kev-latest",
        )
        assert results["department"].mode == "native"
        assert results["department"].n_perms == 4
