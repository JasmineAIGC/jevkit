"""CLI 端到端：六个子命令在临时目录里跑通。"""

import json

import pytest
from conftest import TRIAGE_QUESTIONS, triage_policy

from jevkit import cli


@pytest.fixture(autouse=True)
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def _write_states(path, n=8):
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "state": "ticket number %d about a late delivery and a refund" % i,
                        "state_ref": "row-%d" % i,
                        "questions": {k: q.to_request() for k, q in TRIAGE_QUESTIONS.items()},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_labeled(path, n=60):
    from jevkit import MockBackend

    backend = MockBackend()
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            state = "invoice %d charged twice please refund" % i
            answers = backend.ask(state, TRIAGE_QUESTIONS)
            dep = answers.answers["department"]
            row = {
                "state": state,
                "state_ref": "lab-%d" % i,
                "questions": {k: q.to_request() for k, q in TRIAGE_QUESTIONS.items()},
                "answers": answers.raw,
                "labels": {
                    "department": dep.choice,
                    "escalate": answers.answers["escalate"].noul >= 0.5,
                    "frustration": int(round(answers.answers["frustration"].score)),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestDemo:
    def test_demo_writes_ledger(self, capsys):
        assert cli.main(["demo", "--log", "d.jsonl"]) == 0
        out = capsys.readouterr().out
        assert "AUTO" in out or "DEFER" in out
        lines = open("d.jsonl", encoding="utf-8").read().strip().splitlines()
        assert len(lines) == 3
        rec = json.loads(lines[0])
        for key in ("ts", "model", "raw", "policy", "actions"):
            assert key in rec


class TestCalibrateCoverage:
    def test_calibrate_synthetic(self, capsys):
        assert cli.main(["calibrate", "-n", "200"]) == 0
        out = capsys.readouterr().out
        assert "T = " in out and "ln2" in out

    def test_coverage_synthetic(self, capsys):
        assert cli.main(["coverage", "-n", "300", "--budget", "0.1"]) == 0
        out = capsys.readouterr().out
        assert "覆盖率" in out

    def test_calibrate_on_labeled_data(self, capsys):
        _write_labeled("lab.jsonl")
        assert cli.main(["calibrate", "--data", "lab.jsonl"]) == 0
        out = capsys.readouterr().out
        assert "department（choice" in out

    def test_missing_answers_without_backend_exits(self):
        _write_states("lab.jsonl", n=5)
        with pytest.raises(SystemExit, match="answers"):
            cli.main(["calibrate", "--data", "lab.jsonl"])


class TestPermute:
    def test_permute_with_policy(self, capsys):
        _write_states("states.jsonl")
        with open("p.json", "w", encoding="utf-8") as f:
            json.dump(triage_policy().to_json(), f)
        assert (
            cli.main(
                [
                    "permute",
                    "--data",
                    "states.jsonl",
                    "--policy",
                    "p.json",
                    "--backend",
                    "mock",
                    "--n-perm",
                    "4",
                ]
            )
            == 0
        )
        out = capsys.readouterr().out
        assert "department" in out and "漂移" in out


class TestCompileCheck:
    def test_compile_then_check_ok(self, capsys):
        _write_labeled("lab.jsonl", n=120)
        with open("tpl.json", "w", encoding="utf-8") as f:
            json.dump(triage_policy().to_json(), f)
        assert (
            cli.main(
                [
                    "compile",
                    "--data",
                    "lab.jsonl",
                    "--template",
                    "tpl.json",
                    "--budget",
                    "0.3",
                    "--out",
                    "policy.lock.json",
                ]
            )
            == 0
        )
        lock = json.load(open("policy.lock.json", encoding="utf-8"))
        assert lock["policy"]["version"].endswith("@budget0.30")
        assert lock["evidence"]["data_sha256"]

        assert cli.main(["check", "--data", "lab.jsonl", "--lock", "policy.lock.json"]) == 0
        out = capsys.readouterr().out
        assert "无漂移" in out

    def test_check_drift_returns_nonzero(self, capsys):
        _write_labeled("lab.jsonl", n=120)
        with open("tpl.json", "w", encoding="utf-8") as f:
            json.dump(triage_policy().to_json(), f)
        cli.main(
            [
                "compile",
                "--data",
                "lab.jsonl",
                "--template",
                "tpl.json",
                "--budget",
                "0.3",
                "--out",
                "policy.lock.json",
            ]
        )
        capsys.readouterr()
        # 破坏分布：标签全部翻转
        rows = [json.loads(line) for line in open("lab.jsonl", encoding="utf-8")]
        for r in rows:
            r["labels"]["department"] = (
                "returns" if r["labels"]["department"] != "returns" else "billing"
            )
        with open("lab.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        assert cli.main(["check", "--data", "lab.jsonl", "--lock", "policy.lock.json"]) == 1
        out = capsys.readouterr().out
        assert "漂移" in out


class TestVersion:
    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as e:
            cli.main(["--version"])
        assert e.value.code == 0
