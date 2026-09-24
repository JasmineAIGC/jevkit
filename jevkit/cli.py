"""jevkit 命令行。

    jevkit demo                                   mock 后端跑通三段式路由
    jevkit calibrate --data labeled.jsonl         分题型校准体检 + 温度
    jevkit coverage  --data labeled.jsonl         覆盖率—准确率 + 错误预算选阈值
    jevkit permute   --data states.jsonl --policy p.json   选项排列稳定性
    jevkit compile   --data labeled.jsonl --template tpl.json --budget 0.05
    jevkit check     --data new.jsonl --lock policy.lock.json

评测命令的数据用 kev train 格式 JSONL（每行 = 请求 + 每题 label，
可选内嵌 answers）。行内没有 answers 时必须给 --backend 现场取预测；
不给 --data 则用内置合成数据演示方法本身。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any

from . import __version__
from .core import Choice, Noul, Score, ScoreAnswer, make_backend
from .eval import (
    LabeledExample,
    PolicyLock,
    calibrate_report,
    check_drift,
    compile_policy,
    overconfident_pairs,
    permute_report,
    read_examples,
    sha256_file,
    with_predictions,
)
from .eval.metrics import (
    Pair,
    best_threshold,
    choice_logloss_mc,
    coverage_table,
    pairs_from_examples,
    score_mae,
)
from .policy import JsonlLedger, Policy

# ---------------------------------------------------------------- demo 素材
# 与笔记《大模型之快速决策》3.4 节同源的三条工单 + 三段式政策。

DEMO_QUESTIONS: dict[str, Any] = {
    "department": Choice(
        "Which team should handle this ticket?",
        {
            "returns": "Exchanges, refunds, wrong or damaged items",
            "shipping": "Delivery status, delays, lost packages",
            "billing": "Charges, invoices, payment problems",
        },
    ),
    "escalate": Noul("Does this need urgent human attention?"),
    "frustration": Score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
}

DEMO_FIXTURES = {
    "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.": {
        "department": {
            "type": "choice",
            "choice": "returns",
            "confidence": 0.21,
            "probabilities": {"returns": 0.47, "shipping": 0.28, "billing": 0.25},
        },
        "escalate": {"type": "noul", "noul": 0.93},
        "frustration": {
            "type": "score",
            "score": 1.44,
            "confidence": 0.78,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.00, "1": 0.56, "2": 0.44},
        },
    },
    "The invoice for order #4411 was charged twice. Please refund one of them.": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "confidence": 0.88,
            "probabilities": {"returns": 0.04, "shipping": 0.08, "billing": 0.88},
        },
        "escalate": {"type": "noul", "noul": 0.72},
        "frustration": {
            "type": "score",
            "score": 0.61,
            "confidence": 0.55,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.45, "1": 0.49, "2": 0.06},
        },
    },
    "Hi, about my thing... it's not right. Please check?": {
        "department": {
            "type": "choice",
            "choice": "shipping",
            "confidence": 0.18,
            "probabilities": {"returns": 0.30, "shipping": 0.36, "billing": 0.34},
        },
        "escalate": {"type": "noul", "noul": 0.51},
        "frustration": {
            "type": "score",
            "score": 0.98,
            "confidence": 0.31,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.35, "1": 0.32, "2": 0.33},
        },
    },
}

DEMO_SAMPLES = [
    (
        "shoes",
        "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
    ),
    ("invoice", "The invoice for order #4411 was charged twice. Please refund one of them."),
    ("vague", "Hi, about my thing... it's not right. Please check?"),
]


def demo_policy() -> Policy:
    from .policy import Gate, Signal, Tier

    return Policy(
        version="triage-v1",
        gates=(
            Gate(
                "department",
                Signal.CONFIDENCE,
                (Tier("AUTO", 0.70), Tier("DEFER", 0.50), Tier("HUMAN", 0.0)),
            ),
            Gate(
                "escalate",
                Signal.PROBABILITY,
                (Tier("ALERT", 0.90), Tier("DEFER", 0.60), Tier("NORMAL", 0.0)),
            ),
        ),
    )


# ---------------------------------------------------------------- 数据装载


def ensure_examples(args: argparse.Namespace) -> list[LabeledExample] | None:
    """读标注数据；缺 answers 时按 --backend 现场取预测。无 --data 返回 None。"""
    if not getattr(args, "data", None):
        return None
    examples = read_examples(args.data)
    missing = [ex for ex in examples if ex.answers is None]
    if missing:
        if not getattr(args, "backend", None):
            sys.exit(
                "数据里有 %d 行没有 answers。两个选择：\n"
                "  1) 行内嵌 answers（离线、免费）；\n"
                "  2) 加 --backend <spec> 现场取预测（花 token）" % len(missing)
            )
        backend = make_backend(args.backend)
        examples = with_predictions(examples, backend, model=args.model)
    return examples


def question_names(examples: list[LabeledExample]) -> list[str]:
    names: list[str] = []
    for ex in examples:
        for k in ex.questions:
            if k not in names:
                names.append(k)
    return names


def question_pairs(examples: list[LabeledExample], name: str) -> list[Pair]:
    return pairs_from_examples(
        [(ex.answers, ex.labels) for ex in examples if ex.answers is not None], name
    )


def answer_type_of(examples: list[LabeledExample], name: str) -> str:
    for ex in examples:
        if ex.answers is not None and name in ex.answers.answers:
            return ex.answers.answers[name].type
    return "?"


# ---------------------------------------------------------------- 子命令


def cmd_demo(args: argparse.Namespace) -> int:
    from .policy import decide  # 延迟导入，减少 --help 开销

    backend = make_backend("mock", fixtures=DEMO_FIXTURES)
    policy = demo_policy()
    ledger = JsonlLedger(args.log)
    print("三段式决策路由 demo（mock 后端，无需任何服务）")
    print(
        "政策 %s：路由 0.70/0.50 · 升级 0.90/0.60"
        " —— 模型负责不确定性，代码负责政策" % policy.version
    )
    for ref, state in DEMO_SAMPLES:
        answers = backend.ask(state, DEMO_QUESTIONS, model="mock-1.0")
        rec = decide(
            answers,
            policy,
            backend_name=backend.name,
            state_ref=ref,
            state=state,
            question_set_version="triage-questions-v1",
            question_set=DEMO_QUESTIONS,
        )
        ledger.append(rec)
        print("─" * 72)
        print("[%s] %s" % (ref, state))
        for q, act in rec.actions.items():
            print("  %-12s %-6s %s" % (q, act["action"], act["detail"]))
        fr = answers.answers["frustration"]
        assert isinstance(fr, ScoreAnswer)
        print(
            "  %-12s score=%.2f（%s）"
            % ("frustration", fr.score, fr.legend.get(int(round(fr.score)), "?"))
        )
    print("─" * 72)
    print("决策日志已写入 %s。字段：ts/模型/题集版本/state 哈希/完整概率/政策快照/动作" % args.log)
    print("下一步：jevkit coverage --data <标注数据> 用错误预算重校阈值")
    return 0


def _print_calibrate(
    name: str, qtype: str, pairs: list[Pair], examples: list[LabeledExample] | None
) -> None:
    print("\n══ %s（%s，n=%d）" % (name, qtype, len(pairs)))
    kind = "binary" if qtype == "noul" else "confidence"
    acc_label = "准确率" if kind == "binary" else "一致率"
    report = calibrate_report(pairs, kind=kind)
    print(
        "拟合温度 T = %.3f（fit %d / test %d，分题型各自拟合）"
        % (report.temperature, report.n_fit, report.n_test)
    )
    print()
    print("  %-14s %10s %10s %10s %10s" % ("", acc_label, "ECE", "Brier", "LogLoss"))
    for label, m in (
        ("T = 1（原始）", report.before),
        ("T = %.3f" % report.temperature, report.after),
    ):
        print(
            "  %-14s %9.1f%% %10.4f %10.4f %10.4f"
            % (label, m["accuracy"] * 100, m["ece"], m["brier"], m["logloss"])
        )
    print()
    print(
        "  %s不变：%.1f%% → %.1f%%（温度不改变 argmax，答案一个都不变）"
        % (acc_label, report.before["accuracy"] * 100, report.after["accuracy"] * 100)
    )
    print("  参考：二分类瞎猜 log loss = ln2 = %.3f；接近它说明概率几乎没有信息量" % math.log(2))
    if examples is not None:
        if qtype == "score":
            print(
                "  score 补充：期望档位 MAE = %.3f 档"
                % score_mae([(ex.answers, ex.labels) for ex in examples if ex.answers], name)
            )
        if qtype == "choice":
            print(
                "  choice 补充：多类 NLL = %.4f（整个分布质量，不只 top-1）"
                % choice_logloss_mc(
                    [(ex.answers, ex.labels) for ex in examples if ex.answers], name
                )
            )
    print("  分箱明细（test 集，T = %.3f）：" % report.temperature)
    print("    %-14s %6s %10s %10s" % ("置信区间", "样本数", "平均置信", "实际准确"))
    for lo, hi, cnt, conf, acc in report.reliability_after:
        print(
            "    %-14s %6d %9.1f%% %9.1f%%" % ("%.1f–%.1f" % (lo, hi), cnt, conf * 100, acc * 100)
        )


def cmd_calibrate(args: argparse.Namespace) -> int:
    examples = ensure_examples(args)
    if examples is None:
        print("未给 --data：用内置合成数据演示（过度自信：报 ~0.9，实际 ~0.73）")
        _print_calibrate(
            "synthetic-noul", "noul", overconfident_pairs(args.n, seed=args.seed), None
        )
        return 0
    for name in question_names(examples):
        pairs = question_pairs(examples, name)
        if len(pairs) < 20:
            print("\n══ %s：样本 %d < 20，跳过（校准结果无意义）" % (name, len(pairs)))
            continue
        _print_calibrate(name, answer_type_of(examples, name), pairs, examples)
    return 0


def _print_coverage(name: str, qtype: str, pairs: list[Pair], budget: float) -> None:
    print("\n══ %s（%s，n=%d）" % (name, qtype, len(pairs)))
    print("  覆盖率 — 准确率表（按信号从高到低累计）")
    print("  %-10s %10s %10s %12s" % ("阈值 ≥", "覆盖率", "准确率", "该段错误率"))
    for thr, cov, acc in coverage_table(pairs):
        print("  %10.2f %9.1f%% %9.1f%% %11.1f%%" % (thr, cov * 100, acc * 100, (1 - acc) * 100))
    tau, cov, acc = best_threshold(pairs, budget=budget)
    if cov > 0:
        print(
            "  错误预算 %.0f%%：可自动化 %.1f%%（阈值 %.2f，该段准确率 %.1f%%）"
            % (budget * 100, cov * 100, tau, acc * 100)
        )
    else:
        print("  错误预算 %.0f%%：无可自动化区间——即使阈值 1.0 错误率也超预算" % (budget * 100))
    print("  对照：Jev 在 5% 预算下 0.70，Kev-9B 0.45–0.57（笔记附录；别照抄，画自己的曲线）")


def cmd_coverage(args: argparse.Namespace) -> int:
    examples = ensure_examples(args)
    if examples is None:
        print("未给 --data：用内置合成数据演示")
        _print_coverage(
            "synthetic-noul", "noul", overconfident_pairs(args.n, seed=args.seed), args.budget
        )
        return 0
    for name in question_names(examples):
        pairs = question_pairs(examples, name)
        if len(pairs) < 20:
            print("\n══ %s：样本 %d < 20，跳过" % (name, len(pairs)))
            continue
        _print_coverage(name, answer_type_of(examples, name), pairs, args.budget)
    return 0


def cmd_permute(args: argparse.Namespace) -> int:
    if not args.data:
        sys.exit("permute 需要 --data（行内有 state + questions 即可，label 可选）")
    policy = None
    if args.policy:
        with open(args.policy, encoding="utf-8") as f:
            policy = Policy.from_json(json.load(f))
    backend = make_backend(args.backend)
    examples = read_examples(args.data)
    report = permute_report(
        examples, backend, policy=policy, n_perm=args.n_perm, model=args.model, seed=args.seed
    )
    flips_total = 0
    for ref, results in report:
        print("\n══ %s" % ref)
        for name, r in results.items():
            print("  %-14s %-7s %s" % (name, r.qtype, r.verdict))
            if r.qtype == "choice":
                print(
                    "      漂移 max=%.3f · KL 均值=%.3f · argmax 翻转 %.0f%% · %s"
                    % (
                        r.drift_max,
                        r.kl_mean,
                        r.argmax_flip_rate * 100,
                        ("动作翻转 %.0f%%" % (r.action_flip_rate * 100))
                        if not math.isnan(r.action_flip_rate)
                        else "未给政策",
                    )
                )
                if r.argmax_flip_rate > 0 or (
                    not math.isnan(r.action_flip_rate) and r.action_flip_rate > 0
                ):
                    flips_total += 1
    print("\n结论：%d 个 choice 题存在顺序敏感；阈值卡在漂移带内的题不适合全自动" % flips_total)
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    if not (args.data and args.template):
        sys.exit("compile 需要 --data 与 --template")
    with open(args.template, encoding="utf-8") as f:
        template = Policy.from_json(json.load(f))
    examples = ensure_examples(args)
    assert examples is not None
    lock = compile_policy(
        examples, template, budget=args.budget, data_sha256=sha256_file(args.data)
    )
    lock.save(args.out)
    print(
        "编译完成：%s（预算 %.0f%%，数据 %s，模型 %s）"
        % (
            lock.policy.version,
            args.budget * 100,
            lock.evidence.get("data_sha256") or "?",
            ", ".join(lock.evidence.get("models", [])) or "?",
        )
    )
    print()
    print(
        "  %-14s %4s %8s %9s %9s %8s  %s"
        % ("题目", "n", "新阈值", "覆盖率", "准确率", "ECE", "备注")
    )

    def _pct(v) -> float:
        return v * 100 if isinstance(v, (int, float)) and v == v else 0.0

    for ev in lock.evidence["gates"]:
        print(
            "  %-14s %4d %8.2f %8.1f%% %8.1f%% %8s  %s"
            % (
                ev["question"],
                ev["n"],
                ev["threshold"],
                _pct(ev["coverage"]),
                _pct(ev["accuracy"]),
                ("%.4f" % ev["ece"])
                if isinstance(ev["ece"], (int, float)) and ev["ece"] == ev["ece"]
                else "—",
                ev["note"] or "",
            )
        )
    print()
    print("已写入 %s。运行时：policy = PolicyLock.load(%r).policy" % (args.out, args.out))
    print("提醒：lock 绑定数据与模型——换模型 / 改题集 / 分布漂移都要重新编译")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    if not (args.data and args.lock):
        sys.exit("check 需要 --data 与 --lock")
    lock = PolicyLock.load(args.lock)
    examples = ensure_examples(args)
    assert examples is not None
    report = check_drift(examples, lock)
    print(
        "漂移检查：%d 行新数据 vs lock %s（容差：准确率 ±0.05 / 覆盖率 −0.10 / ECE ±0.05）"
        % (report.n_examples, lock.policy.version)
    )
    print()
    print("  %-14s %-14s %12s %12s  %s" % ("题目", "指标", "lock", "新数据", "判定"))
    for row in report.rows:
        print(
            "  %-14s %-14s %12.4f %12.4f  %s"
            % (
                row.question,
                row.metric,
                row.lock_value,
                row.new_value,
                "✅" if row.ok else "⚠️ 漂移",
            )
        )
    if report.ok:
        print("\n✅ 无漂移，lock 继续有效")
        return 0
    print("\n⚠️ 检出漂移：重新编译（jevkit compile）并复核政策，不要硬扛")
    return 1


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="jevkit", description="Jev / Kev 政策层 + 评测层：概率进，带证据的动作出"
    )
    ap.add_argument("--version", action="version", version="jevkit " + __version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="mock 后端跑通三段式路由")
    d.add_argument("--log", default="demo-decisions.jsonl")
    d.set_defaults(func=cmd_demo)

    for name, help_, _extra in (
        ("calibrate", "分题型校准体检 + 温度拟合", True),
        ("coverage", "覆盖率—准确率表 + 错误预算选阈值", True),
    ):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--data", help="标注 JSONL（kev train 格式）；不给则用合成数据")
        p.add_argument(
            "--backend", help="行内无 answers 时现场取预测：mock / http://… / typesafe://…"
        )
        p.add_argument("--model", default="")
        p.add_argument("-n", type=int, default=600, help="合成模式样本数")
        p.add_argument("--seed", type=int, default=7)
        if name == "coverage":
            p.add_argument("--budget", type=float, default=0.05, help="可接受错误率")
        p.set_defaults(func=(cmd_calibrate if name == "calibrate" else cmd_coverage))

    p = sub.add_parser("permute", help="选项排列稳定性（动作翻转率）")
    p.add_argument("--data", required=True)
    p.add_argument("--policy", help="政策 JSON；给了才报动作翻转率")
    p.add_argument("--backend", default="mock")
    p.add_argument("--model", default="")
    p.add_argument("--n-perm", type=int, default=6)
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_permute)

    p = sub.add_parser("compile", help="标注数据 + 政策模板 → policy.lock.json")
    p.add_argument("--data", required=True)
    p.add_argument("--template", required=True, help="政策模板 JSON")
    p.add_argument("--budget", type=float, default=0.05)
    p.add_argument("--backend", help="行内无 answers 时现场取预测")
    p.add_argument("--model", default="")
    p.add_argument("--out", default="policy.lock.json")
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("check", help="新数据 vs lock：漂移检查")
    p.add_argument("--data", required=True)
    p.add_argument("--lock", required=True)
    p.add_argument("--backend", help="行内无 answers 时现场取预测")
    p.add_argument("--model", default="")
    p.set_defaults(func=cmd_check)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
