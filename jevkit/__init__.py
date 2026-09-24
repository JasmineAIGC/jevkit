"""jevkit —— Jev / Kev（System One）政策层 + 评测层统一框架。

一句话：非结构化状态进，带概率的类型化决策出；
模型负责不确定性，代码（本框架）负责政策与证据。

包结构与三层架构一一对应：

    jevkit.core    三原语（Choice/Score/Noul）、类型化答案、可替换后端
    jevkit.policy  政策 = 数据 + 纯函数；决策日志（完整概率分布）
    jevkit.eval    校准 / 覆盖率 / 排列稳定性 / 阈值编译与漂移检查

    from jevkit import Choice, Noul, Score, Policy, Tier, Gate, Signal, make_backend, decide

    backend = make_backend("http://127.0.0.1:8009")   # kev.serve；或 typesafe://… / mock
    answers = backend.ask(state, questions, model="kev-latest")
    policy = Policy.from_json(policy_dict)
    record = decide(answers, policy, state=state, state_ref="ticket-42")
"""

__version__ = "0.1.0"

from .core import (  # noqa: F401
    Answers,
    AnyAnswer,
    Backend,
    Choice,
    ChoiceAnswer,
    HttpBackend,
    JSONContent,
    MockBackend,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    TypesafeSdkBackend,
    make_backend,
    parse_answer,
    parse_question,
    state_digest,
)
from .errors import (  # noqa: F401
    BackendError,
    JevkitError,
    PolicyError,
    ValidationError,
)
from .eval import (  # noqa: F401
    CalibrationReport,
    DriftReport,
    LabeledExample,
    PermuteResult,
    PolicyLock,
    accuracy,
    apply_temperature,
    best_threshold,
    brier,
    calibrate_report,
    check_drift,
    choice_logloss_mc,
    compile_policy,
    coverage_table,
    ece,
    fit_temperature,
    iter_examples,
    logloss,
    overconfident_pairs,
    pairs_from_examples,
    perfectly_calibrated_pairs,
    permute_report,
    permute_state,
    read_examples,
    reliability_table,
    score_mae,
    sha256_file,
    shuffled_choice_questions,
    split_half,
    with_predictions,
)
from .policy import (  # noqa: F401
    DecisionRecord,
    Gate,
    JsonlLedger,
    Policy,
    Signal,
    Tier,
    decide,
    signal_value,
)
