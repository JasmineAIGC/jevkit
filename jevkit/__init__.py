# -*- coding: utf-8 -*-
"""jevkit —— Jev / Kev（System One）政策层 + 评测层统一框架。

一句话：非结构化状态进，带概率的类型化决策出；
模型负责不确定性，代码（本框架）负责政策与证据。

    from jevkit import Choice, Noul, Score, Policy, Tier, Gate, Signal, make_backend, decide

    backend = make_backend("typesafe://jev-1.13.0")     # 或 http://127.0.0.1:8009 / mock
    answers = backend.ask(state, questions)
    policy = Policy.from_json(policy_dict)
    record = decide(answers, policy, state=state, state_ref="ticket-42")
"""

__version__ = "0.1.0"

from .types import (  # noqa: F401
    Answers, AnyAnswer, BackendError, Choice, ChoiceAnswer, JevkitError,
    Noul, NoulAnswer, PolicyError, Question, Score, ScoreAnswer,
    ValidationError, parse_answer, question_from_request, state_digest,
)
from .backend import (  # noqa: F401
    Backend, HttpBackend, MockBackend, TypesafeSdkBackend, make_backend,
)
from .policy import Gate, Policy, Signal, Tier, decide, signal_value  # noqa: F401
from .record import DecisionRecord, JsonlLedger  # noqa: F401
from .data import LabeledExample, read_examples, sha256_file  # noqa: F401
from .metrics import (  # noqa: F401
    accuracy, best_threshold, brier, choice_logloss_mc, coverage_table, ece,
    logloss, pairs_from_examples, reliability_table, score_mae,
)
from .calibration import (  # noqa: F401
    CalibrationReport, apply_temperature, calibrate_report, fit_temperature,
)
from .permute import (  # noqa: F401
    PermuteResult, permute_report, permute_state, shuffled_choice_questions,
)
from .compile import DriftReport, PolicyLock, check_drift, compile_policy  # noqa: F401
