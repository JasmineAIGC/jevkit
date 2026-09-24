# -*- coding: utf-8 -*-
"""评测层：验证概率本身可信吗、阈值该定在哪。

纪律：一切分题型（三种题型校准误差差一个数量级）；
拟合与检验分离（只看 in-sample 是自欺）；
阈值由标注数据 + 错误预算编译产出，且证据随 lock 保存。
"""

from .data import (  # noqa: F401
    LabeledExample, iter_examples, read_examples, sha256_file,
    split_half, with_predictions,
)
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
from .synthetic import (  # noqa: F401
    overconfident_pairs, perfectly_calibrated_pairs,
)
