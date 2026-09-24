# -*- coding: utf-8 -*-
"""核心层：三原语、类型化答案与可替换的后端。

这是与 Jev / kev 的 /v1/systemone 契约对应的客户端抽象——
非结构化 state 进，带概率的类型化答案出。契约以 kev/api.py 为准。
"""

from .types import (  # noqa: F401
    JSONContent, Answers, AnyAnswer, Choice, ChoiceAnswer, Noul, NoulAnswer,
    Question, Score, ScoreAnswer, parse_answer, question_from_request,
    state_digest,
)
from .backend import (  # noqa: F401
    Backend, HttpBackend, MockBackend, TypesafeSdkBackend, make_backend,
)
