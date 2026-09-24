"""核心层：三原语、类型化答案与可替换的后端。

这是与 Jev / kev 的 /v1/systemone 契约对应的客户端抽象——
非结构化 state 进，带概率的类型化答案出。契约以 kev/api.py 为准。
"""

from .backend import (  # noqa: F401
    Backend,
    HttpBackend,
    MockBackend,
    TypesafeSdkBackend,
    make_backend,
)
from .types import (  # noqa: F401
    Answers,
    AnyAnswer,
    Choice,
    ChoiceAnswer,
    JSONContent,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    parse_answer,
    parse_question,
    state_digest,
)
