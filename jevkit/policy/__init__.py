"""政策层：把概率翻译成动作。

模型负责不确定性，代码负责政策。政策在这里是**数据 + 纯函数**，不是 DSL：
阈值来自业务的误报/漏报成本，框架提供结构校验、确定性执行与完整审计日志。
"""

from .policy import Gate, Policy, Signal, Tier, decide, signal_value  # noqa: F401
from .record import DecisionRecord, JsonlLedger  # noqa: F401
