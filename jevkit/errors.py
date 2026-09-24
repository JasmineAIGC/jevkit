"""异常层级（跨层共享，单独成模块避免环）。"""


class JevkitError(Exception):
    """jevkit 所有异常的基类。"""


class ValidationError(JevkitError):
    """题目定义或响应结构不合法。"""


class BackendError(JevkitError):
    """后端调用失败（连接、鉴权、过载重试耗尽等）。"""


class PolicyError(JevkitError):
    """政策与答案不匹配（缺题、信号不可用等）。"""
