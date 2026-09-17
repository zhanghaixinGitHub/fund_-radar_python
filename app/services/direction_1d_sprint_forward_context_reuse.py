"""单次运行内复用已完成父层的上下文，保持冻结预测函数与原始校验逻辑。

只为每个函数建立独立globals副本，把prior_forward换成严格的父层接续器；
不修改原模块、代码文件、SHA函数或模型。实际run仍按原来的递归顺序进入父层，
因此顶层提前采集、父层写预测、子层读取父回读的顺序保持不变。父层完成后，
其已验证的输入和模型上下文可传给子层，避免再次递归重建整条父链。
每个子层原有validate、原文读取、模型绑定、预测落盘和截止时间检查继续执行。
"""

from collections.abc import Mapping
from copy import deepcopy
from types import FunctionType, ModuleType


def require(condition, reason):
    if not condition:
        raise ValueError("FORWARD_CONTEXT_REUSE_" + reason)


def _copy_context(value):
    """复制数据与已载入模型；CORE辅助脚本携带的导入模块保持原身份，不能deepcopy模块。"""
    memo, seen = {}, set()

    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, ModuleType):
            memo[id(item)] = item
        elif isinstance(item, Mapping):
            for key, entry in item.items():
                visit(key)
                visit(entry)
        elif isinstance(item, (tuple, list, set)):
            for entry in item:
                visit(entry)

    visit(value)
    return deepcopy(value, memo)


class _Parent:
    """只接管父层run/context；validate和其他属性仍指向原冻结实现。"""

    def __init__(self, executor, module):
        self.executor, self.module = executor, module

    def run(self):
        require(not self.executor.audit_only, "RUN_FORBIDDEN_IN_READ_ONLY_AUDIT")
        return self.executor._execute(self.module)

    def context(self, *args, **kwargs):
        require(not args and not kwargs, "UNSUPPORTED_PARENT_CONTEXT_ARGUMENTS")
        state = self.executor.states[self.module.__name__]
        expected = "CONTEXT_READY" if self.executor.audit_only else "COMPLETED"
        require(state["phase"] == expected and state["context"] is not None, "PARENT_NOT_COMPLETED")
        # 子层不得借共享dict/ndarray修改父层已验收输入；模型仅复制内存对象，不替换磁盘模型。
        return _copy_context(state["context"])

    def __getattr__(self, name):
        return getattr(self.module, name)


class Executor:
    """一次性执行容器，调用者负责冻结图/代码摘要、持有原run.lock和记录实际验收。

    此类本身没有网络/文件入口，也不绕过任何模型或来源校验。每次调度必须新建
    实例；不能把缓存带到下次半小时运行。只读审计模式只能组装context，禁止run。
    """

    def __init__(self, terminal, *, audit_only=False):
        self.terminal, self.audit_only, self.used = terminal, audit_only, False
        chain, seen = [], set()
        current = terminal
        while True:
            require(current.__name__ not in seen, "CYCLIC_PARENT_GRAPH")
            seen.add(current.__name__)
            require(all(callable(getattr(current, k, None)) for k in ("run", "context", "report")), "UNKNOWN_LAYER")
            chain.append(current)
            if not hasattr(current, "prior_forward"):
                break
            current = current.prior_forward
        self.modules = tuple(reversed(chain))
        self.states = {
            m.__name__: {"phase": "NEW", "context": None, "result": None, "context_calls": 0} for m in self.modules
        }
        self.events = []
        self.clones = {}
        for module in self.modules:
            environment = dict(vars(module))
            if hasattr(module, "prior_forward"):
                environment["prior_forward"] = _Parent(self, module.prior_forward)
            original_context = self._clone(module.context, module, environment)

            def save_context(*args, _module=module, _context=original_context, **kwargs):
                value = _context(*args, **kwargs)
                state = self.states[_module.__name__]
                state["context"], state["context_calls"] = value, state["context_calls"] + 1
                self.events.append((_module.__name__, "CONTEXT_VALIDATED"))
                return value

            environment["context"] = save_context
            environment["report"] = self._clone(module.report, module, environment)
            self.clones[module.__name__] = {
                "run": self._clone(module.run, module, environment),
                "context": save_context,
            }

    @staticmethod
    def _clone(function, module, environment):
        require(function.__globals__ is vars(module), "FUNCTION_NOT_OWNED_BY_LAYER")
        copied = FunctionType(
            function.__code__, environment, function.__name__, function.__defaults__, function.__closure__
        )
        copied.__kwdefaults__ = function.__kwdefaults__
        return copied

    def _execute(self, module):
        state = self.states[module.__name__]
        if state["phase"] == "COMPLETED":
            return deepcopy(state["result"])
        require(state["phase"] == "NEW", "LAYER_REENTERED_OR_FAILED")
        state["phase"] = "RUNNING"
        self.events.append((module.__name__, "RUN_STARTED"))
        try:
            result = self.clones[module.__name__]["run"]()
            require(state["context"] is not None, "RUN_DID_NOT_VALIDATE_CONTEXT")
        except Exception:
            state["phase"] = "FAILED"
            raise
        state["phase"], state["result"] = "COMPLETED", result
        self.events.append((module.__name__, "RUN_COMPLETED"))
        return result

    def run(self):
        require(not self.audit_only and not self.used, "EXECUTOR_ALREADY_USED_OR_AUDIT_ONLY")
        self.used = True
        return self._execute(self.terminal)

    def audit_contexts(self):
        """只读审计从父到子验证全部上下文；不会宣称父层已经执行或保存预测。"""
        require(self.audit_only and not self.used, "AUDIT_ALREADY_USED_OR_RUN_MODE")
        self.used = True
        for module in self.modules:
            self.clones[module.__name__]["context"]()
            self.states[module.__name__]["phase"] = "CONTEXT_READY"
        return {m.__name__: self.states[m.__name__]["context"] for m in self.modules}
