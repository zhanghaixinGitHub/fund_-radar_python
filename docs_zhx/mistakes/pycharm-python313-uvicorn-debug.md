# PyCharm + Python 3.13 调试 Uvicorn 兼容性

## 适用场景 / 触发条件

本项目使用 Python 3.13、Uvicorn 0.52 及 PyCharm 2025.2.3 的“小虫子”调试启动方式。

## 实际影响

正常运行可以启动，但调试启动在 Uvicorn 调用 `asyncio.run(..., loop_factory=...)` 时退出，错误为：

```text
TypeError: _patch_asyncio.<locals>.run() got an unexpected keyword argument 'loop_factory'
```

## 已确认根因

PyCharm 的 `pydevd_nest_asyncio.py` 会替换 `asyncio.run`，但原函数签名只接受 `debug`，未兼容 Python 3.12+ 新增的 `loop_factory` 参数；Uvicorn 在 Python 3.12+ 会传入该参数。

## 修复 / 规避方式

已修改本机 PyCharm 文件：

`C:\PyCharm 2025.2.3\plugins\python\helpers-pro\pydevd_asyncio\pydevd_nest_asyncio.py`

将 `run` 函数签名改为 `def run(main, *, debug=None, loop_factory=None)`，在提供 `loop_factory` 时创建并设置该事件循环，再将其传给 `asyncio.ensure_future`。

## 预防 / 复查

PyCharm 升级或修复安装可能覆盖该文件。若再次出现相同异常，先检查上述文件是否仍包含 `loop_factory` 参数，再按本记录恢复兼容处理；不需要改动项目业务代码或降低 Python 版本。
