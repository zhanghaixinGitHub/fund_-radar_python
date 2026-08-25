# Windows Celery Worker — 默认进程池不消费任务

### 2026-08-24｜Windows 默认进程池保留任务但不执行
场景：Windows 本地环境使用 Python 3.13、Redis 7.4.7 和 Celery 5.6.3 启动 `fund_ai` Worker。
现象：控制面 `ping` 返回成功，但 `health_probe.delay().get()` 超时；Worker 的 reserved 列表持续存在未确认任务，active 列表为空。
根因：Windows 不支持 Celery 默认 prefork 进程池的完整执行语义，Worker 可从 Redis 预取任务但子进程无法开始执行。
正确做法：本地 Windows Worker 使用 `celery -A app.workers.celery_app worker --pool=solo --loglevel=INFO`；生产 Linux 环境再依据并发和任务类型评估 prefork 或其他池。
关联代码：`app/workers/celery_app.py`、`app/workers/tasks.py`。
关联技术点：Celery Worker、Redis broker、Windows 兼容性、任务结果回收。
标签：⚠️ 反复
