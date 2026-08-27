# 全市场基金雷达 FastAPI AI 服务

Python 服务只处理数据采集、事件理解、AI 评分、回测和 Java 内部接口。它不面对浏览器开放，不处理用户关注/提醒，不保存支付宝凭证，也不执行交易。

## 本地启动

```powershell
C:\anaconda3\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000
```

Windows 本地验证 Celery 时，另开终端使用单进程池：

```powershell
.\.venv\Scripts\celery.exe -A app.workers.celery_app worker --pool=solo --loglevel=INFO
```

重点基金的日常净值补数由独立的 Celery Beat 调度器执行；同一环境只能启动一个 Beat，Windows 本机另开终端运行：

```powershell
.\.venv\Scripts\celery.exe -A app.workers.celery_app beat --loglevel=INFO
```

默认在 `Asia/Shanghai` 工作日 20:30 触发，仅补六只配置基金在 Tushare 来源中缺失的日期。可通过 `.env` 中的 `TUSHARE_FOCUSED_INCREMENTAL_ENABLED`、`TUSHARE_FOCUSED_INCREMENTAL_HOUR`、`TUSHARE_FOCUSED_INCREMENTAL_MINUTE` 调整；中国节假日或当晚尚未发布数据时任务以零变更成功结束。不要同时启动多个 Beat。

Linux 部署的并发池应按任务类型和容量另行评估；不要直接沿用 Windows 的 `solo` 结论。

本机只使用 `.env` 管理基础配置和私有凭据，并与 Java 的 `AI_SERVICE_TOKEN` 对齐。`.env` 已被 Git 忽略；凭据不得写入源码、测试断言、日志或文档。

## 验证

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe
```

内部接口不暴露 Swagger/OpenAPI 页面，所有接口均须携带 `X-Service-Token`，并会拒绝含浏览器 `Origin` 的请求。`/internal/v1/funds` 和 `/internal/v1/funds/{fund_code}` 读取本机持久化目录；有已同步净值时详情会返回净值来源和截至日期，无净值时明确返回 `as_of_date=null` / `nav_status=NOT_SYNCED`，不能被解释为实时行情。`GET /internal/v1/sources` 仅返回无凭证的来源开关、限频、保留期和最近状态。

获授权后，维护人员可显式执行：

```powershell
.\.venv\Scripts\python.exe -m app.commands.sync_tushare_funds catalog
.\.venv\Scripts\python.exe -m app.commands.sync_tushare_funds nav --nav-date 2026-08-25
.\.venv\Scripts\python.exe -m app.commands.sync_tushare_funds focused-incremental --as-of-date 2026-08-27
```

目录同步按市场和存续状态分片；任何分片达到配置行数上限都会失败关闭，绝不将部分结果标为全市场目录。`focused` 用于首次完整回填六只重点基金；日常运行使用 `focused-incremental`，按同一 Tushare 来源中每只基金的最后净值日向后补齐。日常任务不会重新拉取完整历史；任一基金缺少历史基线时失败关闭并要求先执行完整回填。原始响应和 Token 不写入数据库、日志或命令输出。

关联文档：

- `docs_zhx/requirements/fund-radar.md`
- `docs_zhx/implementation/fund-radar.md`
- `docs_zhx/design/fund-radar-api-v1.md`
