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

净值补拉、002112 输入维护和资料补齐已统一改为手动触发：FastAPI 启动不再创建自动维护线程，Celery Beat 也不再注册净值计划。旧的 `TUSHARE_MARKET_INCREMENTAL_ENABLED/HOUR/MINUTE` 配置只保留兼容读取，不会重新启用自动采集；修改代码后需重启旧服务进程才会生效。

同步中心“一键同步”依次执行九项任务：标普500、近期基金公告、基金资料与市场数据更新、基金持仓与公司资料、净值增量、002112 输入维护、历史指标计算、模拟费率、多周期预测与建议核验。输入维护复用原单基金锁，放在资料与净值更新后；忽略旧自动等待时间，但保留研究日期边界、真实取得时间和快照不覆盖规则，不训练或换模。缺数据、来源不可用和错过留存时间均明确报未完成，其余步骤继续；再次点击一键同步可重试。服务关闭期间不采集，也不补造当时的输入。

后台“数据同步”页面经 Java 调用受保护的 `POST /internal/v1/funds/sync-jobs/market-nav-incremental`，由当前 FastAPI 进程直接执行同步，因此不依赖 Beat 或 Worker。同步范围只从 `fund_share_class` 中来源为 Tushare 且状态为 `ACTIVE` 的基金市场记录读取，用户关注列表不会收窄或扩大范围。手动入口共享 PostgreSQL 咨询锁；已有同步运行时接口返回冲突，绝不重复调用 Tushare。该接口只返回安全的任务进度与统计，不返回 Token 或原始响应。

Linux 部署的并发池应按任务类型和容量另行评估；不要直接沿用 Windows 的 `solo` 结论。

本机只使用 `.env` 管理基础配置和私有凭据，并与 Java 的 `AI_SERVICE_TOKEN` 对齐。`.env` 已被 Git 忽略；凭据不得写入源码、测试断言、日志或文档。

## 验证

阶段2净值样本直接调用 `GET /internal/v1/features/historical-nav-samples/preview?fundCode=008888&asOfDate=2025-08-07`，
带现有`X-Service-Token`，Body留空，服务自己读取数据库已有净值并计算。参见[调用说明](docs_zhx/implementation/historical-nav-http-preview.md)。
同路径POST仍支持自备净值的纯计算测试；普通验收不需要导入文件。两种预览均不保存结果、不触发同步、不训练或发布。

小范围批量制作样本使用 `GET /internal/v1/features/historical-nav-samples/dry-run`，传 `fundCode`、`startDate`、
`endDate`、可选 `pageSize`（默认10，1–30）。含首尾最多31个自然日，返回全部样本和统计，仍然只读不保存。
逐步验收见[批量调用说明](docs_zhx/implementation/historical-nav-http-preview.md#6-批量制作练习题只读-dry-run)。

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe
```

内部接口不暴露 Swagger/OpenAPI 页面，所有接口均须携带 `X-Service-Token`，并会拒绝含浏览器 `Origin` 的请求。`/internal/v1/funds` 和 `/internal/v1/funds/{fund_code}` 读取本机持久化目录，列表默认 `pageSize=10`；有已同步净值时详情会返回净值来源和截至日期，无净值时明确返回 `as_of_date=null` / `nav_status=NOT_SYNCED`，不能被解释为实时行情。`POST /internal/v1/funds/sync-jobs/market-nav-incremental` 仅供 Java 创建基金市场同步任务。`GET /internal/v1/sources` 仅返回无凭证的来源开关、限频、保留期和最近状态。

获授权后，维护人员可显式执行：

```powershell
.\.venv\Scripts\python.exe -m app.commands.sync_tushare_funds catalog
.\.venv\Scripts\python.exe -m app.commands.sync_tushare_funds nav --nav-date 2026-08-25
.\.venv\Scripts\python.exe -m app.commands.sync_tushare_funds market-incremental --as-of-date 2026-08-27
```

目录同步按市场和存续状态分片；任何分片达到配置行数上限都会失败关闭，绝不将部分结果标为完整市场目录。将新基金纳入基金市场时，使用 `market-history --ts-code <完整 Tushare 代码>` 显式完成目录和历史净值回填；日常运行使用 `market-incremental`，按同一 Tushare 来源中基金市场每只启用基金的最后净值日向后补齐。日常任务不会重新拉取完整历史；任一基金缺少历史基线或精确来源代码时失败关闭并要求先完成校验。原始响应和 Token 不写入数据库、日志或命令输出。

关联文档：

- `docs_zhx/requirements/fund-radar.md`
- `docs_zhx/implementation/fund-radar.md`
- `docs_zhx/design/fund-radar-api-v1.md`
