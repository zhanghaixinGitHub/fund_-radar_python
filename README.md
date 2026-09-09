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

基金市场的日常净值补数由独立的 Celery Beat 调度器执行；同一环境只能启动一个 Beat，Windows 本机另开终端运行：

```powershell
.\.venv\Scripts\celery.exe -A app.workers.celery_app beat --loglevel=INFO
```

默认在 `Asia/Shanghai` 工作日 20:00 触发，补齐基金市场中所有启用基金在 Tushare 来源中缺失的日期。可通过 `.env` 中的 `TUSHARE_MARKET_INCREMENTAL_ENABLED`、`TUSHARE_MARKET_INCREMENTAL_HOUR`、`TUSHARE_MARKET_INCREMENTAL_MINUTE` 调整；中国节假日或当晚尚未发布数据时任务以零变更成功结束。不要同时启动多个 Beat。

后台“数据同步”页面经 Java 调用受保护的 `POST /internal/v1/funds/sync-jobs/market-nav-incremental`，由当前 FastAPI 进程直接执行同步，因此不依赖 Beat 或 Worker。同步范围只从 `fund_share_class` 中来源为 Tushare 且状态为 `ACTIVE` 的基金市场记录读取，用户关注列表不会收窄或扩大范围。手动和定时任务共享 PostgreSQL 咨询锁；已有同步运行时接口返回冲突，绝不重复调用 Tushare。该接口只返回安全的任务进度与统计，不返回 Token 或原始响应。

Linux 部署的并发池应按任务类型和容量另行评估；不要直接沿用 Windows 的 `solo` 结论。

本机只使用 `.env` 管理基础配置和私有凭据，并与 Java 的 `AI_SERVICE_TOKEN` 对齐。`.env` 已被 Git 忽略；凭据不得写入源码、测试断言、日志或文档。

## 验证

### Chronos-2 与自训练模型的离线比较

入口是 `scripts/compare_chronos2.py`，仅使用本机 `fund_ai` 的明确现金研究批次，数据库事务只读。
Chronos 是可选研究依赖，在独立环境运行；原服务与自训练接口不需要安装它。
先安装固定依赖并下载核验官方权重：

```powershell
.\.venv\Scripts\python.exe -m venv .local-runs\chronos2-runtime
.\.local-runs\chronos2-runtime\Scripts\python.exe -m pip install -r requirements-chronos2.txt
.\.venv\Scripts\python.exe scripts\setup_chronos2.py
```

再按 `freeze → export → smoke → run → verify` 顺序执行。`freeze` 生成新 UUID，后续 `--run` 使用该值；
同一运行包不可覆盖，重新比较须创建新包。以下来源是本轮已核验的原研究：

```powershell
.\.local-runs\chronos2-runtime\Scripts\python.exe scripts\compare_chronos2.py freeze --source-run f70feb1a-129d-4482-b66d-f4e2e3a5425c --dataset-hash 7f482c3f6cb43ac00c7dc635dcaa3397c5795e25e127e5bc8b338c70fb8f1c41
.\.local-runs\chronos2-runtime\Scripts\python.exe scripts\compare_chronos2.py export --run <新UUID>
.\.local-runs\chronos2-runtime\Scripts\python.exe scripts\compare_chronos2.py smoke --run <新UUID>
.\.local-runs\chronos2-runtime\Scripts\python.exe scripts\compare_chronos2.py run --run <新UUID>
.\.local-runs\chronos2-runtime\Scripts\python.exe scripts\compare_chronos2.py verify --run <新UUID>
```

`inputs.jsonl` 只含历史输入，答案另存且按 FIT/CALIBRATION/HISTORY/EXAM 分区；冻结全部考试预测后才读取 EXAM 答案。
`complete.json` 最后生成。模型、虚拟环境、完整锁定依赖及原始资料均位于 Git 忽略的 `.local-runs`。
干净环境按运行包的 `requirements-resolved.txt` 安装后，可用 `replay --run <UUID>` 核验六条真实样本与完整评分；
复跑回执另存，不改已完成运行包。硬件仅用 CPU 1 线程，小批量 8 条，Chronos 子进程有超时退出边界。

2026-09-09 的正式运行是 `4a329ccd-0cfc-4c81-9ecb-164e0fd9762d`：625 道有答案的 2024 考题上，
三基金等权原始方向准确率为自训练 58.07%、Chronos 38.40%；两者校准斜率均为负，按冻结协议拒绝概率输出。
因此没有有效概率的整体赢家，没有读取 2025 数值或替换正式模型。
完整分步标记和中文报告在 Vue 文档仓库 `docs_zhx/implementation/chronos2-model-comparison.md` 及其报告链接。

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
