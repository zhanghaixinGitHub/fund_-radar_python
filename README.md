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

Linux 部署的并发池应按任务类型和容量另行评估；不要直接沿用 Windows 的 `solo` 结论。

`.env` 是本项目唯一的本地运行配置，已与 Java 的 `AI_SERVICE_TOKEN` 对齐；无需复制或切换环境。

## 验证

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe
```

内部接口不暴露 Swagger/OpenAPI 页面，所有接口均须携带 `X-Service-Token`，并会拒绝含浏览器 `Origin` 的请求。M0 暂提供 `/internal/v1/health`、`/internal/v1/funds` 和 `/internal/v1/funds/{fund_code}`；基金数据固定标为 `M0_MOCK`，M1 必须替换为已授权数据源。`GET /internal/v1/sources` 仅返回无凭证的来源开关、限频、保留期和最近状态；未取得授权时返回空数组，不创建来源、不访问第三方、不返回外部内容。

关联文档：

- `docs_zhx/requirements/fund-radar.md`
- `docs_zhx/implementation/fund-radar.md`
- `docs_zhx/design/fund-radar-api-v1.md`
