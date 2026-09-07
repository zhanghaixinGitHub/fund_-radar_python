# 阶段2：用 Postman 验收历史净值样本

现在直接传基金代码和日期，服务会读取数据库已有净值，返回“当时已知条件”和“后来答案”。不需要导入文件、填写净值列表或提交Body。预览不保存结果、不触发补数、不训练模型；`SCORABLE`只表示样本满足计算条件。

关联：Vue文档仓 `docs_zhx/design/free-data-prediction-v1.md` v1.2 和 `docs_zhx/testcase/free-data-prediction-v1.md` TC-FDP-09。

## 0. 推荐：直接读取数据库

在Postman中新建GET请求（服务默认端口8000；独立验收实例若使用8001，则改用8001）：

```text
GET http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/preview?fundCode=008888&asOfDate=2025-08-07
```

Headers填写现有`X-Service-Token`，值为Python配置中的`AI_SERVICE_TOKEN`；Body留空，点击Send。返回的直接是一条样本，不包在`items`列表中。

- `fundCode`：要看哪只基金。
- `asOfDate`：要看哪个净值业务日。非净值日不会自动改成邻近日期，返回404。
- `feature_payload.metrics.return_20d`：该示例实际读库结果为`0.06821425`，约6.8214%。
- `offline_label.future_return_20d`：该示例实际读库结果约`0.1400961881`，约14.0096%；标签为1。

GET读取同一启用来源，基金须为启用的股票型；最多读取60条已知历史、1条起点和20条未来净值。数据读取使用只读一致性事务、独立2连接池及5秒建连/查询超时。缺基金或该日净值为404，来源未就绪/品类不适用为409，数据库不可用为503，参数错误为422。

净值来自数据库当前已存版本。返回的来源运行ID标识当前来源水位，不意味着历史修订版本已恢复。该入口只完成单日查询预览，不等于全量历史样本已构建。

## 1. 可选：POST提交自备净值（普通验收不需要）

以下导入方法仅用于自行构造异常数据、测试纯计算函数；已有数据库数据时直接使用上面的GET即可。POST不读库。

1. Python服务尚未启动时，在本仓库根目录执行：

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```

   如果端口已有服务，不要重复启动；更新代码后需由运行服务的IDE或终端重新加载。调用预览不需要启动Java、Vue、数据库或Celery。

2. 用Postman桌面版的 **Import** 导入[验收集合](../examples/historical-nav-preview.postman_collection.json)。
3. 在集合变量中设置 `base_url`（默认 `http://127.0.0.1:8000`），将 `service_token` 的本地值设为Python项目 `.env` 中 `AI_SERVICE_TOKEN` 的值。不要分享、导出或提交真实Token。
4. 打开集合中的请求，点击 **Send**。81条模拟净值已填好，不用手工录入。

方法：`POST`。完整地址：

```text
http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/preview
```

| 请求头 | 值 |
| --- | --- |
| `Content-Type` | `application/json` |
| `X-Service-Token` | `{{service_token}}` |

不要添加`Origin`请求头；这是内部维护接口，浏览器产品页面仍只调用Java，Swagger仍未开放。也可以自行创建请求，将[完整请求JSON](../examples/historical-nav-preview.request.json)复制到 **Body → raw → JSON**。

## 2. 用简单数字验算

示例 `source_code=SYNTHETIC_DEMO`。代码`000001`只是符合格式的占位符；**整组数据均为模拟教学数据，不代表该代码的真实基金表现**。日期只跳过周末，不模拟交易所节假日。

81条净值从`1.00`逐条增加到`1.80`，每次增加`0.01`。公告日设为净值日次日。请求`as_of_date=2025-03-26`对应第61条，净值为`1.60`。

| 输出位置 | 预期 | 验算方式 |
| --- | --- | --- |
| `mode` | `PREVIEW_ONLY` | 只做计算预览。 |
| `input_nav_count` / `sample_count` | `81` / `1` | 输入81条，只挑一天返回。 |
| `items[0].available_at` | `2025-03-27` | 起点净值次日公布。 |
| `items[0].feature_payload.metrics.return_20d` | `"0.14285714"` | `1.60 / 1.40 - 1`，约14.2857%。 |
| `items[0].offline_label.future_return_20d` | `"0.125"` | `1.80 / 1.60 - 1`，即12.5%。 |
| `items[0].offline_label.label_up_20d` | `1` | 后来收益大于0。 |
| `items[0].offline_label.label_end_date` | `2025-04-23` | 后第20条净值的业务日。 |

收益小数乘100才是百分数。为保留Decimal精度，净值/收益在JSON中序列化为字符串，标签与计数为整数。集合的Tests/Test Results中已包含核心验算。

删除请求最外层`as_of_date`后重发，返回全部81条：前60条`DATA_INSUFFICIENT`（历史不足），第61条`SCORABLE`，最后20条`LABEL_NOT_MATURED`（后面的答案未凑齐）。计数均以实际返回的items为准。

## 3. 请求字段

| 字段 | 必填/默认 | 意思 |
| --- | --- | --- |
| `fund_code` | 必填 | 6位数字字符串，不查库验证真实基金身份。 |
| `fund_type` | 默认`STOCK` | 本阶段只接受股票型。 |
| `source_code` | 默认`MANUAL_PREVIEW` | 调用者声明的来源，不是接口对来源的认证。 |
| `source_sync_run_id` | 可为`null` | 真实同步数据可传对应UUID；教学数据不虚构同步证据。 |
| `nav_points` | 必填 | 单基金1–512条记录，净值业务日严格递增、无重复。 |
| `nav_points[].nav_date` | 必填 | 净值业务日期，`YYYY-MM-DD`。 |
| `nav_points[].ann_date` | 可为`null` | 公告日；缺失时不能把该点当作已知输入。 |
| `nav_points[].unit_nav` | 必填 | 有限正数，最多24位数字、小数最多8位；推荐传字符串。 |
| `nav_points[].accumulated_nav` | 可为`null` | 累计净值，同样只接受有限正数。 |
| `as_of_date` | 可省略 | **只筛选返回的净值业务日**，必须在输入序列内，不改变样本可得日期。 |

“20日”按提交序列的后20条净值计数，不是自然日加20。调用方负责提供连续完整序列；接口不接交易日历，无法发现整条缺失的净值日。已提供但无效的标签记录会导致拒收，不跳过后延终点。

每条样本的可得截止日取起点`ann_date`；特征最多用61条符合可得日期的净值。只有日期精度，按公告日结束后可用解释，不声称盘中已经可见；来源对历史净值的事后修订版本仍未解决。

## 4. 亲手改三个场景

每次先恢复原始请求。

1. 将`2025-03-26`的`ann_date`改为`null`：HTTP 200，样本`DATA_INSUFFICIENT`，原因`MISSING_NAV_ANNOUNCEMENT_DATE`，特征与标签为空。
2. 只保留前61条净值：样本`LABEL_NOT_MATURED`，标签为空，已经算好的特征和哈希保留。
3. 将最后一条`2025-04-23`的`accumulated_nav`改为`null`：过去的`feature_payload`、`feature_hash`与累计净值口径都不变；只有标签因为缺值而不可用。未来缺值不能让过去特征换口径。

总体`eligibility_status`表示特征与标签是否同时合格；`feature_payload.quality.status`只说明特征质量。因此标签待成熟时，前者可为`LABEL_NOT_MATURED`，后者仍为`SCORABLE`。

| HTTP状态 | 怎么处理 |
| --- | --- |
| `200` | 请求已处理，继续看每条样本状态；数据不足也可正常返回200。 |
| `403` | Token未传、错误或带`Origin`，检查请求头。 |
| `503` | 服务未配置`AI_SERVICE_TOKEN`，配置并重新加载服务。 |
| `422` | 日期、顺序、类型、数量、数值或额外字段不符，根据`detail`修改请求。 |

## 5. 对照代码

```text
app/api/routes/features.py → preview_historical_nav_samples：接收HTTP、筛选返回日期和统计
app/schemas/historical_nav.py：检查输入格式、数量和顺序
app/services/historical_nav_samples.py → build_historical_nav_samples：先算特征，再附加离线标签
```

重复相同请求得到相同特征哈希，没有数据库写入。日志记录TraceID、基金代码、数量和耗时，不打印Token或整组净值。512条上限用于控制预览开销，正式历史样本分页读库和持久化仍需单独实施。

自动验证：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_historical_nav_samples.py tests/test_historical_nav_http.py tests/test_stock_feature_snapshot.py -q
```
