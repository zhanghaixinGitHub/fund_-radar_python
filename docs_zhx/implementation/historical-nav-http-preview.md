# 阶段2：用 Postman 验收历史净值样本

现在直接传基金代码和日期，服务会读取数据库已有净值，返回“当时已知条件”和“后来答案”。不需要导入文件、填写净值列表或提交Body。预览不保存结果、不触发补数、不训练模型；`SCORABLE`只表示样本满足计算条件。

关联：Vue文档仓 `docs_zhx/design/free-data-prediction-v1.md` v1.2–v1.5 和
`docs_zhx/testcase/free-data-prediction-v1.md` TC-FDP-09–TC-FDP-12。

单日查询看第0节；批量制作历史练习题看第6节。两者都不是模型预测。

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

重复相同请求得到相同特征哈希，没有数据库写入。日志记录TraceID、基金代码、数量和耗时，不打印Token或整组净值。
512条上限仅用于POST预览；小范围分页读库见第6节。全历史离线批处理、持久化、训练仍未实施。

自动验证：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_historical_nav_samples.py tests/test_historical_nav_http.py tests/test_stock_feature_snapshot.py -q
```

## 6. 批量制作练习题：只读 dry-run

### 6.1 这一步做了什么

原GET回答“这只基金在某一天的练习题是什么”。新GET回答“这只基金在一段日期内，每一天的练习题是什么”。
数据库里存的是净值；服务现场计算特征和后来真实发生的结果，组成样本，返回后不保存样本。

dry-run就是“试着完整跑一遍，但不把计算结果保存到数据库”。仍然没有模型训练或未来预测。

### 6.2 在你现有的接口工具中这样填

如果服务是在改代码之前启动的，先在IDE中重启Python服务，让新路由生效。没有重载的新进程会返回404，
这不代表日期范围没有数据。继续使用你自己的8000端口，不必导入集合或另开一个端口。

方法：GET。地址：

```text
http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/dry-run
```

Params填写：

| 参数名 | 先填这个值 | 通俗解释 |
| --- | --- | --- |
| `fundCode` | `008888` | 要制作哪只基金的练习题。 |
| `startDate` | `2025-08-01` | 从哪天开始出题，包含当天。 |
| `endDate` | `2025-08-31` | 到哪天结束，包含当天。 |
| `pageSize` | `10` | 程序每一批处理几个起点；省略默认10，允许1–30。 |

Headers沿用原接口的`X-Service-Token`；Body留空，不传`Origin`。
一次请求会在服务内部读完所有页，然后返回整个日期范围的结果，不需要你手动请求“第2页”。
首版含首尾最多31个自然日，超出返回422。它不是“预测31天”，也不保证有31条净值。

### 6.3 返回值怎么看

本轮用当前数据库实际读到：008888在2025年8月有21个净值日，21条样本均为SCORABLE。
这只是这一数据版本的验收例子，来源后续修订时应重新核对。

| 返回字段 | 本例值 | 意思 |
| --- | --- | --- |
| `mode` | `DRY_RUN` | 只读试跑，结果未保存。 |
| `fund_code`、`start_date`、`end_date` | 与请求对应 | 这次正在看哪个基金、哪段日期。 |
| `page_size` | `10` | 每批起点数量。 |
| `page_count` | `3` | 实际处理3个非空批次：10份、10份、1份。 |
| `sample_count` | `21` | 整段日期共返回21份练习题。 |
| `scorable_count` | `21` | 特征和历史答案都符合当前计算条件，不代表预测正确21次。 |
| `data_insufficient_count` | `0` | 因历史或标签数据不合格而无法使用的数量。 |
| `label_not_matured_count` | `0` | 已有特征但未来记录/公告还不齐的数量。 |
| `unavailable_reasons` | `{}` | 各不可用原因及数量；本例没有。 |
| `items` | 21个对象 | 按日期排序的全部样本，每项与单日GET的返回结构一致。 |

后三种样本状态数量相加等于`sample_count`，它又等于`items`的长度。
没有净值的日期不会补一份“零收益样本”；整个范围都没净值时正常返回200、0份样本、0页。

### 6.4 怎么验收“分页不改变结果”

1. 先调用原单日GET，`fundCode=008888`、`asOfDate=2025-08-07`，保存完整返回JSON。
2. 调用上述批量GET，在`items`里找`as_of_date=2025-08-07`的对象。
3. 这一对象应与第1步的完整JSON相同，包括`feature_payload`、`offline_label`、状态、净值口径和`feature_hash`。
4. 把批量请求的`pageSize`从10改为30，再发一次；两次`items`应逐项完全相同，样本总数及质量统计也相同。
5. `page_size`和`page_count`本来就会改变，不要求这两个字段相同。本例分别为10/3与30/1。

比较前提：净值、公告日期、来源同步运行记录及计算版本没有变化。来源运行ID也在特征内容里，
如果同步任务在两次请求之间发生，应重新在同一数据状态下对比，不能把正常数据更新误判为分页错误。
单看HTTP 200或SCORABLE不能代替上述内容比对；哈希也不包含单独存放的标签，所以还要比较`offline_label`。

### 6.5 程序内部怎么做

1. 核验基金和来源一次，整次请求使用同一个只读数据库快照。
2. 按日期取一页起点。“游标”就是记住上一页最后一天，下一页从它后面继续，不重复读取起点。
3. 给每个起点查找它自己的最多60条已知历史和后20条净值；这些资料可以在请求日期范围之外。
   公告迟到的历史记录不能提前使用；未来缺公告或坏值的记录不能跳过，否则会偷换第20日终点。
4. 每一页的窗口使用`UNION ALL`合并成一次数据库请求。它只是“把多个有界查询的结果放在一起”，
   不会把不同样本混成一条长序列。每个非空页最多2条净值SQL，没有逐日网络查库。
5. 每份完整资料交给原构建器，再取出对应日期的样本，最后汇总最多31份结果。

只读连接池仍为2个连接，连接等待/建连/单SQL各5秒限制。批量另有15秒页间耗时预算，
不是精确到15秒强制中断；正在执行的一页仍受单SQL超时约束。中途异常或超时整次返回失败，
不把前几页冒充完整结果。日志只有整次摘要和异常堆栈，不打印逐日净值或Token。

基金不存在404；基金不是启用股票型或来源未就绪409；非法参数422；数据库异常或批量超时503。
没有新表、DDL、DML或外部采集调用。它仍不具备历史修订版本回放，也不能检测来源整日漏数。

阅读代码顺序：`app/api/routes/features.py` 的 `dry_run_historical_nav_samples` →
`app/schemas/historical_nav.py` 的两个 Batch 类 → `app/services/historical_nav_preview.py` 的批量函数 →
`app/repositories/historical_nav.py` 的 `read_historical_nav_input_page`。现有收益公式未改。

自动测试（临时内存库，不改项目数据库）：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_historical_nav_batch.py tests/test_historical_nav_repository.py tests/test_historical_nav_http.py tests/test_historical_nav_samples.py tests/test_stock_feature_snapshot.py tests/test_feature_read.py -q
```

本轮验证：83项自动测试通过；真实数据库中每页1/10/30的21份样本一致，并逐日与单日服务完整对照；
24个真实事务核验为只读、可重复读。新代码的接口测试客户端未替换数据库依赖，实际验证了200、403、422、
空范围返回0及单日JSON一致。上述接口验证运行在独立进程内，不表示现有8000/8001进程已重载新路由。

**学习停顿点：** 看懂“起点日期范围”“每份样本自己的窗口”“pageSize只影响批次”后再决定下一步。
本轮不扩展到所有基金、全历史离线任务、样本持久化或模型训练。

## 7. 出题前检查：这条净值是不是已经过时了

### 7.1 规则与两个新名字

“出题日”仍是起点净值的公告日，按当天结束后可见解释。如果当时同基金、同来源已有业务日期更晚的净值公布，旧起点就不能用。
不设“晚3天/7天就异常”的阈值，也不把公告日改成净值日加一天。20日仍是从起点向后数第20条净值，不是从公告日另加20天。

- `STALE_NAV_AT_CUTOFF`：出题时已有更新净值公布，起点已过时。它放在既有`unavailable_reason`里，不是服务器报错。
- `sample_rule_version`：本次新增的响应字段，当前为`HISTORICAL_NAV_SAMPLE_RULE_V2`，表示采用新版样本筛选规则。不是模型版本，不参与特征哈希；没有该字段的旧响应属于原规则V1。

收益公式版本和标签版本未改；正常样本仅多了上述版本字段，原有内容不变。过时样本还没进入公式阶段，故口径为`UNDETERMINED`，特征指标和标签都为空，而不是伪造为0。

### 7.2 直接调用现有GET验收

先让运行Python的IDE或终端重载代码；无需修改数据库或启动同步。Headers仍用现有`X-Service-Token`，Body留空。

```text
GET http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/preview?fundCode=008888&asOfDate=2025-03-31
```

当前数据库把这条净值的公告记为2025-04-22，而那时已经有更新净值公布。响应关键字段应为：

```json
{
  "as_of_date": "2025-03-31",
  "available_at": "2025-04-22",
  "nav_value_basis": "UNDETERMINED",
  "eligibility_status": "DATA_INSUFFICIENT",
  "unavailable_reason": "STALE_NAV_AT_CUTOFF",
  "offline_label": null,
  "sample_rule_version": "HISTORICAL_NAV_SAMPLE_RULE_V2"
}
```

这是响应节选；`feature_payload.metrics`也应为null。源数据若发生修订，需要重新核对，不能硬编码日期结果。
再把`asOfDate`改回熟悉的`2025-08-07`：其原有指标、答案和哈希应与修改前一致，仍不是模型预测。

批量验收改查`2025-03-01`至`2025-03-31`，分别设`pageSize=1/10/30`。3月31日不能被丢掉，必须与单日GET完整JSON相同；新原因计入`unavailable_reasons`，全部items及质量统计不受分页影响。

### 7.3 这几段代码分别负责什么

| 位置 | 通俗解释 |
| --- | --- |
| `historical_nav.py` → `_has_newer_announced_nav` | 对每个起点问数据库：同基金、同来源有没有更新净值在当时已公告？`EXISTS`只返回是/否，不把全历史取回来。 |
| `HistoricalNavSampleInput.stale_anchor_dates` | 把数据库确认过时的起点日期交给构建器。`frozenset`表示不可增删的集合。它是内部事实，不是新增HTTP参数或模型特征。 |
| `historical_nav_samples.py` → `_find_stale_anchor_dates` | 对已提供的序列从后往前检查一次；记住后面最早的有效公告日，因此不会只盯住第20条。 |
| `_build_one_sample` | 先检查起点日期，再检查起点是否过时；拒收时保留原日期并返回原因，正常时继续使用原指标公式。 |

GET的数据库检查不受后20条、当前页或请求日期范围限制；净值计算资料仍最多81条，批量每个非空页仍最多2条净值SQL。
POST不查数据库，只能检查你提交的记录，不能证明遗漏的记录不存在。已公告的判断只认可有效公告日期；缺失或公告早于净值日的记录不能充当证据。

**边界与停顿点：** 这一步只排除旧起点，不恢复历史修订、不证明`ann_date`一定是首次可得日。先看懂这道检查，再讨论保存训练样本；本步不训练、不发布。

### 7.4 本次验收结果（2026-09-07）

102项相关测试及Ruff通过。真实数据库中，三只基金七个固定月份的429份正常样本，除新增规则版本字段外原有内容全部保持一致；
21条季度末滞后起点均返回`STALE_NAV_AT_CUTOFF`，核对的原始数据未变。
008888的2025年3月：21份样本中20份可用、1份过时；分页1/10/30的items与统计一致，3月31日与单日完整JSON一致。
新代码通过真实数据库的只读事务和进程内HTTP验证；现有8000服务仍返回旧版本，手工验收前需要在启动它的IDE或终端重载，本轮未自动重启。

## 8. 保存前的一小步：先定义练习册的格式（2026-09-07）

**本步只准备实体和建表脚本，未在真实数据库执行迁移。** 没有保存接口，没有样本落库，没有模型训练。
现有单日和批量GET仍然现场读取净值、计算、返回；调用它们不会写入下面三张新表。

### 8.1 三张表分别装什么

```text
historical_nav_sample_batch（封面：一只基金、日期段、规则、统计）
  └─ historical_nav_sample（题目：这个起点已知的条件，或不能出题的原因）
       └─ historical_nav_sample_label（答案：后来20条净值的真实结果；可以没有）
```

以008888的2025-08-07为例：封面记基金与本次日期范围，题目记8月7日净值、8月8日公告这一时间边界及历史指标，
答案另记20条净值后的实际收益和上涨标签。题目表不额外复制原始净值数值，完整输入说明保留在`feature_payload`。
这仍是整理历史题目，不是预测未来；保存也无法把当前`nav_daily`恢复成过去的首次公告版本。

| 表/字段组 | 用人话解释 |
| --- | --- |
| 封面：`batch_id` / `request_key` | 前者是练习册编号；后者是一次保存请求的凭证。重试沿用凭证，明确重算换新凭证、产生新批次。 |
| 封面：`fund_code` / `fund_type` | 这份练习册属于哪只基金；当前固定股票型。 |
| 封面：`start_date` / `end_date` | 题目起点的日期范围，含首尾最多31个自然日，不是读取辅助净值的全部范围。 |
| 封面：`source_code` / `source_sync_run_id` | 本次使用的数据来源和同步水位，不是每条净值的首次可得证明。 |
| 封面：三个`*_version` | 分别登记指标公式、样本筛选、答案算法的版本。不是训练出来的模型版本。 |
| 封面：`purpose` / `created_at` | 固定`LEARNING_ONLY`；记录何时保存，不是何时公告。 |
| 封面：四个`*_count` / `unavailable_reasons` | 总数、完整数、不合格数、答案未齐数，以及各不可用原因的数量。 |
| 题目：`sample_id` / `batch_id` | 题目编号、所属练习册编号。基金和共享版本从封面读取。 |
| 题目：`as_of_date` / `available_at` | 净值属于哪天、公告截止是哪天；拒收样本可以保留缺失或错误的公告事实。 |
| 题目：`nav_value_basis` | 本题采用累计净值还是单位净值；无法出题时可以尚未确定。 |
| 题目：`eligibility_status` / `unavailable_reason` | 题目是否完整，不完整时说明原因；不合格样本也保留。 |
| 题目：`feature_payload` / `feature_hash` | 已知条件包及其内容指纹。哈希只核对特征包，不含答案或批次编号。 |
| 答案：`sample_id` / `horizon_trading_days` | 回答哪道题；固定向后数20个净值区间。 |
| 答案：`label_end_date` / `label_available_at` | 第20条净值属于哪天、答案所需记录何时全部公告完毕。 |
| 答案：`future_return_20d` / `label_up_20d` | 后来实际收益，以及上涨1/非上涨0；不是概率。收益用十进制数，不固定截成8位。 |

例如`DATA_INSUFFICIENT`或`LABEL_NOT_MATURED`仍有题目行，但**没有答案行**。
缺失不能写`label_up_20d=0`，因为0已经有明确含义：后来确实没有上涨。标签以后成熟，可明确重算成新批次，不能暗改旧练习册。

### 8.2 新的代码术语与阅读顺序

1. 先读`app/models/historical_nav_sample.py`。三个类依次是`HistoricalNavSampleBatch`、`HistoricalNavSampleRecord`、`HistoricalNavSampleLabel`。
   **ORM实体**就是“Python属性与数据库列的对应表”；`Mapped[date]`表示Python属性是日期，`mapped_column(Date, ...)`声明数据库日期列。
   `comment="..."`不是无用文字，迁移执行后可在数据库工具里看到中文字段说明。
2. 再看同文件的`__table_args__`。**主键**是本行编号；**外键**要求关联编号存在；**唯一约束**拦住重复；**检查约束**限制字段值；**索引**帮助按基金快速找批次。
   外键不配置级联删除，防止误删封面时顺带清掉题目、答案。
3. 最后看`alembic/versions/20260907_13_add_historical_nav_sample_storage.py`。
   **迁移**就是有编号的数据库结构变更步骤；`upgrade()`建空表，`downgrade()`回退空表。导入实体不会自动调用它们。
   迁移独立保存本次定义，不直接导入会继续变化的实体；这是保留历史版本，不是又写了一遍预测公式。
4. `tests/test_historical_nav_storage_schema.py`核对实体、迁移、原生SQL的一致性，防止以后只改其中一份。

这里的实体与以前纯计算模块的`dataclass`不是同一种职责：`dataclass`临时装计算结果，ORM说明将来怎样存。
本步**没有编写从计算结果到三张表的保存转换**，创建Python对象也不等于保存成功。`uuid4`默认值在ORM插入时生成，不是数据库自动生成编号。

### 8.3 哪些规则已经写进结构，哪些还没实现

| 层次 | 当前情况 |
| --- | --- |
| 结构已声明、尚未在真实数据库生效 | 请求凭证唯一；同批同日唯一；一道题最多一份答案；基金/批次/题目的关联存在；日期窗口上限、计数非负与总和、用途、状态及答案正负规则。 |
| 下一步保存服务必须核对 | 明细日期在封面范围内；所有样本的基金/来源/版本一致；计数与真实明细一致；原因JSON中的数量准确；特征哈希重算一致且特征包不含答案。 |
| 下一步保存服务必须核对 | 只有SCORABLE有且必须有答案；答案终点晚于起点、答案可得日晚于输入截止；净值口径一致；后续读库保存仍要求来源已就绪。 |
| 下一步事务与重试 | 同一凭证同一请求直接返回原批次，凭证相同但基金/日期/规则不同应拒绝；明确重算换凭证；整批成功才提交，任一失败不留半批次。 |
| 下一步不可覆盖约定 | 保存代码只新增批次，不更新旧批次；本步没有数据库触发器或权限配置来保证物理不可修改，不能据此宣称绝对不可变。 |

不存`page_size/page_count`，因为它们是读取过程信息，不是题目内容，不应成为重试去重或重算依据。
版本和来源元数据不是模型指标；未来训练只从允许的指标字段读取X，将标签单独作为y，不能把整个JSON或答案表当特征喂入。

### 8.4 现在如何验收，不需要调新HTTP接口

在Python项目目录运行：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_historical_nav_storage_schema.py -q
```

此测试不读`.env`、不连接项目数据库，输出通过说明“结构定义一致”，不是“真实数据库已建表”。
现有GET响应不会因这一步新增批次编号或保存状态，这恰好是本步兼容边界。

原生PostgreSQL脚本位于`docs_zhx/sql/20260907_13_historical_nav_sample_storage_upgrade.sql`与同名`downgrade.sql`。
**DDL**表示建表等结构操作；**DML**表示插入/修改业务数据，本步没有DML脚本。
常规建表以后通过Alembic执行，不能再把原生SQL重复执行一遍；原生SQL不自动登记Alembic版本。

后续获准执行前，确认目标`fund_ai`、当前迁移为`20260905_12`，并先在隔离PostgreSQL测试库验证建表与非法数据拦截。
本次迁移只新增三张空表，不修改`nav_daily`和`feature_snapshot`；仍需确认窗口、权限与备份。
回退仅允许三张表全部为0行：先尝试加锁，任一表有数据或锁不可得就报错停止；事务失败要回滚。
有数据时保留表，单独评审备份和回退，不能删除保护检查强行清表。当前没有执行上述任何迁移或回退操作。

**停顿点：** 先能回答“为什么没有答案不能填0”“题目怎样找到所属批次”“重试与重算为什么用不同凭证策略”。
看懂后再确认实际建表，随后才逐步实现整批保存；不直接进入训练。

本步验证：13项新结构检查和102项原有相关测试，共115项通过；改动Python文件Ruff通过。
有一条既有TestClient依赖弃用警告，未升级依赖。真实数据库建表、数据保存和回退执行仍未验证。
