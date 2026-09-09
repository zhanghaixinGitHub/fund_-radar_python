# 历史净值样本与离线候选模型：HTTP 验收手册

当前入口：旧单日预览看第0节，旧批量dry-run看第6节，保存样本看第9节，基线评估看第10节，候选训练看第11节，校准与滚动研究验证看第12节，只读失败诊断脚本看第13节，交易日窗口预览看第14节，净值口径审计看第15节，现金再投单日样本看第16节，**新增现金再投批量dry-run看第17节**。新版不能直接混入旧存储/训练接口。
前面的“只制作样本、不训练”描述的是对应预览/存储接口；新候选训练接口会真正拟合一个研究模型，但不写数据库、不发布预测。

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

本节保留建表前的设计说明；其中“未执行”“下一步”“停顿点”均为当时状态。当前已完成建表与保存，调用方法以第9节为准。

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

## 9. 当前功能：整批保存与按批次查询（2026-09-07）

已按用户要求取消逐段学习停顿，完成样本保存闭环：从已有净值制作样本、整批保存、重复请求复用、按编号读回。
数据库`fund_ai`已执行Alembic `20260907_13`，三张表及全部33个字段的中文注释已核验。不要再重复执行原生建表SQL。
这一步保存的是历史样本及后来发生的实际答案，**不是模型预测；尚未训练、回测或发布模型**。

### 9.1 先重启Python服务，再调接口

本次新代码已通过连接真实数据库的进程内HTTP验收，但原8000监听进程实测仍未加载新路由，返回404。
请在启动它的IDE或终端重启/重载Python服务，再使用下列接口。Headers沿用现有`X-Service-Token`；不要带`Origin`。
Token只在本地接口工具填写，不写到文档、前端或仓库。保存请求Body选择JSON，`Content-Type: application/json`。

| 接口 | 用途 |
| --- | --- |
| `GET /internal/v1/features/historical-nav-samples/preview` | 原单日预览，实时计算、不保存。 |
| `GET /internal/v1/features/historical-nav-samples/dry-run` | 原日期段预览，实时计算、不保存。 |
| `POST /internal/v1/features/historical-nav-samples/batches` | 新接口：读库计算并保存一整批。不是自备净值的原POST预览。 |
| `GET /internal/v1/features/historical-nav-samples/batches/{batch_id}` | 新接口：只查已经保存的内容，不重新计算。 |

### 9.2 立即查验已保存的真实例子

```text
GET http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/batches/75b8258a-7cfd-4bfc-b2e8-2b5f05d9f8b3
```

该批次是本次验收保留的`008888 / 2025-08-07`。真实三表核验为1条批次、1条样本、1条答案。
`mode=STORED`、`purpose=LEARNING_ONLY`、`sample_count=1`、`scorable_count=1`。
`items[0]`与保存当时的同日单日预览完整JSON一致；答案收益为`0.140096188101175632347702173`，表示后来实际上涨约14.0096%，不是预测概率。

### 9.3 保存请求：只传基金、日期与请求凭证

```text
POST http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/batches
```

```json
{
  "fundCode": "008888",
  "startDate": "2025-08-07",
  "endDate": "2025-08-07",
  "pageSize": 10,
  "requestKey": "70bd8a7d-e1d4-4fbb-9e9e-d5eeadab62aa"
}
```

上面的凭证已经用于9.2的批次，所以直接复制会返回**200和原批次**，不会再新增。
想新建一批，使用接口工具生成一个新UUID填入`requestKey`；日期可以换成需要的范围，含首尾最多31个自然日。
只保存这一只基金在范围内实际存在的净值日期，不会导入全历史。没有净值的范围也会保存一条总数为0的批次，不伪造样本。
`pageSize`为1–30的整数，默认10，只控制内部每页读取数量；无需自己传净值、特征或标签JSON。

**重试和重算的区别：** 网络超时、点击重发，沿用原凭证；明确要重新用当前数据库计算，才换新凭证。
同一凭证、相同基金/日期/规则，无论改页大小还是来源随后更新，都返回旧结果。相同凭证改日期或基金返回409。
以后代码升级导致规则版本不同，同一凭证也会冲突；旧批次仍可按编号查询，新规则另建批次。

### 9.4 返回结果与错误

保存首次成功为201；相同请求重试为200；按编号GET为200。三者返回同一份批次结构，重试与GET的完整JSON相同。
相较dry-run，增加`batch_id`、`request_key`、来源水位、三个规则版本、`purpose`和保存时间`created_at`；不保存或返回分页过程字段。
`created_at`带`+08:00`时区；业务日期仍为`YYYY-MM-DD`。`items`沿用原样本结构，字段解释见前文。

| 状态码 | 含义与处理 |
| --- | --- |
| 403 | 缺少/错误服务Token，或请求带Origin；不访问数据库。 |
| 404 | 创建时基金不存在，或查询时批次编号不存在。 |
| 409 | `REQUEST_KEY_CONFLICT`表示凭证被另一范围/规则使用；也可能是基金品类或来源不满足准入。先看`detail.code`。 |
| 422 | UUID、日期、页大小、额外字段等不合法，或构建结果未通过保存一致性校验。 |
| 503 | 数据库/超时等失败，不返回半批成功；保存重试沿用原凭证。若为`STORED_BATCH_INCONSISTENT`，表示已存内容校验失败，应排查，不能靠重试覆盖旧数据。 |

只有`SCORABLE`样本写答案；不合格或答案未成熟的样本仍保留，但不写答案行，更不能用0冒充未知。
批次、样本、答案处于一个事务中：全部成功才提交，任何一步失败整体回滚。服务不更新或删除旧批次。
这是应用层“不覆盖”的约定，不是数据库层防篡改保证；本轮没有增加不可修改触发器或数据库角色权限。

### 9.5 验收与代码位置

- 读回9.2的批次，再重发9.3，完整JSON应一致。仅将`pageSize`改为1或30，应仍为同一`batch_id`。
- 原凭证保持不变，将`endDate`改为`2025-08-08`，应返回409，旧结果仍可GET查询。
- 与现场预览比较时，要求原始净值、来源水位和计算规则未变化；来源更新后保存快照不跟着变，不应强求它永远等于最新预览。
- 数据库验收用下面只读SQL。正常业务写入由上述HTTP事务服务完成，本轮没有额外数据迁移DML，不手工拼造题目或答案。

```sql
SELECT batch_id, request_key, fund_code, start_date, end_date,
       sample_count, scorable_count, purpose, created_at
FROM public.historical_nav_sample_batch
WHERE batch_id = '75b8258a-7cfd-4bfc-b2e8-2b5f05d9f8b3'::uuid;

SELECT s.as_of_date, s.eligibility_status, s.unavailable_reason,
       l.label_end_date, l.future_return_20d, l.label_up_20d
FROM public.historical_nav_sample s
LEFT JOIN public.historical_nav_sample_label l ON l.sample_id = s.sample_id
WHERE s.batch_id = '75b8258a-7cfd-4bfc-b2e8-2b5f05d9f8b3'::uuid
ORDER BY s.as_of_date;
```

| 代码位置 | 职责 |
| --- | --- |
| `app/api/routes/historical_nav_storage.py` | 两个HTTP入口、认证、状态码与带TraceID的摘要日志。 |
| `app/schemas/historical_nav_storage.py` | 请求参数校验、响应字段中文说明、保存时间时区输出。 |
| `app/services/historical_nav_storage.py` | 重试识别、调用公共构建器、整批事务、三表读回。 |
| `app/services/historical_nav_storage_validation.py` | 基金/日期/版本/来源、哈希、字段白名单、数量及答案关系校验。 |
| `app/repositories/historical_nav_storage.py` | 按索引读批次，批量插入明细，一次关联查询读回，最多32行用于发现非法超量。 |
| `app/services/historical_nav_preview.py` | dry-run与保存共用的分页计算，不另写预测或收益公式。 |

离线回归与显式启用的PostgreSQL测试分开运行，具体清单见跨端测试文档TC-FDP-14。
`tests/test_historical_nav_storage_postgres.py`默认跳过；仅设置`RUN_NAV_STORAGE_PG_TESTS=1`后运行，限制本机`fund_ai`，只创建随机独立测试schema，结束清理自己创建的测试数据，不写public业务表。
本次另用真实public净值做了上述最小保存验收，核对原始起点数据未变；没有批量保存全基金历史。

建表时已核验原有168张表、2310个字段定义及注释未变；空表升级/回退和非空拒绝回退在隔离事务验证。
现在已存在真实样本，迁移回退会按设计拒绝；需要保留数据，不能跳过保护强行删表。

后续功能是训练数据准备与基线评估；当前`LEARNING_ONLY`不证明首次可得日期、历史修订和严格交易日历已补齐，不能直接发布为有效预测。

## 10. 当前功能：候选训练数据准备与基线验证

现已在保存闭环之上实现：选择已存批次、核对内容、去重、按时间划分X/y、隔离跨界答案、比较四种简单方法。
**这不是逻辑回归训练，也不是模型发布。** 报告继续为`LEARNING_ONLY / MODEL_NOT_RELEASED`，`training_eligible=false`。
本轮先给验证段打分；独立测试段只准备，不返回成绩、上涨比例或样本答案。它不是数据库权限层面的封存，维护人员仍可能直接查原始样本。

### 10.1 调用新接口

重启Python服务后，沿用`X-Service-Token`，不要带`Origin`。Body选择JSON：

```text
POST http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/baseline-evaluation
```

接口使用POST是为了传批次清单，**全过程只读，不生成新批次、不训练、不补数据**。
完整试点请求在本机`.local-runs/nav-baseline-2022-2025-2a87b245/evaluation.request.json`，复制其内容到Body即可；无需手填144个编号。
这些本地请求/报告文件不含Token，已通过`.gitignore`排除，其他环境需使用自己的已存批次编号。

下面的短例子只选先前保存的1条样本，可用于验收“数量不足”的响应，不是完整试点请求：

```json
{
  "batchIds": ["75b8258a-7cfd-4bfc-b2e8-2b5f05d9f8b3"],
  "trainStartDate": "2022-01-01",
  "trainEndDate": "2023-12-31",
  "validationEndDate": "2024-12-31",
  "testEndDate": "2025-12-31",
  "previewSize": 5
}
```

| 参数 | 含义 |
| --- | --- |
| `batchIds` | 明确选择已经保存的批次，1–512个不同UUID。不自动扫描全历史或选最新批次。 |
| `trainStartDate/trainEndDate` | 训练输入的可得日期范围，含首尾。 |
| `validationEndDate` | 训练截止日之后至此日为验证输入段。 |
| `testEndDate` | 验证截止日之后至此日为保留测试输入段。 |
| `previewSize` | 返回0–20条训练/验证样本，默认5；只影响预览，不影响指纹或成绩。 |

### 10.2 系统怎样准备数据

1. 在一次只读一致性事务中读取批次、样本和答案；每组32个批次查询一次明细，避免逐批、逐样本查询。最多15,872条输入，不加载全部基金历史。
2. 复用存储完整性校验，再要求来源编码和三个规则版本统一、水位存在。基金类型固定股票型。
3. 用“基金+净值业务日”去重：完整内容（包括标签与来源水位）相同才合并，固定保留编号最小的批次作为追溯入口；内容不同返回409，不自动取最新或挑选可用答案。
4. 只纳入`SCORABLE`且采用累计净值的样本。单位净值回退、质量不足等单独计剔除原因，不把缺值补成0。
5. 按`available_at`（输入何时能看到）划段，不按`as_of_date`（净值属于哪天）、批次保存时间或随机比例划分。
6. 再检查`label_available_at`：训练答案不得晚于训练截止，验证答案不得晚于验证截止，测试答案不得晚于测试截止。跨界就剔除，不能只比较标签终点，也不拿20个自然日代替净值窗口。

所以，对本次试点，2023年末尚未公布答案的样本不会进入训练；2024年末同理不会进入验证。
每只基金在这些过滤之后，至少需要训练252、验证120、测试120条样本。任何一只不足，整份报告返回`INSUFFICIENT_DATA`和缺口，`baselines=[]`；不静默丢掉不足的基金。
这里是**可用样本数**，不是原始净值点数；比早期设计按原始点数估算更明确、保守，HTTP不能下调门槛。

内部`PreparedDataset`分开保存`train/validation/test`，每行`x`只含固定7列指标，`y`单独保存历史方向。
基金编号、日期、来源水位和标签元数据仅用于追溯，不能整体喂给模型。矩阵当前只在内存生成，后续可由相同请求和已存批次重建，不是已保存的模型文件。

### 10.3 四种对照和怎样读成绩

| 对照编号 | 做法 |
| --- | --- |
| `ALWAYS_UP` | 不看输入，始终判断上涨，分数为1。 |
| `TRAIN_UP_FREQUENCY` | 每只基金使用自己训练段的“上涨次数/总数”，进入验证后固定不变。 |
| `MOMENTUM_20D` | 最近20日历史收益大于0则判断上涨，否则非上涨，分数为1或0。 |
| `FIXED_MOMENTUM_SCORE` | 复用原固定公式`0.5 + 2 × 最近20日收益`，限制在0.05–0.95，四位小数。 |

为统一比较，四种方法都以“分数严格大于0.5”为上涨，恰好0.5算非上涨；旧评分流程的0.45/0.55中性区间不用于本次二分类比较，原评分流程本身未改。
历史上涨频率与固定公式只是参考分数，不是经过概率校准的未来预测。

- `accuracy`：方向判断正确比例，越高越好。
- `balanced_accuracy`：上涨和非上涨分别算识别正确比例，再平均；验证段只有一种答案时返回null，而不是满分。
- `brier_score`：逐条算`(分数 - 真实0/1答案)²`再平均，越低越好。数值低也不证明模型已经校准或可以发布。

指标定义参考[平衡准确率官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)和[Brier官方说明](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html)。实现使用现有Python/Decimal，不安装scikit-learn。
报告同时给总体和逐基金验证结果，不选冠军、不自动调参；本轮是一次固定划分的学习验证，不是完整滚动回测或经济收益回测。

### 10.4 返回字段和异常

| 字段 | 含义 |
| --- | --- |
| `status` | `INSUFFICIENT_DATA`说明数量不足；`BASELINE_EVALUATED`说明已完成本轮验证段比较，不表示预测模型合格。 |
| `protocol` | 时间边界、固定数量门槛、净值口径、标签隔离与测试保留规则，带独立版本。 |
| `batch_ids/source_sync_run_ids/versions` | 使用哪些保存批次、来源水位和计算规则。 |
| `dataset_hash` | 完整输入批次和时间规则的内容指纹，包含标签；批次顺序、预览数量不影响它。不是落库报告ID。 |
| `train_hash` | 只覆盖实际训练样本；验证/测试标签改变而训练数据不变时，它不应变化。 |
| `input_sample_count/duplicate_sample_count/unique_sample_count` | 去重前、合并重复数量、去重后数量。 |
| `included_sample_count/excluded_reasons` | 纳入三个时间段的总数及互斥剔除原因；合计必须等于去重后数量。 |
| `funds` | 每只基金的三段数量、缺口、剔除原因和训练上涨频率。只有一种训练答案会提示`TRAIN_SINGLE_CLASS`。 |
| `feature_names/sample_preview` | 固定X列顺序及少量训练/验证X和y，测试样本不预览。 |
| `baselines` | 四种方法的总体与逐基金验证结果；数据不足时为空。 |
| `persisted/training_eligible/publication_status/limitations` | 报告未写数据库、不授予正式训练资格、模型未发布，以及首次可得/修订/交易日历等边界。 |

无/错Token或带Origin为403；任一批次不存在为404；内容/来源/规则冲突为409；请求参数错误为422；存储损坏、超时或数据库异常为503。
异常不返回半份报告、连接信息或Token。内部阶段预算15秒、数据库单查询5秒，不宣称是严格HTTP总时长硬中断。

### 10.5 本次真实试点与可恢复执行

用户已确认`001632/006730/008888`、2022–2025年：2022–2023训练，2024验证，2025保留测试。
执行脚本固定此范围，每基金每月一批，共144批；不接受任意日期或其他基金，且限制本机`fund_ai`。
默认只读预检；加`--execute`后也先预检，数量不足时不新增样本。每月独立事务，某月失败不删除此前已完成月份，沿用run-key续跑。

```powershell
# 只读预检，不保存
.\.venv\Scripts\python.exe -m scripts.historical_nav_baseline_pilot --run-key 2a87b245-dae5-45a5-a547-d34409ad0c2b

# 实际执行；本次已执行过，沿用同一run-key会复用已存批次
.\.venv\Scripts\python.exe -m scripts.historical_nav_baseline_pilot --run-key 2a87b245-dae5-45a5-a547-d34409ad0c2b --execute
```

不要为了重试生成新run-key，否则会被解释为一轮新的明确重算。脚本不建表、不改原始净值、不调用外部同步、不训练或发布。
本轮无新增DDL/迁移DML；业务INSERT继续通过已验收的保存事务。原始净值内容在执行前后作范围指纹核验。
当前代码入口：`app/services/historical_nav_evaluation.py`（准备/评估），`app/repositories/historical_nav_evaluation.py`（有界读取），
`app/schemas/historical_nav_evaluation.py`（字段说明），`app/services/momentum_baseline.py`（原固定公式），`scripts/historical_nav_baseline_pilot.py`（显式试点操作）。

## 11. 真正训练第一个候选模型（2026-09-08）

### 11.1 这次做的事情

上一步是“整理历史题目，并给四种简单猜法考试”。这一步让逻辑回归从TRAIN的七个指标和历史答案中学习权重，再对VALIDATION答题。不是查询今天的基金预测，也不是让大语言模型编一个百分比。
三只基金共用一个模型；使用2022–2023年1377条训练行、2024年660条验证行，2025年的657条仍留作最终测试，不出成绩。均值、尺度、系数全部只用训练段计算。

**逻辑回归：** 将七项指标按学习到的权重相加，再转换成0–1的上涨分数。**标准化：** 先减去训练期平均值，再除以训练期尺度，让不同单位的指标便于共同学习。**正则化：** 限制权重不要过分放大，减少“记住历史偶然现象”。
参数方案已在首次真实成绩产生前固定，只训练一次方案，不自动搜索参数。模型输出尚未校准，不能把0.7解释成已经证实的“未来有70%上涨概率”。

### 11.2 HTTP 怎么调用

```text
POST http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/candidate-training
```

Headers沿用现有`X-Service-Token`和`Content-Type: application/json`，不带Origin。服务须加载本次新代码。

Body在第10节基线请求的基础上，只增加一个字段：

```json
{
  "expectedDatasetHash": "67e1bd1ab68226a1e8ef29cfd5f369cd3879eb96d57a2e8ce20276ca243e1e10"
}
```

这只是**新增字段示意，不是完整Body**；原来的batchIds、四个日期和previewSize仍需保留。该字段复制自基线返回的`dataset_hash`，作用是确认“训练的仍然是这一本练习册”。不匹配返回409，不训练。

本机已准备完整可复制文件，无需手抄144个编号：

- [本次训练请求](C:/pythonProject/workSpace06/.local-runs/nav-candidate-6e25d583-5ab3-443c-bd65-8aa7ae2ce5d1/request.json)
- [完整训练报告](C:/pythonProject/workSpace06/.local-runs/nav-candidate-6e25d583-5ab3-443c-bd65-8aa7ae2ce5d1/report.json)
- [模型JSON](C:/pythonProject/workSpace06/.local-runs/nav-candidate-6e25d583-5ab3-443c-bd65-8aa7ae2ce5d1/model.json)

这些是本机忽略文件，不随Git提交；其他环境要使用自己的已保存批次及基线指纹。HTTP只返回JSON，不自动写服务器文件，不保存到模型数据库。

### 11.3 返回内容先看哪些

| 字段 | 通俗解释 |
| --- | --- |
| status=CANDIDATE_EVALUATED | 模型已经训练并完成验证考试，不等于通过上线门槛。 |
| status=INSUFFICIENT_DATA | 有基金样本不够，本次未训练；model/candidate为null，缺口见preparation。 |
| preparation | 上一步的数据筛选、数量、指纹和四种基线成绩，直接复用，没有换另一套样本。 |
| protocol | 事先固定的训练方案，如正则强度、最多迭代次数、未做概率校准。 |
| model | 训练成果，下面的均值、尺度、权重等保存在这里，不是一串无法阅读的二进制。 |
| candidate.validation | 新模型在全部2024验证样本上的成绩；accuracy看方向，Brier看分数与答案的偏差。 |
| candidate.per_fund | 分基金成绩，防止总体均值掩盖某只基金表现差。 |
| baseline_deltas | 新模型成绩减去每条基线成绩。准确率差值正数更好；Brier差值负数更好。0.01准确率差值代表1个百分点。 |
| prediction_preview | 最多previewSize条历史验证预测，包含分数、方向和当年真实答案；不是今日预测。 |
| artifact_persisted=false | 训练服务未写模型文件；CLI可另行保存，保存状态看complete.json。 |
| database_written=false | 没有修改数据库中的净值、样本或模型表。 |
| training_eligible=false | 严格历史数据准入仍未通过，不是说这次离线实验没有训练成功。 |
| MODEL_NOT_RELEASED | 研究模型不能展示为“我的关注”正式预测。 |

模型中的字段：

| 字段 | 含义 |
| --- | --- |
| feature_names | 七个指标的固定顺序，必须与mean、scale、coefficients逐项对应。 |
| mean / scale | 只从训练段学出的平均值和尺度；常量列尺度为1，避免除零。 |
| coefficients | 七项标准化指标的权重；正数推高上涨得分、负数压低。相关指标共同作用，不能把绝对值直接当因果重要性。 |
| intercept | 基础得分，和指标加权结果相加后才转换为0–1。 |
| classes | [0,1]分别是非上涨、上涨；up_score取类别1的输出。 |
| iterations | 求解器实际计算了多少轮，本次21轮；不是训练了21天，也不是发布门槛。 |
| train_count / train_counts_per_fund | 总训练量及逐基金数量，本次1377条、每基金459条。 |
| sample_weight_per_fund | 每基金每条训练行的权重，使基金总贡献相等；本次数量相同，所以每行权重都是1。 |
| train_class_counts | 仅训练段的0/1答案数量，不含验证和测试。 |
| train_hash / train_start_date / train_end_date | 模型从哪份训练内容、哪个时间范围学来。 |
| versions / runtime_versions | 特征/标签规则及Python、数值库版本，便于以后复现。 |
| model_hash | 模型内容指纹；只变验证或测试答案不改变模型指纹。不是防恶意篡改的签名或发布授权。 |

### 11.4 如何保存、读回和重跑

HTTP响应本身包含完整model，可以在API工具保存响应。要生成完整本机实验包，在Python项目根目录运行：

```powershell
$candidateRunKey = [guid]::NewGuid().ToString()
.\.venv\Scripts\python.exe -m scripts.historical_nav_candidate --request .local-runs/nav-baseline-2022-2025-2a87b245/candidate.request.json --run-key $candidateRunKey
```

CLI只允许当前本机`fund_ai`，新建`.local-runs/nav-candidate-<run-key>/`；写request.json、report.json、model.json，最后写complete.json。清单包含文件SHA256；清单可解析且各文件指纹匹配，才是完整保存。失败目录保留、不删除旧文件；换新run-key可重新执行。相同run-key目录已存在则拒绝覆盖，不静默重新训练或替换。
这与“样本保存接口同requestKey复用”的语义不同：这里的run-key是**新实验目录编号**，不写样本批次，不需要重新补存144个月。

代码入口：

- `app/services/historical_nav_training.py`：只读准备→TRAIN拟合→JSON重算核对→VALIDATION成绩。`restore_logistic_artifact`校验模型，`predict_artifact_scores`只接收七列X，不接收答案，不挂线上预测路由。
- `app/schemas/historical_nav_training.py`：请求/协议/模型/返回字段中文说明。
- `scripts/historical_nav_candidate.py`：本机实验包保存；没有pickle反序列化。

### 11.5 本次真实结果与边界

| 方法 | 2024验证准确率 | 平衡准确率 | Brier，越小越好 |
| --- | --- | --- | --- |
| 逻辑回归候选 | 53.18%（351/660） | 52.80% | 0.26914973 |
| 历史训练上涨频率 | 50.61%（334/660） | 50.00% | 0.25820037 |
| 原固定公式 | 48.48%（320/660） | 48.43% | 0.27739452 |

候选逐基金：001632为48.18%，006730为59.09%，008888为52.27%。总体方向准确率比历史频率高约2.58个百分点，但Brier反而高约0.01095；不能只挑准确率宣布“全面提升”。未改参数追求更漂亮的成绩，未使用2025测试答案选模型。

模型指纹`05c5873aecb134651222011ce8a282e69c1812de62f4a9bf3022c96e26ebebac`；完整数据指纹仍为第10节那一份。真实运行前后三表数量保持145/2923/2875（含早期单日批次），三基金2022–2025的2922条原始净值内容指纹一致。
新增依赖组已固定在requirements.txt：scikit-learn及其数值依赖，使用当前项目虚拟环境；原FastAPI等依赖未升级。

训练库API口径参考官方[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)与[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)；`l1_ratio=0`表达L2，标准化支持训练样本权重。持久化仅导出数值JSON，不引入[模型持久化文档](https://scikit-learn.org/stable/model_persistence.html)所提醒的未知pickle执行风险。

下一阶段建议做训练期内部的时间隔离校准与滚动验证设计，仍不提前打开2025测试成绩。正式首次可得/历史修订核验、严格日历和发布闸门尚未完成，不能直接接入用户预测卡。

本轮验证：265项离线测试、18项隔离PostgreSQL测试通过；真实数据库的进程内HTTP和独立临时18011网络HTTP均为200，与保存报告一致，无Token/Origin均403。临时服务验收后关闭；验收时8000未连通，你需在IDE或原启动方式中启动/重载Python服务后，再使用上述8000 URL。没有改动已有其他监听进程。

## 12. 校准与滚动研究验证（2026-09-08）

### 12.1 这次做到了什么

现在能把“基础模型学习、分数调整、最后考试”分成三个先后时间段，换两次历史窗口再考试，并在2024年做固定验收。基础模型和校准器各有自己能看的答案，考试答案不能用于学习。
2025继续保留，未算成绩。没有同步新净值、补存样本、写模型表或激活线上预测；第11节原模型和报告也未覆盖。

**校准不是自动变准按钮。** 校准器只学习“原始得分z应该怎样映射到0–1”，本次固定`sigmoid(a*z+b)`的一维L2逻辑回归，既不搜索参数，也不比较多个校准方法择优。基础模型保持冻结，不能校准后再偷偷重训基础模型。
概率可靠性参考官方[校准说明](https://scikit-learn.org/stable/modules/calibration.html)：Brier同时受区分能力和数据不确定性影响，不能仅看它判断校准。因此这里还输出固定分箱和ECE。

### 12.2 固定的三次考试

| 窗口 | 基础模型学到哪天 | 独立校准段 | 考试段 | 每基金实际拟合/校准/考试数 |
| --- | --- | --- | --- | --- |
| DEV_2023_Q3 | 2022-01-01至2023-02-28 | 2023-03-01至06-30 | 2023-07-01至09-30 | 254 / 61 / 44 |
| DEV_2023_Q4 | 2022-01-01至2023-05-31 | 2023-06-01至09-30 | 2023-10-01至12-31 | 315 / 64 / 40 |
| VALIDATION_2024 | 2022-01-01至2023-06-30 | 2023-07-01至12-31 | 2024年 | 335 / 104 / 220 |

前两轮是原TRAIN内部的扩展窗口，后轮可以用已经成为历史的前轮信息；不是每轮都永远不碰任何曾经考过的样本，而是任何时点都不能看自己未来的答案。2024则仍只用于验收，不参与任何基础模型/校准器的拟合。
所有日期指输入可得日期，不是保存时间；每段答案也必须已在段末披露，否则剔除。全局筛选先执行，窗口再筛选，不能用20个自然日机械代替。季度考试40条是短窗研究诊断下限，不替代全局252/120/120门槛；校准至少60条，基础拟合仍至少252条/基金。

### 12.3 怎么调接口

```text
POST http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/calibration-evaluation
```

Headers与第11节相同，Body也**完全沿用第11节候选训练请求**：batchIds、四个日期、previewSize、expectedDatasetHash。不需要你再配置三组窗口。
本版只接受固定2022-01-01 / 2023-12-31 / 2024-12-31 / 2025-12-31边界；改日期、自定义windows、method、最低样本量、includeTest或输出路径均422。坏指纹409；与候选训练共享计算槽，忙时429。

本机完整文件：

- [可直接复制的请求](C:/pythonProject/workSpace06/.local-runs/nav-calibration-f0f9f709-b82b-4af1-8326-f09b11abac50/request.json)
- [三轮完整报告](C:/pythonProject/workSpace06/.local-runs/nav-calibration-f0f9f709-b82b-4af1-8326-f09b11abac50/report.json)
- [三轮模型及校准参数](C:/pythonProject/workSpace06/.local-runs/nav-calibration-f0f9f709-b82b-4af1-8326-f09b11abac50/models.json)

这些文件仅存在本机忽略目录，不会随Git提交。HTTP不自动保存文件，重跑不会覆盖它们。代码入口为`app/api/routes/historical_nav_calibration.py`、`app/services/historical_nav_calibration.py`和同名schemas文件，字段都有中文说明。

### 12.4 返回字段怎么读

| 字段 | 通俗解释 |
| --- | --- |
| CALIBRATION_EVALUATED | 三轮都算完了，不是三轮都变好了，更不是允许发布。 |
| PARTIAL_EVALUATION | 部分窗口有缺口或被拒绝；没有偷偷丢掉失败窗口。 |
| INSUFFICIENT_DATA / NO_VALID_WINDOWS | 全部不足，或没有可用的完整考试窗口；看每窗reason，不能当成功模型。 |
| preparation | 原来的全局筛选报告；其中基线是全局2024基线，不要混同逐窗基线。 |
| protocol.windows | 事先冻结的时间窗口，不是根据结果挑出的好时段。 |
| windows[].funds.counts / missing | 每基金本窗FIT、CALIBRATION、EXAM数量与缺口。 |
| windows[].funds.purged | 全局筛选后本窗口新增的跨界答案剔除数。比如2023年末跨界已在全局剔除，本窗CAL/EXAM可以是0，不代表没有隔离。 |
| before / after | 同一个基础模型校准前、后的总体现时段及逐基金成绩；里面复用的validation字段指当前窗考试，不一定是2024。 |
| baselines | 四种简单方法在同一考试段的成绩；历史频率只用校准截止前已成熟的历史答案，考试段不更新。 |
| reliability_before / reliability_after | 校准前后总体分箱与ECE；逐基金版本见reliability_per_fund。 |
| brier_delta / ece_delta | 校准后减校准前，负数更好，正数更差；不能据此自动发布。 |
| prediction_preview | 每窗最多previewSize条历史考试例子：before_score、after_score和实际答案。不是2025测试或今日预测。 |
| evaluated_window_count | 本次完整完成的窗口数；不是训练成功率。 |
| brier_improved_window_count | Brier严格改善的窗口数量；本次1，不是“通过1票就上线”。 |
| model.base_model | 本窗冻结的基础七指标模型，字段沿用第11节。 |
| model.calibrator | 校准器参数：slope是a，intercept是b；另有校准时间、数量、基金权重、内容指纹和绑定的基础模型哈希。 |
| model.model_hash | 基础模型和校准器整体指纹；考试答案变化不会改变同窗模型。 |
| REVERSED_PENDING_VALIDATION | 反向校准，需要独立数据验证；允许离线计算/评分，不表示算法失败或模型合格。 |
| CONSTANT_PENDING_VALIDATION | 零斜率只输出常数概率，未利用原始分数区分样本；按固定概率对照验证。 |
| NON_POSITIVE_CALIBRATION_SLOPE | 历史报告保留的旧提示码，新生成报告使用上面两个明确状态；原记录不改写。 |

**可靠性分箱：** 分数固定分为0–20%、20–40%、40–60%、60–80%、80–100%五档，前四档不含右端点，最后包含1。`count`看多少条，`mean_score`看平均报多少，`observed_up_rate`看实际涨多少，`absolute_gap`看两者差多少。空档返回null，不伪造0%；`enough_samples=false`表示不足30条。
**ECE：** 把各档偏差按样本数量加权平均。它越小通常表示这些分档中报出的分数更贴近实际比例，但分箱和小样本都会影响它，不能单独证明未来概率准确。重叠20日标签并非独立试验，所以没有提供误导性的独立样本置信区间。

### 12.5 真实结果：没有稳定改善

| 考试段 | 校准前准确率 → 校准后 | Brier前 → 后（越小越好） | ECE前 → 后（越小越好） |
| --- | --- | --- | --- |
| 2023第三季度 | 64.39% → 67.42% | 0.25659806 → 0.25797493 | 0.17831184 → 0.20043181 |
| 2023第四季度 | 55.83% → 61.67% | 0.30338054 → 0.21464207 | 0.23039864 → 0.15585841 |
| 2024固定验证 | 51.82% → 51.21% | 0.27513393 → 0.33669545 | 0.13196165 → 0.26291714 |

只有2023第四季度的两个分数偏差指标变好，但它的映射斜率为-0.7566；2024映射斜率也为负（-0.6864），且指标明显变差。三轮历史频率基线准确率分别69.70%、62.50%、50.61%，Brier分别0.23976427、0.23678793、0.25820037。不能只展示一轮较好的成绩，也不能只看准确率。
本轮2024“校准前51.82%”不是上一轮全TRAIN模型的53.18%：这里特意留了一段历史给校准，基础模型只学了1005条，而不是1377条。因此校准效果必须在本轮51.82%与51.21%之间比较。上一轮模型未变，重跑原接口仍返回原完整报告。
目前结果只支持“这套固定校准方案没有在所检验窗口稳定改善”，不能确定根因就是某项数据差或某种市场状态；需要另做失败归因。没有根据本次成绩重选时间、强制正斜率、改校准器或改参数。

### 12.6 本机保存与下一步

```powershell
$calibrationRunKey = [guid]::NewGuid().ToString()
.\.venv\Scripts\python.exe -m scripts.historical_nav_calibration --request .local-runs/nav-candidate-6e25d583-5ab3-443c-bd65-8aa7ae2ce5d1/request.json --run-key $calibrationRunKey
```

只允许本机fund_ai，写新`.local-runs/nav-calibration-<run-key>/`。request.json、report.json、models.json写完并回读核验后才写complete.json，完成清单记录文件SHA256及三组模型指纹。重复目录拒绝覆盖；磁盘失败保留未完成目录，不删除旧实验。部分窗口失败也可以保存诊断，但退出码2、不伪装全部通过。
没有新依赖、表、外部数据接入、Java/Vue改动或模型发布；原样本和旧实验文件保留。下一步先做**失败归因和历史数据准入核验**，再决定是否需要扩大历史/基金覆盖或调整训练方案；本轮不自动执行这些扩展，不提前打开2025最终测试。

### 12.7 本次实际验收到哪里

- 303项离线相关测试、19项真实PostgreSQL隔离测试通过，共322项；本轮新增38项离线校准用例和1项隔离库端到端用例。Ruff与pip check通过，保留既有TestClient弃用警告。
- 新接口连接真实数据库，重复请求、批次倒序和保存报告完整JSON一致；原候选训练接口仍返回上一轮完整报告。
- 临时18012端口实际HTTP返回200，无Token/Origin为403，错误指纹409；完成后临时服务已关闭。**验收时8000未连通，你使用8000调用前需要启动或重载Python服务。** 没有擅自重启用户已有进程。
- 原始净值范围2922行全列内容指纹不变；三表仍为145批次、2923样本、2875答案。上一轮实验四文件指纹均不变，新实验完成清单和三组模型回读验证通过。
- 这些检查证明接口、隔离规则和结果保存按预期工作；不证明模型预测已经准确。详细用例见主项目测试文档TC-FDP-17。

## 13. 只读失败诊断：不重新训练，也不新增HTTP

本步复盘已保存的校准实验，使用旧模型对原来的训练/校准/考试段重算分数，核验成绩完全复现后再拆分统计。只检查已有三基金，读取当前原始净值上限为2024-12-31；2021数据仅提供2022起点历史。既有样本准备层仍会读取2025做完整性/指纹校验，但诊断计算不访问TEST对象、不输出测试成绩。

完整人话结论见[主项目诊断报告](C:/WebStormProject/workSpace05/docs_zhx/implementation/free-data-prediction-v1-diagnostics.md)。2024出现明显偏向不涨；同时发现周末净值、累计/复权口径不等价及首次版本证据缺口。诊断完成不表示正式准入通过。

### 13.1 怎么运行

```powershell
$env:PYTHONIOENCODING = 'utf-8'
$diagnosticRunKey = [guid]::NewGuid().ToString()
.\.venv\Scripts\python.exe -m scripts.historical_nav_diagnostics --calibration-run-key f0f9f709-b82b-4af1-8326-f09b11abac50 --run-key $diagnosticRunKey
```

- `calibration-run-key`：你要复盘的旧校准实验编号，目录必须有完整request/report/models/complete四文件。
- `run-key`：新诊断编号。只新建`.local-runs/nav-diagnostics-<编号>/`，拒绝覆盖既有目录；不需要8000端口启动，不通过HTTP重训模型。
- 本机fund_ai限制不变；不接受任意模型路径、输出目录、基金扩展或训练参数。文件上限2MiB，固定文件名，核验完成清单SHA256及各JSON关联，错误则不交付完成清单。
- 成功写report.json，回读核验后最后写complete.json。失败退出码1，无有效完成清单；不删除旧数据或旧实验文件。

本次[完整JSON报告](C:/pythonProject/workSpace06/.local-runs/nav-diagnostics-1054d538-301e-44a1-858c-0deca79a32f6/report.json)可直接打开，运行产物不随Git提交。

### 13.2 主要字段

| 字段 | 人话解释 |
| --- | --- |
| source_calibration_run_key / source_files_sha256 | 复盘哪次旧实验、旧文件有没有变化。 |
| model_fitted=false / test_scored=false | 没有再训练，也没有给2025考试。 |
| windows[].stages.FIT / CALIBRATION / EXAM | 对应原来三个时间块；前两块的成绩是样本内描述，不能作为独立验收。 |
| overall / per_fund | 总体和每只基金分别统计，不能只挑成绩好的基金。 |
| actual_up_rate | 这段历史真实上涨样本比例。 |
| mean_score_before / mean_score_after | 校准前后平均报出的分数，不是准确率，也不等于判涨数量。 |
| up_minus_down_mean_logit | 真涨与不涨样本的平均基础线性得分差；负数表示平均关系反向，单类时null，不是因果结论。 |
| before / after | 原来相同公式的方向准确率、平衡准确率、Brier等，EXAM必须与旧报告完整一致。 |
| features[].mean_shift_in_fit_scale | 该段指标均值相比冻结FIT均值偏了多少个FIT尺度；没有用考试数据重新标准化。 |
| outside_fit_range_rate | 该段有多少输入超出FIT观察过的最小/最大范围，不等同于坏数据比例。 |
| mean_absolute_linear_contribution | 对此模型线性得分的平均绝对数值贡献；不是特征因果重要性。 |
| exam_quarters | 原2024考试集合按available_at拆四季，数量相加与原考试一致；不是四个新独立回测。 |
| nav_audit[].current_replay_mismatches | 当前源净值重算输入、方向及可得日期，与已存样本是否不同；空对象表示未发现这些字段差异。 |
| weekend_nav_dates / evaluated_windows_containing_weekend_future_nav | 源净值的周末日期，以及多少已纳入样本的未来20条包含周末；不自动删除来源记录。 |
| accumulated_vs_adjusted_direction_difference_count | 同起终点改看复权净值时，有多少历史方向不同；只做敏感性比较，不替换标签。 |
| mean/max_absolute_return_gap_bps | 两口径收益差绝对值，单位基点；100基点=1个百分点，不能与涨跌准确率混用。 |
| raw_snapshot_hash | 当前诊断原始行快照指纹；范围/字段与旧净值完整性MD5不同，不能相互比较。 |
| catalog_and_source | 当前目录、来源登记和本机表清单；不代表历史分类、供应商账号权限或法律授权被重新核验。 |
| admission.status=NOT_APPROVED | 首次版本、交易日历、分红口径等仍未闭环，不得发布。 |

代码入口：`scripts/historical_nav_diagnostics.py`负责旧文件核验和新报告保存，`app/repositories/historical_nav_diagnostics.py`负责同来源有界只读查询，`app/services/historical_nav_diagnostics.py`负责纯统计和样本重放。没有改变已有HTTP或训练代码。

## 14. 交易日窗口预览：先确定日期，不计算涨跌

这是新入口，旧GET不变。旧GET读取某个净值业务日的样本；新GET回答：“站在某一天结束时，我已经知道哪个交易日的净值？历史资料必须对应哪61天？未来20交易日准确截止哪天？”

### 14.1 怎么调用

```text
GET http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/trading-window-preview?fundCode=008888&cutoffDate=2025-08-08
```

Headers使用已有`X-Service-Token`，不带Origin，Body留空。只支持001632、006730、008888；不传`asOfDate`、`pageSize`、`horizon`、`includeTest`、日历路径或训练参数。

本次验收时8000未监听；在Python仓`C:\pythonProject\workSpace06`用原环境启动新代码：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

若你已有别的端口服务，先确认它加载新代码并改URL端口，不必重复启动。此次临时18013网络验收服务已关闭，未重启其他进程。

`cutoffDate`按自然日结束解释。8月7日的净值8月8日公告，就不能作为8月7日已知信息；传8月8日时才可能选中它。只有日期没有准确发布时间，不能拿这个接口证明8月8日上午也已经看得到。

### 14.2 返回字段用人话解释

| 字段 | 含义 |
| --- | --- |
| mode | 固定只读交易日窗口预览，不是样本保存或模型预测。 |
| window_rule_version | 本次独立日期规则版本`TRADING_WINDOW_AFTER_CUTOFF_V1`；不是旧样本版本改名。 |
| status | `WINDOW_DATES_COMPLETE`日期及公告齐全；`INPUT_DATES_INCOMPLETE`起点/历史有问题；`FUTURE_DATES_INCOMPLETE`历史齐但未来日期不齐。只看日期，不代表可训练。 |
| fund_code / cutoff_date | 当前基金和信息截止日。 |
| source_code / source_sync_run_id | 当前净值来源及成功同步水位；不是日历来源，也不证明每行历史首次版本。 |
| anchor_nav_date / anchor_ann_date | 最新已公告且属于交易日的净值日期，以及它的公告日。没有可用起点则null。 |
| anchor_lag_sessions | 这个起点比截止时最近交易日落后几天，按交易日而非自然日数；最多允许1天。 |
| anchor_issue | null表示起点正常；`NO_KNOWN_TRADING_NAV`没有已知交易日起点；`STALE_ANCHOR_OVER_ONE_SESSION`超过1交易日滞后；`CALENDAR_HISTORY_SHORTAGE`起点前历史日历不够。异常时不输出一份假装完整的历史。 |
| history_dates | 截至起点的61个准确交易日；61个数值才对应60段相邻变化，后续算60日指标要用。 |
| history_issues | 历史要求日期中缺失/公告有问题的项目；不会向前多读一天来顶替。 |
| future_dates / future_end_date | 严格在cutoff之后的20个交易日，以及第20天。缺净值也不往后挪终点。 |
| future_issues | 未来这些日期是否已有源记录及合法公告，仅作离线检查，不参与选择历史输入。 |
| future_navs_available_at | 未来20天都有合法公告日期时的最晚公告日；缺失则null。这不是收益，也不是已生成的标签。 |
| ignored_non_trading_nav_dates | 本次读取范围里有净值但不开市的日期；仅不参与本规则计数，原记录没有删。可能含历史和未来的诊断日期。 |
| anchor_based_20th_trading_date | 用日历从起点净值日后数20天的对照终点，不是新规则使用的终点。 |
| legacy_20th_nav_date_from_anchor | 旧规则从起点后数第20条源记录的对照终点；只看本次有界读取范围，不足20条返回null，不扩查或计算旧标签。 |
| database_written / feature_generated / label_generated | 固定false：没写库、没生成模型输入指标、没算后来涨跌答案。 |
| nav_values_verified / training_eligible | 固定false：没有读取和核验净值数值，更未获训练资格。 |
| publication_status / limitations | 固定`MODEL_NOT_RELEASED`及明确限制，日期齐全不会放行发布。 |

`history_issues`和`future_issues`里的`nav_date`是准确出问题日期；`reason`分别是`MISSING_NAV`缺该日记录、`MISSING_ANN_DATE`缺公告日、`ANN_BEFORE_NAV_DATE`公告早于净值日、`NOT_KNOWN_AT_CUTOFF`截止时还不知道（只用于历史）。

`calendar`说明这把“日期尺子”从哪来：`version`版本、`content_hash`固定内容指纹、`market`适用市场、`coverage_start/end`覆盖起止、`reviewed_on`本地核验日、`construction`按官方休市事实派生的方法、`source_urls`所涉年度沪深官方公告。运行时只读本地静态JSON，不联网抓取日历。

`calendar.future_schedule_known_at_cutoff=false`尤其要注意：表示所用未来年度安排在cutoff之后才公告。接口允许看这份事后日期诊断，但不会宣称当时已知或可以训练。试传`cutoffDate=2023-12-08`可看到此情况。日历不是基金申赎或境外市场日历；缺少历史首次公告版本的问题没有因此解决。

### 14.3 两组真实结果怎么验收

请求008888、`cutoffDate=2025-08-08`：

- `anchor_nav_date=2025-08-07`，`anchor_ann_date=2025-08-08`，`anchor_lag_sessions=1`。
- 历史61天，从2025-05-14至2025-08-07；未来20天，从2025-08-11至2025-09-05。
- `future_end_date=2025-09-05`，两种从净值日起算的对照终点都是2025-09-04；差异在于信息截止日和净值业务日不是同一天。
- 状态日期完整，问题列表为空，未来日期公告最晚2025-09-06；无净值数值、涨跌答案和概率。

再改`cutoffDate=2024-03-07`：

- 起点2024-03-06；旧记录计数终点2024-04-02；从起点按交易日数终点2024-04-03；新规则从截止日之后数终点2024-04-08。
- 忽略列表含2023-12-31与2024-03-31；3月31日是周末，有净值也不占交易日位置。新终点还跨过清明休市，故不是简单把旧终点机械加一天。

同一数据/来源/日历版本重复请求完整JSON一致。这里只返回有限日期，**没有分页参数**；原批量接口的页大小一致性要求仍由原接口验收。

### 14.4 错误、实际检查与代码入口

- 无Token、错Token或带Origin：403。基金不在三试点、日期格式错或多传参数：422。
- 缺基金404；基金/来源不适用、日历无法覆盖完整前后窗口409，日历错误码`CALENDAR_COVERAGE_INSUFFICIENT`。例如传2026-01-01或2025-12-31；2021年初历史不足也拒绝，不猜日期。
- 数据库或日历文件损坏：503，业务错误码`TRADING_WINDOW_UNAVAILABLE`，不输出内部连接或文件细节。日期范围通过但源记录有洞则200带不完整状态，不伪装接口宕机。
- 新增50项测试；相关离线382项、真实隔离PostgreSQL19项，共401项通过。真实网络200/403/422/409、TraceID传递、重复结果均验收。新入口7次真实读取共35条SET/SELECT，净值查询只含`nav_date`和`ann_date`。旧样本145批次/2923行/2875答案及净值、旧实验指纹未变。

本次[日期验收JSON](C:/pythonProject/workSpace06/.local-runs/nav-trading-window-0147bbd8-f375-40a3-aabc-fc1bea4a3955/report.json)是本地只读验收记录，不是可再次训练的样本数据包。

调用链：`app/api/routes/trading_nav_window.py`负责鉴权/参数/错误；`app/services/trading_nav_window.py`负责日期和只读事务；`app/repositories/trading_nav_window.py`只查两个日期字段；`app/services/trading_calendar.py`加载并校验固定日历；`app/schemas/trading_nav_window.py`说明输入输出。原构建器、旧标签和模型不变，下一步才核准净值口径并接入独立版本样本。

## 15. 净值口径审计：看清楚不同净值到底算出了什么

上一节只看日期，这一节读取2022–2024范围内的净值数值和分红进行对照。它不是新版样本接口，也不改变旧GET；没有生成特征、20日答案或模型概率。

### 15.1 请求

```text
GET http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/nav-basis-audit?fundCode=006730&startDate=2023-01-01&endDate=2023-12-31
```

Headers用现有`X-Service-Token`，不带Origin，Body留空。只支持三只既有试点，起止均在2022–2024内，最多366自然日；2025、反向/过大范围、`navBasis/includeTest/cutoffDate`等额外参数均422，不查询数据库。没有交易日的范围409；缺基金404，来源不适用409，内部错误503，不展示私密连接细节。

本次验收时8000没有监听。启动方式沿用第14.1节；实际验收用18014，完成后已停止，未重启你的其他进程。

### 15.2 主要返回字段

| 字段 | 人话解释 |
| --- | --- |
| mode / audit_rule_version | 固定只读口径审计及V1规则，不是训练或样本版本。 |
| status | `DIFFERENCES_FOUND`发现现金/复权日差异超过固定1基点；`NO_LARGE_DAILY_DIFFERENCE_FOUND`未发现这种大日差异；`AUDIT_INCOMPLETE`日期、数值、公告或事件存在问题。不是上线状态。 |
| source_code / source_sync_run_id | 当前净值与分红共同来源、净值同步水位；不能证明分红已完整或当时首次版本正确。 |
| calendar_version / calendar_hash | 使用上一步独立日历数交易日；不是用源记录条数代替。 |
| snapshot_hash | 当前有界净值、分红、范围、来源和日历的内容指纹；同一快照重复一致，不是特征哈希。 |
| trading_day_count | 范围内准确交易日数量。 |
| raw_nav_count | 实际净值行数，包含用于分母的前一个交易日，以及被忽略的非交易日。 |
| ignored_non_trading_dates | 有净值但不占日序列位置的日期，源记录未删除。 |
| invalid_nav_counts | 三列分别检查，缺日期/缺值/非正/NaN或Infinity都算无效；不会用别的净值列补上。 |
| accumulated_dividend_missing_count | 所需日期里累计分红字段为空的数量；空值不是没有分红。 |
| dividend_record_count | 范围内可能有关的源分红记录数，包括冲突或未知进度的记录。 |
| checked_daily_pairs | 实际完成对照的相邻交易日数量；缺数据不能跳过后假装全数完成。 |
| daily_gap_threshold_bps / daily_gap_over_threshold_count / max_daily_gap_bps | 固定1基点阈值、超过阈值的日数、最大绝对差。100基点=1个百分点。这是现金算式与来源复权的差，不是预测误差或发布标准。 |
| issues | 需要核对的具体日期与原因；不是自动修复清单。 |
| period_comparison | 同一基准日和终点的三列比值，以及条件性现金再投资连乘对照，见下文。 |
| dividend_comparisons | 每个合法且唯一实施分红日的原始数值与公式结果，便于拿计算器核对。 |
| admission_status / publication_status | 所有状态均`NOT_APPROVED`、`MODEL_NOT_RELEASED`，HTTP200不代表准入通过。 |
| automatic_basis_fallback_allowed | false：本入口不允许把净值列互相替换。旧学习样本的已有规则并未被暗改。 |
| database_written / feature_generated / label_generated / model_fitted / test_scored | 全false：没写库、没生成输入/答案、没训练、没给2025评分。 |
| evidence_urls / limitations | 官方字段说明链接及计算/事件完整性/历史版本限制；接口运行时不联网。 |

`period_comparison.base_date`是首个审计交易日前一个交易日；2023年度示例为2022-12-30，`end_date=2023-12-29`。因此一年242个交易日需要243个所需净值点，数据库还可能多出周末记录，三个数量不矛盾。

`unit_nav_ratio_return/accumulated_nav_ratio_return/source_adjusted_nav_ratio_return`分别计算同端点“对应列终点值/基准值-1”；只表示各列的数学变化，不表示经济含义等价。`cash_reinvestment_candidate_return`将`(当日单位净值+现有每份现金分红)/前日单位净值`逐日连乘后减1，假设现金当日按单位净值再投。它不同于把多年现金简单加在最后，不含到账/申赎/个人持仓信息；没有现有事件时暂按现金0只是条件性假设。有缺数、异常事件或无法解释的日差异则返回null，不能用部分天数拼一个全年结果。

### 15.3 真实示例和分红字段

006730、2023年度返回`DIFFERENCES_FOUND`，242个交易日完成对照，1条实施事件，最大日差26.152031198640基点。原始读取246行，含243个所需交易日点和3条非交易日记录。

其中2023-06-21这一项：

- `previous_trading_date=2023-06-20`，`unit_nav_before=1.4188`，`unit_nav_after=1.2235`，`cash_per_share=0.1676`。
- `unit_nav_ratio_return≈-0.137651536510`，约跌13.7652%，没计入派息。
- `cash_inclusive_day_return≈-0.019523541021`，约跌1.9524%，明确按`(1.2235+0.1676)/1.4188-1`算。
- `source_adjusted_day_return≈-0.022138744140`，约跌2.2139%；两者差`absolute_gap_bps≈26.152031198640`。
- `ex_cash_denominator_hypothesis_return≈-0.022138746803`是另一种公式`1.2235/(1.4188-0.1676)-1`的假设对照，数值接近不等于供应商确认该算法。

本例全年同端点结果分别为：单位净值比值约-17.7952%，累计净值比值约-4.1216%，来源复权比值约-6.7837%，按现有事件的条件性现金再投约-6.5344%。这些是历史口径对照，不是模型成绩，更不是未来预测或个人实际收益。

日期字段优先`net_ex_date`映射值，缺失时用`ex_date`；同时存在且冲突则拒绝对照。只计算唯一“实施”事件；不能把“预案”、重复事件或不明现金直接当成已实施分红求和。

常见`issues[].code`：`TRADING_NAV_MISSING`缺准确交易日，`INVALID_*_NAV`对应数值无效，`NAV_ANN_DATE_INVALID`公告异常，`DIVIDEND_DATE_CONFLICT`生效日期冲突，`MULTIPLE_DIVIDENDS_REQUIRE_REVIEW`同日多条待核对，`DIVIDEND_NOT_IMPLEMENTED`非实施进度，`DIVIDEND_CASH_INVALID`派息无效，`UNEXPLAINED_ADJUSTMENT_WITHOUT_DIVIDEND`没有对应现有事件却出现较大复权差异。完整原因码在服务与测试中维护，不会据此自动补数。

### 15.4 实际验收和代码位置

64项新增用例，相关离线446项、真实隔离PG19项通过。三基金×三年九组真实审计及一次TestClient实库请求共60条SET/SELECT；未读样本表。临时网络200/403/422/409、重复JSON及TraceID均通过。旧数据和旧实验指纹不变，未执行新训练或保存样本。

本次[审计验收JSON](C:/pythonProject/workSpace06/.local-runs/nav-basis-audit-f6931bd0-5860-4320-abf5-a79882f02c84/report.json)包含九组完整结果及保护快照。它位于Git忽略目录，不是训练数据包。

`app/api/routes/nav_basis_audit.py`负责HTTP，`app/repositories/nav_basis_audit.py`负责单基金/同来源/有界查询，`app/services/nav_basis_audit.py`负责固定算式和拒收，`app/schemas/nav_basis_audit.py`含字段注释。检查入口：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nav_basis_audit.py -q --tb=short
```

下一步建议先明确现金分红再投资这一研究目标，再实现独立版本回报序列和样本；不根据2024哪种口径分数更好来决定，也不宣称缺失的事件完整性或首次版本证据已恢复。

## 16. 新版现金再投样本：已经能返回“历史输入和后来答案”

本节是在上一节建议后继续实现的结果，不是把旧样本改名。新规则有自己的样本/特征/口径/标签版本，使用明确现金公式和严格交易日，不使用供应商复权列，不保存、不训练、不发布。

### 16.1 请求与服务

```text
GET http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/cash-reinvestment-preview?fundCode=006730&cutoffDate=2023-06-20
```

Headers仍填已有`X-Service-Token`，不带Origin，Body留空。`fundCode`固定001632、006730、008888；`cutoffDate`是**信息截止日的日末**，不是旧`asOfDate`净值日。只支持2022–2024；即使截止在2024，之后20交易日跨2025也拒绝。不支持分页、批量、任意收益口径、交易日数或includeTest参数。

验收时8000未监听；需要在本仓库启动加载本次代码的服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

实际验证使用临时54479端口且已关闭，不表示你的8000进程已启动或更新。使用现有运行方式也可以，不需新增数据库表、依赖、采集任务或导入文件。

### 16.2 先分清三个日期

以006730、截止2023-06-20为例：

| 日期 | 当前值 | 用途 |
| --- | --- | --- |
| `cutoff_date` | 2023-06-20 | 假设站在这天结束，划定当时能知道什么。 |
| `anchor_nav_date` | 2023-06-19 | 最新已公布的净值日；历史输入截至它，不能使用6月20日尚未公布的净值。 |
| `label_base_date` | 2023-06-20 | 答案从这天的单位净值作分母；这是后来核对用的值，不能进特征。 |
| `label_end_date` | 2023-07-20 | 严格在cutoff之后第20个交易日，缺数也不能顺延。 |

历史取3月21日至6月19日共61个精确交易日；未来为6月21日至7月20日20个交易日。标签序列含基准日，所以是**21个点、20段变化**。若把6月19日作为分母一直算至7月20日，就变成21段，这正是新旧起算语义需要分开的原因。周末/节假日截止，标签基准改用最近交易日，但仍只数截止之后20天。

### 16.3 怎么计算收益和指标

```text
每天收益 = (当日单位净值 + 当日每份现金分红) / 前一交易日单位净值 - 1
研究指数 = 从100开始，每天乘以(1 + 当天收益)
这段累计收益 = 末日研究指数 / 100 - 1
```

现金假设生效当天按当天单位净值买回份额，所以多次分红的影响是连乘，不是把历次现金简单加到终点。没有当时可见事件的日期按现金0只是研究假设，不是证明没有漏分红。每段第一点只是基准，当天分红属于之前的区间，不重复加一次；不模拟费用、到账延迟或个人持有收益。

实际6月21日：`(1.2235 + 0.1676) / 1.4188 - 1 = -0.019523541021`，约－1.9524%。本次20日最终指数`97.630933385748`，答案`-0.023690666143`（约－2.3691%）、方向0。**这些是历史事实在当前快照及固定假设下计算出的答案，不是模型预测。**

输入中的7项`metrics`仍用已有纯数学公式，但在这条新研究指数上计算：

| 指标 | 人话解释 |
| --- | --- |
| `return_5d/return_20d/return_60d` | 历史最近5/20/60个交易日区间的收益。`0.01`就是1%，不是1元。 |
| `volatility_20d` | 最近20次日收益的起伏大小，越大越不稳定；总体标准差，不年化。 |
| `max_drawdown_60d` | 最后60点中从之前高点跌得最深的幅度；负数，－0.05是回撤5%。 |
| `relative_position_60d` | 最新值在最后60点的最高/最低之间处于什么位置，0最低、1最高；不是上涨概率。全段相等则拒收，不伪造位置。 |
| `consecutive_decline_days` | 从末尾向前连续下跌了几个区间，遇到持平或上涨停止。 |

计算内部固定Decimal40位及ROUND_HALF_UP；日收益连乘中途不按输出位数截断。输出点的净值、现金、日收益和指数12位；指标从这些显示指数计算，小数字段8位、连续下跌整数。答案按末日显示指数计算并固定12位，方向依据同一个返回值是否大于0，避免“显示0却判上涨”。

### 16.4 返回字段怎么读

| 字段 | 说明 |
| --- | --- |
| `mode` | `READ_ONLY_CASH_REINVESTMENT_SAMPLE_PREVIEW`，仅新版研究样本预览。 |
| `sample_rule_version` | `CASH_REINVESTMENT_SAMPLE_RULE_V1`，这一版样本选择及隔离规则。 |
| `status` | `RESEARCH_SAMPLE_READY`题和答案可计算；`INPUT_UNAVAILABLE`输入有问题；`LABEL_UNAVAILABLE`输入可用但答案不可用。都不是发布状态。 |
| `calendar` | 固定交易日历版本、内容指纹、来源及未来年度安排当时是否已公告。 |
| `anchor_lag_sessions` | 最新已知净值落后截至日最近交易日几天，最多允许1天。 |
| `history_dates/future_dates` | 必须精确匹配的61个历史日/20个未来日；缺记录不能跳过换下一天。 |
| `input_issues/label_issues` | 输入和答案各自的问题日及原因；答案问题不会回写历史输入。 |
| `feature_payload` | 历史输入包，包含可见日期、来源、日历、研究序列和指标，没有未来答案值。真正训练时只能挑规定指标列，不能把整个响应当输入。 |
| `feature_payload.feature_version/nav_value_basis` | 分别为`CASH_REINVESTMENT_FEATURE_V1`与`CASH_REINVESTED_UNIT_NAV_V1`；不同于旧累计净值样本。 |
| `feature_payload.available_at` | 历史整段所用信息最晚公告日，必须不晚于cutoff。 |
| `feature_payload.source_code/source_sync_run_id` | 同一启用来源及当前净值同步水位；不是分红完整性或历史首次版本证明。 |
| `history_series/label_series`中的`unit_nav` | 真实单位净值；没有读取累计或供应商复权列来替换。 |
| 每个点的`cash_per_share/dividend_event_keys` | 当天实际计入的每份现金及对应来源事件标识；基准日/无可见事件日现金0、标识为空。 |
| 每个点的`daily_return/growth_index` | 当天收益及从100起步的累计研究指数；基准日收益null。 |
| 每个点的`available_at` | 从本段起点到这一点，净值和已使用分红的最晚公告日；是累计可得日期。 |
| `feature_hash/label_hash` | 两份内容分别生成的SHA256指纹；未来答案变化不应改变历史输入指纹。不是加密或历史快照签名。 |
| `offline_label` | 后来答案，独立于输入；无答案时null，不能当成0或“预测下跌”。 |
| `offline_label.label_version` | `CASH_REINVESTMENT_FORWARD_20TD_V1`，新版答案定义。 |
| `horizon_trading_days` | 固定20，是交易日收益区间数，不是自然日或源表行数。 |
| `future_return_20d/label_up_20d` | 后来20日收益、涨跌答案；大于0为1，持平/下跌为0。 |
| `label_available_at` | 21个答案净值点及所用事件均已公告的最早自然日；后续训练分段仍须按此隔离。 |
| `ignored_non_trading_nav_dates` | 查询范围里多出来的非交易日净值，仅用于诊断；不删除，也不进入特征包。 |
| `usage/training_eligible` | `LEARNING_ONLY/false`：可研究复算，不具备正式训练准入，也不自动进入旧训练器。 |
| `historical_versions_verified/dividend_history_complete_verified` | 都是false：没有证明首次公开版本及分红历史完整。 |
| `database_written/model_fitted/test_scored` | 全是false：未保存或改数，未训练，未做2025测试评分。 |
| `publication_status/limitations` | 未发布，以及现金再投、市场日历、时点版本和事件覆盖的具体限制。 |

### 16.5 分红信息怎样避免“事后偷看”

历史阶段要求方案公告日存在，若有实施公告日则取两者较晚者，必须已到cutoff。未来才公告或公告缺失的事件不回填历史输入；如果后来才获知旧分红，旧题特征也不追溯改写。当前“实施”字段没有历史版本，仍保留未核准状态。

只有生效日在当前序列基准日之后、末日以内的事件影响该段。两个除息日期不一致、非交易日、同日多个事件、非实施、现金无效等拒收整段；当时已公告但两种生效日期都没有，无法判断和历史是否无关，历史也拒收。有效日优先净值除权日，缺失才用除息日，不自动顺延。

标签允许使用cutoff之后到2024年底才公告的相关事件，并同步推迟`label_available_at`。如果只有答案资料有问题，历史payload与hash保留不变。缺公告事件不用于历史计算，但不因此获得“没有遗漏”的证明；事件完整性、拆分/折算、源覆盖修订仍未解决。

### 16.6 拒绝方式、实际证据与代码入口

- 无Token、错误Token或带Origin403；非试点、非法日期、2025截止或额外参数422。被拒请求不读库。
- 截止2024-12-20虽然年份允许，但未来窗口进入2025：409 `TEST_PERIOD_PROTECTED`，开数据库前拒绝。来源/日历不满足409，基金不存在404。
- 数据库/数据结构等内部异常503 `CASH_SAMPLE_UNAVAILABLE`并脱敏；数据缺失、异常或不可得通常200带不可用状态与问题清单。2023-12-08跨2024、当时下一年度安排未公告，保留特征、标签不可用。
- 一致性只读事务内按基金/同来源取净值三列和分红七列；净值最多192行、事件100行，LIMIT193/101作哨兵，超量拒绝；无分页、无隐式全表返回。
- 新增80项测试；相关离线526项、真实PostgreSQL隔离19项，共545项通过，Ruff/pip check通过。实际三基金×三个截止日，加006730/2023-06-22共10组可计算；真实HTTP重复结果与直接服务调用一致，鉴权/非法参数/测试期保护和TraceID已核对。
- 最终验收10次直接调用+2次网络成功请求共72条SET/SELECT（12个只读事务、12条净值、12条分红查询）；失败网络请求不查库，无业务写入、无样本表读取。业务三表145/2923/2875、原始净值及旧候选/校准/诊断/日期/口径审计文件指纹未变。
- 本地只读验收JSON：[report.json](C:/pythonProject/workSpace06/.local-runs/nav-cash-sample-d87bc4e4-0e2b-457c-b629-0834dda861f8/report.json)，SHA256为`ed7148d9742ccdb49354e20cce16c18998a461d54786cc8105a5c934d63cb8c3`，Git忽略，不是已存样本批次。

调用链：`app/api/routes/cash_reinvestment_samples.py`处理HTTP与错误，`app/services/cash_reinvestment_samples.py`冻结输入/独立附答案，`app/repositories/cash_reinvestment_samples.py`只读所需资料，`app/schemas/cash_reinvestment_samples.py`有每个字段的中文解释。复用已有日历和7项纯数学函数，不执行旧样本选择或标签规则。

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cash_reinvestment_samples.py -q --tb=short
```

下一步是新版本的有界批量dry-run，核对单日/批量一致及拒收分布；不是再次训练，不能把新响应直接发给旧保存、基线或训练接口。

## 17. 新版批量dry-run：一只基金、一段日期，一次返回全部练习题

### 17.1 直接调用，不用导入数据

```text
GET http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/cash-reinvestment-dry-run?fundCode=006730&startDate=2023-06-01&endDate=2023-06-30&pageSize=7
```

Headers沿用现有`X-Service-Token`，不带Origin，Body留空。服务未启动时仍按第16.1节启动；本次检查8000未监听，临时57807已在验收后关闭，未重启用户进程。

| 参数 | 含义与限制 |
| --- | --- |
| `fundCode` | 一只基金，仅001632、006730、008888。 |
| `startDate/endDate` | 要出题的信息截止日范围，含首尾最多31自然日，均在2022–2024；不是读取原始净值的范围。 |
| `pageSize` | 内部每页处理的题目数，1–30，默认10；不是预测周期，也不是每页原始净值行数。响应仍返回全部页。 |

只对固定日历的交易日出题，周末/休市日列在`skipped_cutoff_dates`；它们不会被算成“缺少样本”。交易日即使缺净值，仍返回问题样本。单日GET支持周末截止，但本批量固定只选交易日，避免同一休市期重复出题。

任何一道题的完整20日答案进入2025，整次在开数据库前409，不只返回前半段，也不靠换页大小绕过保护。例：2024-11-20至2024-12-20参数范围合法，但被未来窗口保护拒绝。2025截止、多余参数、反向/过长日期、非法页大小为422。

### 17.2 怎么验收分页不变、单日与批量一致

上面这段2023年6月真实数据有20道题、10个跳过日，三只基金均有20道输入/答案可计算。以006730为例：

| pageSize | page_count | sample_count | ready_count | batch_hash |
| --- | --- | --- | --- | --- |
| 1 | 20 | 20 | 20 | `454558c6cec14e04b1b93a24ba3abc27f17570799714f181eca8672cb3f4cfb9` |
| 7 | 3 | 20 | 20 | 与上一行完全相同 |
| 30 | 1 | 20 | 20 | 与上一行完全相同 |

操作：先发送原请求，再改页大小为1和30。**只有`page_size/page_count`应不同，其他响应字段包括items和batch_hash应相同。** 然后找`items`中`cutoff_date=2023-06-20`，与下面单日接口的完整返回逐字段比较：

```text
GET http://127.0.0.1:8000/internal/v1/features/historical-nav-samples/cash-reinvestment-preview?fundCode=006730&cutoffDate=2023-06-20
```

该题答案仍是第16节的`-0.023690666143`、方向0，特征/标签指纹也不变。不是拿旧累计净值`preview?asOfDate=...`作对照，新旧定义不同。此一致性以相同源数据、来源水位、日历和规则为前提，源同步更新后指纹可能合理变化。

### 17.3 返回的新字段

每个`items`对象内部仍是第16节的完整单日结构，字段含义不变。外层批量新增：

| 字段 | 人话说明 |
| --- | --- |
| `mode` | 固定`READ_ONLY_CASH_REINVESTMENT_BATCH_DRY_RUN`，已经算了，但没有保存。 |
| `batch_rule_version` | `CASH_REINVESTMENT_BATCH_RULE_V1`，本批量选题/分页/统计规则；逐题仍是原新版单日规则。 |
| `status` | `DRY_RUN_COMPLETED`表示全部题目处理完，不代表每道题都可用；`NO_TRADING_CUTOFFS`表示范围内全是休市日。 |
| `fund_code/start_date/end_date` | 本次基金和请求的信息截止日范围。 |
| `cutoff_selection` | `TRADING_DAYS_ONLY`，按固定日历选题，不按数据库有无记录选题。 |
| `skipped_cutoff_dates` | 请求范围内不出题的周末/休市日；与每条样本的非交易日源记录忽略列表不是一回事。 |
| `source_code/source_sync_run_id` | 本次共同来源及当前同步水位，一次检查、整批共用；不能证明历史首次版本或事件完整。 |
| `calendar_version/calendar_hash` | 同一份固定研究日历及内容指纹。 |
| `page_size/page_count` | 每页题数、实际非空页数；不计入batch_hash，不是SQL数量。 |
| `sample_count` | 总题数，等于items长度；不因题目不可用而减少。 |
| `input_available_count` | 历史输入算得出的题数，有输入不一定有后来答案。 |
| `label_available_count` | 后来答案算得出的题数；不是预测命中次数。 |
| `ready_count` | 输入、答案都可计算的题数，不等于正式训练准入。 |
| `input_unavailable_count` | 历史输入不可用的题数，这些题也不会附答案。 |
| `label_unavailable_count` | 输入可用但答案不可用的题数；与上一项互斥。 |
| `input_issue_counts/label_issue_counts` | 按问题代码统计受影响题数；同题同代码只算一次，多类问题分别计数。详细日期见items里的问题列表。 |
| `items` | 全部样本，按cutoff_date升序；不是只返回当前页。 |
| `batch_hash` | 整批内容校验指纹；不是已保存的batch_id，不能拿它去批次查询接口查数据。包含后来答案，不能当模型历史输入。 |
| `usage/training_eligible` | 固定研究用途、正式训练资格false；批量成功不会解除单日的完整性/历史版本限制。 |
| `database_written/model_fitted/test_scored/publication_status` | 不写库、不训练、不做2025评分、未发布。 |

数量关系：`sample_count = ready_count + input_unavailable_count + label_unavailable_count`；输入可用数是`ready_count + label_unavailable_count`，答案可用数等于`ready_count`。原因统计可能一题多类，不能把所有原因数直接相加当总题数。

### 17.4 空范围、有问题和失败时怎样返回

- 006730、2023-06-17至06-18仅周末：200，`NO_TRADING_CUTOFFS`，0题0页；仍核验基金和来源，但不读净值或分红。
- 006730、2023-12-01至12-08：本次6题全部有输入、1题有答案、5题答案不可用，原因`FUTURE_CALENDAR_NOT_KNOWN_AT_CUTOFF`。说明当时跨年安排尚未公布，不是预测下跌；`DRY_RUN_COMPLETED`仍可成立。
- 缺净值、无公告、分红冲突等按单日规则保留在每道题里，不因批量而放宽。单日与批量都只是假设下可研究计算，来源事件完整性、历史首次版本及拆分/折算仍未核准。
- 无Token/错Token/带Origin403；缺基金404、来源/日历/测试期保护409；范围/页大小/额外参数422。参数或未来保护拒绝后不读业务资料。
- 中途SQL/计算/15秒阶段软预算失败：503 `CASH_BATCH_UNAVAILABLE`，不返回已做好的前半段items，不保存半批，不展示内部连接细节。

### 17.5 实现与真实验收

调用链是`app/api/routes/cash_reinvestment_batch.py` → `app/services/cash_reinvestment_batch.py` → 共享现金仓储；参数/字段注释在`app/schemas/cash_reinvestment_batch.py`。先按日历形成全部计划，然后在一个REPEATABLE READ、READ ONLY事务内检查一次来源、读取一次全批分红、每页一条净值SELECT，最后逐题复用原单日纯构建器。没有复制收益公式，不调用旧累计净值样本/训练流程。

每页只取所需的日期/公告/单位净值，最多192行、LIMIT193超量拒绝；全批合并资料跨度也不足192自然日。分红按全批固定上限100、LIMIT101，不能通过分更小的页改变上限。净值和分红都裁回每道题的单日窗口，较大页的相邻题资料不会混入当前题。15秒是页间/题间软检查，数据库单条仍5秒超时，不承诺请求硬实时截止。

- 新增57项用例；583项相关离线、19项真实PostgreSQL隔离回归，共602项通过，Ruff和pip check通过。既有TestClient弃用警告保留。
- 三基金×1/7/30共9组真实批量对照，与60次真实单日查询逐条相同；006730旧单日完整示例与上轮验收包未发生变化。
- 最终现场验收共75个只读事务，532条SET/SELECT（包含单日对照、批量、空范围和跨年示例），无样本表读取、无DDL/DML；158条净值SELECT只读单位净值值域、74条事件SELECT包含实施公告日期。参数/鉴权/测试期保护失败未产生SQL。
- 实际网络200/403/422/409、TraceID、三种页大小、单日一致和空范围已验收。旧三表仍145/2923/2875，原始净值与旧候选/校准/诊断/日期/口径/单日现金报告指纹未变；2025只做完整性指纹核对，不读取为模型输入或答案。
- 本地验收包：[report.json](C:/pythonProject/workSpace06/.local-runs/nav-cash-batch-23ea0dca-afdf-4e45-81df-260d667ef13b/report.json)，SHA256为`f34b3d73b63d7caaea8d979a1d94f7d4def46c690aaf600a42011cdb884e260b`，已回读、Git忽略；保留对照摘要及一条完整样本，不是保存批次。

复跑本步：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_cash_reinvestment_batch.py tests/test_cash_reinvestment_samples.py -q --tb=short
```

下一步接独立新版样本保存和按批次读回；本次没有建表、保存、重训、发布或提交/推送。

## 18. 新版保存、研究报告与关注页面状态（2026-09-08）

**现在能够把新版练习题存起来，实际做研究训练，再让关注页面读取结果状态。当前模型未发布，页面没有真实上涨概率。** 旧累计净值接口和批次保留，新链路只用第16–17节的现金再投版本，不能混用。

### 18.1 先用一个只读请求看真实结果

启动加载新代码的8000服务，Header填写原有X-Service-Token，不带Origin：

```text
GET http://127.0.0.1:8000/internal/v1/features/cash-reinvestment/research-runs/f70feb1a-129d-4482-b66d-f4e2e3a5425c
```

这个接口**只读已经保存的报告**，不会重新训练。数据库里现在是108批、2118道题，其中1750道输入和答案都能算出；不是预测正确1750次。没有将2025最终测试答案带入研究。

| 返回部分 | 通俗含义 |
| --- | --- |
| run_id / created_at | 这次研究的编号和保存时刻；不是预测的日期。 |
| database_written=true | 这份研究已经入库，不表示本次GET又写了一遍。 |
| report.preparation | 本次用了哪些批次、七列指标、资料指纹、可用题数、各基金时间段数量及证据缺口。 |
| report.windows | 三轮时间隔离考试，各自包含基础训练、校准、考试数量、模型和对照成绩。 |
| status=PARTIAL_EVALUATION | 有窗口做完，但没有全部做完；本次前两个DEV窗口数量不足，2024验证做完。 |
| model_fitted=true | 至少一轮真的训练和校准过，不是占位响应。 |
| release_gate=BLOCKED | 不允许发布，当前协议不能通过请求参数改成通过。 |
| publication_status=MODEL_NOT_RELEASED | 不向产品提供模型概率。与“已经训练过”不矛盾。 |
| report_hash | 整份报告的内容校验码，防止模型和成绩错配。 |

本次2024基础拟合778条、校准272条、考试625条，方向准确率52.64%，Brier误差0.35242478。简单训练期涨跌频率对照的Brier是0.25321469，越小越好；当前模型没有证明比简单对照可靠。历史首次版本、分红完整性、拆分/折算及最终独立测试证据仍不足。

### 18.2 保存一小批，再按编号读取

`POST /internal/v1/features/cash-reinvestment/batches`，JSON字段：

| 请求字段 | 含义和限制 |
| --- | --- |
| fundCode | 三试点之一：001632、006730、008888。 |
| startDate / endDate | 信息截止日范围，首尾包含，2022–2024，最多31自然日；未来20交易日答案不得跨入2025。 |
| pageSize | 内部每页处理题数，1–30，不决定最终只返回几条。 |
| requestKey | 本次保存操作新生成的UUID；网络重试沿用，想重新生成才换一个。 |

返回`batch_id`后，用`GET /internal/v1/features/cash-reinvestment/batches/{batch_id}`读取。完整计算内容放在`preview`中，跟第17节逐题字段含义相同。

外层`database_written=true`表示“资料已保存”；内层`preview.database_written=false`保留原纯计算结果的语义，表示“这一步计算本身没有写库”。不是自相矛盾，也不是GET产生了写操作。内层hash是内容指纹，外层batch_id才是保存编号。

同requestKey、同基金日期会返回第一次保存的快照，换pageSize也不重算；同key改基金或日期409。整批事务保存，出错不留半批。重复保存不是覆盖旧批次。源数据后来变化不会改已保存快照；读回发现损坏503，不能自动修补。

### 18.3 资料准备和固定研究

1. `POST /internal/v1/features/cash-reinvestment/preparation`，Body只含`batchIds`数组，1–128个不同批次UUID。它只检查和组织资料，返回`dataset_hash`，不写库或训练。
2. `POST /internal/v1/features/cash-reinvestment/research-runs`，Body含同样的`batchIds`、刚才的`expectedDatasetHash`、本次研究的`requestKey`。它实际运行固定方案并保存；重试返回已有研究，不重复训练。
3. 用第一节GET按run_id读回完整报告。

X是七列历史指标组成的“输入表”；y是后来涨没涨的“答案列”。基金代码、日期、后来答案不会混进X。时间不随机打散；答案跨出当前训练/校准/考试段的题目要剔除，避免提前知道后来的结果。

`RESEARCH_READY`只说明总研究数量够，不代表每个小时间窗都够，更不代表`training_eligible=true`。当前正式训练准入固定false，但允许带证据缺口说明的研究拟合。新代码没有开放降低数量、传入自定义答案、查看2025考试或强制发布参数。

本轮月批次生成脚本`python -m scripts.cash_reinvestment_pilot`默认只显示计划；`--execute`才会写本机fund_ai。它限定既定三基金和2022-01-01至2024-12-03，以固定幂等键重跑不会新增第二套。正常验收看上面已有GET即可，不必再次补存或重训。

### 18.4 页面读的不是整份研究报告

Python内部：`GET /internal/v1/predictions/008888`。Java对用户：`GET /api/v1/watchlist/008888/prediction`。

浏览器只能访问Java。Java先验证登录、权限和本人是否关注，再向Python取最小状态。Python服务Token、模型系数、考试答案和批次内容不会交给前端。非法基金参数400，未登录401，未关注403；响应禁止共享缓存。上游失败时Java降级为可读的UNAVAILABLE，不泄露连接信息。

Python字段使用下划线，Java页面字段使用驼峰，例如`up_probability → upProbability`：

| 字段 | 含义 |
| --- | --- |
| status | MODEL_NOT_RELEASED未发布、DATA_INSUFFICIENT资料不足、NOT_APPLICABLE不适用、UNAVAILABLE暂时无法读取。 |
| horizon_trading_days | 固定20交易日整体方向，不是逐天预测20次。 |
| up_probability / direction | 当前只能null，不能填0或50代替未知，也不能把研究分数显示成已发布概率。 |
| latest_nav_date | 本地当前来源最新已公告净值日，不是预测截止日；只读日期元数据。 |
| research_run_id / research_evaluated_at | 支撑状态的研究编号和保存时间，不是今日预测生成时间。 |
| model_version | 研究协议版本，不等于已发布模型。 |
| reason_codes / reasons / message | 稳定原因代码、中文原因、主提示。 |
| disclaimer | 研究用途说明；未发布不表示会下跌。 |

真实三端及浏览器验收见[实施手册第21节](C:/WebStormProject/workSpace05/docs_zhx/implementation/free-data-prediction-v1.md:644)和[TC-FDP-23](C:/WebStormProject/workSpace05/docs_zhx/testcase/free-data-prediction-v1.md)。639项Python离线、27项真实隔离PG、18项Java测试通过；Vue检查/构建及真实页面权限、故障恢复、非股票型、窄屏与键盘验收通过。测试服务和合成账户schema已清理，真实账户/关注未变。

正式发布、2025独立测试及在线实时特征/预测生成仍未启用。后续必须先补证据、按预先约定的条件验证效果，再实现发布后的在线能力；不能仅修改status字段上线。

## 19. 检查一份研究是否允许生成预测（2026-09-09）

```text
POST http://127.0.0.1:8000/internal/v1/predictions/generation-check
Header: X-Service-Token: 当前内部服务令牌
Content-Type: application/json
```

```json
{
  "fundCode": "008888",
  "researchRunId": "f70feb1a-129d-4482-b66d-f4e2e3a5425c",
  "expectedReportHash": "db527ccca8015a4f41af2ee68608dae27ec5ab86c2627ef158d39f2f6b067795"
}
```

**这不是生成预测的接口，而是生成前的只读检查。** 指定已有报告和指纹，避免不小心检查了另一版。它不重训、不读原始净值数值、不打开2025、不保存，也不提供force之类绕过参数。

返回 `GENERATION_BLOCKED` 表示不允许进入预测计算；`inference_executed=false` 表示没有执行模型，`forecast_created=false` 表示没有创建产品预测。上涨概率和方向均为null。

`comparisons` 是已有考试成绩的对照，不是当前基金的未来概率：按窗口、整体ALL及每只基金各列四个对照，`accuracy_delta`是模型正确率减对照（正数较好），`brier_delta`是模型误差减对照（负数较好）。`strict_gain_observed=true`只说明该组两项同时改善；所有组改善也不能代替完整历史版本证据或最终独立测试。缺窗列入`incomplete_window_ids`，不补一个假成绩。

参数/额外字段422；认证或Origin403；基金/指定研究不存在404；来源、试点范围或指纹不匹配409；数据库不可用或报告损坏503脱敏。只有实际检查完成才返回200的检查结果，数据库连不上不能伪装成“正常未发布”。

新增20项测试通过，相关离线总计659项通过。完整目标及未实现的数值生成/正式发布链路见[实施第22节](C:/WebStormProject/workSpace05/docs_zhx/implementation/free-data-prediction-v1.md)。现场验证不能由人工测试成绩替代，详见TC-FDP-24。

## 20. 历史输入计算器不是一个新的HTTP操作（2026-09-09）

本轮新增服务内`app.services.cash_prediction_features.read_cash_prediction_feature`，为后续实际预测准备已知历史资料。你不需要在接口工具里再加一个操作：现有HTTP契约和页面行为不变。

它与练习题构建器共用历史选窗和现金再投资指标，只看cutoff以前已公布的信息，不读取或生成未来答案；未公告价格在SQL层屏蔽。支持旧研究年份和独立核验的2026日历，仍禁止2025数值读取。日历不足、当日未结束、原始资料缺失均不猜数。

`INPUT_READY`只表示历史输入可以算出，不表示已有合格模型，更不表示预测上涨。计算器不训练、不保存、不返回概率。完整契约和2026官方公告来源见[实施第23节](C:/WebStormProject/workSpace05/docs_zhx/implementation/free-data-prediction-v1.md)。该阶段待验的真实SQL已在Docker恢复后补验，见下节，不能倒写成此前已取得实库证据。

## 21. Docker恢复后的实际返回与页面复验（2026-09-09）

29项隔离PostgreSQL测试通过。第19节generation-check已用真实HTTP和冻结报告复验：三基金都正常200，GENERATION_BLOCKED、16组对照、6项阻止原因；没有执行模型或保存预测。报告指纹仍与第19节示例一致，force参数422、错误指纹409。

第20节历史输入实际读库验证：三基金在09-04/09-05历史cutoff都能得到61点输入；以09-08为cutoff时，最新已知净值仍为09-04、落后2个交易日，因此DATA_INSUFFICIENT，不生成输入指纹。不能把旧截止日的成功当作今日数据已就绪。

Java到Python、本人关注权限、4种状态页面、公开页不展示、故障恢复、窄屏和键盘均已复验。临时服务和测试账户已清理，真实账号/关注及原净值前后指纹不变。最新实际证据见[实施第24节](C:/WebStormProject/workSpace05/docs_zhx/implementation/free-data-prediction-v1.md)和[TC-FDP-26](C:/WebStormProject/workSpace05/docs_zhx/testcase/free-data-prediction-v1.md)。这是状态链路验收，不是正式上涨概率发布或全部数值生成能力已完成。

## 22. 保存一次生成拒绝回执（2026-09-09）

`generation-check`仍然只是只读体检；现在另有一个会保存**拒绝回执**的内部接口：

```http
POST /internal/v1/predictions/generation-attempts
X-Service-Token: <本机配置的服务Token，不要写入代码或提交>
Content-Type: application/json
```

```json
{
  "requestKey": "4e7be4bf-c382-4e48-96b1-8c0e06bbd10e",
  "fundCode": "008888",
  "cutoffDate": "2026-09-04",
  "researchRunId": "f70feb1a-129d-4482-b66d-f4e2e3a5425c",
  "expectedReportHash": "db527ccca8015a4f41af2ee68608dae27ec5ab86c2627ef158d39f2f6b067795"
}
```

这里选择9月4日是为了复验一个已经结束的历史截止日，不表示今天的资料已经足够。`requestKey`是你给这次尝试起的唯一编号：相同请求重试保持不变，要重新检查则换一个新UUID。首次201，重试200；同编号换参数409，不覆盖旧回执。

返回里的关键字段：

| 字段 | 通俗含义 |
| --- | --- |
| attempt_id | 回执编号，可用下方GET读回 |
| cutoff_date | 当时计划使用哪一天的信息；被拒绝后没有继续取输入 |
| check | 当时实际执行的检查及比较证据，含报告编号/指纹和6条拒绝原因 |
| receipt_hash | 回执内容的“校验码”，读回时检查有没有错配或损坏 |
| historical_receipt=true | 这是历史回执，不是重新检查后的当前结论 |
| database_written=true | 回执已经保存，不是预测已经完成 |
| forecast_created=false | 没有生成产品预测，概率和方向仍为空 |

内层`check.database_written=false`说的是检查步骤自己不写库；外层true说的是把检查结果保存成回执。不能只看外层true就理解成模型已经作答。

```http
GET /internal/v1/predictions/generation-attempts/bc5a1092-e131-4379-a14a-4bc315a0bdd2
X-Service-Token: <本机配置的服务Token，不要写入代码或提交>
```

这个已验证的编号属于008888，读回会得到2026-09-09保存的拒绝回执。读回不重新训练、检查或计算。当前服务禁止force、上传模型/指标和2025截止日；未结束的截止日409，无权限403，记录不存在404，异常503脱敏。

模型纯数值内核也已经实现并用人工资料核对公式，但没有向HTTP或页面开放，不能越过闸门先算分数。本轮真实16项HTTP与41项隔离PG通过；临时验收服务已停止，若你的8000未启用自动重载，需要按原项目方式加载新代码后再调用。完整复跑命令与剩余边界见[实施第25节](C:/WebStormProject/workSpace05/docs_zhx/implementation/free-data-prediction-v1.md)和TC-FDP-27。

## 23. 受正式授权保护的结果生成与读取（2026-09-09，迁移与拒绝路径已实测）

这两个新接口用于“真正获准后生成结果”和“查询结果现在能否展示”，不是训练入口。**本机业务库已执行`alembic upgrade 20260909_16`，并使用临时服务完成真实调用。** 其他环境需先执行迁移、加载新服务；迁移只增加独立结果表，保留旧资料。用户常驻服务本轮未重启。

```http
POST /internal/v1/predictions/cash-forecasts
X-Service-Token: <本机配置的服务Token，不要提交>
Content-Type: application/json
```

Body沿用第22节的`requestKey/fundCode/cutoffDate/researchRunId/expectedReportHash`，另须提供`expectedModelHash`：从明确指定研究的固定`VALIDATION_2024`窗口模型取得64位`model_hash`，不是报告指纹，也不能自行填写任意模型。只支持001632、006730、008888，不接受模型参数、输入X、客户端授权或force。

当前真实模型仍未获正式授权，因此应返回200、`status=MODEL_NOT_RELEASED`、非空`reason_codes`、`created=false`，`forecast_id/up_probability/direction`为空；不执行数值计算，不保存结果。200仅代表请求被正常处理，不能理解成预测成功。正式证据驱动的授权签发仍未实现。

将来只有真正获准且资料齐全才会首次201并得到`forecast_id`；相同key重试200，同key改参数409，相同业务结果换key重复生成409。GET `/internal/v1/predictions/cash-forecasts/{forecastId}`查询已保存结果；不存在404。它每次重新核验当前授权和数据时效，但不重新训练或计算：

| status | 页面含义 | 数字 |
| --- | --- | --- |
| AVAILABLE | 授权有效、数据及预测期限仍有效 | 才可返回0至1的上涨概率 |
| MODEL_NOT_RELEASED | 没有正式授权，或授权已更换/撤销 | 概率和方向为空 |
| DATA_INSUFFICIENT | 本次生成输入或来源未就绪 | 不生成结果 |
| STALE | 已保存结果因资料更新、过期等原因不可再展示 | 概率和方向为空 |

`target_base_date/target_end_date`描述原本20交易日区间；不因查询日期变化而延期。分红同步即使没改净值同步编号也会使旧来源快照失效，同步中或最近失败同样不能展示旧概率。

相关离线790项、隔离PG68项通过；成功计算/存储分支仅用人工授权和人工资料验证。真实TCP26项通过，三基金均未发布、不推理、不写结果，原数据摘要不变；正式授权签发尚未实现。最新边界见[实施第27节](C:/WebStormProject/workSpace05/docs_zhx/implementation/free-data-prediction-v1.md)及TC-FDP-29。

## 24. 页面读取已保存结果与当前状态（2026-09-09）

Java仍通过`GET /internal/v1/predictions/{fundCode}`读取本人关注卡片资料。该接口增加AVAILABLE/STALE及`forecast_id/cutoff_date/target_base_date/target_end_date/generated_at/model_hash`，但不会在GET时生成结果。没有记录沿用研究状态；已有最新记录则重新检查授权、来源更新与时效，不能在最新记录损坏或失效后选旧数字替代。

只有完整且仍有效的AVAILABLE才有`up_probability/direction`。其他状态必须为空；返回不含模型参数、完整历史输入、未来答案和授权凭证。Java继续先验证本人关注，浏览器不能直连内部接口或传force。

本轮完成真实未发布状态及明确标注的人工结果投影三端验收。人工页面样例仅验证展示，不写真实预测表；新增结果表当前0条，真实模型仍未获发布资格。查看[实施第27节](C:/WebStormProject/workSpace05/docs_zhx/implementation/free-data-prediction-v1.md)区分隔离数据库计算、人工HTTP投影及真实研究三种证据。

## 25. 发布规则审查：把“哪里没过”逐项列出来（2026-09-09）

这个接口是**给已有研究报告检查成绩和证据**，不是再训练一次，也不生成未来涨跌预测。与generation-check相比，它增加逐条数量、概率分组和规则缺口；单项通过不等于模型可以发布。两入口共用报告数学一致性校验，但generation-check不会启用未批准的草案阈值。

在加载了本次代码的Python服务上发送`POST http://127.0.0.1:8000/internal/v1/predictions/release-review`，沿用已有内部服务Token请求头（不要把实际Token写进文档或代码）。Body选JSON：

```json
{
  "fundCode": "008888",
  "researchRunId": "f70feb1a-129d-4482-b66d-f4e2e3a5425c",
  "expectedReportHash": "db527ccca8015a4f41af2ee68608dae27ec5ab86c2627ef158d39f2f6b067795"
}
```

请求指向008888，但审查不会只挑这一只基金的好成绩：仍完整核验原报告的三试点、三个固定窗口、四种简单对照。因此切换为另外两只试点时，整份规则检查明细相同是正常的。

| 返回字段 | 通俗含义 |
| --- | --- |
| mode / status | 只读规则审查 / BLOCKED。HTTP 200只表示检查正常完成，不表示预测成功。 |
| checked_at | 此次检查时刻；重复请求会变化，不是新的训练时间。 |
| policy.approval_state | 当前DRAFT，规则草案尚未批准；即使以后APPROVED，也不能替代持久冻结和其他证据。 |
| policy_hash | 此次服务器规则内容的指纹，用于核对用的是哪套规则，不是发布凭证。 |
| check_counts | PASS、FAIL、MISSING各有多少项，是检查项数量，不能算成模型准确率或发布通过率。 |
| checks[].window_id / scope | 哪轮考试、总体ALL还是哪只基金。 |
| checks[].code / baseline_id / bin_index | 检查什么、跟哪条简单规则比较、固定五档中的哪档（0至4）；不相关字段为null。 |
| checks[].status | PASS=符合这一项；FAIL=有实际数字但没达到；MISSING=没有足够证据，不能下结论。 |
| checks[].operator | GE=至少，LE=至多，GT=严格大于，LT=严格小于，PRESENT=须有证据，NONDECREASING=顺序不得降低。 |
| checks[].actual / required | 实际数字/规则边界。小数通常按字符串保留精度；缺证据时actual是null，不拿0代替。 |
| blocking_codes | 仍阻止正式发布的原因，不因某几项PASS而清空。 |
| policy_persisted / publication_allowed | 当前均false；前者仅在读到与服务器已批准规则匹配的数据库快照时为true，后者仍为false，保存规则不是批准模型。 |
| policy_freeze_id / policy_frozen_at / policy_freeze_hash / policy_binding_hash | 匹配的规则快照编号、保存时间及内容/比较约定指纹；未冻结时均为null。 |
| inference_executed / independent_test_read / database_written | 均false：没有计算产品预测，没有读2025测试数值，没有写业务库。 |

例如`ANNUAL_ECE`检查的不是“答对多少题”，而是平均报出的分数与实际上涨比例有多大差距。`actual="0.12"、required="0.10"、operator="LE"`表示误差12个百分点，大于候选上限10个百分点，应为FAIL。`EXAM_COVERAGE`缺少事前应考日历计划时，actual为null、required为0.80、状态MISSING；绝不能用现有可用题数自己除自己凑成100%。

当前候选规则：年度至少两个非空档、每个非空档至少30条、加权平均误差最多10个百分点、任一档误差最多15个百分点、实际上涨比例不逆序。季度短窗口仍使用原有每基金40题及对照比较门槛，不硬套完整年度分档门槛。这些是**待确认的项目规则，不是已批准的金融标准或可靠收益保证**。

### 本次实测与复跑

2026-09-09，本机真实冻结报告三只基金均返回118项检查：PASS=50、FAIL=47、MISSING=21，状态BLOCKED；草案指纹为`4b2610983b03076b3a19e90d7b570f447434a8f365c9694a71a382333f965dad`。重复请求除检查时间外一致。

```powershell
Set-Location C:\pythonProject\workSpace06
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe scripts/cash_release_review_acceptance.py --research-run-id f70feb1a-129d-4482-b66d-f4e2e3a5425c --expected-report-hash db527ccca8015a4f41af2ee68608dae27ec5ab86c2627ef158d39f2f6b067795
```

脚本只允许本机fund_ai，使用自己占有的随机回环端口，不占用或重启用户8000服务。21项HTTP/完整性检查通过，包含三只基金原generation-check回归；捕获55条SET/SELECT，未查询净值数值、样本/标签或预测表；鉴权/额外参数失败没有查库。缺报告404、指纹冲突409、无/错Token或浏览器Origin403、额外policy/coverage/force/includeTest参数422；损坏报告或规则文件503且脱敏，不回退到宽松规则。

本轮相关离线875项、八个隔离PostgreSQL测试文件71项通过；新增85项离线和3项PG规则审查测试。包括新旧入口一致拒绝损坏报告，以及用原分档重算ECE比较门槛，防止略超限值被8位小数舍入成达标。PG包含故意在隔离只读事务中尝试UPDATE，被数据库拒绝且原报告不变。原研究内容MD5仍`93549dcd6522737534ed5d9c7a0a3920`，业务计数108批/2118题/1750答案/1研究/3拒绝回执不变，现金预测、旧预测和旧发布表仍各0条。

这次没有冻结规则、签发发布授权或重跑模型，没有改Java/Vue运行代码。此前浏览器验收仍是第24节记录的那次；本节只新增内部只读审查验证，不能冒称正式发布成功。

## 26. 保存已确认的规则：给“考试标准”留一份不能改写的底稿（2026-09-09）

这一步保存的是规则，不是训练模型，也不是发布预测。先确认标准，再固定当时的内容，以后才能追溯一次评估到底用了哪套标准。**当前真实规则仍为DRAFT，不能保存为已确认快照**；本机迁移17已落地，新表`cash_policy_freeze`为0条。

以下路径均以`http://127.0.0.1:8000/internal/v1/predictions`开头，沿用内部服务Token，不允许浏览器Origin。须先让服务加载本次代码；提交或推送不会自动重启常驻服务。

1. `GET /release-policy`：查看服务器当前规则、比较约定及各自的指纹。`approval_ready=false`表示尚未确认；即使true也不表示已保存或模型合格。此接口不查库。
2. `POST /release-policy/freezes`：只允许提交下面三个字段，两个指纹须从上一接口取得；示例中的文字必须替换为实际64位指纹。调用者不能在请求里提供APPROVED或规则正文。

```json
{
  "requestKey": "3b4245c1-d12d-4fd1-9111-e005f3ae42de",
  "expectedPolicyHash": "替换为GET返回的policy_hash",
  "expectedBindingHash": "替换为GET返回的binding_hash"
}
```

当前用合法指纹请求，预期409、错误码`CASH_POLICY_APPROVAL_REQUIRED`，不访问数据库。只有业务真实确认并在服务器配置中留下确认依据后，才允许保存；不要为了试通接口自行改成APPROVED。

3. 获准后的首次POST返回201，`created/database_written=true`，表示只新增一条规则快照；同requestKey重试返回200且两个标志false。同版本换请求号或换内容返回409，不覆盖旧记录。
4. `GET /release-policy/freezes/{freeze_id}`：读回当时快照；不存在404、内容损坏503。服务器当前配置改变也不会重写历史。`release-review`只有匹配到当前规则快照才返回`policy_persisted=true`，但其余数据、测试和发布门槛继续保留。

`freeze_id`是规则记录号，不是模型授权号；`frozen_at`来自数据库时钟。`content_hash`用于核对完整内容，`binding_hash`核对窗口/口径/分档约定，都不是审批签名。数据库阻止UPDATE、DELETE、TRUNCATE，非空表拒绝降级删除。

成功保存、重试、并发和不可变性使用隔离PostgreSQL人工确认资料验证；真实规则没有获批或被冻结。最终独立测试的执行协议、数据证据和模型发布管理仍未完成，这份快照不会解封2025答案。最新验证记录见实施第29节及TC-FDP-31。

## 27. 考试资料准备：先确定应有多少题，再看实际有多少（2026-09-09）

先调用`GET /internal/v1/predictions/release-policy`，现在`binding.version`为`CASH_RELEASE_RULE_BINDING_V2`，`binding.exam_plan`内有日期清单和`plan_hash`。V1历史快照仍能原样读回，但不能当作V2匹配；已冻结记录不会被覆盖。真实规则仍DRAFT。

日期计划不看数据库资料好坏。每个窗口先列全部交易日，再排除未来20交易日终点必定越界的尾部20日；前三窗应考数依次44、40、222。迟公告、缺输入、少传批次都不能进一步缩小分母。2025计划223个日期，仅来自静态日历，不表示已经读取测试数据。

然后发送`POST http://127.0.0.1:8000/internal/v1/features/cash-reinvestment/exam-preparation`，使用已有内部服务Token，Body选JSON：

```json
{
  "batchIds": ["替换为preparation中明确选择的现金批次UUID"],
  "expectedDatasetHash": "替换为原preparation返回的dataset_hash",
  "expectedPlanHash": "替换为GET返回的binding.exam_plan.plan_hash"
}
```

示例文字必须替换为实际UUID/64位指纹，不能直接发送占位符。最多128个不重复批次，不接受日期范围、模型、coverage、force或includeTest。服务需加载本次代码；本轮没有主动重启常驻8000服务。

| 返回字段 | 通俗含义 |
| --- | --- |
| plan.windows[].planned_cutoffs | 不看数据好坏预先列出的应考日期，所有三基金相同。 |
| plan.windows[].boundary_purged_cutoffs | 答案终点一定越过考试末日的20个尾部日期，不因成绩变更。 |
| preparation | 原资料准备结果与指纹，规则和批次未被改写。 |
| coverage[].planned_count | 本窗口本基金理论上应有多少道题。 |
| coverage[].usable_count | 所选批次中，输入完整且答案在窗口内可得的题数；不是答对多少题，也没有调用模型。 |
| coverage[].coverage | 可用题数除以计划题数，是资料覆盖率，不是准确率。 |
| coverage[].missing_cutoffs | 哪些应考日尚无可用题；可能未选到批次、输入/答案不可用或公告太晚，此处不臆测具体来源原因。 |
| coverage[].late_label_count | 缺口中已有研究答案、但公布日超过本窗口末日的数量；不是所有缺口都由晚公告造成。 |
| TEST_PERIOD_PROTECTED | 2025只列计划；usable_count、coverage、missing_cutoffs、late_label_count均null，未检查不能报0或空列表。 |
| ex_ante_evidence | 固定false；今天的预览不是旧研究训练前曾冻结计划的证明。 |
| model_fitted / independent_test_read / database_written / publication_allowed | 均false：没有训练、读取2025答案、保存结果或批准发布。 |

实测原108批：2023第三季度三基金为30/44、31/44、31/44；第四季度均40/40；2024为208/222、209/222、208/222。2024资料覆盖约94%不等于预测正确率94%，也不会解除旧报告的发布限制。

同一只读事务每8批先检查批次和标签日期，再读取允许范围的输入/答案正文。无/错Token、Origin403；非法或额外参数422；计划/资料指纹冲突409；缺批次404；保护期409；损坏日期或正文503并脱敏。计划冲突不查库，不能靠客户端提供一个覆盖率绕过。

本轮937项相关离线、93项隔离PG和17项真实TCP/完整性检查通过；真实报告仍保留九项覆盖证据MISSING。日期计划须在后续新研究训练开始前真实绑定已确认冻结快照；尚不能给旧报告事后补盖。复跑命令及验证范围见TC-FDP-32。

## 28. 计划先行的新研究：不允许给旧成绩补手续（2026-09-09）

新增`POST http://127.0.0.1:8000/internal/v1/features/cash-reinvestment/planned-research-runs`。它与原research-runs不同：先核验已经确认并存下来的规则/日期计划，再读取资料和研究；本次结果与绑定回执一起保存。原接口及旧报告保留，不能把旧报告编号传给新接口补绑。

沿用内部服务Token，Body字段如下，示例文字需换成实际UUID和64位指纹：

```json
{
  "requestKey": "替换为本次新研究请求UUID",
  "batchIds": ["替换为明确的现金批次UUID"],
  "expectedDatasetHash": "替换为preparation返回的dataset_hash",
  "policyFreezeId": "替换为已确认且保存的规则freeze_id",
  "expectedPolicyFreezeHash": "替换为规则快照的完整content_hash"
}
```

**当前真实规则是DRAFT，没有真实可用的freeze_id。** 合法格式的请求会在连接数据库前409拒绝，代码`CASH_POLICY_APPROVAL_REQUIRED`；不要为了试通接口自行把规则改成APPROVED。本轮成功分支仅在隔离人工资料中验证。未来确认规则并正常冻结后，才能使用实际快照；不能传客户端审批标记、模型、旧研究或计算时间。

| 返回字段 | 通俗含义 |
| --- | --- |
| binding_id / request_key | 本次绑定记录号/安全重试号；不是模型发布号。 |
| evaluation_started_at | 核验计划和资料后、进入研究评估函数前从数据库取得的时间。资料不足可能不实际拟合模型。 |
| completed_at | 评估结束、准备保存结果时从数据库取得的时间。 |
| policy_freeze | 计算前已真实保存的已确认规则及日期计划；此操作不创建规则。 |
| preparation | 计算前独立复制的资料覆盖底稿，不会因算法意外修改元数据而被改写。 |
| research | 本流程新建的研究报告。内部研究请求号由服务器生成，与binding_id相同，不会命中调用者的旧研究请求。 |
| plan_bound_before_evaluation | true只说明服务器按计划先行的流程执行，不是数据已准入、已通过考试或一定训练过。 |
| created / database_written | 首次201时均true，表示研究与关联一起新增；已完成重试和GET均false。嵌套research.database_written沿用“报告已保存”的历史含义。 |
| publication_allowed / independent_test_read | 始终false，不批准模型，不读取2025答案。 |
| content_hash | 本次关联、时刻及资料/报告指纹的完整内容指纹，不是审批签名。 |

GET同路径加`/{binding_id}`只读保存内容；不查样本、不重新训练、不依赖今天的审批配置。规则快照先保存，**绑定回执是在计算结束后与研究一起保存**，不能把回执时间说成训练前持久化。回读还核验冻结/开始/完成时间顺序、明确模型关联、资料底稿和各窗口数量。

已完成请求重试200不重新训练；改参数409，缺记录404，损坏关联503且脱敏；当前进程训练忙429。跨进程并发可能重复计算，但唯一约束和事务保证最多一份完整绑定结果。单次准备/研究仍有既有阶段预算，真实长请求需给客户端足够等待时间，不要超时后换requestKey反复开新研究。

本机迁移18已新增`cash_planned_research_binding`且当前0条；所有字段有注释，数据库拒绝更改、删除、清空及非空降级。970项相关离线、111项隔离PG、28项真实TCP/完整性检查通过，真实规则仍草案、旧报告未变。尚需将绑定证据接入发布审查，补齐数据及行情证据、独立考试和正式发布凭证；本轮不解除旧报告的任何发布门槛。复跑见TC-FDP-33。

## 29. 发布审查开始使用研究绑定（2026-09-09）

还是原来的`POST http://127.0.0.1:8000/internal/v1/predictions/release-review`，Body仍只传`fundCode`、`researchRunId`、`expectedReportHash`，内部Token要求不变。**不用额外传绑定编号、计划或覆盖率**：系统在数据库中按本次研究编号查找并核验，自己填“已通过”的参数会被422拒绝。

新返回字段解释：

| 字段 | 通俗含义 |
| --- | --- |
| ex_ante_plan_verified | 是否已从真实存储记录核验本次研究先选定了当前冻结计划。true不是模型合格，也不是一定完成了所有窗口的考试。 |
| planned_research_binding_id | 核验的绑定记录号，可用上一节GET追溯；未核验或旧报告为null。 |
| planned_research_binding_hash | 绑定内容指纹，防止拿另一份记录混用；不是审批签名。 |
| exam_plan_hash | 该研究事前选定的固定日期计划指纹，不按最后能用多少资料缩小计划。 |
| exam_coverage_evidence | 只列已完成评估的窗口，每基金一条实际评分覆盖明细。尚未完成的窗口不列入，全部未完成时为空列表。 |
| 明细中的 planned_count / scored_count | 计划应考题数/模型实际已评分题数。EXAM_COVERAGE.actual等于后者除以前者，不是预测准确率。 |
| 明细中的 plan_version / plan_hash / dataset_hash | 这项覆盖所用的计划版本、计划指纹、研究资料指纹，用于追溯而不是涨跌判断。 |
| 明细中的 window_id / fund_code | 在哪段固定时间、对哪只基金检查；不拿其他基金的数量补足。 |

有完整绑定且窗口完成评分时，覆盖达到规则为PASS、不足为FAIL；缺绑定或尚未评分时为MISSING且actual=null。例如准备好了40道题，但模型因训练资料不足没能考试，这里不能填“评分40道、覆盖100%”。

当前真实规则仍DRAFT，接口不采信历史绑定，实际返回`ex_ante_plan_verified=false`、三个引用null、明细空列表，旧报告九项EXAM_COVERAGE仍MISSING。当前APPROVED配置若与冻结内容冲突409；绑定损坏503，不把坏记录当成“没有记录”继续检查。没有修改审批文件的HTTP捷径。

所有审查仍为只读、不训练、不执行预测、不读取2025答案、不签发模型。计划覆盖合格也不会解除历史数据、行情证据、最终测试或其他成绩门槛。988项相关离线、121项隔离PG、32项真实TCP/完整性检查通过；成功采信绑定仅在隔离人工资料中验证，真实模型仍未发布。复跑见TC-FDP-34。

## 30. 检查本地源值观察记录（2026-09-09）

POST `http://127.0.0.1:8000/internal/v1/features/cash-reinvestment/source-observation-check`，沿用内部服务Token，Body示例：

```json
{
  "fundCode": "008888",
  "startDate": "2022-01-01",
  "endDate": "2024-12-31",
  "cutoffDate": "2024-12-31"
}
```

这是数据准入的**基础诊断**，不是批准入口。startDate/endDate是要检查的源业务日期范围，cutoffDate是要核对的研究截点；开始≤结束≤截点，研究日期不进入2025。只按基金和启用来源聚合观察日志的计数/时间，不取前后净值、分红金额、答案或模型。

| 返回字段 | 通俗含义 |
| --- | --- |
| status | NO_LOCAL_RECORDS：区间内没有未过期留档；ONLY_AFTER_CUTOFF：有留档但都晚于研究截点；LOCAL_WRITE_RECORDS_ONLY：有截点前本地写入记录，仍不代表首次公开或正式准入。 |
| active_record_count | 区间内未过保留期的本地变更日志条数，不是净值天数或事件完整数。 |
| recorded_before_cutoff_count | 其中数据库写入时间早于截点次日上海零点的条数，不代表那时已经提交可见。 |
| expired_record_count | 已过保留期的日志条数，不能作为有效证据。 |
| first_observed_at | 未过期日志中最早的实际数据库写入时间；无记录为null，不填写旧净值业务日。 |
| historical_first_versions_verified / event_completeness_verified | 均false。本地日志不是供应商首次公开档案，零分红记录也不等于没有分红或拆分。 |
| training_eligible / publication_allowed / database_written / source_payload_read | 均false；此接口不放行、不写库、不读取源值正文。 |

新增表`cash_source_observation`由三试点源行变化触发器写入，记录本次现金相关旧值/新值。只记录安装之后的变化，不复制旧库；当前真实0条，三只基金都返回NO_LOCAL_RECORDS。迁移19已在本机执行，但常驻服务未主动重启，新路由需要运行包含本轮代码的服务；本轮真实HTTP验收使用临时本机服务。

观察时间不是首次公告或事务提交时间；日志不覆盖管理员TRUNCATE/禁用触发器/改DDL，不能声称事件完整性。无可归属日期的事件不会被计入任意日期区间。现有旧研究的历史版本、分红/拆分等正式准入缺口没有被解除。

记录按观察时来源登记的保留期到期，当前365天。保留期工具默认只预览：

```powershell
.venv\Scripts\python.exe scripts/cash_source_observation_retention.py
```

明确需要清理已到期日志时，运维可加`--execute --max-records 1000`，每次最多1000条；这会不可恢复地删除选中的过期观察日志，**不删除源值、历史样本或模型**。未到期日志仍被数据库保护；没有自动清理任务。本轮仅预览，0条可清理、0条删除。

1005项相关离线、135项隔离PG、44项真实TCP/完整性检查及11项原同步测试通过；没有新来源网络请求、真实模型或预测概率。复跑见TC-FDP-35。
