"""从本轮已验证产物生成中文交付报告；不训练、不调参，不读取外部最新数据。"""

from __future__ import annotations

import html
from collections import Counter
from urllib.parse import urlparse

from scripts import fund_002112_evening_fusion_v2 as m


def main():
    root = m.ROOT
    verification = m.read(root / "independent-verification.json")
    assert verification["passed"]
    scores = m.read(root / "scores.json")
    sources = m.read(root / "sources.json")
    predictions = m.lines(root / "historical-predictions.jsonl")
    lineage = m.lines(root / "daily-lineage.jsonl")
    public = [e for e in sources["public_events"] if e["source_kind"] in ("news", "policy")]
    domains = Counter(urlparse(e.get("source_url") or "").hostname for e in public)
    used = {e["id"] for row in lineage for e in row.get("information", {}).get("events", [])}
    flips = {}
    for baseline in ["history_only", "price_history"]:
        details = []
        for row in predictions:
            guess = m.CLASSES[max(range(3), key=lambda i: row["weighted_55"][i])]
            old = m.CLASSES[max(range(3), key=lambda i: row[baseline][i])]
            if guess != old:
                details.append({"target": row["target"], "before": old, "after": guess, "actual": row["label"]})
        flips[baseline] = {
            "changed": len(details),
            "corrected": sum(d["after"] == d["actual"] for d in details),
            "spoiled": sum(d["before"] == d["actual"] for d in details),
            "details": details,
        }
    m.write(root / "information-impact.json", flips)
    m.write(
        root / "delivery.json",
        {
            "completed_at": m.now(),
            "model": str(root / "002112-evening-1d-news55.joblib"),
            "model_sha256": m.sha(root / "002112-evening-1d-news55.joblib"),
            "current_version": "20261003-v2",
            "training_complete": True,
            "current_version_fits": 73,
            "superseded_fits": 72,
            "total_actual_fits": 145,
            "weights": m.WEIGHTS,
            "final_training_rows": verification["final_training_rows"],
            "historical_screen_passed": False,
            "future_validation": False,
            "adopted": False,
            "unique_used_documents": len(used),
            "public_source_domains": dict(domains),
            "full_scope": "全部本地可用样本；不是全网完整新闻历史；新闻/行情共665日，净值2582日",
            "prior_artifacts_preserved": True,
            "database_writes": 0,
            "paid_requests": 0,
        },
    )
    labels = {
        "weighted_55": "新闻55%＋行情30%＋净值15%",
        "price_history": "行情＋净值对照",
        "history_only": "仅净值对照",
        "always_up": "始终预测上涨",
    }
    table = ""
    for method, label in labels.items():
        cells = "".join(
            f"<td>{scores[p][method]['correct']}/{scores[p][method]['n']}<br>"
            f"<strong>{scores[p][method]['accuracy']:.2%}</strong></td>"
            for p in ["2025", "2026", "ALL"]
        )
        table += f"<tr><th>{label}</th>{cells}</tr>"
    examples = ""
    for row in predictions[-3:]:
        probs = row["branches"]
        formula = " + ".join(f"{m.WEIGHTS[b]:.0%} × {probs[b][2]:.4%}" for b in m.WEIGHTS)
        actual = "涨" if row["label"] == "UP" else "跌" if row["label"] == "DOWN" else "平"
        examples += f"<tr><td>{row['target']}<br><small>截至{row['as_of'][:16].replace('T', ' ')}</small></td>"
        examples += f"<td>{formula}<br>= <b>{row['weighted_55'][2]:.4%}</b></td>"
        examples += f"<td>涨</td><td>{actual}（{row['return']:+.2%}）</td></tr>"
    off_calendar = sorted(set(sources["nav"]) - set(sources["sessions"]))
    note = f"""# 002112 晚间一日模型全量训练结果

当前可用历史的全量训练、模型保存和独立验证已完成。新闻权重55%的方案，历史424日正确202日（47.64%），未证明优于行情加净值对照（48.11%），未采用到现用系统。

## 训练范围与方法

- 预测截点：目标交易日前一自然日23:00，价格基日为前一交易日。2026-09-28的截点为09-27 23:00，包含周末消息；09-29的截点为09-28 23:00。
- 标签：目标日单位净值与前一交易日单位净值精确比较，分涨、平、跌。使用真实净值，不使用前两次人工判断作答案。单位净值包含除息影响；已公告的分红另作输入。
- 历史净值：原始2659条；14条休市日估值不当成额外交易日，首个交易日无前值，62日缺足够且已可用的61日连续窗口，余2582个样本，2016-02-19至2026-09-30。
- 信息、行情分支：当前具有对应历史材料的全部665日，2024-01-03至2026-09-30。没有把2016—2023年缺失新闻当“当天无新闻”补进该分支。
- 检查5322份材料（既有5222份＋后续补充100份），实际进入各日输入的唯一材料{len(used)}份。新闻570份、政策91份；有合格公司事件264日、新闻648日、政策624日，窗口为1/5/20个预测交易日；公司数值用5日窗口。
- 信息分支188项：核对过的营收/利润同比、业绩预告、回购等事实与缺失状态，以及新闻政策的原文方向、阶段、类别、影响途径。关联持仓使用当时公开报告；场景相关不自动视为公司受益。
- 行情分支22项：当日已披露持仓的价格贡献、覆盖权重、成交活跃度和沪深300/中证500行情；净值分支15项：涨跌、波动、回撤、连续下跌、可用滞后与已公告分红。
- 三个分支分别学习下一日真实方向，采用固定随机森林配方（200棵树、深度4、叶节点至少10个训练样本）。每20个检验交易日更新，训练标签须在当时已成熟；预处理仅从训练样本计算，全缺列删除，其他缺失保留标志并以训练中位数作计算替代。
- 最终概率向量 = 0.55×信息分支 + 0.30×行情分支 + 0.15×净值分支；取最大项作为方向。55%是合成比例，不是可解释归因百分比，也不是经过校准的胜率。
- 固定一种方案，未根据本轮成绩搜索更好权重。对照为同日期同截点的纯净值、行情2/3＋净值1/3、始终上涨。

## 结果及影响

2025：新闻增强115/243（47.33%）；行情＋净值122/243（50.21%）；纯净值118/243（48.56%）；始终上涨128/243（52.67%）。

2026截至09-30：新闻增强87/181（48.07%）；行情＋净值82/181（45.30%）；纯净值83/181（45.86%）；始终上涨95/181（52.49%）。

合计：新闻增强202/424（47.64%）；行情＋净值204/424（48.11%）；纯净值201/424（47.41%）；始终上涨223/424（52.59%）。

相对行情＋净值，加入55%信息权重实际改动102日方向，改对50日、改错52日，净少对2日。相对纯净值改动165日，改对83日、改错82日，净多对1日，不足以证明提升。

2026下跌识别率仅25/86（29.07%），低于行情＋净值34/86（39.53%）和纯净值42/86（48.84%）。概率误差Brier合计0.504635，略好于行情＋净值0.506690，但不能抵消方向正确率及下跌识别的问题。

以上均为已看过答案的历史研究。2025/2026不能称未见测试，也不代表未来实际收益。此前08:00实验的分数因时间口径、训练范围和输入不同，不直接与本表相减。

## 不能说已完整复刻人工方法的原因

661份新闻政策中649份来自国家医保局，其他来源共12份；连续海外市场、海外新闻和科技产业新闻历史明显不足。当前材料并非全网完整集合，早年采集选择可能偏向后来关注的持仓。人工分析里的海外20%等六小项没有被完整复刻；本模型的55%对应现有整体信息分支。

原文有可核对引文不等于历史首次公开版本已证实。部分正文只处理节选，日期只有日粒度的公告等下一日0点再用；已知较晚版本继续后移。历史收盘价采用当日16点可得的重建假设，真实首见存档仍不齐全。净值保留既有更晚版本约束，近期实际收取时间也参与限制。

因此本次是“当前可用数据全量训练完成”，不是“完整新闻版正式一日模型已通过验收”。历史门槛未过、未来验证未做、现用模型未替换。

## 工程证据与交付

最终三个分支以全部成熟历史重新拟合：信息665行、行情665行、净值2582行。最终模型是当前时点的训练产物，不能拿它回推09-28/09-29并声称没有偷看；报告中的历史判断来自当时训练截止的滚动模型。

最终版本72次训练＋1次同种子重放，共73次；初版72次因周末截点和最近单日行情回执格式修正而作废，完整保留且未据其分数选择方案。本任务实际总拟合145次，不隐去作废计算。

19项边界测试与Ruff通过；72个正式产物模型回读、1272行分支概率独立复算，最大误差{verification["maximum_probability_error"]:.3g}；重放误差{verification["replay_maximum_error"]:.3g}，均小于1e-12。2582行输入全量重建一致、33145条事件时间关系通过、186个工作区文件保护核对通过（当前新增验证器允许修正数值容差，验证器指纹已记录）。

未改数据库、业务接口、Java/Vue、现用绑定、后台任务；无付费调用、提交或推送。训练器、来源、输入、逐日判断、完整模型和回读证据都保存在本目录。

- 模型：002112-evening-1d-news55.joblib
- 日级预测与三个分支：historical-predictions.jsonl
- 全量输入及逐日依据：dataset.jsonl、daily-lineage.jsonl
- 规则与来源：protocol.json、source-manifest.json、feature-names.json
- 成绩与验收：scores.json、decision.json、independent-verification.json、delivery.json
- 休市日估值：{", ".join(off_calendar)}。
"""
    with (root / "结果说明.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(note)
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>002112｜晚间一日模型训练结果</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f5f6f8;color:#182229;font:16px/1.75 system-ui,'Microsoft YaHei',sans-serif}}
main{{max-width:1100px;margin:auto;padding:40px 24px 72px}}h1{{font-size:32px;line-height:1.35;margin:12px 0}}h2{{font-size:22px;margin:34px 0 12px}}
.eyebrow,small{{color:#596976}}.lead{{font-size:19px}}.summary{{border-left:4px solid #b36f28;background:#fff7eb;padding:18px 22px;margin:25px 0}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;background:#fff;margin:15px 0}}th,td{{padding:14px 16px;border-bottom:1px solid #dce2e6;text-align:left;vertical-align:top}}thead{{background:#eaf0f2}}strong,b{{font-weight:650}}
.weights{{display:flex;gap:4px;margin:22px 0;min-height:74px}}.weights div{{display:flex;flex-direction:column;justify-content:center;padding:10px 14px;color:#fff}}.w55{{width:55%;background:#285d64}}.w30{{width:30%;background:#546d86}}.w15{{width:15%;background:#69747d}}
a{{color:#176b76}}details{{margin-top:26px;border-top:1px solid #cbd4d9;padding-top:16px}}summary{{cursor:pointer;font-weight:600}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:15px/1.8 system-ui,sans-serif}}
@media(max-width:600px){{main{{padding:24px 16px 50px}}h1{{font-size:26px}}th,td{{padding:10px 8px;font-size:14px}}.weights div{{padding:8px;font-size:14px}}.w15{{min-width:64px}}}}
</style><main><div class="eyebrow">德邦鑫星价值C · 002112 · 2026-10-03</div><h1>一日模型已训练，尚未证明新闻加权更准</h1>
<p class="lead">按当前可用历史完成全量训练。424个历史检验日，判断对202天，准确率<strong>47.64%</strong>。</p>
<div class="summary">保留为研究模型，现用模型未替换。2025与2026均是已经看过答案的历史研究，不能当作未见测试或未来胜率。</div>
<h2>这次怎么训练</h2><p>在目标交易日前一晚23点截断信息，预测目标日相对前一交易日的单位净值方向。周末新闻会进入周一的预测；晚于截点的材料和净值不进当日输入。</p>
<div class="weights" aria-label="信息55%，行情30%，净值15%"><div class="w55"><b>55% 新闻、公告、政策</b><span>学习原文事实、阶段与方向</span></div><div class="w30"><b>30% 当日行情</b><span>持仓与市场</span></div><div class="w15"><b>15% 净值历史</b></div></div>
<p>三个分支分别训练后合成方向判断。人工打分没有作为训练答案；权重先固定，再看历史成绩。</p>
<table><thead><tr><th>分支</th><th>最终训练范围</th><th>样本</th></tr></thead><tbody><tr><td>新闻、公告、政策</td><td>2024-01-03 至 2026-09-30</td><td>665日</td></tr><tr><td>行情</td><td>2024-01-03 至 2026-09-30</td><td>665日</td></tr><tr><td>净值历史</td><td>2016-02-19 至 2026-09-30</td><td>2,582日</td></tr></tbody></table>
<h2>与同日期对照相比</h2><div class="table-wrap"><table><thead><tr><th>方案</th><th>2025</th><th>2026截至9月30日</th><th>合计</th></tr></thead><tbody>{table}</tbody></table></div>
<p>相对“行情＋净值”，加入信息分支后改变了102天的判断：<b>改对50天，改错52天</b>。2026年下跌识别率仅29.07%，未通过预先固定的采用条件。</p>
<h2>最主要的数据缺口</h2><p>661份新闻政策中，<b>649份来自国家医保局</b>。海外市场、海外新闻和科技产业信息历史不足，材料分布明显偏医药。当前55%是已有整体信息分支，尚未完整复刻人工分析里的海外、产业、公司、政策六小项。</p>
<p>历史首次公开版本也未完全证实，部分正文为节选。因此“全量”指全部本地可用样本，不能理解为全互联网信息已经补齐。</p>
<h2>逐日计算示例</h2><p>以下使用各历史截点前训练的滚动模型。上涨分值是模型输出，未作胜率校准；完整三个方向及每个分支均已保存。</p>
<div class="table-wrap"><table><thead><tr><th>目标日／截点</th><th>上涨分值的合成</th><th>判断</th><th>事后真实方向</th></tr></thead><tbody>{examples}</tbody></table></div>
<p>当前全量模型已学到9月30日的历史，不能拿它回推上述日期并声称没有看到未来答案。</p>
<h2>交付与核对</h2><p>最终版本完成73次拟合（含一次重放），19项边界测试通过；72个模型回读、1,272行分支概率独立复算通过，最大误差小于1e-12。</p>
<p><a href="{(root / "002112-evening-1d-news55.joblib").as_uri()}">完整研究模型</a> · <a href="{(root / "scores.json").as_uri()}">全部成绩</a> · <a href="{(root / "historical-predictions.jsonl").as_uri()}">逐日判断与分支</a> · <a href="{(root / "independent-verification.json").as_uri()}">独立验证</a></p>
<details><summary>查看完整范围、规则、限制与工程记录</summary><pre>{html.escape(note)}</pre></details></main></html>"""
    output = m.INVENTORY.parents[1] / "002112-training-20261003" / "002112-training-results.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    m.write(root / "report-location.json", {"html": str(output), "sha256": m.sha(output)})
    print(output, flush=True)


if __name__ == "__main__":
    main()
