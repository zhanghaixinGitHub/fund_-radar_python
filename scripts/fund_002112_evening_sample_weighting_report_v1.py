"""生成本轮历史样本加权对照报告；只读取已完成且通过复核的独立研究产物。"""

from pathlib import Path

from scripts import fund_002112_evening_sample_weighting_v1 as m

old = m.old
REPORT = Path(r"C:\Users\a\.codex\visualizations\2026\10\02\01a0fc78-9a4f-7df0-90e5-703ebd5bb196") / (
    "002112-sample-weighting-20261003/002112-sample-weighting-results.html"
)
LABELS = {"ALL": "原方案：全部历史", "RECENT": "更重视近期历史", "SIMILAR": "更重视持仓相近的历史"}


def main():
    verified = old.read(m.ROOT / "independent-verification.json")
    assert verified["passed"]
    scores = old.read(m.ROOT / "scores.json")
    pairs = old.read(m.ROOT / "paired-comparisons.json")
    decision = old.read(m.ROOT / "decision.json")
    assert not any(decision["passed_all"].values()), "REVIEW_NARRATIVE_IF_RESULTS_CHANGE"
    table = ""
    for policy in m.POLICIES:
        cells = "".join(
            f"<td><b>{scores[p][policy]['correct']} / {scores[p][policy]['n']}</b>"
            f"<small>{scores[p][policy]['accuracy']:.2%}</small></td>"
            for p in ("2025", "2026", "ALL")
        )
        table += f'<tr class="{"base" if policy == "ALL" else ""}"><th>{LABELS[policy]}</th>{cells}</tr>'
    quarters = ""
    for year in ("2025", "2026"):
        for q in range(1, 5):
            period = year + "Q" + str(q)
            if period not in scores:
                continue
            cells = "".join(f"<td>{scores[period][p]['correct']} / {scores[period][p]['n']}</td>" for p in m.POLICIES)
            quarters += f"<tr><th>{year}年{q}季度</th>{cells}</tr>"
    changes = ""
    for policy in m.POLICIES[1:]:
        pair = pairs["ALL"][policy]
        lo, hi = pair["descriptive_block_interval95"]
        changes += (
            f"<tr><th>{LABELS[policy]}</th><td>{pair['corrected']}天</td><td>{pair['spoiled']}天</td>"
            f"<td>{pair['net_correct']:+d}天</td><td>{lo * 100:+.2f} 至 {hi * 100:+.2f} 个百分点</td></tr>"
        )
    weights = {
        p: {b: old.read(m.ROOT / "fits" / (p + "_2026_180_" + b + ".json"))["weight_summary"] for b in m.WEIGHTS}
        for p in m.POLICIES[1:]
    }
    links = "".join(
        f'<li><a href="{(m.ROOT / file).as_uri()}">{label}</a></li>'
        for file, label in (
            ("protocol.json", "训练前固定的比较规则"),
            ("predictions.jsonl", "三组逐日预测和真实结果"),
            ("scores.json", "各年、各季度及三部分的详细成绩"),
            ("paired-comparisons.json", "哪些错误被改正，哪些正确判断被改错"),
            ("decision.json", "按原门槛逐项判断的结果"),
            ("independent-verification.json", "模型、权重、时点和原文件保护复核"),
            ("fit-ledger.jsonl", "138次新训练的记录"),
        )
    )
    detail_rows = ""
    for year in ("2025", "2026"):
        for policy in m.POLICIES:
            score = scores[year][policy]
            detail_rows += (
                f"<tr><th>{year} · {LABELS[policy]}</th><td>{score['brier']:.6f}</td>"
                f"<td>{score['down_recall']:.2%}</td><td>{score['predicted_up']} / {score['n']}</td></tr>"
            )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>002112 · 旧历史怎么用，实测结果</title>
<style>
*{{box-sizing:border-box}}
body{{margin:0;background:#f3f6f8;color:#183342;font:16px/1.85 'Microsoft YaHei',sans-serif}}

main{{max-width:1050px;padding:28px 24px 50px;margin:auto}}
header{{background:#183d4c;color:#fff;border-radius:18px;padding:32px}}

h1{{font-size:30px;line-height:1.45;margin:10px 0}}
header p{{color:#dbe9eb}}
.eyebrow{{font-size:13px;color:#afdacf;letter-spacing:1px}}

section{{background:white;margin-top:20px;padding:27px;border:1px solid #dde6e9;border-radius:15px}}

h2{{font-size:22px;margin:0 0 14px}}
p{{margin:10px 0}}
small{{display:block;color:#697f8a;font-size:13px}}
.muted{{color:#697f8a}}

.scroll{{overflow:auto}}
table{{border-collapse:collapse;width:100%;min-width:580px;font-size:15px}}

td,th{{padding:12px;border-bottom:1px solid #dfe7eb;text-align:left}}
thead{{background:#eef4f6}}
th{{font-weight:600}}

.base{{background:#edf6f3}}
.callout{{background:#fff7e7;border-left:4px solid #c89b46;padding:16px 18px;margin-top:18px}}

.cards{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.card{{padding:19px;background:#f0f6f5;border-radius:12px}}

.number{{font-size:29px;font-weight:bold;color:#146e67}}
details{{margin-top:16px}}
summary{{cursor:pointer;color:#146e67;font-weight:600}}

li{{margin:7px 0}}
a{{color:#126a85;overflow-wrap:anywhere}}
footer{{font-size:13px;color:#697f8a;margin-top:22px}}

@media(max-width:650px){{main{{padding:14px 12px 32px}}
header,section{{padding:21px}}
h1{{font-size:25px}}
h2{{font-size:20px}}
.cards{{grid-template-columns:1fr}}
}}

</style>
</head>
<body>
<main>
<header>
<div class="eyebrow">002112 · 三种历史资料用法 · 截至2026年9月30日</div>
<h1>两种新用法都没有稳定改善，保留原方案</h1>
<p>只改变每条历史资料的影响大小，新闻、公告、政策的内容不动，信息40%＋行情40%＋净值20%也不动。
两组新训练均未通过事前固定的历史合格标准，现用模型未替换。
</p>
</header>
<section>
<h2>1. 实际判对了多少天？</h2>
<p>下表为“判对天数／评估天数”。
2026年只统计截至9月30日的181个评估日。
</p>
<div class="scroll">
<table>
<thead>
<tr>
<th>怎么使用历史资料</th>
<th>2025年</th>
<th>2026年截至9月底</th>
<th>全部424天</th>
</tr>
</thead>
<tbody>{table}</tbody>
</table>
</div>
<div class="callout">
<b>近期组：2026年多对4天，但2025年少对6天，合计少对2天。
持仓组：两年分别少对3天，合计少对6天。
</b>
<br>
所以本轮没有证据支持替换原方案，也不能把“旧资料太多”认定为已解决的问题。
</div>
<p class="muted">原方案本身也未被证明可靠。
全部424天即使每天猜涨，也会判对223天（52.59%）；
原方案是214天。
这个简单对照同样不能作为未来收益保证。
</p>
</section>
<section>
<h2>2. 最想改善的第三季度，仍然没有改善</h2>
<div class="cards">
<div class="card">
<div class="number">25 → 22 天</div>
<b>更重视近期历史后的第三季度正确数</b>
<p>原方案判对25／65天，近期组22／65天。
实际下跌的34天，原方案抓到7天，近期组只抓到3天。
</p>
</div>
<div class="card">
<div class="number">25 → 25 天</div>
<b>更重视持仓相近历史后的第三季度正确数</b>
<p>持仓组改正2天，也把原本正确的2天改错，整体没有净改善。
</p>
</div>
</div>
<p>近期组2026年的进步主要来自第二季度：29／60天升到35／60天。
这个阶段的改善，没有延续到第三季度。
</p>
<details>
<summary>展开全部季度，查看结果是否稳定</summary>
<div class="scroll">
<table>
<thead>
<tr>
<th>季度</th>
<th>原方案</th>
<th>近期组</th>
<th>持仓组</th>
</tr>
</thead>
<tbody>{quarters}</tbody>
</table>
</div>
</details>
</section>
<section>
<h2>3. 新规则确实进入了训练</h2>
<p>
<b>近期组：</b>每旧126个交易日，约半年的资料，影响减半；
每条合格历史都保留。
在预测9月30日所用的净值训练里，一年前资料的总影响从约90.2%降到{weights["RECENT"]["history"]["older_than_year_weight_share"]:.1%}。
</p>
<p>
<b>持仓组：</b>比较各日期当时已公开的前十股票，越接近训练时点已公开的持仓，影响越大。
已知持仓里，完全重合与完全不重合的权重之比为4倍。
</p>
<p>
<b>资料缺失仍有边界：</b>净值训练的2,580条资料中，1,807条缺少可核对的历史持仓，持仓组将其按中性处理，保留其约70%的总权重；
没有假装知道这些日期持有什么。
因此这轮只检验了“可观察到的披露持仓相似度”。
</p>
<p class="muted">每次训练只使用此前已成熟的涨跌答案；
各历史样本用其当时已公开的报告，训练参考持仓也只取该次训练前已公开的报告。
季度披露无法还原真实每日完整持仓。
缺值处理、算法参数、随机种子、训练日期和更新频率都与原方案保持一致。
</p>
</section>
<section>
<h2>4. 没有把“改了很多判断”当成“变好了”</h2>
<div class="scroll">
<table>
<thead>
<tr>
<th>新用法</th>
<th>把错改对</th>
<th>把对改错</th>
<th>净增加正确</th>
<th>历史重抽样差值范围</th>
</tr>
</thead>
<tbody>{changes}</tbody>
</table>
</div>
<p>两组都修正了一部分错误，也引入了更多错误。
差值范围都包含零，整体稳定优势没有得到支持。
</p>
<p class="muted">范围来自以连续20日为单位的4,000次成块重抽样，只描述这段已观察历史的不确定性。
它没有消除多轮使用同一历史数据的偏差，也不是未见未来的统计证明。
</p>
<details>
<summary>查看概率误差与下跌判断</summary>
<p>概率误差越小越好；
下跌识别率表示实际下跌时猜对的比例。
合格标准同时看正确天数、概率误差、下跌判断和季度稳定性，且要同时胜过原方案及之前修正过输入的对照。
</p>
<div class="scroll">
<table>
<thead>
<tr>
<th>时期与方案</th>
<th>概率误差</th>
<th>下跌识别率</th>
<th>猜涨天数</th>
</tr>
</thead>
<tbody>{detail_rows}</tbody>
</table>
</div>
</details>
</section>
<section>
<h2>本轮决定</h2>
<p>
<b>两种新用法均不采用；
保留现有模型。
</b>本轮固定的两项假设已经检验完，不根据这份成绩反复改变半年期限或持仓倍率，直到凑出更好看的历史分数。
</p>
<p>这证明的是“本次两种具体规则没有稳定改善”，并不能证明所有近期加权或持仓加权都无效。
要判断后续是否真的有用，仍需要提前保存预测，再等新的真实结果。
</p>
<details>
<summary>查看比较规则、逐日结果和复核记录</summary>
<ul>{links}</ul>
<p class="muted">新增138次拟合、复用69个旧模型；
独立回读207个模型，核对27,600棵树的加权抽样和3,816行分支概率；
11项边界检查通过。
原实验和三个工作区的既有文件均保持不变。
没有补采数据、付费调用、部署、提交或推送。
</p>
</details>
</section>
<footer>历史对照研究 · {verified["at"]} · 2025与2026均为已观察历史，历史首见存档仍不完整，本轮不是未来盲测。
</footer>
</main>
</body>
</html>"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(page)
    old.write(
        m.ROOT / "delivery.json",
        {
            "report": str(REPORT),
            "report_sha256": old.sha(REPORT),
            "new_fits": 138,
            "adopted": False,
            "future_validation": False,
        },
    )
    print(old.json.dumps({"report": str(REPORT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
