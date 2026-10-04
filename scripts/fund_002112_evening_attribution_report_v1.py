"""交付两阶段固定实验的成绩、分组增量和逐日权重，不根据结果再选择方案。"""
# ruff: noqa: E501

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import fund_002112_evening_attribution_v1 as m

REPORT = Path(r"C:\Users\a\.codex\visualizations\2026\10\02\01a0fc78-9a4f-7df0-90e5-703ebd5bb196") / (
    "002112-attribution-20261003/002112-attribution-results.html"
)
POLICY_NAMES = {
    "FIXED_25": "信息25%／市场50%／净值25%",
    "FIXED_40": "信息40%／市场40%／净值20%",
    "FIXED_55": "信息55%／市场30%／净值15%",
    "FIXED_70": "信息70%／市场20%／净值10%",
    "QUALITY_GATE": "按证据状态：信息55%／25%／0%",
}


def pct(value):
    return f"{value:.2%}"


def paired(a: dict, b: dict, targets: list) -> dict:
    counts = Counter()
    rows = []
    for target in targets:
        before, after = a[target], b[target]
        p = m.old.CLASSES[int(np.argmax(before["probabilities"]))]
        q = m.old.CLASSES[int(np.argmax(after["probabilities"]))]
        state = "UNCHANGED" if p == q else "CORRECTED" if q == after["label"] else "SPOILED"
        counts[state] += 1
        rows.append({"target": target, "before": p, "after": q, "actual": after["label"], "status": state})
    return {"counts": dict(counts), "net_correct": counts["CORRECTED"] - counts["SPOILED"], "days": rows}


def score_table(scores, names, selected):
    output = ""
    for key, label in names.items():
        cells = "".join(
            f"<td><b>{pct(scores[p][key]['accuracy'])}</b><small>{scores[p][key]['correct']} / {scores[p][key]['n']}</small></td>"
            for p in ("2025", "2026", "ALL")
        )
        output += f'<tr class="{"selected" if key == selected else ""}"><th>{html.escape(label)}</th>{cells}</tr>'
    return (
        '<div class="scroll"><table><thead><tr><th>方案</th><th>2025 · 选择期</th><th>2026至9月30日 · 历史诊断</th><th>合计</th></tr></thead><tbody>'
        + output
        + "</tbody></table></div>"
    )


def main():
    root, old = m.ROOT, m.old
    verified = old.read(root / "independent-verification.json")
    assert verified["passed"]
    selection, decision = old.read(root / "selection.json"), old.read(root / "decision.json")
    factorial, weights, scores = [
        old.read(root / name) for name in ("factorial-scores.json", "weight-scores.json", "scores.json")
    ]
    records = old.lines(root / "factorial-predictions.jsonl")
    by_group = {g: {r["target"]: r for r in records if r["group"] == g} for g in m.GROUPS}
    weighted = old.lines(root / "weight-predictions.jsonl")
    by_policy = {p: {r["target"]: r for r in weighted if r["policy"] == p} for p in m.POLICIES}
    audit = {}
    for period in ("2025", "2026", "ALL"):
        targets = sorted(t for t in by_group["CONTENT"] if m.matches(t, period))
        for before, after in [("CONTENT", "TIMING"), ("CONTENT", "REACTION"), ("TIMING", "BOTH"), ("REACTION", "BOTH")]:
            audit[f"{period}:{before}->{after}"] = paired(by_group[before], by_group[after], targets)
        audit[period + ":WEIGHT_CHANGE"] = paired(by_policy["FIXED_55"], by_policy[selection["policy"]], targets)
    old.write(root / "paired-impact.json", audit)
    quality = {r["target"]: r for r in old.lines(root / "quality-evidence.jsonl")}
    strata = {}
    for period in ("2025", "2026", "ALL"):
        for state in ("STRONG", "CONTEXT", "NONE"):
            targets = [t for t in by_group["CONTENT"] if m.matches(t, period) and quality[t]["state"] == state]
            strata[period + ":" + state] = (
                {p: m.metric([by_policy[p][t] for t in targets]) for p in m.POLICIES} if targets else {}
            )
    old.write(root / "quality-strata.json", strata)
    delta_rows = ""
    names = {
        ("CONTENT", "TIMING"): "加入新增内容与时效",
        ("CONTENT", "REACTION"): "加入价格反应",
        ("TIMING", "BOTH"): "已有时效，再加价格反应",
        ("REACTION", "BOTH"): "已有价格反应，再加时效",
    }
    for (before, after), label in names.items():
        cells = "".join(f"<td>{audit[f'{p}:{before}->{after}']['net_correct']:+d} 天</td>" for p in ("2025", "2026"))
        delta_rows += f"<tr><th>{label}</th>{cells}</tr>"
    quarter_rows = ""
    for period, values in scores.items():
        if "Q" in period:
            cells = "".join(
                f"<td>{values[k]['correct']}/{values[k]['n']} · {pct(values[k]['accuracy'])}</td>"
                for k in ("A_FIX", "PRIOR_C", "SELECTED")
            )
            quarter_rows += f"<tr><th>{period}</th>{cells}</tr>"
    error_rows = ""
    for period in ("2025", "2026", "ALL"):
        values = scores[period]
        error_rows += (
            f"<tr><th>{period}</th><td>{values['PRIOR_C']['brier']:.6f} → {values['SELECTED']['brier']:.6f}</td>"
            f"<td>{pct(values['PRIOR_C']['down_recall'])} → {pct(values['SELECTED']['down_recall'])}</td></tr>"
        )
    links = "".join(
        f'<li><a href="{(root / name).as_uri()}">{label}</a></li>'
        for name, label in [
            ("protocol.json", "训练前冻结的规则与上限"),
            ("selection.json", "2025两步选择回执"),
            ("factorial-scores.json", "四组表达的全部成绩"),
            ("weight-scores.json", "五种配比的全部成绩"),
            ("paired-impact.json", "逐日改对、改错和分组增量"),
            ("quality-evidence.jsonl", "每个日期的证据状态及公告引用"),
            ("weight-predictions.jsonl", "每个日期的概率与实际融合权重"),
            ("quality-strata.json", "按证据状态分层的成绩"),
            ("independent-verification.json", "独立模型回读与时间边界验证"),
            ("002112-evening-attribution.joblib", "本轮研究候选包"),
        ]
    )
    baseline_names = {
        "A_FIX": "上轮修正数字基准",
        "PRIOR_C": "上轮候选：信息55%",
        "SELECTED": "本轮候选：信息40%",
        "PRICE": "市场＋净值历史",
        "NAV": "仅净值历史",
        "ALWAYS_UP": "每天均猜上涨",
    }
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>002112 · 信息贡献与融合权重实验</title><style>
:root{{--ink:#172c3c;--muted:#617182;--line:#dce5e9;--accent:#14756d;--bg:#f3f6f8}}*{{box-sizing:border-box}}
body{{margin:0;color:var(--ink);background:var(--bg);font:16px/1.8 'Microsoft YaHei',sans-serif}}main{{max-width:1120px;margin:auto;padding:32px 24px 60px}}
header{{background:#173548;color:white;border-radius:18px;padding:36px}}.eyebrow{{font-size:13px;color:#afd4cc;letter-spacing:2px}}
h1{{font-size:32px;line-height:1.4;margin:12px 0}}h2{{font-size:22px;margin:0 0 16px}}h3{{font-size:17px;margin:14px 0 6px}}p{{margin:10px 0}}
header p{{color:#dce8ed}}.badge{{display:inline-block;color:#6b4b09;background:#ffe6ad;padding:4px 12px;border-radius:20px;font-size:14px}}
section{{background:white;border:1px solid var(--line);border-radius:16px;padding:28px;margin-top:20px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:20px}}
.card{{background:white;border:1px solid var(--line);padding:20px;border-radius:14px}}.number{{color:var(--accent);font-size:30px;font-weight:700;line-height:1.4}}
small{{display:block;color:var(--muted);font-size:13px}}.muted{{color:var(--muted)}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:15px;min-width:660px}}th,td{{text-align:left;padding:12px 14px;border-bottom:1px solid var(--line)}}thead{{background:#edf3f5}}th{{font-weight:600}}tr.selected{{background:#e8f5ef}}
.callout{{background:#fff8e7;border-left:4px solid #c9973f;padding:14px 18px;margin-top:16px}}.two{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.panel{{padding:18px;background:#f3f7f8;border-radius:12px}}
a{{color:#066c8b;overflow-wrap:anywhere}}li{{margin:7px 0}}details{{margin-top:18px}}summary{{cursor:pointer;color:var(--accent);font-weight:600}}footer{{color:var(--muted);font-size:13px;margin-top:24px}}
@media(max-width:650px){{main{{padding:14px 12px 32px}}header,section{{padding:22px}}h1{{font-size:25px}}.cards,.two{{grid-template-columns:1fr}}.card{{display:flex;gap:22px;align-items:center}}.number{{min-width:95px}}}}
</style></head><body><main>
<header><div class="eyebrow">002112 · 同日期、同截点、同市场与净值输入</div><h1>拆清信息贡献，再检验融合权重</h1>
<p>两阶段实验已完成。2025选出40%／40%／20%，但2026未改善，仍不替换现用模型。</p><span class="badge">历史检验未通过 · 研究候选已保存</span></header>
<div class="cards"><div class="card"><div class="number">4 + 5</div><div>四组表达、五种配比<small>先选表达，再只测试所选表达的配比</small></div></div>
<div class="card"><div class="number">46</div><div>实际新增拟合<small>两个旧方案及三个全量模型直接复用</small></div></div>
<div class="card"><div class="number">56 / 56</div><div>边界检查通过<small>全部模型回读及概率重算也通过</small></div></div></div>
<section><h2>第一步：改善来自时效，还是价格反应？</h2>
<p>四组都用信息55%／市场30%／净值历史15%，只改变信息分支的输入。内容包含已核实公司公告及有关联的行业背景；时效组还包含新增比例、衰减和非价格的覆盖字段；价格反应组包含截点前已完成的股价变化及其可观察状态。</p>
{score_table(factorial, m.LABELS, selection["group"])}
<p>仅按2025成绩选择“全部合并”。2026的反应组虽然高一些，也不据此改选。</p>
<div class="scroll"><table><thead><tr><th>在其他条件相同时的增量</th><th>2025净多判对</th><th>2026净多判对</th></tr></thead><tbody>{delta_rows}</tbody></table></div>
<div class="callout"><b>能支持的结论：</b>新增与时效处理并非完全没有价值；改善也不全来自价格。但两组效果不能简单相加，合并在2026反而弱于分别加入。这里只是这套模型下的预测差异，不能认定经济因果或长期有效。</div></section>
<section><h2>第二步：固定55%是否合适？</h2><p>只对第一步选中的“全部合并”比较五种规则。剩余市场与净值比例始终为2:1；权重比较复用原概率，不增加拟合。</p>
{score_table(weights, POLICY_NAMES, selection["policy"])}
<p><b>2025选出的配比为信息40%／市场40%／净值历史20%。</b>相对原55%，2025多判对3天，2026少判对1天；合计只多2天。提高到70%也没有表现出跨年份的一致优势。</p>
<div class="two"><div class="panel"><h3>按证据调整的规则是什么？</h3><p>两个预测窗口内，营收/利润/订单的已核实数字覆盖至少3%持仓：信息占55%；其他合格公告或相关背景：25%；没有本轮可用证据：0%。余下权重给市场与净值。</p></div>
<div class="panel"><h3>这条规则是否有效？</h3><p>在424个评估日中，强证据29天、背景178天、无可用证据217天。条件规则合计判对210天，低于原55%的212天，未选用。降低信息占比不能替代补齐证据。</p></div></div>
<p class="muted">这里的0%是融合占比，不是把缺失数据填成0；“无可用证据”不代表市场没有消息。信息分支含价格反应，40%不能解释为纯新闻的因果贡献。</p>
<details><summary>查看概率误差和下跌识别变化</summary><div class="scroll"><table><thead><tr><th>时期</th><th>概率误差：原55% → 本轮40%，低者较好</th><th>下跌识别：原55% → 本轮40%</th></tr></thead><tbody>{error_rows}</tbody></table></div>
<p>概率误差采用三分类Brier分数；下跌识别是实际下跌日期中判对的比例。两者与整体正确率可能出现取舍，不能只展示提高的一项。概率尚未做独立校准。</p></details></section>
<section><h2>为何没有通过替换条件？</h2>{score_table(scores, baseline_names, "SELECTED")}
<p>冻结规则要求2025、2026均超过修正基准，并兼顾概率误差、下跌识别和季度表现。2026候选为86/181，低于修正基准89/181，季度支持也未通过。</p>
<div class="callout"><b>合计50.47%不足以宣布成功。</b>相较上轮候选50.00%只多判对2天，而且2026更差。“每天均猜上涨”的合计方向正确率仍为52.59%，但它无法识别任何下跌，只作为必要的简单对照。</div>
<details><summary>查看逐季度成绩</summary><div class="scroll"><table><thead><tr><th>季度</th><th>修正数字基准</th><th>上轮55%</th><th>本轮40%</th></tr></thead><tbody>{quarter_rows}</tbody></table></div></details></section>
<section><h2>验证范围与交付边界</h2><p>使用现有665个共同输入日期，没有新增外部数据或付费请求。新训两组各23次，共46次；最终选中旧的合并表达，因此复用已验证的全量模型，将新配比保存在独立候选包里。</p>
<p>独立回读46个新模型、92个复用历史模型和3个全量模型；重算2,544行分支概率，核对1,696条四组判断、2,120条权重判断，重建2,660行输入和665天证据状态。全量包的665行输出也已核对，最大分支概率误差为{verified["maximum_probability_error"]:.2e}。</p>
<p>194个既有工作区文件和前两轮产物的字节保持不变。没有提交、推送、部署、替换现用模型或开启定时任务。</p>
<div class="callout"><b>时间边界：</b>两步选择在2026新拟合前保存，全部信息仍按目标交易日前一晚23点筛选。2025已反复用于选择，2026也是已见历史；两者均不能替代未见未来验证。历史首见存档仍不完整，本轮结果是保守时点重建下的研究结论。</div></section>
<section><h2>这轮留下的判断</h2><p>继续细调固定权重，暂时没有稳定收益证据；按证据强弱降权也没有通过检验。下一项更值得研究的是旧样本与当时基金持仓风格是否匹配，再检查概率是否过度自信；本轮没有扩大到这些新实验。</p><ul>{links}</ul></section>
<footer>本地研究报告 · {html.escape(decision["at"])} · 正确率不等于可交易收益，候选文件仅用于复核。</footer>
</main></body></html>"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(page)
    old.write(
        root / "delivery.json",
        {
            "at": old.now(),
            "report": str(REPORT),
            "model": str(root / "002112-evening-attribution.joblib"),
            "new_fits": decision["actual_new_fits"],
            "selected_group": selection["group"],
            "selected_policy": selection["policy"],
            "historical_gate_passed": decision["historical_gate"]["passed"],
            "adopted": False,
            "verification": str(root / "independent-verification.json"),
        },
    )
    print(
        old.json.dumps(
            {"report": str(REPORT), "weights": POLICY_NAMES[selection["policy"]], "adopted": False}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
