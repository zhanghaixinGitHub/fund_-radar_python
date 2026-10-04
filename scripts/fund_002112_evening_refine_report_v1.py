"""将冻结实验结果整理成可复查的本地研究报告；不新增拟合、不修改历史成绩。"""
# ruff: noqa: E501

from __future__ import annotations

import html
from collections import Counter
from pathlib import Path

import numpy as np

from scripts import fund_002112_evening_refine_v1 as m

REPORT = Path(r"C:\Users\a\.codex\visualizations\2026\10\02\01a0fc78-9a4f-7df0-90e5-703ebd5bb196") / (
    "002112-optimization-20261003/002112-optimization-results.html"
)
NAMES = {
    **m.LABELS,
    "OLD": "优化前晚间模型",
    "PRICE": "市场＋净值历史",
    "NAV": "仅净值历史",
    "ALWAYS_UP": "每天均猜上涨",
}


def pct(value):
    return f"{value:.2%}"


def main():
    root, old = m.ROOT, m.old
    verification = old.read(root / "independent-verification.json")
    assert verification["passed"]
    scores, coverage, decision = [old.read(root / name) for name in ("scores.json", "coverage.json", "decision.json")]
    records = old.lines(root / "predictions.jsonl")
    by_group = {g: {r["target"]: r for r in records if r["group"] == g} for g in m.GROUPS}
    categories = {r["target"]: r["category"] for r in old.lines(root / "all-day-diagnosis.jsonl")}
    paired = {}
    for group in m.GROUPS:
        for period in ("2025", "2026", "ALL"):
            counts = Counter()
            days = []
            for target, row in by_group[group].items():
                if period != "ALL" and not target.startswith(period):
                    continue
                a = old.CLASSES[int(np.argmax(by_group["A_FIX"][target]["probabilities"]))]
                b = old.CLASSES[int(np.argmax(row["probabilities"]))]
                category = "UNCHANGED" if a == b else "CORRECTED" if b == row["label"] else "SPOILED"
                counts[category] += 1
                days.append(
                    {"target": target, "original_category": categories[target], "against_corrected_baseline": category}
                )
            paired[group + ":" + period] = {"counts": dict(counts), "days": days}
    old.write(root / "paired-impact.json", paired)
    baseline_audit = []
    original_inputs = {r["target"]: r for r in old.lines(old.ROOT / "dataset.jsonl") if "information" in r["groups"]}
    for row in old.lines(root / "A_FIX-inputs.jsonl"):
        original = original_inputs[row["target"]]["groups"]["information"]
        values = row["groups"]["information"]
        changed = [i for i in range(len(values)) if values[i] != original[i]]
        if changed:
            baseline_audit.append(
                {
                    "target": row["target"],
                    "as_of": row["as_of"],
                    "changes": [
                        {"feature": m.feature_names("A_FIX")[i], "old": original[i], "new": values[i]} for i in changed
                    ],
                }
            )
    old.write_lines(root / "numeric-input-changes.jsonl", baseline_audit)
    material = old.read(root / "verified-business-material.json")
    proofs = old.lines(root / "feature-lineage.jsonl")
    used_ids = {e["id"] for row in proofs for e in row["public_events"]}
    used = [e for e in material["public"] if e["id"] in used_ids]
    source_summary = {
        "retained_public": dict(Counter(e["source_kind"] for e in used)),
        "year_related_days": {
            year: sum(bool(p["public_events"]) for p in proofs if p["target"].startswith(year))
            for year in ("2024", "2025", "2026")
        },
        "numeric_changed_rows": len(baseline_audit),
        "buyback_removed_rows": sum(
            any(c["feature"] == "buyback_execution_ratio" for c in r["changes"]) for r in baseline_audit
        ),
    }
    old.write(root / "report-evidence.json", source_summary)
    table = ""
    for key in ("OLD", "A_FIX", "B_LINK", "C_EVENT", "PRICE", "NAV", "ALWAYS_UP"):
        cells = []
        for period in ("2025", "2026", "ALL"):
            s = scores[period][key]
            cells.append(f"<td><b>{pct(s['accuracy'])}</b><small>{s['correct']} / {s['n']}</small></td>")
        table += (
            f'<tr class="{"chosen" if key == decision["selected"] else ""}"><th>{NAMES[key]}</th>{"".join(cells)}</tr>'
        )
    selected = decision["selected"]
    metric_table = ""
    for period in ("2025", "2026", "ALL"):
        a, b = scores[period]["A_FIX"], scores[period][selected]
        change = paired[selected + ":" + period]["counts"]
        metric_table += (
            f"<tr><th>{period}</th><td>{change.get('CORRECTED', 0)} / {change.get('SPOILED', 0)}</td>"
            f"<td>{a['brier']:.6f} → {b['brier']:.6f}</td>"
            f"<td>{pct(a['down_recall'])} → {pct(b['down_recall'])}</td></tr>"
        )
    quarterly = ""
    for period in scores:
        if "Q" in period:
            a, b = scores[period]["A_FIX"], scores[period][selected]
            quarterly += (
                f"<tr><th>{period}</th><td>{a['correct']}/{a['n']} · {pct(a['accuracy'])}</td>"
                f"<td>{b['correct']}/{b['n']} · {pct(b['accuracy'])}</td></tr>"
            )
    documents = "".join(
        f"<tr><td>{e['available_at'][:10]}</td><td>{'新闻' if e['source_kind'] == 'news' else '政策'}</td>"
        f'<td><a href="{html.escape(e["source_url"], quote=True)}">{html.escape(e["title"])}</a></td></tr>'
        for e in used
    )
    links = "".join(
        f'<li><a href="{(root / name).as_uri()}">{label}</a></li>'
        for name, label in [
            ("protocol.json", "冻结的规则与验收条件"),
            ("scores.json", "全部时期成绩"),
            ("feature-lineage.jsonl", "逐日原文、持仓和时间依据"),
            ("all-day-diagnosis.jsonl", "424天原模型错误审查"),
            ("numeric-input-changes.jsonl", "数字输入修正前后对照"),
            ("paired-impact.json", "逐日改对与改错明细"),
            ("independent-verification.json", "独立回读验证结果"),
            ("002112-evening-refined.joblib", "研究候选模型文件"),
        ]
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>002112 · 一日模型优化结果</title>
<style>
:root{{--ink:#162536;--muted:#657183;--line:#dce4e9;--paper:#fff;--bg:#f3f6f8;--accent:#146b66}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);background:var(--bg);font:16px/1.8 'Microsoft YaHei',sans-serif}}
main{{max-width:1080px;margin:auto;padding:32px 24px 60px}}header{{background:#142e42;color:#fff;padding:36px;border-radius:20px}}
.eyebrow{{font-size:13px;letter-spacing:2px;color:#a7cec9}}h1{{font-size:32px;line-height:1.35;margin:12px 0}}h2{{font-size:22px;margin:0 0 16px}}
p{{margin:10px 0}}.lead{{font-size:18px;color:#deeaee}}.badge{{display:inline-block;background:#fae3aa;color:#624509;padding:3px 12px;border-radius:20px;font-weight:600}}
section{{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:28px;margin-top:20px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:20px}}.card{{border:1px solid var(--line);background:#fff;padding:20px;border-radius:14px}}
.number{{font-size:30px;line-height:1.3;font-weight:700;color:var(--accent)}}.muted,small{{color:var(--muted)}}small{{display:block;font-size:13px}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:15px;min-width:590px}}th,td{{text-align:left;padding:12px 14px;border-bottom:1px solid var(--line)}}thead{{background:#eef4f6}}th{{font-weight:600}}tr.chosen{{background:#eaf6f3}}.callout{{border-left:4px solid #c99535;background:#fff8e8;padding:14px 18px;margin-top:16px}}
.chain{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.chain>div{{background:#f4f7f8;padding:18px;border-radius:12px}}.step{{font-size:12px;letter-spacing:1px;color:var(--accent)}}
a{{color:#096d8a;overflow-wrap:anywhere}}li{{margin-bottom:8px}}details{{margin-top:18px}}summary{{cursor:pointer;color:var(--accent);font-weight:600}}code{{font-size:13px;background:#edf1f3;padding:2px 5px;border-radius:4px}}footer{{font-size:13px;color:var(--muted);margin-top:22px}}
@media(max-width:650px){{main{{padding:16px 12px 36px}}header{{padding:24px}}h1{{font-size:26px}}section{{padding:20px}}.cards,.chain{{grid-template-columns:1fr}}.card{{display:flex;gap:20px;align-items:center}}.number{{min-width:100px}}}}
</style></head><body><main>
<header><div class="eyebrow">002112 · 晚间 23:00 → 下一交易日</div><h1>一轮优化完成，尚未达到替换条件</h1>
<p class="lead">候选在 2025 年有所改善，在 2026 年未超过修正后的基准。结果已保存，现用模型保持不变。</p>
<span class="badge">研究候选 · 历史检验未通过</span></header>
<div class="cards"><div class="card"><div class="number">665</div><div>同日期训练样本<small>2024-01-03 至 2026-09-30</small></div></div>
<div class="card"><div class="number">71</div><div>本轮实际拟合次数<small>包含全量训练和一次复现</small></div></div>
<div class="card"><div class="number">39 / 39</div><div>边界检查通过<small>独立回读验证也通过</small></div></div></div>
<section><h2>本轮具体改了什么</h2><div class="chain">
<div><div class="step">A · 先修正对照</div><b>数字输入真正进入模型</b><p>修复利润同比字段名不一致的问题，64 天补回有效值。另将 5 天无法确认累计口径的回购比例保留为未知。</p></div>
<div><div class="step">B · 再检验关联</div><b>新闻必须对应当时的持仓业务</b><p>核对 682 条实际业务引文，只使用当时已公开的持仓和业务资料。无关联消息排除，行业关联方向保持未知；工商经营范围不代替实际业务。</p></div>
<div><div class="step">C · 最后表达事件</div><b>加入新增内容、时效及已观察到的反应</b><p>重复原文按新增比例处理，信息按 5 个交易日半衰期衰减；价格反应只读截点前的完整交易日。无可比预告、回购进度仍是未知。</p></div>
</div><p class="muted">三组使用同一组日期、标签、价格及净值分支；仍为信息 55%／市场 30%／净值历史 15%。事件分支含价格反应，因此它的改善不能全部归因于新闻。</p></section>
<section><h2>同日期对照：方向判断正确率</h2><div class="scroll"><table><thead><tr><th>方案</th><th>2025 · 选择期</th><th>2026 · 已知历史诊断</th><th>合计</th></tr></thead><tbody>{table}</tbody></table></div>
<p>仅按 2025 年结果选择 C，再查看其 2026 年表现。相比修正后的 A，C 在 2025 年多判对 8 天，2026 年少判对 2 天；合计多判对 6 天。相比原版，合计多判对 10 天。</p>
<div class="callout"><b>未达到预先冻结的条件：</b>要求 2025、2026 两年都比修正对照判对更多。2026 未通过，因此不能用合计改善宣布成功。“每天均猜上涨”的方向正确率也仍高于候选，但它完全不识别下跌；该项只作最基本的对照。</div>
<details><summary>概率误差、下跌识别和逐季明细</summary><div class="scroll"><table><thead><tr><th>时期</th><th>C 相对 A：改对／改错</th><th>概率误差 A → C，越低越好</th><th>下跌识别率 A → C</th></tr></thead><tbody>{metric_table}</tbody></table></div>
<p class="muted">概率误差采用三分类 Brier 分数；下跌识别率为实际下跌日中被正确判跌的比例。</p><div class="scroll"><table><thead><tr><th>季度</th><th>A 修正对照</th><th>C 事件候选</th></tr></thead><tbody>{quarterly}</tbody></table></div></details></section>
<section><h2>新闻和政策实际用了多少</h2><p>对已有的 651 篇合格公开材料核对业务关系，19 篇与当时披露持仓存在行业关联，覆盖 248 个预测日。其余 632 篇未匹配到本轮可验证关系。</p>
<p>其中新闻覆盖 111 天、政策覆盖 182 天，两者有重叠。全部 1,129 条“事件—日期—持仓”关联均为行业背景，没有证实公司订单、收入或利润传导，不能自动标成利好。仅加入关联的 B，合计正确数还从 A 的 206 降到 200。</p>
<p class="muted">本輪使用已保存材料，未新增付费调用。新闻来源覆盖仍有明显偏向；有行业关系也不等于消息足够新、足够强或已改变市场预期。</p>
<details><summary>查看实际关联的 19 篇原始材料</summary><div class="scroll"><table><thead><tr><th>本轮保守可用日</th><th>类型</th><th>来源标题</th></tr></thead><tbody>{documents}</tbody></table></div>
<p class="muted">日期沿用已冻结的历史可得时间重建；有些旧材料在本轮可验证的更晚日期才进入输入，不能将其当作当日新发布。原站链接可能更新，训练依据是本地存档的固定版本。</p></details></section>
<section><h2>怎样确认没有把答案带进输入</h2><p>预测截点固定为目标交易日前一个自然日 23:00。训练只使用当时已经成熟的标签；持仓、公告、公司业务资料和日线都按各自可用时间筛选。周末消息不能获得尚未发生的周一价格。</p>
<p>回读 71 个新模型、46 个复用历史模型，重算 2,128 行分支概率，最大误差 {verification["maximum_probability_error"]:.2e}；重建 1,995 行候选输入并核对 1,129 条公共事件关联。190 个既有工作区文件及上一轮产物字节未变。</p>
<div class="callout"><b>证据边界：</b>2026 年历史答案已在此前研究中看过，不能称为真正未见样本；部分原始历史首见时间仍没有完整存档证明。因此，即使历史分数更高，也还需要事前保存预测的未来检验。本轮不启用定时任务、不更换现用模型。</div></section>
<section><h2>交付文件与下一步判断</h2><p>全量研究候选已保存：信息分支 665 条、市场分支复用 665 条、净值历史分支复用 2,582 条。该文件供复核，不作为已通过验证的正式预测能力。</p>
<p>这轮更明确的短板是：与科技持仓直接相关的事件证据不足。后续优先补齐对应公司的业务事件及发布时间，并建立事前预测记录；继续放大新闻固定权重，暂时没有证据支持。</p><ul>{links}</ul></section>
<footer>生成自本地冻结实验 · {html.escape(decision["at"])} · 非收益保证；方向正确率也不等同于可交易收益。</footer>
</main></body></html>"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(page.replace("本輪", "本轮"))
    delivery = {
        "report": str(REPORT),
        "model": str(root / "002112-evening-refined.joblib"),
        "selected": selected,
        "scores": str(root / "scores.json"),
        "verification": verification,
        "historical_gate_passed": False,
        "adopted": False,
        "paid_calls": 0,
        "new_scripts": [
            str(Path(__file__).resolve()),
            str(Path(m.__file__).resolve()),
            str(old.PY / "scripts/fund_002112_evening_refine_verify_v1.py"),
            str(old.PY / "scripts/test_fund_002112_evening_refine_v1.py"),
        ],
    }
    old.write(root / "delivery.json", delivery)
    print(old.json.dumps({"report": str(REPORT), "evidence": source_summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
