"""用自然语言展示排查事实与尚未证实的原因，不引入新训练或新方案选择。"""
# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts import fund_002112_evening_failure_audit_v1 as a

m, old = a.m, a.old
REPORT = Path(r"C:\Users\a\.codex\visualizations\2026\10\02\01a0fc78-9a4f-7df0-90e5-703ebd5bb196") / (
    "002112-diagnosis-20261003/002112-diagnosis-results.html"
)


def main():
    root = a.ROOT
    verified = old.read(root / "verification.json")
    assert verified["passed"]
    scores = old.read(root / "branch-diagnosis.json")
    inputs = old.read(root / "input-coverage-diagnosis.json")
    folds = old.lines(root / "training-and-probe-diagnosis.jsonl")
    last, first = folds[-1]["branches"], folds[0]["branches"]
    daily = old.lines(root / "daily-diagnosis.jsonl")
    ranges = old.read(root / "input-range-diagnosis.json")
    dataset = old.lines(m.ROOT / "BOTH-inputs.jsonl")
    supplement = {"volatility": {}, "outside_range_q3": {}}
    for period in ("2026Q1", "2026Q2", "2026Q3"):
        values = [
            r["groups"]["history"][old.HISTORY_NAMES.index("vol_60")] for r in dataset if m.matches(r["target"], period)
        ]
        supplement["volatility"][period] = {"median_60day_daily_std": float(np.median(values)), "n": len(values)}
    for flag in (True, False):
        rows = [
            r
            for r in daily
            if m.matches(r["target"], "2026Q3")
            and bool(ranges[r["target"]]["history"]["outside_training_range"]) == flag
        ]
        supplement["outside_range_q3"][str(flag)] = {
            "full": a.metrics(rows, "full"),
            "history": a.metrics(rows, "history"),
        }
    old.write(root / "supplementary-diagnosis.json", supplement)
    table = ""
    names = {
        "full": "三部分一起用",
        "without_information": "暂时拿掉信息部分",
        "without_market": "暂时拿掉行情部分",
        "without_history": "暂时拿掉净值历史部分",
    }
    for method, name in names.items():
        cells = "".join(
            f"<td><b>{scores[p]['metrics'][method]['correct']}/{scores[p]['metrics'][method]['n']}</b><small>{scores[p]['metrics'][method]['accuracy']:.2%}</small></td>"
            for p in ("2026Q2", "2026Q3", "ALL")
        )
        table += f'<tr class="{"base" if method == "full" else ""}"><th>{name}</th>{cells}</tr>'
    branch_table = ""
    for branch, label in (("information", "信息部分"), ("market", "行情部分"), ("history", "净值历史部分")):
        metric = scores["2026Q3"]["metrics"][branch]
        branch_table += f"<tr><th>{label}</th><td>{metric['correct']}/65</td><td>{metric['predicted_up']}/65</td></tr>"
    coverage_table = ""
    for period in ("2026Q1", "2026Q2", "2026Q3"):
        row = inputs[period]
        coverage_table += (
            f"<tr><th>{period}</th><td>{row['related_public_days']}/{row['n']}</td>"
            f"<td>{row['qualified_company_days']}/{row['n']}</td><td>{scores[period]['metrics']['full']['correct']}/{row['n']}</td></tr>"
        )
    old_share = last["history"]["older_than_252_sessions"] / last["history"]["training_count"]
    mismatch_share = last["information"]["overlap_under_half_rows"] / last["information"]["training_count"]
    links = "".join(
        f'<li><a href="{(root / name).as_uri()}">{label}</a></li>'
        for name, label in [
            ("daily-diagnosis.jsonl", "每一天谁判对、谁判错，以及相关消息"),
            ("branch-diagnosis.json", "拿掉各部分后的完整对照"),
            ("training-and-probe-diagnosis.jsonl", "每次模型所用历史资料和固定输入检查"),
            ("holdings-timeline.json", "当时已公开持仓的变化"),
            ("input-coverage-diagnosis.json", "消息覆盖和输入缺失检查"),
            ("supplementary-diagnosis.json", "波动变化及超出旧范围的分组检查"),
            ("verification.json", "只读排查验证结果"),
        ]
    )
    page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>002112 · 为什么预测不稳定</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;color:#172d3c;background:#f3f6f8;font:16px/1.8 'Microsoft YaHei',sans-serif}}main{{max-width:1020px;margin:auto;padding:28px 24px 54px}}
header{{padding:32px;background:#173b4b;color:white;border-radius:18px}}h1{{font-size:31px;line-height:1.4;margin:10px 0}}header p{{color:#dfebed}}.eyebrow{{font-size:13px;color:#a8d4ca;letter-spacing:2px}}
section{{margin-top:20px;padding:28px;border:1px solid #dce5e9;border-radius:15px;background:white}}h2{{font-size:22px;margin:0 0 12px}}p{{margin:10px 0}}small,.muted{{color:#68798a}}small{{display:block;font-size:13px}}
.callout{{padding:15px 18px;border-left:4px solid #d2a14a;background:#fff8e8;margin-top:16px}}.facts{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}.fact{{background:#f0f6f5;padding:18px;border-radius:12px}}.number{{font-size:30px;font-weight:700;color:#147069}}
.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;min-width:560px;font-size:15px}}th,td{{text-align:left;padding:12px;border-bottom:1px solid #dce5e9}}thead{{background:#eef4f6}}th{{font-weight:600}}tr.base{{background:#e9f5f1}}
details{{margin-top:18px}}summary{{cursor:pointer;color:#147069;font-weight:600}}li{{margin:7px 0}}a{{color:#076b89;overflow-wrap:anywhere}}footer{{font-size:13px;color:#68798a;margin-top:22px}}
@media(max-width:650px){{main{{padding:14px 12px 32px}}header,section{{padding:21px}}h1{{font-size:25px}}h2{{font-size:20px}}.facts{{grid-template-columns:1fr}}}}
</style></head><body><main>
<header><div class="eyebrow">002112 · 本轮只排查，新增训练 0 次</div><h1>问题会换地方，删掉某一部分不能解决</h1>
<p>已确认：各部分的表现会随时期变化，旧样本与当前持仓存在差异，2026年第三季度净值波动明显变大。尚未证明：旧资料造成了预测变差。</p></header>
<section><h2>1. 到底是谁把判断带偏了？</h2><p>将同一天的预测重新组合，暂时拿掉一部分，看整体多判对还是少判对。下表数字是“判对天数／总天数”。</p>
<div class="scroll"><table><thead><tr><th>怎么组合</th><th>2026年第二季度</th><th>2026年第三季度</th><th>全部424天</th></tr></thead><tbody>{table}</tbody></table></div>
<p><b>第二季度行情部分拖累较明显，第三季度信息部分拖累较明显。</b>但是看全部424天，拿掉任何一部分都会少判对。所以没有查出一个“永久没用、删掉就好”的部分。</p>
<p class="muted">这只是用已保存概率重新计算，剩余部分按原比例分配权重。没有重新训练，也没有据此删掉任何模型。信息部分含公告、行业背景及事件后的价格反应，不能把它的失误全归因于新闻。</p></section>
<section><h2>2. 旧经验与现在这只基金，确实不太一样</h2><p>以预测2026年9月30日所用的训练资料为例：</p><div class="facts">
<div class="fact"><div class="number">{old_share:.1%}</div><b>净值训练样本早于一年前</b><p>{last["history"]["training_count"]}个样本中，有{last["history"]["older_than_252_sessions"]}个早于252个交易日。最早追溯到2016年，现有训练没有按新旧程度另加权。</p></div>
<div class="fact"><div class="number">{mismatch_share:.1%}</div><b>信息、行情样本的披露持仓差别较大</b><p>{last["information"]["training_count"]}个样本中，{last["information"]["overlap_under_half_rows"]}个与预测时点已披露前十持仓的加权重合度不足一半。</p></div></div>
<p>持仓存档显示：2023年末前十持仓与2024年一季度前十持仓没有相同股票；此后也多次变化。这说明“只要是002112以前的数据，就同样适合现在”值得检验。</p>
<div class="callout"><b>不能直接说旧资料害了模型。</b>早期旧披露持仓阶段的样本，在信息和行情训练中一直是68个，占比反而从2025年初的{68 / first["information"]["training_count"]:.1%}降到截至2026年9月底的最后一次训练的{68 / last["information"]["training_count"]:.1%}。错误变多不能简单解释成“医药时期的资料越来越多”。</div>
<p class="muted">这里比较的是各日期当时已公开的前十持仓，按股票代码和权重归一化，不是基金真实每日完整持仓。更早的净值训练样本中，有1,807个无法用现有报告核对当时持仓，不能当作“完全不相似”。</p></section>
<section><h2>3. 第三季度环境变了，三部分都没有跟好</h2><p>在本地净值资料中，过去60天日涨跌的波动程度，其中位数从第二季度约3.03%升至第三季度约4.84%。第三季度65天里，有36天的部分净值指标超出了当时训练样本见过的范围。</p>
<div class="scroll"><table><thead><tr><th>单独看哪部分</th><th>第三季度判对天数</th><th>第三季度判涨天数</th></tr></thead><tbody>{branch_table}</tbody></table></div>
<p>实际65天中只有31天上涨，三部分却都更偏向判涨；合在一起只判对25天。这一阶段整体适应得不好。</p>
<p class="muted">“见到了以前少见的情况”是线索，仍不是原因证明。超出旧范围的36天，整体判对14天；其余29天判对11天，两组都差，不能把错误全归到极端波动。</p>
<details><summary>同样的旧资料，换一个时期的模型会怎样？</summary><p>把同一批60条旧输入交给不同日期保存的模型，不改变任何输入：行情部分在2025年初判涨37条，截至2026年9月底的最后一个模型判涨46条；信息部分从22条变成28条。这确认判断变化也来自模型随历史重训后的变化。</p><p class="muted">这60条旧资料曾参与训练，只用于观察模型变化，不能当作预测正确率，更不能证明新模型更好。</p></details></section>
<section><h2>4. 消息确实稀少，但没有突然断档</h2><div class="scroll"><table><thead><tr><th>时期</th><th>有相关公开新闻、政策的天数</th><th>有合格公司公告的天数</th><th>整体判对天数</th></tr></thead><tbody>{coverage_table}</tbody></table></div>
<p>第三季度相关消息覆盖增加了，表现却更差。因此问题不只是“没读到消息”，还包括这些材料是否真正能帮助判断明天涨跌。</p>
<p>已进入模型的行情和净值列，在这些评估日没有缺值；也未发现新出现的非空信息列因训练期全缺而被丢弃。信息列长期稀疏，仍是证据不足的问题，但本轮未发现近期突然大面积丢数据。</p></section>
<section><h2>排查后的下一步</h2><p><b>最值得做的对照是：保持信息内容和配比不动，比较“全部旧资料”“主要看近期资料”“更重视相近披露持仓的资料”。</b>这能直接检验旧经验是否用得不合适，而不是继续猜新闻该占多少。</p><p>本轮停在排查结果：没有新训练、没有改配比、没有替换现用模型。若后续对照仍不改善，就应承认现有材料尚未证明可靠的一日预测能力。</p>
<details><summary>查看逐日证据与验证文件</summary><ul>{links}</ul><p class="muted">回读69个历史模型，重算1,272行概率，最大差异为{verified["maximum_error"]:.2e}；5项统计口径检查通过。旧产物及已记录工作区文件保持不变。2025、2026均为已观察历史，原始历史首见存档仍不完整；本报告不是未见未来验证。</p></details></section>
<footer>本地只读排查 · {verified["at"]} · 保留证据不足和未知项，未作新模型选择。</footer></main></body></html>"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("x", encoding="utf-8") as handle:
        handle.write(page)
    old.write(
        root / "delivery.json",
        {
            "report": str(REPORT),
            "new_fits": 0,
            "model_changed": False,
            "next_hypothesis": "样本新旧及披露持仓相似性是否影响泛化；尚未通过新训练验证",
            "tests": 5,
            "verification": str(root / "verification.json"),
        },
    )
    print(old.json.dumps({"report": str(REPORT), "new_fits": 0}, ensure_ascii=False))


if __name__ == "__main__":
    main()
