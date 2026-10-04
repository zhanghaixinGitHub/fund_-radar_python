"""002112 持仓与正文优化的共同边界：停牌保留未知，旧研究只读，新试验独立保存。"""

from __future__ import annotations

import argparse
import bisect
import math
from collections import Counter
from datetime import date, datetime

from scripts import fund_002112_price_sources_v1 as price
from scripts import fund_002112_recent_bodies_v1 as previous

io = previous.io
ROOT = io.RESEARCH / "semantic-holdings-optimization/20261001-v1"
OLD = previous.ROOT
H_NAMES = ["observed_nav_return_1d", "observed_nav_return_5d", "observed_up_nav_weight",
           "observed_amount_change_20d", "disclosed_concentration", "disclosed_nav_weight",
           "reported_stock_nav_weight", "report_age_days", "full_disclosure",
           "return_1d_covered_nav_weight", "return_5d_covered_nav_weight", "amount_covered_nav_weight",
           "suspended_nav_weight", "unexplained_missing_nav_weight"]


def plan():
    path = ROOT / "plan.json"
    if path.exists():
        return io.read(path)
    value = {
        "at": io.now(), "version": "SEMANTIC_HOLDINGS_OPTIMIZATION_V1",
        "authorization": "用户要求按已讨论的全部五步执行完毕",
        "steps": ["停牌与行情缺口", "100篇正文抽样核验", "相关新闻政策及事实提取",
                  "固定四组2025比较", "2026暴露审计与固定方案验证"],
        "candidates": ["BASE", "HOLDINGS", "FACTS", "HOLDINGS_FACTS"],
        "recipe": {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 10, "random_state": 0},
        "cadence": 20, "maximum_supervised_fits": 100, "maximum_replay_fits": 1,
        "labels": "成熟时刻严格早于每个更新点；同一下一交易日预测口径",
        "selection": "2025开发集正确数降序，平局优先较少特征；2026不再用于选择",
        "public_requests_max": 4400, "llm_requests_max": 12000, "llm_attempts_per_document": 2,
        "llm_provider": "EXISTING_DEEPSEEK_CONFIGURATION", "llm_sample_count": 100,
        "llm_source_limit_characters": 10000, "llm_output_max_tokens": 1800,
        "sample_gate": "机器校验原文引用、日期数值出处；100篇逐项审阅，错误字段隔离，不伪造通过",
        "news_scope": "既有医保局政策/动态/媒体目录2024至2026-09-29及固定技术产业官方来源",
        "event_window_sessions": [1, 5, 20], "duplicate": "相同来源/正文及原文事实指纹仅首次计入",
        "holdings_missing": "分指标保留可观察净资产权重贡献；未知部分单独保留，不归一、不填零收益",
        "independence": "旧2026成绩涉及的日期明确标记已暴露；新增未参与选择日期单列样本量",
        "no_production_write": True, "adoption": False, "commit_or_push": False,
    }
    io.save(path, value)
    return value


def partial_holdings(report, stocks, sessions, cutoff, suspension_events):
    """按指标计算已知部分，缺一只股票不删除其他股票；结果不是完整基金收益。

    一日收益只需最后一天，五日需完整五日，成交活跃度需21日。每个贡献均保留
    原净资产权重分母，并附带覆盖权重。没有任何有效股票时该指标为None。
    停牌状态只依据当时已公开的开始/复牌信息，不能借未来复牌公告提前结束停牌。
    """
    if report is None:
        return [None] * len(H_NAMES), {"reason": "NO_AVAILABLE_REPORT"}
    instant = datetime.fromisoformat(cutoff)
    end = bisect.bisect_left(sessions, cutoff[:10])
    days = sessions[end - 21:end]
    values = [0.0] * 4
    covered = [0.0] * 3
    suspended = unknown = 0.0
    details = []
    for h in report["holdings"]:
        weight = float(h["nav_weight_pct"]) / 100
        if weight <= 0:
            continue
        code = h["stock_code"]
        qs = [stocks.get(day, {}).get(code) for day in days]
        valid = [q is not None and previous.holdings.quote_available(q, day, sessions, instant)
                 for q, day in zip(qs, days, strict=True)]
        missing = [day for day, ok in zip(days, valid, strict=True) if not ok]
        if valid[-1] and math.isfinite(float(qs[-1]["pct_chg"])):
            values[0] += weight * qs[-1]["pct_chg"] / 100
            values[2] += weight if qs[-1]["pct_chg"] > 0 else 0
            covered[0] += weight
        if all(valid[-5:]):
            values[1] += weight * (math.prod(1 + q["pct_chg"] / 100 for q in qs[-5:]) - 1)
            covered[1] += weight
        if all(valid):
            mean = sum(q["amount"] for q in qs[:-1]) / 20
            if mean > 0:
                values[3] += weight * (qs[-1]["amount"] / mean - 1)
                covered[2] += weight
        if missing:
            justified = []
            for day in missing:
                # 日期在历史窗口内，停牌信息可晚于该日但不能晚于当前预测时刻。
                event = next((e for e in suspension_events if e["code"] == code
                              and e["start_available_at"] <= cutoff and day >= e["start_date"]
                              and (e["resume_available_at"] > cutoff or day < e["resume_date"])), None)
                justified.append(event is not None)
            if not all(justified):
                unknown += weight
            active = any(e["code"] == code and e["start_available_at"] <= cutoff
                         and e["start_date"] <= days[-1]
                         and (e["resume_available_at"] > cutoff or days[-1] < e["resume_date"])
                         for e in suspension_events)
            suspended += weight if active else 0
            details.append({"code": code, "nav_weight": weight, "missing_days": missing,
                            "all_missing_explained_by_public_suspension": all(justified),
                            "suspension_known_at_cutoff": active})
    for feature, coverage in ((0, 0), (1, 1), (2, 0), (3, 2)):
        if covered[coverage] == 0:
            values[feature] = None
    values += [sum(w ** 2 for w in previous.weights(report).values()),
               float(report["disclosed_nav_pct"]) / 100, float(report["stock_nav_pct"]) / 100,
               float((instant.date() - date.fromisoformat(report["report_end"])).days),
               float(report["full_stock_disclosure"]), *covered, suspended, unknown]
    return values, {"report_hash": report["raw"]["sha256"], "incomplete_stocks": details}


def prepare():
    if (ROOT / "holdings-audit.json").exists():
        return io.read(ROOT / "holdings-audit.json")
    plan()
    bundle = io.read(previous.BUNDLE)
    reports = io.read(OLD / "reports.json")
    # 2026报告沿用既有验证包的更晚可用时点。不会把报告期末日期当成公开日期。
    known = {r["raw"]["sha256"] for r in reports}
    reports += [r for r in bundle["reports"] if r["report_end"] >= "2025-12-31" and r["raw"]["sha256"] not in known]
    io.save(ROOT / "reports.json", reports)
    suspension_path = io.RESEARCH / "recent-input-completion/20260930-v1/suspension-evidence.json"
    evidence = io.read(suspension_path)
    io.save(ROOT / "suspension-evidence.json", evidence)
    rows = previous.read_lines(OLD / "base-inputs.jsonl")
    old_days = previous.read_lines(io.RESEARCH / "next-trading-day-optimization/20260930-v1/inputs.jsonl")
    for row in old_days:
        if not row["target"].startswith("2026"):
            continue
        base = row["as_of"][:10]
        nav_end, _, n = previous.holdings.choose_nav_window(bundle["sessions"], bundle["nav"], base)
        pos = bundle["sessions"].index(nav_end)
        days = bundle["sessions"][pos - 60:pos + 1]
        ne = price.nav_extra([float(bundle["nav"][d]["unit_nav"]) for d in days])
        assert n == row["groups"]["N"]
        rows.append({k: row[k] for k in ("as_of", "base", "target", "session_index", "label_mature_at")} |
                    {"groups": {"N": n, "NE": ne}})
    audit = []
    for row in rows:
        report = previous.choose_report(reports, row["as_of"])
        h, proof = partial_holdings(report, bundle["stocks"], bundle["sessions"], row["as_of"], evidence["events"])
        row["H"] = h
        audit.append({"target": row["target"], "as_of": row["as_of"], **proof})
    io.save_lines(ROOT / "numeric-inputs.jsonl", rows)
    io.save_lines(ROOT / "holdings-lineage.jsonl", audit)
    old = previous.read_lines(OLD / "daily-lineage.jsonl")
    affected = {r["target"] for r in old if r["holding_missing"]}
    gaps = [r for r in audit if r["target"] in affected]
    result = {"at": io.now(), "old_group_missing_days": len(affected),
              "recovered_partial_input_days": sum(all(v is not None for v in r["H"][:4])
                                                   for r in rows if r["target"] in affected),
              "old_missing_explanation": dict(Counter(x["code"] for r in gaps for x in r["incomplete_stocks"])),
              "unexplained_missing_days": sum(any(not x["all_missing_explained_by_public_suspension"]
                                                  for x in r.get("incomplete_stocks", [])) for r in audit),
              "rows": len(rows), "feature_names": H_NAMES, "zero_fill_suspended_returns": False,
              "source_hashes": {str(p): io.sha(p) for p in (previous.BUNDLE, suspension_path, OLD / "reports.json")}}
    io.save(ROOT / "holdings-audit.json", result)
    return result


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(io.canonical(prepare()), flush=True)
