"""002112 已冻结模型的零拟合诊断：分数分解是模型解释，不能当成因果结论。"""

from collections import Counter

import numpy as np

from app.services import fund_002112_training as original
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read

BASE_RUN = "002112-b9fcc6eb85cce6e19466db3a"


def distribution(rows, index):
    """按合格目标日期统计输入分布；仓位来自当时公开报告，不冒充当日实际持仓。"""
    values = [r["x"][index] for r in rows]
    return {"count": len(values), "min": min(values), "median": float(np.median(values)), "max": max(values)}


def margin_parts(model, values):
    """拆解上涨相对下跌的线性分数差；精确和等于 log(UP 分数 / DOWN 分数)。"""
    normalized = [(v - m) / s for v, m, s in zip(values, model["mean"], model["scale"], strict=True)]
    terms = [(up - down) * x for up, down, x in zip(model["coef"][2], model["coef"][0], normalized, strict=True)]
    intercept = model["intercept"][2] - model["intercept"][0]
    return {
        "intercept": intercept,
        "features": dict(zip(model["features"], terms, strict=True)),
        "margin": intercept + sum(terms),
    }


def diagnose(data, directory):
    """保留每一天及相对 A 的得失，不依据诊断移除困难日期或选择参照基金。"""
    models = {v: read(directory / "models" / f"{v}.json") for v in original.numerical.VARIANTS}
    forecasts = {v: read(directory / "predictions" / f"{v}.json") for v in models}
    target = [r for r in data["train"] if r["fund_code"] == "002112"]
    funds = {}
    for row, weight in zip(data["train"], data["weights"], strict=True):
        item = funds.setdefault(row["fund_code"], {"records": 0, "weight": 0.0, "flat": 0})
        item["records"] += 1
        item["weight"] += weight
        item["flat"] += row["actual_direction"] == "FLAT"
    for item in funds.values():
        item["share"] = item["weight"] / sum(data["weights"])
    daily = []
    for i, row in enumerate(data["development"]):
        parts = {v: margin_parts(m, row["x"][: len(m["features"])]) for v, m in models.items()}
        for rows in forecasts.values():
            if rows[i]["input_hash"] != digest(row):
                raise ValueError("OPT_DIAGNOSTIC_ROW_CHANGED")
        a, c = forecasts["NAV7"][i], forecasts["NAV7_HOLDINGS_MARKET"][i]
        ac, cc = a["direction"] == row["actual_direction"], c["direction"] == row["actual_direction"]
        pa, pc = parts["NAV7"], parts["NAV7_HOLDINGS_MARKET"]
        features = models["NAV7_HOLDINGS_MARKET"]["features"]
        delta = {
            "intercept": pc["intercept"] - pa["intercept"],
            "refitted_nav": sum(pc["features"][f] - pa["features"][f] for f in features[:7]),
            "holdings": sum(pc["features"][f] for f in features[7:16]),
            "market": sum(pc["features"][f] for f in features[16:]),
        }
        if abs(sum(delta.values()) - (pc["margin"] - pa["margin"])) > 1e-12:
            raise ValueError("OPT_DIAGNOSTIC_DECOMPOSITION")
        daily.append(
            {
                "target": row["target"],
                "actual": row["actual_direction"],
                "directions": {v: forecasts[v][i]["direction"] for v in models},
                "parts": parts,
                "C_minus_A": delta,
                "outcome": "both_correct" if ac and cc else "A_only" if ac else "C_only" if cc else "both_wrong",
                "report_age_days": row["exposure"]["report_age_days"],
                "reported_stock_weight": row["x"][13],
                "disclosed_stock_weight": row["x"][12],
                "full_disclosure": row["exposure"]["full_disclosure"],
            }
        )
    summaries = {}
    for name, rows in {
        "all": daily,
        "C_gained": [r for r in daily if r["outcome"] == "C_only"],
        "C_lost": [r for r in daily if r["outcome"] == "A_only"],
        "Q4": [r for r in daily if r["target"] >= "2024-10-01"],
    }.items():
        means = (
            {
                f: float(np.mean([r["parts"]["NAV7_HOLDINGS_MARKET"]["features"][f] for r in rows]))
                for f in models["NAV7_HOLDINGS_MARKET"]["features"]
            }
            if rows
            else {}
        )
        summaries[name] = {
            "dates": len(rows),
            "actual": dict(Counter(r["actual"] for r in rows)),
            "C_predicted": dict(Counter(r["directions"]["NAV7_HOLDINGS_MARKET"] for r in rows)),
            "C_mean_feature_margin": means,
            "mean_C_minus_A": {
                k: float(np.mean([r["C_minus_A"][k] for r in rows]))
                for k in ("intercept", "refitted_nav", "holdings", "market")
            }
            if rows
            else {},
        }
    return {
        "baseline_run": directory.name,
        "dataset_hash": digest(data),
        "fits": 0,
        "fund_weights": funds,
        "target_distribution": {
            name: {original.package.PLAN["features"][i]: distribution(rows, i) for i in (12, 13, 14, 15)}
            for name, rows in (("training", target), ("development", data["development"]))
        },
        "summaries": summaries,
        "daily": daily,
        "limitations": ["DESCRIPTIVE_NOT_CAUSAL", "2024_ALREADY_OBSERVED", "REPORTED_NOT_DAILY_ACTUAL_HOLDINGS"],
    }


def report(result):
    q4 = result["summaries"]["Q4"]
    lines = [
        "# 002112 错例、仓位变化和权重诊断",
        "",
        "本诊断零拟合；分数分解解释已有模型的计算，不证明因果。",
        "",
        "## 训练权重",
        "",
        "| 基金 | 训练记录 | 持平记录 | 总学习权重占比 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for code, value in result["fund_weights"].items():
        lines.append(f"| {code} | {value['records']} | {value['flat']} | {value['share']:.2%} |")
    lines += ["", "## 仓位与披露输入", "", "| 输入 | 自身训练中位数 | 2024 中位数 |", "| --- | ---: | ---: |"]
    dist = result["target_distribution"]
    for key in dist["training"]:
        lines.append(f"| {key} | {dist['training'][key]['median']:.4f} | {dist['development'][key]['median']:.4f} |")
    lines += [
        "",
        "股票仓位与披露覆盖分别统计。报告仓位只是当时最新公开报告的数值，不是预测日实际仓位。",
        "",
        "## 相对 A 的分数变化",
        "",
        "正值推向上涨，负值推向下跌；净值部分也重新拟合过，不能把全部差异归因于新增字段。",
        "",
        "| 范围 | 天数 | 截距变化 | 净值系数变化 | 持仓项 | 大盘项 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, item in result["summaries"].items():
        values = item["mean_C_minus_A"]
        lines.append(
            f"| {name} | {item['dates']} | "
            + " | ".join(f"{values[k]:.5f}" for k in ("intercept", "refitted_nav", "holdings", "market"))
            + " |"
        )
    ranked = sorted(q4["C_mean_feature_margin"].items(), key=lambda item: -abs(item[1]))[:6]
    lines += ["", "四季度 C 分数中绝对平均贡献最大的六项："]
    lines += [f"- {key}：{value:.5f}（正值偏上涨，负值偏下跌）" for key, value in ranked]
    lines += [
        "",
        "## 两个待检验假设",
        "",
        "1. 固定目标基金总权重为 50%，其余 50% 在参照家族间等分；不搜索多个比例。",
        "2. 保持家族等权，在 C 原 20 项后增加 4 项‘报告股票仓位 × 大盘涨跌’，单独检验仓位对市场信号的调节。",
        "两项不合并；保留原净值 A 和大盘 C 对照，名单/日期/三分类/完整性门槛不变。",
        "2024 错例已知，下一轮仍是探索开发；不得宣称独立验证。",
        "",
    ]
    return "\n".join(lines)
