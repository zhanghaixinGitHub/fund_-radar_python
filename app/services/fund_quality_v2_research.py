"""真实产品资料的隔离校准和诊断工具，不给出正式验收或激活结论。

每产品仅取一份额、一个校准期末；验证产品及验证月份同时隔离。样本不足
返回精确缺口，不创建虚拟产品，不将25个重叠窗口算成25个独立产品。
"""

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from statistics import correlation, covariance, variance

from app.services.fund_quality_v2_anchor_rules import rule_points
from app.services.fund_quality_v2_materials import build_cohort
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_quality_v2 import (
    HIGHER_BETTER,
    VERSION,
    calculate_candidate,
    candidate_policy,
    metric_names,
)
from app.services.fund_rating_rules import decimal

CALIBRATOR = "PRODUCT_EMPIRICAL_MIDRANK_WITH_RULE_STATES_2"


def representatives(packets: list[dict]) -> list[dict]:
    """先固定每产品最新期末，再固定代码最小的完整份额；不看得分挑样本。"""
    selected = {}
    for packet in sorted(packets, key=lambda p: (p["as_of_date"], p["fund_code"]), reverse=True):
        key = packet["product_id"]
        if key not in selected or selected[key]["as_of_date"] == packet["as_of_date"]:
            selected[key] = packet
    return [selected[key] for key in sorted(selected)]


def empirical_anchors(packets: list[dict], kind: str) -> dict:
    """连续指标用独立产品中秩，明确定义的状态用冻结候选语义，不伪造观测端点。"""
    reps = representatives(packets)
    anchors = {}
    config = candidate_policy(kind)
    for name in sorted(metric_names(candidate_policy(kind))):
        counts = Counter(decimal(p["metrics"][name]["value"]) for p in reps)
        if name in config["rule_defined_anchors"]["metrics"]:
            anchors[name] = rule_points(name, list(counts))
            continue
        if len(counts) < 2:
            raise ValueError("QUALITY_V2_CONSTANT_CALIBRATION_METRIC:" + name)
        previous = 0
        points = []
        for value, count in sorted(counts.items()):
            rank = 100 * (Decimal(previous) + Decimal(count) / 2) / len(reps)
            points.append([str(value), str(rank if name in HIGHER_BETTER else 100 - rank)])
            previous += count
        anchors[name] = points
    return anchors


def correlations(rows: list[dict[str, str]]) -> dict:
    """只做描述统计；恒定列、样本不足明确为空，不把相关性当因果或无用证明。"""
    if len(rows) < 3:
        return {"product_count": len(rows), "pairs": [], "reason": "FEWER_THAN_THREE_PRODUCTS"}
    columns = {key: [float(r[key]) for r in rows] for key in rows[0]}
    pairs = []
    for i, left in enumerate(sorted(columns)):
        for right in sorted(columns)[i + 1 :]:
            a, b = columns[left], columns[right]
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "pearson": correlation(a, b) if len(set(a)) > 1 and len(set(b)) > 1 else None,
                }
            )
    return {"product_count": len(rows), "pairs": pairs, "interpretation": "DESCRIPTIVE_NOT_INDEPENDENCE_PROOF"}


def partial_observations(packets: list[dict]) -> dict:
    """已核原始字段的真实观察；逐对按产品去重，不将不完整产品变成校准样本。

    由experiment完成全部原件重建核验后调用。每个产品每月先取代码最小的
    可用份额，再取共同有值的产品计算相关性；明确保留每对的实际产品数。
    """
    supported = {"ACTIVE_EQUITY", "PASSIVE_INDEX", "INDEX_ENHANCED", "GLOBAL_ACTIVE_EQUITY"}
    groups = defaultdict(list)
    for packet in packets:
        if packet.get("product_id") and packet.get("kind") in supported:
            groups[packet["kind"], packet["as_of_date"]].append(packet)
    results = []
    for (kind, month), members in sorted(groups.items()):
        metrics = sorted(metric_names(candidate_policy(kind)))
        values = {metric: {} for metric in metrics}
        for packet in sorted(members, key=lambda p: p["fund_code"]):
            for metric in metrics:
                if metric in packet["metrics"]:
                    values[metric].setdefault(packet["product_id"], packet["metrics"][metric]["value"])
        pairs = []
        for i, left in enumerate(metrics):
            for right in metrics[i + 1 :]:
                common = sorted(values[left].keys() & values[right].keys())
                if len(common) < 3:
                    continue
                rows = [{left: values[left][p], right: values[right][p]} for p in common]
                pairs.append(correlations(rows) | {"left": left, "right": right, "product_ids": common})
        results.append(
            {
                "kind": kind,
                "as_of_date": month,
                "products": len({p["product_id"] for p in members}),
                "metric_product_counts": {k: len(v) for k, v in values.items()},
                "pairwise": pairs,
            }
        )
    # 历史原文观察以(产品、报告期、原件哈希)去重；A/C不能翻倍，跨月不计独立产品。
    historical = {}
    for packet in sorted(packets, key=lambda p: p["fund_code"]):
        if not packet.get("product_id"):
            continue
        for report in packet.get("parsed_facts", {}).get("historical_asset_reports", {}).get("reports", []):
            key = packet["product_id"], report["observed_on"], tuple(report["source_hashes"])
            historical.setdefault(key, {"product_id": packet["product_id"], "fund_code": packet["fund_code"], **report})
    return {
        "status": "PARTIAL_RAW_OBSERVATIONS_ONLY",
        "groups": results,
        "historical_report_observations": list(historical.values()),
        "limits": [
            "字段相关性不能证明总分去重、权重或类型公平性合格",
            "历史月份不增加独立产品数量",
            "不同份额费用按固定代码选择，仅描述该情景，不代表所有份额费用相同",
            "完整八维不足时不生成锚点、总分、等级、权重敏感性或新旧等级差异",
        ],
    }


def diagnostics(packets: list[dict], anchors: dict, kind: str) -> dict:
    """分月去重，保留每只算例、关键短板上限和逐维±20%敏感性，不预设等级方向。"""
    config = candidate_policy(kind)
    months = defaultdict(list)
    for packet in packets:
        months[packet["as_of_date"]].append(packet)
    output = []
    for month, members in sorted(months.items()):
        reps = representatives(members)
        results, dimension_rows, metric_rows, contributions = [], [], [], []
        for packet in reps:
            values = {k: v["value"] for k, v in packet["metrics"].items()}
            result = calculate_candidate(values, config, anchors)
            dimension_rows.append(result["dimensions"])
            metric_rows.append(values)
            weighted = {
                key: float(decimal(result["dimensions"][key]) * weight / 100)
                for key, weight in config["weights"].items()
            }
            contributions.append(weighted)
            scenarios = []
            for dimension in config["weights"]:
                for factor in (Decimal("0.8"), Decimal("1.2")):
                    changed = deepcopy(config)
                    weights = {k: Decimal(w) * (factor if k == dimension else 1) for k, w in config["weights"].items()}
                    changed["weights"] = {k: w * 100 / sum(weights.values()) for k, w in weights.items()}
                    outcome = calculate_candidate(values, changed, anchors)
                    scenarios.append(
                        {
                            "dimension": dimension,
                            "factor": str(factor),
                            "score": outcome["score"],
                            "grade": outcome["grade"],
                            "grade_cap_reasons": outcome["grade_cap_reasons"],
                        }
                    )
            results.append(
                {
                    "fund_code": packet["fund_code"],
                    "product_id": packet["product_id"],
                    "packet_hash": packet["hash"],
                    "baseline": result,
                    "weight_sensitivity": scenarios,
                }
            )
        totals = [sum(r.values()) for r in contributions]
        effective = None
        if len(totals) >= 3 and len(set(totals)) > 1:
            effective = {
                k: covariance([r[k] for r in contributions], totals) / variance(totals) for k in contributions[0]
            }
        output.append(
            {
                "as_of_date": month,
                "product_count": len(reps),
                "results": results,
                "metric_correlations": correlations(metric_rows),
                "dimension_correlations": correlations(dimension_rows),
                "variance_contribution": effective,
            }
        )
    return {
        "kind": kind,
        "months": output,
        "independent_acceptance": "NOT_PERFORMED",
        "limits": [
            "相关指标组扰动及分类公平性需要逐案解释",
            "月份内产品去重不消除跨月份相关性",
            "协方差贡献描述当前样本分布，可为负；不代表因果贡献或最优权重",
        ],
    }


def experiment(packets: list[dict], assignments: dict[str, str], cutoff: date, root: Path, frozen: datetime) -> dict:
    """冻结真实校准和验证输入并诊断；没有完整资料时交付失败清单，不输出伪锚点。"""
    cohort = build_cohort(packets, assignments, root)
    observations = partial_observations(packets)
    groups = defaultdict(lambda: {"calibration": [], "validation": []})
    gaps = list(cohort["gaps"])
    for packet in packets:
        if not packet["raw_metrics_complete"] or packet["product_id"] not in assignments:
            continue
        partition = assignments[packet["product_id"]]
        if (partition == "calibration") != (packet["as_of_date"] <= str(cutoff)):
            gaps.append({"fund_code": packet["fund_code"], "reason": "CALIBRATION_VALIDATION_TIME_OVERLAP"})
            continue
        if datetime.fromisoformat(packet["frozen_at"]) > frozen:
            raise ValueError("QUALITY_V2_EXPERIMENT_FUTURE_PACKET")
        groups[packet["kind"]][partition].append(packet)
    results = []
    for kind in sorted(
        {
            p["kind"]
            for p in packets
            if p.get("kind") in {"ACTIVE_EQUITY", "PASSIVE_INDEX", "INDEX_ENHANCED", "GLOBAL_ACTIVE_EQUITY"}
        }
    ):
        train, validation = groups[kind]["calibration"], groups[kind]["validation"]
        products = representatives(train)
        config = candidate_policy(kind)
        counts = {"calibration_products": len(products), "validation_products": len(representatives(validation))}
        if len(products) < config["minimum_calibration_products"] or not validation:
            results.append(
                {
                    "kind": kind,
                    "status": "NOT_READY",
                    **counts,
                    "reason": "REAL_CALIBRATION_AND_LATER_HOLDOUT_PRODUCTS_INSUFFICIENT",
                }
            )
            continue
        try:
            anchors = empirical_anchors(products, kind)
        except ValueError as exc:
            results.append({"kind": kind, "status": "NOT_READY", **counts, "reason": str(exc)})
            continue
        calibration = {
            "version": VERSION,
            "calibrator": CALIBRATOR,
            "kind": kind,
            "policy_hash": digest(config),
            "cutoff": str(cutoff),
            "frozen_at": frozen.isoformat(),
            "product_ids": [p["product_id"] for p in products],
            "validation_product_ids": sorted({p["product_id"] for p in validation}),
            "source_hashes": sorted({h for p in products for h in p["sources"]}),
            "calibration_packets": products,
            "validation_packets": validation,
            "anchors": anchors,
            "anchor_derivation": {
                name: {
                    "method": "RULE_DEFINED_NOT_EMPIRICAL_ENDPOINTS"
                    if name in config["rule_defined_anchors"]["metrics"]
                    else "EMPIRICAL_PRODUCT_MIDRANK",
                    "observed_products": len(products),
                    "observed_distinct_values": len({p["metrics"][name]["value"] for p in products}),
                    "rule_points_counted_as_products": 0,
                }
                for name in anchors
            },
            "independent_review": None,
            "statistical_sufficiency": "NOT_ESTABLISHED",
        }
        calibration["hash"] = digest(calibration)
        results.append(
            {
                "kind": kind,
                "status": "TRIAL_ONLY_NOT_ACCEPTED",
                **counts,
                "calibration": calibration,
                "calibration_diagnostics": diagnostics(train, anchors, kind),
                "validation_diagnostics": diagnostics(validation, anchors, kind),
            }
        )
    return {
        "version": VERSION,
        "checked_at": frozen.isoformat(),
        "cutoff": str(cutoff),
        "status": "ISOLATED_RESEARCH_NOT_ACCEPTANCE",
        "groups": results,
        "gaps": gaps,
        "partial_observations": observations,
        "formal_ratings_changed": 0,
    }
