"""基金质量第二版的独立计算器：先验证指标与校准证据，不覆盖现行评级。

八维仍全部计分；Calmar退出总分，目标执行使用独立的配置或复制偏离证据。
候选权重不是行业标准。没有独立校准、完整来源和关键字段时返回缺口，
不能把第一版分项分数当作第二版原始输入，也不能由本模块激活正式方法。
"""

from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, localcontext
from pathlib import Path

from app.services.fund_quality_v2_anchor_rules import rule_anchor_specs
from app.services.fund_quality_v2_contract import contract_hash
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_public_policy import interpolate
from app.services.fund_rating_rules import DIMENSIONS, decimal, grade, stored_score

VERSION = "FUND_QUALITY_V2"
KINDS = frozenset({"ACTIVE_EQUITY", "INDEX_ENHANCED", "GLOBAL_ACTIVE_EQUITY", "PASSIVE_INDEX"})
KEY_DIMENSIONS = ("risk", "efficiency", "management", "liquidity")
HIGHER_BETTER = frozenset(
    {
        "relative_return_1y",
        "relative_return_3y",
        "rolling_target_success",
        "monthly_target_success",
        "governance_state",
        "key_person_coverage",
        "team_months",
    }
)
LABELS = {**DIMENSIONS, "efficiency": "目标执行与运作效率"}


def candidate_policy(kind: str) -> dict:
    """给出可审查候选配置；锚点必须另附真实校准证据，不内置通用收益满分线。"""
    if kind not in KINDS:
        raise ValueError("QUALITY_V2_KIND_UNSUPPORTED")
    passive = kind == "PASSIVE_INDEX"
    weights = (10, 10, 25, 10, 10, 10, 10, 15) if passive else (15, 20, 10, 10, 15, 10, 10, 10)
    terms = {
        "return": [("relative_return_1y", 30), ("relative_return_3y", 70)],
        "risk": [("extra_drawdown", 70), ("extra_downside", 30)],
        "efficiency": [("tracking_error", 100)]
        if passive
        else [("allocation_deviation", 60), ("mandate_deviation", 40)],
        "stability": [("rolling_target_success", 70), ("monthly_target_success", 30)],
        "management": [("governance_state", 50), ("key_person_coverage", 30), ("team_months", 20)],
        "holdings": [("issuer_replication_deviation", 50), ("industry_replication_deviation", 50)]
        if passive
        else [("issuer_excess_concentration", 50), ("industry_excess_concentration", 50)],
        "liquidity": [
            ("restricted_ratio", 40),
            ("cash_target_deviation", 30),
            ("holder_pressure_band", 20),
            ("capacity_pressure", 10),
        ],
        "cost": [("cost_1y", 100)],
    }
    return {
        "version": VERSION,
        "raw_contract_hash": contract_hash(),
        "kind": kind,
        "status": "CANDIDATE_REQUIRES_INDEPENDENT_CALIBRATION",
        "rule_defined_anchors": rule_anchor_specs(),
        "weights": dict(zip(DIMENSIONS, weights, strict=True)),
        "terms": terms,
        "critical_dimensions": list(KEY_DIMENSIONS),
        "weak_limit": "50",
        "excellent_minimum": "70",
        "history_months": 36,
        "minimum_calibration_products": 30,
    }


def metric_names(config: dict) -> set[str]:
    """一个计算指标只有一个主要计分归属，共享底层净值不被伪称为独立证据。"""
    names = [name for terms in config["terms"].values() for name, _ in terms]
    if len(names) != len(set(names)):
        raise ValueError("QUALITY_V2_DUPLICATE_METRIC")
    return set(names)


def validate_calibration(calibration: dict, config: dict, as_of: date) -> dict:
    """只接纳有独立验证集、原件摘要和明确分类的校准清单。

    产品数是当前候选方法的准入要求，不是统计显著性保证。参数冻结时间是
    真实形成方法的时间，历史重建不能称为当时已经发表的评分。
    """
    value = deepcopy(calibration)
    supplied_hash = value.pop("hash", None)
    if supplied_hash != digest(value) or value.get("version") != VERSION:
        raise ValueError("QUALITY_V2_CALIBRATION_HASH")
    if value.get("kind") != config["kind"] or value.get("policy_hash") != digest(config):
        raise ValueError("QUALITY_V2_CALIBRATION_SCOPE")
    if date.fromisoformat(value["cutoff"]) > as_of or not value.get("source_hashes"):
        raise ValueError("QUALITY_V2_CALIBRATION_FUTURE_OR_SOURCE")
    if datetime.fromisoformat(value["frozen_at"]).tzinfo is None:
        raise ValueError("QUALITY_V2_CALIBRATION_TIME")
    products = value.get("product_ids", [])
    validation = value.get("validation_product_ids", [])
    if (
        len(set(products)) < config["minimum_calibration_products"]
        or len(products) != len(set(products))
        or not validation
        or set(products) & set(validation)
    ):
        raise ValueError("QUALITY_V2_CALIBRATION_NOT_INDEPENDENT")
    anchors = value.get("anchors", {})
    if set(anchors) != metric_names(config):
        raise ValueError("QUALITY_V2_CALIBRATION_METRICS")
    for name, points in anchors.items():
        if len(points) < 2:
            raise ValueError("QUALITY_V2_CALIBRATION_ANCHORS")
        xs = [decimal(x) for x, _ in points]
        ys = [decimal(y) for _, y in points]
        if xs != sorted(set(xs)) or any(not 0 <= y <= 100 for y in ys):
            raise ValueError("QUALITY_V2_CALIBRATION_ANCHORS")
        if len(set(ys)) < 2 or ys != sorted(ys, reverse=name not in HIGHER_BETTER):
            raise ValueError("QUALITY_V2_CALIBRATION_NOT_MONOTONIC")
        fixed = config["rule_defined_anchors"]["metrics"].get(name)
        if fixed and [(decimal(x), decimal(y)) for x, y in points] != [
            (decimal(x), decimal(y)) for x, y in fixed["points"]
        ]:
            raise ValueError("QUALITY_V2_RULE_ANCHORS_CHANGED")
    return value


def assess(packet: dict, calibration: dict | None, *, root: Path | None = None) -> dict:
    """返回独立试算或明确缺口；不查询/写入正式批次，不复用旧版分项凑新评级。

    packet.metrics 每项包含原始值、来源哈希、观察日；sources 另保留公开和
    实际取得时间。取得时间可晚于历史考察日，但不得晚于本次真实冻结时刻。
    """
    config = candidate_policy(packet["kind"])
    as_of = date.fromisoformat(packet["as_of_date"])
    frozen = datetime.fromisoformat(packet["frozen_at"])
    if frozen.tzinfo is None or as_of > frozen.date():
        raise ValueError("QUALITY_V2_FREEZE_TIME")
    reasons = []
    if packet.get("history_months", 0) < config["history_months"]:
        reasons.append("HISTORY_INCOMPLETE")
    # 标志和自填来源摘要不能证明原件。正式试算须重新读取文件、重建事实和指标。
    if root is None or not packet.get("source_bundle"):
        reasons.append("ORIGINALS_REBUILD_REQUIRED")
    else:
        from app.services.fund_quality_v2_materials import verify_packet

        verify_packet(packet, root)
    if not packet.get("benchmark_evidence", {}).get("expression"):
        reasons.append("BENCHMARK_UNVERIFIED")
    if not packet.get("classification_evidence", {}).get("product_id"):
        reasons.append("CLASSIFICATION_UNVERIFIED")
    sources = packet.get("sources", {})
    metrics = packet.get("metrics", {})
    values = {}
    for name in sorted(metric_names(config)):
        evidence = metrics.get(name)
        if not evidence:
            reasons.append("MISSING:" + name)
            continue
        refs = evidence.get("source_hashes", [])
        if not refs or any(ref not in sources for ref in refs):
            reasons.append("SOURCE_MISSING:" + name)
            continue
        if date.fromisoformat(evidence["observed_on"]) > as_of:
            raise ValueError("QUALITY_V2_FUTURE_OBSERVATION")
        for ref in refs:
            source = sources[ref]
            received = datetime.fromisoformat(source.get("acquired_at") or source["received_at"])
            published = source.get("published_on")
            if received.tzinfo is None or received > frozen or published and date.fromisoformat(published) > as_of:
                raise ValueError("QUALITY_V2_SOURCE_TIME")
        values[name] = decimal(evidence["value"])
    if calibration is None:
        reasons.append("INDEPENDENT_CALIBRATION_REQUIRED")
    else:
        calibrated = validate_calibration(calibration, config, as_of)
        if datetime.fromisoformat(calibrated["frozen_at"]) > frozen:
            raise ValueError("QUALITY_V2_CALIBRATION_NOT_YET_KNOWN")
        # 校准产品身份来自逐包原件，不能靠30个字符串、source_hashes或非空评语通关。
        samples = calibrated.get("calibration_packets", [])
        holdouts = calibrated.get("validation_packets", [])
        if root is None or not samples or not holdouts:
            reasons.append("CALIBRATION_ORIGINAL_PACKETS_REQUIRED")
        else:
            from app.services.fund_quality_v2_materials import verify_packet

            for sample in samples + holdouts:
                verify_packet(sample, root)
                if (
                    not sample["raw_metrics_complete"]
                    or sample["kind"] != config["kind"]
                    or sample["as_of_date"] > str(as_of)
                    or datetime.fromisoformat(sample["frozen_at"]) > frozen
                ):
                    raise ValueError("QUALITY_V2_CALIBRATION_SAMPLE_SCOPE")
            if any(
                p["as_of_date"] > calibrated["cutoff"]
                or datetime.fromisoformat(p["frozen_at"]) > datetime.fromisoformat(calibrated["frozen_at"])
                for p in samples
            ) or any(p["as_of_date"] <= calibrated["cutoff"] for p in holdouts):
                raise ValueError("QUALITY_V2_CALIBRATION_TIME_ISOLATION")
            if {p["product_id"] for p in samples} != set(calibrated["product_ids"]) or {
                p["product_id"] for p in holdouts
            } != set(calibrated["validation_product_ids"]):
                raise ValueError("QUALITY_V2_CALIBRATION_PRODUCT_PROOF")
            from app.services.fund_quality_v2_research import CALIBRATOR, empirical_anchors

            if (
                calibrated.get("calibrator") != CALIBRATOR
                or calibrated["anchors"] != empirical_anchors(samples, config["kind"])
                or set(calibrated["source_hashes"]) != {h for p in samples for h in p["sources"]}
            ):
                raise ValueError("QUALITY_V2_CALIBRATION_ORIGINAL_ANCHORS")
    if reasons:
        return {"status": "NOT_READY", "version": VERSION, "reasons": reasons, "score": None, "grade": None}
    result = calculate_candidate(values, config, calibrated["anchors"], packet.get("material_event_pending", False))
    return result | {
        "status": "REVIEW_REQUIRED" if result["status"] == "REVIEW_REQUIRED" else "TRIAL_ONLY",
        "policy_hash": digest(config),
        "calibration_hash": calibration["hash"],
        "independent_acceptance": "NOT_PERFORMED",
    }


def calculate_candidate(values: dict, config: dict, anchors: dict, material_event_pending: bool = False) -> dict:
    """纯算术内核，仅供隔离运算验证；调用它不构成原件、校准或正式评级验收。"""
    with localcontext() as context:
        context.prec = 40
        dimensions = {
            key: stored_score(
                sum(interpolate(decimal(values[name]), anchors[name]) * weight / 100 for name, weight in terms)
            )
            for key, terms in config["terms"].items()
        }
        score = stored_score(sum(dimensions[k] * Decimal(config["weights"][k]) / 100 for k in DIMENSIONS))
    raw_grade = grade(score)
    final_grade, caps = raw_grade, []
    weak = [key for key in KEY_DIMENSIONS if dimensions[key] < decimal(config["weak_limit"])]
    if weak and raw_grade in {"GOOD", "EXCELLENT"}:
        final_grade = "AVERAGE"
        caps.extend("KEY_DIMENSION_WEAK:" + key for key in weak)
    elif raw_grade == "EXCELLENT":
        below = [key for key in KEY_DIMENSIONS if dimensions[key] < decimal(config["excellent_minimum"])]
        if below:
            final_grade = "GOOD"
            caps.extend("EXCELLENT_FLOOR:" + key for key in below)
    if material_event_pending:
        return {
            "status": "REVIEW_REQUIRED",
            "version": VERSION,
            "reasons": ["MATERIAL_EVENT_PENDING"],
            "score": None,
            "grade": None,
        }
    return {
        "status": "ARITHMETIC_ONLY_NOT_VALIDATED",
        "version": VERSION,
        "policy_hash": digest(config),
        "score": str(score),
        "raw_grade": raw_grade,
        "grade": final_grade,
        "grade_cap_reasons": caps,
        "dimensions": {key: str(value) for key, value in dimensions.items()},
    }
