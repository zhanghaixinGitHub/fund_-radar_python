"""R5八维计算：复用已冻结经济差距公式，正式发布由上层事务负责。

同行位置保留为解释信息。新点值关注差距大小、合同目标与实际成本；参数是
冻结的规范性候选，不是已完成校准的行业标准。历史状态项仍从原始事实重算。
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path
from statistics import median

from app.services.fund_rating_r5_rules import check_refs, continuity_score, holders_score, payment_score
from app.services.fund_rating_reviewed_quality import _node, policy
from app.services.material_content import content_digest, source_bytes, source_exists

D = Decimal
RELATIVE_NONNEGATIVE = {
    "extra_drawdown",
    "extra_downside",
    "issuer_concentration",
    "industry_concentration",
    "market_capacity",
}


def number(value) -> Decimal:
    """拒绝缺失、布尔值和非有限数字；未知资料不能由数值转换默认为0。"""
    if value is None or isinstance(value, bool):
        raise ValueError("MISSING_NUMERIC_OBSERVATION")
    result = D(str(value))
    if not result.is_finite():
        raise ValueError("NONFINITE_OBSERVATION")
    return result


def clamp(value: Decimal) -> Decimal:
    return max(D(0), min(D(100), value))


def peer_gap_score(target, peers, *, neutral=75, amplitude=25) -> Decimal:
    """非负损失/暴露的相对差距；没有把参照最大最小值映射成质量0/100。

    分母为双方实际量级之和，所以参照中新增一个远端极值不直接改写量尺。
    同行中位数仅表达相对交付中性，不证明绝对安全。双方都0没有相对优势，取中性。
    """
    x = number(target)
    values = [number(v) for v in peers]
    if not values or x < 0 or any(v < 0 for v in values):
        raise ValueError("NONNEGATIVE_PEERS_REQUIRED")
    m = median(values)
    gap = D(0) if x + m == 0 else (m - x) / (m + x)
    return clamp(number(neutral) + number(amplitude) * gap)


def signed_gap_score(delta, unit, *, neutral=75, amplitude=25) -> Decimal:
    """有符号差距平滑换分；单位与经济幅度必须显式提供，不能从目标结果拟合。"""
    delta = number(delta)
    unit = number(unit)
    if unit <= 0:
        raise ValueError("POSITIVE_GAP_UNIT_REQUIRED")
    return clamp(number(neutral) + number(amplitude) * delta / (delta * delta + unit * unit).sqrt())


def window_frequency_score(delta, unit, *, neutral=75, amplitude=25) -> Decimal:
    """R3只记录有意义领先是否持续，不重复奖励已超过宽容带后的收益幅度。"""
    delta = number(delta)
    unit = number(unit)
    if unit <= 0:
        raise ValueError("POSITIVE_WINDOW_UNIT_REQUIRED")
    direction = 1 if delta > unit else -1 if delta < -unit else 0
    return clamp(number(neutral) + number(amplitude) * direction)


def tracking_score(actual, contract_limit) -> Decimal:
    """跟踪误差与截止日已核准的力争目标比较；目标位置75并非合规优秀。

    实际及目标均为比例，不接受百分数字段直接混入。超过目标连续下降，单次
    观察跨目标不自动构成法定违规；严重事实由独立的事项复核处理。
    """
    x = number(actual)
    limit = number(contract_limit)
    if x < 0 or limit <= 0:
        raise ValueError("INVALID_TRACKING_ANCHOR")
    # 同式通分后只除一次，合同目标处精确得到75，避免两次有限精度除法误差。
    return D(300) * limit / (3 * limit + x)


def fee_score(cost_cny, unit_cny=100) -> Decimal:
    """一万元持有365日的公开基金层成本，单位元；未知佣金不填0。"""
    cost = number(cost_cny)
    unit = number(unit_cny)
    if cost < 0 or unit <= 0:
        raise ValueError("INVALID_COST")
    return clamp(D(100) - D(25) * cost / unit)


def exposure_factor(name: str, value) -> Decimal:
    """R2补上实际暴露：仅相对平滑不能让极端风险仍保底50。

    DD和HHI有真实0..1边界；额外下行及K无有限统一安全线，采用连续衰减，
    不把1这个量纲单位声称为监管线或未来能变现的期限。
    """
    x = number(value)
    if x < 0:
        raise ValueError("NEGATIVE_EXPOSURE")
    if name in {"extra_drawdown", "issuer_concentration", "industry_concentration"}:
        if x > 1:
            raise ValueError("BOUNDED_EXPOSURE_EXCEEDS_ONE")
        return 1 - x
    if name in {"extra_downside", "market_capacity"}:
        return 1 / (1 + x)
    raise ValueError("UNSUPPORTED_EXPOSURE_FACTOR")


def normalize_packet(packet: dict) -> dict:
    """按每个资料包显式单位约定统一，保留源值，不按基金代码或大小猜单位。"""
    unit = packet.get("unit_contract")
    if unit not in {"PERCENT_RANKS_PERCENT_WINDOWS", "PERCENT_RANKS_RATIO_WINDOWS", "RATIO_RANKS_RATIO_WINDOWS"}:
        raise ValueError("EXPLICIT_INPUT_UNIT_CONTRACT_REQUIRED")
    result = deepcopy(packet["quality_metrics"])
    with localcontext() as context:
        context.prec = 80
        for name, node in result["observations"].items():
            if name == "cost_365" and node["kind"] == "standard_cost":
                # R5费用是绝对万元场景，不需要制造无关同行费用来通过旧rank格式。
                from app.services.fund_rating_market_review import annual_fee_score

                node["target"] = annual_fee_score(node["fee_inputs"])["cost_cny"]
            if (
                unit != "RATIO_RANKS_RATIO_WINDOWS"
                and node["kind"] == "rank"
                and (name.startswith("return_") or name in {"extra_drawdown", "extra_downside", "tracking_error"})
            ):
                node["target"] = str(number(node["target"]) / 100)
                for peer in node["peers"]:
                    peer["value"] = str(number(peer["value"]) / 100)
            if node["kind"] == "windows" and unit == "PERCENT_RANKS_PERCENT_WINDOWS":
                for row in node["rows"]:
                    for field in ["target", "reference"]:
                        row[field] = str(number(row[field]) / 100)
    return result


def verified_tracking_limit(anchor: dict | None, asof: str) -> Decimal:
    if not anchor or anchor.get("source_verified") is not True:
        raise ValueError("TRACKING_CONTRACT_ANCHOR_MISSING")
    if date.fromisoformat(anchor["published_on"]) > date.fromisoformat(asof):
        raise ValueError("ANCHOR_AFTER_CUTOFF")
    source = Path(anchor["path"])
    raw = source_bytes(source)
    if content_digest(raw) != anchor["sha256"]:
        raise ValueError("CONTRACT_ANCHOR_HASH_MISMATCH")
    return number(anchor["annual_tracking_error_limit_ratio"])


def _calculate_frozen(
    packet: dict,
    protocol: dict,
    anchors: dict,
    *,
    parameters: dict | None = None,
    scenario="main",
    selected_dimensions: set[str] | None = None,
) -> dict:
    """返回研究点值、旧分与贡献变化；新正式等级保持空，不触碰业务存储。

    逐节点验证必要观察和同产品排除。未修订项使用冻结R3原始观察公式，
    三个修订节点按真实本金/义务、责任日期及结构履职事实计算，不借旧分项。
    """
    config = policy(packet["basis"])
    selected = set(config["terms"]) if selected_dimensions is None else selected_dimensions
    if not selected or not selected <= config["terms"].keys():
        raise ValueError("R5_DIMENSION_SCOPE")
    complete = selected == config["terms"].keys()
    required = {name for key, terms in config["terms"].items() if key in selected for name, weight in terms}
    raw_nodes = packet["quality_metrics"]["observations"]
    aliases = set(packet["quality_metrics"]["target_product_aliases"])
    if (
        set(raw_nodes) != required
        or not aliases
        or packet["quality_metrics"].get("material_event_pending") is not False
    ):
        raise ValueError("R4_REQUIRED_OBSERVATIONS_OR_EVENT_SCOPE")
    registry = packet["r4_evidence_registry"]
    # 支付覆盖必须来自同一财报日，不能用新账户余额覆盖旧负债。
    payment_date = raw_nodes.get("known_payment", {}).get("as_of")
    if "known_payment" in required and (
        payment_date != packet["quality_public"]["dataDates"]["holdings"] or payment_date > packet["as_of_date"]
    ):
        raise ValueError("R4_PAYMENT_REPORT_DATE_MISMATCH")
    new_kinds = {
        "known_payment": "purpose_covered_payment",
        "continuity": "responsibility_timeline",
        "holders": "holder_execution_r5",
    }
    old_kinds = {
        "experience": "state",
        "governance": "state",
        "allocation": "state",
        "mandate": "state",
        "asset_quality": "asset_quality",
        "monthly_success": "windows",
        "annual_success": "windows",
        "restricted": "restricted",
    }
    for name, node in raw_nodes.items():
        if name == "cost_365" and node.get("kind") == "standard_cost":
            from app.services.fund_rating_market_review import annual_fee_score

            if any(key in node for key in ("target", "peers", "expected_products", "score")):
                raise ValueError("R5_COST_RAW_FEES_REQUIRED")
            annual_fee_score(node["fee_inputs"])
            check_refs(node, registry, packet["as_of_date"])
        elif name in new_kinds:
            if node.get("kind") != new_kinds[name]:
                raise ValueError("R4_NODE_KIND:" + name)
            check_refs(node, registry, packet["as_of_date"])
            for event in node.get("confirmed_functional_defects", []) or node.get("confirmed_pressure_effects", []):
                check_refs(event, registry, packet["as_of_date"])
            for interval in node.get("intervals", []):
                check_refs(interval, registry, packet["as_of_date"])
        else:
            if node.get("kind") != old_kinds.get(name, "rank"):
                raise ValueError("R4_NODE_KIND:" + name)
            if node["kind"] == "rank" and node.get("direction") != (
                "higher" if name.startswith("return_") else "lower"
            ):
                raise ValueError("R4_DIRECTION:" + name)
            if name in {"monthly_success", "annual_success"} and node["expected_windows"] != (
                36 if name == "monthly_success" else 25
            ):
                raise ValueError("R4_WINDOW_TYPE:" + name)
            _node(node, aliases=aliases)
    metrics = normalize_packet(packet)
    nodes = metrics["observations"]
    p = parameters or protocol["parameters"]
    subscores = {}
    details = {}
    aliases = set(metrics["target_product_aliases"])
    with localcontext() as context:
        context.prec = 60
        for name, node in nodes.items():
            kind = node["kind"]
            method = "EXISTING_FACT_ANCHORED_STATE"
            if name.startswith("return_"):
                ref = median(number(r["value"]) for r in node["peers"])
                delta = number(node["target"]) - ref
                value = signed_gap_score(
                    delta, p["annual_return_gap_unit"], neutral=p["relative_neutral"], amplitude=p["relative_amplitude"]
                )
                method = "ANNUAL_RETURN_GAP"
                details[name] = {
                    "target_ratio": node["target"],
                    "reference_median_ratio": str(ref),
                    "difference_ratio": str(delta),
                }
            elif name in RELATIVE_NONNEGATIVE:
                ref = median(number(r["value"]) for r in node["peers"])
                value = peer_gap_score(
                    node["target"],
                    [r["value"] for r in node["peers"]],
                    neutral=p["relative_neutral"],
                    amplitude=p["relative_amplitude"],
                )
                method = "RELATIVE_MAGNITUDE"
                details[name] = {
                    "target": node["target"],
                    "reference_median": str(ref),
                    "peer_count": len(node["peers"]),
                    "absolute_safety_not_inferred": True,
                }
                if protocol.get("absolute_exposure_adjustment"):
                    factor = exposure_factor(name, node["target"])
                    details[name].update(relative_component=str(value), exposure_factor=str(factor))
                    value *= factor
                    method = "RELATIVE_GAP_WITH_ABSOLUTE_EXPOSURE"
            elif name == "tracking_error":
                limit = verified_tracking_limit(anchors.get(packet["fund_code"]), packet["as_of_date"])
                value = tracking_score(node["target"], p["tracking_economic_anchor_ratio"])
                method = "COMMON_ECONOMIC_TRACKING_DISTANCE"
                details[name] = {
                    "actual_ratio": node["target"],
                    "contract_limit_ratio": str(limit),
                    "economic_anchor_ratio": str(p["tracking_economic_anchor_ratio"]),
                    "contract_attainment_ratio": str(number(node["target"]) / limit),
                    "contract_used_for_score": False,
                    "source": anchors[packet["fund_code"]],
                }
            elif name in {"issuer_replication", "industry_replication"}:
                tv = number(node["target"])
                if not 0 <= tv <= 1:
                    raise ValueError("INVALID_TOTAL_VARIATION")
                value = 100 * (1 - tv)
                method = "PUBLIC_WEIGHT_MATCHING_RATIO"
                details[name] = {"TV": str(tv), "not_hidden_precision": True}
            elif kind == "windows":
                unit = number(p["annual_return_gap_unit"]) / (12 if name == "monthly_success" else 1)
                scorer = window_frequency_score if protocol.get("stability_margin_frequency") else signed_gap_score
                deltas = [number(r["target"]) - number(r["reference"]) for r in node["rows"]]
                values = [
                    scorer(d, unit, neutral=p["relative_neutral"], amplitude=p["relative_amplitude"]) for d in deltas
                ]
                value = sum(values) / len(values)
                method = (
                    "MARGIN_WINDOW_FREQUENCY" if protocol.get("stability_margin_frequency") else "SOFT_WINDOW_SUCCESS"
                )
                details[name] = {
                    "count": len(values),
                    "gap_unit_ratio": str(unit),
                    "hard_wins": sum(d > 0 for d in deltas),
                    "hard_ties": sum(d == 0 for d in deltas),
                    "margin_wins": sum(d > unit for d in deltas),
                    "margin_losses": sum(d < -unit for d in deltas),
                    "margin_neutral": sum(abs(d) <= unit for d in deltas),
                    "rows": [
                        {"end": r["end"], "target": r["target"], "reference": r["reference"], "quality_point": str(v)}
                        for r, v in zip(node["rows"], values, strict=True)
                    ],
                }
            elif name == "cost_365":
                value = fee_score(node["target"], p["annual_cost_unit_per_10000_CNY"])
                method = "ABSOLUTE_STANDARD_COST"
                details[name] = {
                    "cost_CNY_per_10000_365d": node["target"],
                    "unknown_investor_transaction_costs_not_zero": True,
                }
            elif name == "known_payment":
                value, details[name] = payment_score(node)
                method = "PURPOSE_MATCHED_DISCLOSED_PRINCIPAL_COVERAGE"
            elif name == "continuity":
                value, details[name] = continuity_score(node)
                method = "PUBLIC_RESPONSIBILITY_COVERAGE"
            elif name == "holders":
                value, details[name] = holders_score(node)
                method = "DISCLOSED_STRUCTURE_AND_ACTUAL_EXECUTION"
            else:
                fact = _node(node, aliases=aliases)
                value = number(str(fact.numerator)) / number(str(fact.denominator))
            subscores[name] = value
            details.setdefault(name, {})["method"] = method
            details[name]["point"] = str(value)
        dimensions = {
            k: sum(subscores[name] * weight / 100 for name, weight in terms)
            for k, terms in config["terms"].items()
            if k in selected
        }
        total = sum(dimensions[name] * weight / 100 for name, weight in config["weights"].items()) if complete else None
        reference_grade = (
            ("EXCELLENT" if total >= 90 else "GOOD" if total >= 70 else "AVERAGE" if total >= 50 else "POOR")
            if total is not None
            else None
        )
        events = anchors.get("material_events", {}).get(packet["fund_code"], [])
        applicable = [
            e
            for e in events
            if e.get("applicability") == "FUND_CONFIRMED"
            and e.get("status") == "UNRESOLVED"
            and e.get("severity") in {"MATERIAL", "SEVERE"}
        ]
        allrefs = {r["sha256"] for r in packet["raw_archive"] if r["kind"] == "ORIGINAL"}
        if any(not e.get("source_refs") or not set(e["source_refs"]) <= allrefs for e in applicable):
            raise ValueError("MATERIAL_EVENT_ORIGINAL_MISSING")
        severe = any(e["severity"] == "SEVERE" for e in applicable)
        return {
            "fund_code": packet["fund_code"],
            "as_of": packet["as_of_date"],
            "revision": protocol["revision"],
            "scenario": scenario,
            "status": "WITHHELD_MATERIAL_FAILURE" if severe else "RESEARCH_CANDIDATE_ONLY",
            "score": None if severe or total is None else str(total.quantize(D(".000000000001"))),
            "raw_score": str(total) if total is not None else None,
            "grade": None,
            "old_threshold_reference_grade": reference_grade,
            "reference_grade_is_validated": False,
            "confirmed_material_event_restrictions": applicable,
            "dimensions": {k: str(v) for k, v in dimensions.items()},
            "subscores": {k: str(v) for k, v in subscores.items()},
            "details": details,
            "weights": config["weights"],
            "source_dates": packet["quality_public"]["dataDates"],
            "parameters": p,
            "independent_product_validation": False,
            "publishable": False,
            "limits": [
                "比较收益中性75及经济幅度为冻结的研究取舍，尚非实证最优。",
                "管理经验、治理及资产状态保留原事实判断；接续只按本轮统一规则复查。",
                "滚动年度互相重叠；相同收益信息的重复贡献仍须独立核验。",
                "账面覆盖不是未来赎回安全，正常接续不是内部交接审计；新等级暂不产生。",
            ],
        }


# 公式和参数来自2026-10-20先冻结后计算的R5协议；本次仅将隔离实现迁入标准模块。
# 新等级未校准：完整数字分使用SCORED，不把旧阈值套成新等级。
PROTOCOL = {
    "revision": "eight-dimension-iterative-research-20261020-r5",
    "parameters": {
        "relative_neutral": 75,
        "relative_amplitude": 25,
        "annual_return_gap_unit": ".10",
        "annual_cost_unit_per_10000_CNY": "100",
        "tracking_contract_anchor_score": 75,
        "tracking_economic_anchor_ratio": ".01",
    },
    "absolute_exposure_adjustment": True,
    "stability_margin_frequency": True,
}


def calculate(packet: dict, anchors: dict, *, root: Path) -> dict:
    """从已归档原始观察复算完整八维；所有文件仅允许位于正式来源目录。

    单位、参照成员、窗口和状态证据沿原冻结契约核验。原始取得/核验日期
    保持不变；调用本函数不会将过期资料改成新资料，也不会写库或采集。
    """
    root = root.resolve()
    packet = deepcopy(packet)
    anchors = deepcopy(anchors)

    if packet.get("direct_index_compilation") or packet.get("passive_index_compilation"):
        from app.services.fund_rating_direct_index import compile_observations

        if packet.get("direct_index_compilation") and packet.get("passive_index_compilation"):
            raise ValueError("R5_PASSIVE_INDEX_SINGLE_COMPILATION_REQUIRED")
        inputs = packet.get("direct_index_compilation") or packet["passive_index_compilation"]
        if any(inputs[k] != packet[k] for k in ("fund_code", "as_of_date", "verified_at")):
            raise ValueError("R5_DIRECT_INDEX_PACKET_SCOPE_CONFLICT")
        compiled = compile_observations(inputs, root=root)
        if compiled["observations"] != packet["quality_metrics"]["observations"]:
            raise ValueError("R5_DIRECT_INDEX_ORIGINAL_OBSERVATIONS_CONFLICT")
        anchor = anchors.get(packet["fund_code"], {})
        if any(
            anchor.get(k) != compiled["anchor"].get(k)
            for k in ("sha256", "annual_tracking_error_limit_ratio", "published_on", "actual_clause")
        ):
            raise ValueError("R5_DIRECT_INDEX_CONTRACT_ANCHOR_CONFLICT")

    if packet.get("thematic_complete_compilation"):
        from app.services.fund_rating_thematic_complete import check_binding

        if packet.get("direct_index_compilation") or packet.get("passive_index_compilation"):
            raise ValueError("R5_SINGLE_COMPLETE_COMPILATION_REQUIRED")
        check_binding(packet, anchors, root=root)

    def source_path(value):
        path = (root / value).resolve()
        if not path.is_relative_to(root / "sources") or not source_exists(path):
            raise ValueError("R5_CANONICAL_ARCHIVE_REQUIRED")
        return str(path)

    for ref in packet["r4_evidence_registry"].values():
        ref["path"] = source_path(ref["path"])
    for code, anchor in anchors.items():
        if code != "material_events":
            anchor["path"] = source_path(anchor["path"])
    result = _calculate_frozen(packet, PROTOCOL, anchors)
    if result["score"] is None:
        raise ValueError("R5_MATERIAL_EVENT_WITHHOLDS_COMPLETE_SCORE")
    return {
        **result,
        "status": "SCORED",
        "grade": None,
        "publishable": True,
        "calculation_revision": PROTOCOL["revision"],
    }


def calculate_passive_components(packet: dict, anchors: dict, *, root: Path) -> dict:
    """只复算明确完整的七维，缺流动性时总分始终为空，不延长底稿有效期。

    原件编译和七维子项全部重放；不接受预填分数。完整发布仍须走calculate，
    不能将本函数的七维结果、覆盖率或折算分当成完整总分。
    """
    from app.services.fund_rating_direct_index import compile_observations

    root, packet, anchors = root.resolve(), deepcopy(packet), deepcopy(anchors)
    inputs = packet["passive_index_compilation"]
    if (
        inputs.get("component_scope") != "NON_LIQUIDITY_SEVEN_DIMENSIONS"
        or packet.get("direct_index_compilation")
        or any(inputs[k] != packet[k] for k in ("fund_code", "as_of_date", "verified_at"))
    ):
        raise ValueError("R5_PASSIVE_COMPONENT_PACKET_SCOPE")
    compiled = compile_observations(inputs, root=root)
    if compiled["observations"] != packet["quality_metrics"]["observations"]:
        raise ValueError("R5_PASSIVE_COMPONENT_ORIGINAL_OBSERVATIONS_CONFLICT")
    anchor = anchors.get(packet["fund_code"], {})
    if any(
        anchor.get(k) != compiled["anchor"].get(k)
        for k in ("sha256", "annual_tracking_error_limit_ratio", "published_on", "actual_clause")
    ):
        raise ValueError("R5_PASSIVE_COMPONENT_CONTRACT_ANCHOR_CONFLICT")
    for ref in [*packet["r4_evidence_registry"].values(), anchor]:
        path = (root / ref["path"]).resolve()
        if not path.is_relative_to(root / "sources") or not source_exists(path):
            raise ValueError("R5_CANONICAL_ARCHIVE_REQUIRED")
        ref["path"] = str(path)
    selected = set(policy(packet["basis"])["terms"]) - {"liquidity"}
    result = _calculate_frozen(packet, PROTOCOL, anchors, selected_dimensions=selected)
    if result["confirmed_material_event_restrictions"]:
        raise ValueError("R5_PASSIVE_COMPONENT_MATERIAL_EVENT_REVIEW_REQUIRED")
    assert result["score"] is None and result["raw_score"] is None and result["grade"] is None
    return result
