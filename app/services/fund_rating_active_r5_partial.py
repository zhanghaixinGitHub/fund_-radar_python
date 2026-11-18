"""主动R5月评的独立维度原件重放；缺维度与总分始终为空。"""

from copy import deepcopy
from pathlib import Path
from datetime import datetime, UTC
from decimal import Decimal, localcontext

from app.services.fund_rating_thematic_source import RawArchive, plain, require, iso
from app.services.fund_rating_monthly_r5 import rolling_window

VERSION = "R5_ACTIVE_MONTHLY_PARTIAL_ORIGINALS_V1"


def _monthly_components(inputs, root):
    source = RawArchive(inputs, Path(root))
    observations, ready, missing = {}, [], {}
    if inputs.get("monthly_performance_ref"):
        from app.services.fund_rating_002112_monthly_performance import compile_observations as performance

        digest = inputs["monthly_performance_ref"]
        raw = source.json(digest)
        require(raw["fund_code"] == inputs["fund_code"] and raw["as_of_date"] == inputs["as_of_date"],
                "ACTIVE_PARTIAL_PERFORMANCE_COMPONENT_SCOPE")
        result = performance(raw, root=Path(root))
        observations.update({key: dict(value, source_refs=[digest]) for key, value in result["observations"].items()})
        ready.extend(result["ready_dimensions"])
        missing.update({key: rows for key, rows in result["missing"].items() if rows})
    if inputs.get("monthly_portfolio_ref"):
        from app.services.fund_rating_002112_monthly_portfolio import compile_observations as portfolio

        digest = inputs["monthly_portfolio_ref"]
        raw = source.json(digest)
        require(raw["fund_code"] == inputs["fund_code"] and raw["as_of_date"] == inputs["as_of_date"],
                "ACTIVE_PARTIAL_PORTFOLIO_COMPONENT_SCOPE")
        result = portfolio(raw, root=Path(root))
        observations.update({key: dict(value, source_refs=[digest]) for key, value in result["observations"].items()})
        ready.extend(result["ready_dimensions"])
        missing.update(result["missing"])
    return dict(observations=observations, ready_dimensions=sorted(ready), missing=missing)


def _missing_reason(dimension, rows):
    labels = dict(return_="收益", risk="风险", stability="持续性", holdings="持仓", liquidity="流动性")
    label = labels.get(dimension, "收益" if dimension == "return" else dimension)
    details = []
    for row in rows[:12]:
        code = row.get("code") or row.get("security", "")
        dates = row.get("dates", [])
        details.append(code + (f"缺{len(dates)}个实际必要日期（" + "、".join(dates[:5]) + "）" if dates else
                              str(row.get("reason") or row.get("error") or row)))
    return label + "本期生产原件重放尚不能唯一计算：" + "；".join(details) + "。完整名单保留，不删同行、不插值、不用旧分。"


def compile_observations(inputs, *, root):
    """仅消费明确当前原件与已核历史责任，不接收手工分项或缺项补零。"""
    from app.services.fund_rating_r5 import calculate, normalize_packet
    from app.services.fund_rating_market_review import fee_profile
    from app.services.fund_quality_v2_disclosures import report_period

    require(inputs.get("version") == VERSION and inputs.get("fund_code") == "002112"
            and inputs.get("share_class") == "C", "ACTIVE_PARTIAL_LEGAL_SCOPE")
    selected = inputs["selected_dimensions"]
    require(isinstance(selected, list) and len(selected) == len(set(selected)) and bool(selected)
            and set(selected) <= {"management", "cost", "efficiency", "return", "risk", "stability", "holdings", "liquidity"},
            "ACTIVE_PARTIAL_DIMENSION_SCOPE")
    source = RawArchive(inputs, Path(root))
    if inputs.get("monthly_admission_ref"):
        from app.services.fund_rating_active_monthly_admission import replay_monthly_admission

        admission_inputs = source.json(inputs["monthly_admission_ref"])
        require(admission_inputs["fund_code"] == inputs["fund_code"]
                and admission_inputs["as_of_date"] == inputs["as_of_date"], "ACTIVE_PARTIAL_MONTHLY_ADMISSION_SCOPE")
        # 独立原件图超过1800源；按其自身严格读取器逐件重放，不扩主图限额、
        # 不把一次脚本导出的已选名单或旧期池当作本期准入事实。
        replay_monthly_admission(admission_inputs, root=Path(root))
    base = source.json(inputs["previous_packet_ref"])
    anchors = source.json(inputs["previous_anchors_ref"])
    require(base["fund_code"] == inputs["fund_code"] and base["family"] == "ACTIVE_EQUITY"
            and base["as_of_date"] < inputs["as_of_date"]
            and not base.get("active_r5_partial_compilation"), "ACTIVE_PARTIAL_ORIGINAL_BASE")
    calculate(base, anchors, root=Path(root))
    old = normalize_packet(base)["observations"]
    days, _ = rolling_window([iso(r["cal_date"]) for r in source.provider_rows(
        inputs["calendar_ref"], ["exchange", "cal_date", "is_open"]) if r["exchange"] == "SSE" and r["is_open"] == 1],
        inputs["as_of_date"])
    current = inputs["current_report_ref"]
    source.original(current)
    body = source.body(current)
    header = plain(body)
    require(all(plain(value) in header[:20000] for value in [inputs["fund_code"], inputs["legal_name"], inputs["manager_name"]])
            and str(report_period(body)[1]) <= inputs["as_of_date"], "ACTIVE_PARTIAL_CURRENT_REPORT_IDENTITY")
    require(inputs["manager_report_date"] == str(report_period(body)[1])
            and inputs["fee_report_date"] == source.sources[inputs["fee_summary_ref"]]["published_on"],
            "ACTIVE_PARTIAL_DISCLOSURE_DATES")
    nav_ref = inputs["target_nav_ref"]
    source.original(nav_ref)
    points = [row for row in source.table(nav_ref, symbol=inputs["fund_code"] + ".OF")
              if iso(row["nav_date"]) == inputs["as_of_date"]]
    require(len(points) == 1 and Decimal(str(points[0]["unit_nav"])) > 0
            and iso(points[0]["ann_date"]) <= inputs["research_on"], "ACTIVE_PARTIAL_ACTUAL_CURRENT_NAV")
    for claim in inputs["current_claims"]:
        require(claim["source_hash"] == current, "ACTIVE_PARTIAL_CURRENT_CLAIM_BINDING")
        source.claim(claim)
    require(any("基金经理" in plain(c["quote"]) for c in inputs["current_claims"])
            and any("合法合规" in plain(c["quote"]) for c in inputs["current_claims"]), "ACTIVE_PARTIAL_POSITIVE_CURRENT_EXECUTION")
    for digest in inputs["current_catalog_refs"]:
        source.original(digest)
        obj = source.json(digest)
        require(obj.get("ErrCode") == 0 and all(row["FUNDCODE"] == inputs["fund_code"] for row in obj["Data"]),
                "ACTIVE_PARTIAL_CURRENT_CATALOG_IDENTITY")
        changes = [row for row in obj["Data"] if base["as_of_date"] < row["PUBLISHDATEDesc"][:10] <= inputs["as_of_date"]
                   and any(word in row["TITLE"] for word in ["基金经理变更", "暂停履职", "恢复履职", "处罚", "调查", "重大诉讼"])]
        require(not changes, "ACTIVE_PARTIAL_NEW_EVENT_REVIEW_REQUIRED")
    nodes = {}
    if "management" in selected:
        assignment = inputs["manager_assignment_ref"]
        ref = source.original(assignment)
        require((ref.get("params_without_credentials") or ref.get("params"))["ts_code"] == inputs["fund_code"] + ".OF",
                "ACTIVE_PARTIAL_MANAGER_SHARE_REQUEST")
        manager_rows = source.table(assignment, symbol=inputs["fund_code"] + ".OF")
        def end_on(row):
            value = row.get("end_date")
            return None if value is None or isinstance(value, str) and not value.strip() else iso(value)
        active = {row["name"]: row for row in manager_rows
                  if iso(row["begin_date"]) <= inputs["as_of_date"]
                  and (end_on(row) is None or end_on(row) > inputs["as_of_date"])}
        continuity = deepcopy(old["continuity"])
        expected = {row["name"]: row for row in continuity["intervals"] if row["end"] is None}
        require(set(active) == set(expected) and all(iso(active[name]["begin_date"]) == expected[name]["start"]
                                                  for name in expected), "ACTIVE_PARTIAL_MANAGER_HISTORY_CONFLICT")
        for interval in continuity["intervals"]:
            require(any(row["name"] == interval["name"] and iso(row["begin_date"]) == interval["start"]
                        and end_on(row) == interval["end"]
                        for row in manager_rows), "ACTIVE_PARTIAL_HISTORICAL_MANAGER_INTERVAL")
        for name, row in expected.items():
            require(name in header, "ACTIVE_PARTIAL_CURRENT_ROLE_NAME")
            row["source_refs"] = sorted(set(row["source_refs"] + [current, assignment]))
        continuity.update(period_start=days[0], period_end=inputs["as_of_date"], current_role_source_ref=current,
                          source_refs=sorted(set(continuity["source_refs"] + [current, assignment])),
                          source_note=inputs["management_reason"])
        governance = deepcopy(old["governance"])
        claim = inputs["known_governance_claim"]
        source.claim(claim)
        require(claim["source_hash"] in governance["source_refs"]
                and all(word in plain(claim["quote"]) for word in ["合规内控", "整改", "监管"])
                and governance["state"] == "REMEDIATION_DISCLOSED", "ACTIVE_PARTIAL_KNOWN_GOVERNANCE_NOT_ERASED")
        governance.update(reason=inputs["governance_reason"], source_refs=sorted(set(governance["source_refs"] + [current])))
        nodes.update(experience=dict(kind="state", state="BASIC_ROLE_EVIDENCED", reason=inputs["management_reason"],
                                     source_refs=[current]), continuity=continuity, governance=governance)
    if "cost" in selected:
        summary, prospectus = inputs["fee_summary_ref"], inputs["fee_prospectus_ref"]
        source.original(summary)
        source.original(prospectus)
        profile = fee_profile(source.body(summary), code=inputs["fund_code"], share_class="C", legal_name=inputs["legal_name"],
                              supporting_text=source.body(prospectus), published_on=source.sources[summary]["published_on"])
        nodes["cost_365"] = dict(kind="standard_cost", fee_inputs=profile["fees"], source_refs=sorted([summary, prospectus]))
    if "efficiency" in selected:
        from app.services.fund_rating_002112_execution import compile_observations as compile_execution

        execution = compile_execution(inputs, root=Path(root))
        require(execution["status"] == "COMPLETE" and not execution["missing"], "ACTIVE_PARTIAL_EXECUTION_FACTS_UNKNOWN")
        nodes.update(execution["observations"])
        source.used.update(execution["used_source_refs"])
    components = _monthly_components(inputs, Path(root))
    require(set(selected) - {"management", "cost", "efficiency"} <= set(components["ready_dimensions"]),
            "ACTIVE_PARTIAL_COMPONENT_NOT_READY")
    if inputs.get("monthly_performance_ref") or inputs.get("monthly_portfolio_ref"):
        require(inputs.get("monthly_component_gaps") == components["missing"], "ACTIVE_PARTIAL_COMPONENT_MISSING_REPLAY")
        require(set(components["ready_dimensions"]) <= set(selected), "ACTIVE_PARTIAL_READY_COMPONENT_NOT_SELECTED")
    nodes.update(components["observations"])
    return dict(observations=nodes, anchors=anchors, source_dates=dict(nav=inputs["as_of_date"],
                manager=str(report_period(body)[1]), fees=source.sources[inputs["fee_summary_ref"]]["published_on"],
                holdings=inputs["holding_date"]), used_source_refs=sorted(source.used))


def packet_for(inputs, *, root):
    compiled = compile_observations(inputs, root=root)
    source = RawArchive(inputs, Path(root))
    packet = deepcopy(source.json(inputs["previous_packet_ref"]))
    packet.update(as_of_date=inputs["as_of_date"], verified_at=inputs["verified_at"],
                  unit_contract="RATIO_RANKS_RATIO_WINDOWS", active_r5_partial_compilation=deepcopy(inputs))
    packet["quality_metrics"]["observations"] = compiled["observations"]
    packet["quality_public"]["dataDates"] = compiled["source_dates"]
    packet["raw_archive"] = [{**ref, "kind": ref["archive_kind"], "path": ref["file"]} for ref in inputs["sources"].values()]
    packet["r4_evidence_registry"] = {sha: {**ref, "path": ref["file"]} for sha, ref in inputs["sources"].items()}
    return packet, compiled["anchors"]


def calculate_inputs(inputs, *, root):
    from app.services.fund_rating_r5 import calculate

    with localcontext() as context:
        context.prec = 80
        packet, anchors = packet_for(inputs, root=root)
        return calculate(packet, anchors, root=Path(root), selected_dimensions=set(inputs["selected_dimensions"]))


def validate_bundle(bundle, *, root):
    from app.services.fund_rating_reviewed_snapshot import _read

    raw = bundle["active_r5_partial_calculation"]
    inputs = raw["inputs"]
    require(_read(raw["input_archive"], Path(root)) == inputs, "ACTIVE_PARTIAL_INPUT_ARCHIVE")
    require(bundle["family"] == "ACTIVE_EQUITY" and bundle["share_class"] == inputs["share_class"] == "C"
            and bundle["fund_code"] == inputs["fund_code"] and bundle["as_of_date"] == inputs["as_of_date"]
            and bundle["identity"]["legal_name"] == inputs["legal_name"]
            and bundle["identity"]["manager_name"] == inputs["manager_name"]
            and bundle["holding_date"] == inputs["holding_date"]
            and bundle["original_report_archive"]["sha256"] == inputs["primary_ref"], "ACTIVE_PARTIAL_BUNDLE_BINDING")
    require(not any(value for key, value in bundle.items() if key != "active_r5_partial_calculation"
                    and (key.endswith("_calculation") or key.endswith("_calculations"))), "ACTIVE_PARTIAL_SINGLE_METHOD")


def replay_snapshot(bundle, dimensions, *, root):
    from app.services.fund_rating_reviewed_snapshot import KEYS

    validate_bundle(bundle, root=root)
    inputs = bundle["active_r5_partial_calculation"]["inputs"]
    result = calculate_inputs(inputs, root=root)
    for key, internal in KEYS.items():
        value, actual = result["dimensions"].get(internal), dimensions[key]
        if value is None:
            require(actual["candidate"] is None and Decimal(actual["lower"]) == 0 and Decimal(actual["upper"]) == 100,
                    "ACTIVE_PARTIAL_MISSING_DIMENSION_NOT_NULL")
        else:
            require(all(actual[field] is not None and Decimal(actual[field]) == Decimal(value)
                        for field in ["lower", "upper", "candidate"]),
                    "ACTIVE_PARTIAL_DIMENSION_REPLAY:" + key + ":" + str(value) + ":" + str(actual))
    require(result["score"] is None and result["raw_score"] is None, "ACTIVE_PARTIAL_TOTAL_MUST_STAY_NULL")
    return result


def prepare(session, inputs, *, root):
    from app.services.fund_rating_pipeline import archive_json, current_inputs
    from app.services.fund_rating_reviewed_snapshot import BASIS, KEYS, validate_sources, calculate

    inputs = deepcopy(inputs)
    if inputs.get("monthly_performance_ref") or inputs.get("monthly_portfolio_ref"):
        components = _monthly_components(inputs, Path(root))
        inputs["selected_dimensions"] = sorted(set(inputs["selected_dimensions"]) | set(components["ready_dimensions"]))
        inputs["monthly_component_gaps"] = components["missing"]
        for key in components["ready_dimensions"]:
            inputs["missing_dimensions"].pop(key, None)
            inputs["dimension_reasons"].setdefault(key, "本期实际完整原件、当前准入与原R5公式独立重放；原报告日期和真实未知均保留。")
        for key, rows in components["missing"].items():
            inputs["missing_dimensions"][key] = _missing_reason(key, rows)
    result = calculate_inputs(inputs, root=root)
    previous = current_inputs(session, [inputs["fund_code"]])[inputs["fund_code"]]
    bundle = deepcopy(next(row["bundle"] for row in previous["input_snapshot"]["members"] if row["fund_code"] == inputs["fund_code"]))
    for key in list(bundle):
        if key.endswith("_calculation") or key.endswith("_calculations"):
            bundle.pop(key)
    missing = set(KEYS.values()) - set(inputs["selected_dimensions"])
    require(set(inputs["missing_dimensions"]) == missing, "ACTIVE_PARTIAL_EXPLICIT_MISSING_DIMENSIONS")
    document = dict(code=inputs["fund_code"], as_of=inputs["as_of_date"],
                    outer_weights_percent={key: result["weights"][internal] for key, internal in KEYS.items()},
                    dimensions={key: result["dimensions"][internal] if internal in result["dimensions"] else
                                dict(lower="0", upper="100", candidate=None, reason=inputs["missing_dimensions"][internal])
                                for key, internal in KEYS.items()},
                    specific_blocker=inputs["missing_dimensions"].get("return", "本维度本期必要原件尚未闭合"),
                    full_total=None, grade=None, calculation_revision=result["calculation_revision"])
    bundle.update(as_of_date=inputs["as_of_date"], verified_at=inputs["verified_at"], research_format="single",
                  research_archive=archive_json(Path(root), document), reviewed_document=document,
                  active_r5_partial_calculation=dict(inputs=deepcopy(inputs), input_archive=archive_json(Path(root), inputs)),
                  data_dates=dict(nav=inputs["as_of_date"], holdings=inputs["holding_date"], manager=inputs["manager_report_date"],
                                  fees=inputs["fee_report_date"]))
    bundle["public_explanations"] = {key: (inputs["dimension_reasons"][internal] if internal in inputs["selected_dimensions"]
                                              else inputs["missing_dimensions"][internal]) for key, internal in KEYS.items()}
    bundle["public_limitations"] = ["缺项维度和总分保持空值，不删347收益同行、不插值、不按已覆盖维度折算总分。",
                                    "本期只发布已由实际新期来源和原R5公式独立重放的维度；旧完整批次保留在历史中。"]
    validate_sources(bundle, datetime.now(UTC), root=Path(root))
    checked = calculate(bundle, root=Path(root))
    require(checked["score"] is None and checked["status"] == "PARTIAL", "ACTIVE_PARTIAL_PREPARE_MUST_NOT_COMPLETE")
    return dict(basis=BASIS, entries=[{key: value for key, value in bundle.items() if key not in {"basis", "reviewed_document"}}])
