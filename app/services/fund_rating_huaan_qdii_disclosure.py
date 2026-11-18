"""014978持仓/流动性补项：复用已启用披露规则，保留三年业绩组件。"""

import re
from copy import deepcopy
from decimal import Decimal, localcontext

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals, _flows
from app.services.fund_rating_market_review import normalized_text

VERSION = "HUAN_QDII_DISCLOSURE_COMPLETION_20261111_V1"
MONTHLY_VERSION = "HUAN_QDII_MONTHLY_DISCLOSURE_ONLY_20261117_V1"
MONTHLY_EXTENDED_VERSION = "HUAN_QDII_MONTHLY_DISCLOSURE_COMPONENTS_20261117_V2"
MONTHLY_COMPLETE_VERSION = "HUAN_QDII_MONTHLY_EIGHT_COMPONENTS_20261117_V3"
LEGAL = "华安纳斯达克100交易型开放式指数证券投资基金联接基金(QDII)"
TARGET = "华安纳斯达克100交易型开放式指数证券投资基金(QDII)"


def prepare_monthly_disclosure_only(bundle, *, root, period_review, notice_reviews, missing_input_facts):
    """生成明确两维新期候选；旧成功文案、指标和点值均不带入本期缺项。"""
    from app.services.fund_rating_pipeline import archive_json
    from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy
    from app.services.fund_rating_reviewed_snapshot import KEYS

    candidate = deepcopy(bundle)
    inputs = candidate["huaan_qdii_disclosure_calculation"]["inputs"]
    if (
        period_review.get("fund_code") != candidate["fund_code"]
        or period_review.get("as_of_date") != candidate["as_of_date"]
        or period_review.get("holding_date") != candidate["holding_date"]
        or len(period_review.get("reason", "")) < 60
        or not period_review.get("evidence")
        or not notice_reviews
        or not missing_input_facts
        or any(len(str(missing_input_facts.get(key, ""))) < 8 for key in ("cohort", "publisher", "peer_adjusted_nav"))
        or any(item.get("source_ref") not in inputs["sources"] for item in notice_reviews)
    ):
        raise ValueError("QDII_MONTHLY_ACTUAL_REVIEW_AND_CAUSES_REQUIRED")
    for key in ("huaan_qdii_relative_calculation", "huaan_qdii_benchmark_calculation"):
        candidate.pop(key, None)
    candidate["monthly_disclosure_only"] = True
    extended = bool(inputs.get("management_inputs") or inputs.get("monthly_fee_inputs"))
    inputs.update(
        version=MONTHLY_EXTENDED_VERSION if extended else MONTHLY_VERSION,
        monthly_disclosure_only=True,
        as_of_date=candidate["as_of_date"],
        monthly_period_review=deepcopy(period_review),
    )
    inputs["monthly_notice_reviews"] = deepcopy(notice_reviews)
    candidate["monthly_missing_input_facts"] = deepcopy(missing_input_facts)
    candidate["monthly_notice_reviews"] = deepcopy(notice_reviews)
    result = calculate(inputs, root=root)
    candidate["public_explanations"] = {
        "return": "本期未生成收益评分。"
        + missing_input_facts["cohort"]
        + missing_input_facts["peer_adjusted_nav"]
        + missing_input_facts["publisher"],
        "risk": "本期未生成风险评分。" + missing_input_facts["publisher"] + missing_input_facts["peer_adjusted_nav"],
        "execution": "本期未生成目标执行评分。"
        + missing_input_facts["publisher"]
        + "本期合同参照及跟踪误差尚未完成重放。",
        "stability": "本期未生成稳定性评分。"
        + missing_input_facts["cohort"]
        + missing_input_facts["publisher"]
        + "本期月度和滚动年度比较尚未完成。",
        "management": missing_input_facts.get("management", "本期管理原件仍须通过实际评分核验。")
        + "历史整改事实与旧评分仅保留为历史，不作为本期管理评分。",
        "fee": missing_input_facts.get("fee", "本期费用原件仍须通过实际评分核验。")
        + "旧期持有情景估计不直接作为本期费用分数。",
        "holdings": (
            "按最新完整报告期重新核对联接与唯一目标ETF投资金额、目标ETF完整权益明细"
            "及主要股票的同日指数成分匹配。完整持仓日期仍为"
        )
        + candidate["holding_date"]
        + "，本期新披露不被当作实时全仓。",
        "liquidity": (
            "按最新完整报告的公开申赎对账、巨额赎回安排及托管履职重新计算，"
            "并核对本期新公告的实际适用范围。只评价公开履职，不承诺未来实际到账。"
        ),
    }
    # 数值为空但沿用旧成功指标/来源说明同样会误导；缺六维一并清除旧展示投影。
    for key in ("return", "risk", "execution", "stability", "management", "fee"):
        candidate.get("public_metrics", {}).pop(key, None)
        candidate.get("dimension_sources", {}).pop(key, None)
    for outer, internal in [("management", "management"), ("fee", "cost")]:
        if result["dimensions"].get(internal) is not None:
            candidate["monthly_missing_input_facts"].pop(outer, None)
            candidate["public_explanations"][outer] = (
                "本期已按实际经理职责、连续任职及六个半年度治理原件独立重算；已披露整改按既有观察档评价。"
                if outer == "management"
                else "本期已按本C份额当前公开收费原件及一万元持有365天、净值不变的既有情景独立重算。"
                "运作综合费率不重复叠加基础费率；这是公开费用估计。"
            )
    count = len([point for point in result["dimensions"].values() if point is not None])
    candidate["public_limitations"] = [
        "本期资料截止"
        + candidate["as_of_date"]
        + "，已有"
        + str(count)
        + "项按实际原件重新复算，"
        + "其余"
        + str(8 - count)
        + "项及总分为空。",
        "缺项原因和真实补取失败另有记录；本版本不表示自然月末官方估值证明已经齐全。",
    ]
    raw = candidate["huaan_qdii_disclosure_calculation"]
    raw["input_archive"] = archive_json(root, inputs)
    weights = policy(PASSIVE_BASIS)["weights"]
    document = {
        "code": candidate["fund_code"],
        "as_of": candidate["as_of_date"],
        "outer_weights_percent": {key: weights[value] for key, value in KEYS.items()},
        "dimensions": {key: result["dimensions"].get(value) for key, value in KEYS.items()},
    }
    candidate.update(research_format="single", research_archive=archive_json(root, document))
    candidate.pop("reviewed_document", None)
    return candidate


def check_binding(bundle, inputs):
    if (
        any(inputs.get(k) != bundle.get(k) for k in ("fund_code", "share_class", "as_of_date", "holding_date"))
        or inputs.get("primary_ref") != bundle["original_report_archive"]["sha256"]
        or inputs.get("legal_name") != bundle["identity"]["legal_name"]
        or inputs.get("manager_name") != bundle["identity"]["manager_name"]
        or bundle.get("family") != "INDEX"
    ):
        raise ValueError("QDII_COMPLETION_BINDING")
    if bundle.get("monthly_disclosure_only") is True:
        if (
            inputs.get("version") not in {MONTHLY_VERSION, MONTHLY_EXTENDED_VERSION}
            or inputs.get("monthly_disclosure_only") is not True
            or any(
                bundle.get(key)
                for key in (
                    "huaan_qdii_relative_calculation",
                    "huaan_qdii_benchmark_calculation",
                    "evidence_v1_calculation",
                    "r5_calculation",
                )
            )
        ):
            raise ValueError("QDII_MONTHLY_DISCLOSURE_ONLY_BINDING")
        return
    if (
        not bundle.get("huaan_qdii_relative_calculation")
        or not bundle.get("huaan_qdii_benchmark_calculation")
        or bundle.get("evidence_v1_calculation")
        or bundle.get("r5_calculation")
    ):
        raise ValueError("QDII_COMPLETION_EXISTING_REPLAY_REQUIRED")
    if inputs.get("version") == MONTHLY_COMPLETE_VERSION and (
        inputs.get("monthly_disclosure_only") is not False
        or not inputs.get("management_inputs")
        or not inputs.get("monthly_fee_inputs")
    ):
        raise ValueError("QDII_MONTHLY_COMPLETE_REQUIRES_INDEPENDENT_MANAGEMENT_AND_FEES")


def _pages(source, digest, code, legal, period):
    pages = source.pages(digest)
    body = normalized_text("\n".join(pages))
    if (
        code not in body[:15000]
        or normalized_text(legal) not in body[:15000]
        or str(report_period("\n".join(pages))[1]) != period
    ):
        raise ValueError("QDII_COMPLETION_REPORT_IDENTITY_PERIOD")
    return pages, body


def calculate(inputs, *, root):
    if (
        inputs.get("version") not in {VERSION, MONTHLY_VERSION, MONTHLY_EXTENDED_VERSION, MONTHLY_COMPLETE_VERSION}
        or inputs.get("fund_code") != "014978"
        or inputs.get("share_class") != "C"
    ):
        raise ValueError("QDII_COMPLETION_SCOPE")
    if (
        inputs.get("version") in {MONTHLY_VERSION, MONTHLY_EXTENDED_VERSION}
        and inputs.get("monthly_disclosure_only") is not True
    ):
        raise ValueError("QDII_MONTHLY_DISCLOSURE_ONLY_MARKER_REQUIRED")
    if inputs.get("version") == MONTHLY_COMPLETE_VERSION and inputs.get("monthly_disclosure_only") is not False:
        raise ValueError("QDII_MONTHLY_COMPLETE_MARKER_REQUIRED")
    source = Originals(inputs, root, lazy=True)
    from app.services.fund_rating_period import validate_period

    class SourceBody:
        def body(self, reference):
            return "\n".join(source.pages(reference))

    validate_period(inputs, SourceBody())
    pages, own = _pages(source, inputs["primary_ref"], "014978", LEGAL, inputs["holding_date"])
    target_pages, target = _pages(source, inputs["target_ref"], "159632", TARGET, inputs["holding_date"])
    relation = re.search(r"2\.1\.1目标基金基本情况(.+?)2\.2基金产品说明", own)
    if not relation or "159632" not in relation[1] or TARGET not in relation[1]:
        raise ValueError("QDII_COMPLETION_TARGET_RELATION")
    # 仅接受本联接唯一基金投资的原表；全部基金投资与目标明细金额精确对账。
    fund = re.search(r"基金投资([\d,]+\.\d{2})", own[own.find("资产负债表") :])
    fund_tables = list(re.finditer(r"7\.2期末投资目标基金明细(.+?)7\.3期末", own))
    fund_table = fund_tables[-1] if fund_tables else None
    matched_money = bool(fund and fund_table and fund[1] in fund_table[1])
    if not matched_money and fund and fund_table:
        # 本报告金额末千分组被同一行的序号和多列表头打断，只拼原金额已有片段。
        prefix, tail = fund[1].rsplit(",", 1)
        matched_money = bool(re.search(re.escape(prefix) + r".{0,160}?" + re.escape("," + tail), fund_table[1]))
    if (
        not fund
        or not fund_table
        or "华安纳斯达克100" not in fund_table[1]
        or not matched_money
        or re.findall(
            r"(?m)^\s*([1-9])\s+", "\n".join(pages).rsplit("7.2 期末投资目标基金明细", 1)[1].split("7.3 ", 1)[0]
        )
        != ["1"]
    ):
        raise ValueError("QDII_COMPLETION_TARGET_AMOUNT")
    official = source.json(inputs["constituents_ref"])
    meta = source.sources[inputs["constituents_ref"]]
    if meta.get("url") != "https://indexes.nasdaq.com/Index/WeightingData" or meta.get("params") != {
        "id": "NDX",
        "tradeDate": inputs["holding_date"],
        "timeOfDay": "EOD",
    }:
        raise ValueError("QDII_COMPLETION_OFFICIAL_DATE")
    members = [r["Symbol"] for r in official["aaData"]]
    if not members or len(set(members)) != len(members) or official["iTotalRecords"] != len(members):
        raise ValueError("QDII_COMPLETION_CONSTITUENTS_CLOSURE")
    original = "\n".join(target_pages)
    table = original.rsplit("7.4.1 期末指数投资按公允价值占基金资产净值比例大小排序的所有权益投资明细", 1)[1].split(
        "7.4.2", 1
    )[0]
    amounts = list(re.finditer(r"(\d{1,3}(?:,\d{3})*\.\d{2})\s+(\d+\.\d{2})", table))
    positions = []
    for i, m in enumerate(amounts):
        chunk = table[amounts[i - 1].end() if i else 0 : m.end()]
        rank = re.findall(r"(?m)^\s*(\d{1,3})\s+", chunk)
        symbols = [s for s in members if re.search(r"(?<![A-Z])" + re.escape(s) + r"(?![A-Z])", chunk)]
        if rank != [str(i + 1)] or len(symbols) != 1 or "美国" not in chunk:
            raise ValueError("QDII_COMPLETION_STOCK_ROW_AMBIGUOUS")
        positions.append(
            {"rank": i + 1, "stock_code": symbols[0], "market_value_cny": m[1].replace(",", ""), "nav_percent": m[2]}
        )
    balances = list(re.finditer(r"7\.1期末基金资产组合情况.+?权益投资([\d,]+\.\d{2})", target))
    balance = balances[-1] if balances else None
    if (
        not balance
        or not positions
        or sum(Decimal(r["market_value_cny"]) for r in positions) != Decimal(balance[1].replace(",", ""))
    ):
        raise ValueError("QDII_COMPLETION_FULL_EQUITY_RECONCILIATION")
    top = sorted(positions, key=lambda r: -Decimal(r["market_value_cny"]))[:10]
    with localcontext() as context:
        context.prec = 60
        amount = sum(Decimal(r["market_value_cny"]) for r in top)
        matched = sum(Decimal(r["market_value_cny"]) for r in top if r["stock_code"] in members)
        holding_score = 100 * matched / amount
    flows, flow_proof = _flows(pages, allow_reversed=True, allow_interleaved=True, allow_period_variant=True)
    for text in (own, target):
        if not all(
            s in text
            for s in [
                "在基金合同中设计了巨额赎回条款",
                "进行了认真的复核",
                "进行了监督,未发现基金管理人有损害基金份额持有人利益的行为",
                "本基金在本报告期内流动性情况良好",
            ]
        ):
            raise ValueError("QDII_COMPLETION_PUBLIC_REDEMPTION_EXECUTION")
    if inputs.get("actual_effect_review_complete") is not True or inputs.get("confirmed_pressure_effects") is not False:
        raise ValueError("QDII_COMPLETION_ACTUAL_REVIEW_REQUIRED")
    result = {
        "dimensions": {"holdings": str(holding_score), "liquidity": "75"},
        "facts": {
            "target_code": "159632",
            "fund_investment_cny": fund[1].replace(",", ""),
            "complete_equity_value_cny": balance[1].replace(",", ""),
            "positions_count": len(positions),
            "official_components_count": len(members),
            "top_ten": top,
            "share_flows": flows,
            "flow_proof": flow_proof,
            "scope": "主要股票成分匹配及报告期公开申赎履职，不声称全资产权重偏离或实际到账保证。",
            "futures_preserved": (
                "联接及目标ETF期货净列示为零不代表没有名义敞口；本组件只按已启用主要股票匹配规则评价持仓。"
            ),
            "source_sha256": sorted(source.raw),
        },
    }
    if inputs["version"] in {MONTHLY_EXTENDED_VERSION, MONTHLY_COMPLETE_VERSION}:
        independent = _independent_monthly_components(inputs, source, root=root)
        result["dimensions"].update(independent["dimensions"])
        result["facts"].update(independent["facts"])
        if inputs["version"] == MONTHLY_COMPLETE_VERSION and any(
            result["dimensions"].get(key) is None for key in ("management", "cost")
        ):
            raise ValueError("QDII_MONTHLY_COMPLETE_INDEPENDENT_COMPONENT_NOT_READY")
    return result


def _independent_monthly_components(inputs, source, *, root):
    """只从已验收管理原始输入/当前费用原件复算；不消费旧标量。"""
    result = {"dimensions": {}, "facts": {}}
    from app.services.fund_rating_management_inputs import calculate as management_calculate
    from app.services.fund_rating_market_review import annual_fee_score, fee_profile

    errors = result["facts"].setdefault("evidence_errors", {})
    result["dimensions"].update(management=None, cost=None)
    if inputs.get("management_inputs"):
        try:
            observed = management_calculate(
                inputs["management_inputs"],
                root=root,
                manager_name=inputs["manager_name"],
                as_of=inputs["as_of_date"],
                fund_code=inputs["fund_code"],
                legal_name=inputs["legal_name"],
                primary_report_hash=inputs["primary_ref"],
            )
            result["dimensions"]["management"] = observed["score"]
            result["facts"]["management"] = observed
        except (ValueError, KeyError, FileNotFoundError) as error:
            errors["management"] = str(error)
    if inputs.get("monthly_fee_inputs"):
        try:
            fee = inputs["monthly_fee_inputs"]
            if set(fee) != {"summary_ref", "prospectus_ref"}:
                raise ValueError("QDII_MONTHLY_FEE_ORIGINAL_REFS_REQUIRED")
            profile = fee_profile(
                "\n".join(source.pages(fee["summary_ref"])),
                code=inputs["fund_code"],
                share_class=inputs["share_class"],
                legal_name=inputs["legal_name"],
                supporting_text="\n".join(source.pages(fee["prospectus_ref"])),
                published_on=source.sources[fee["summary_ref"]]["published_on"],
            )
            observed = annual_fee_score(profile["fees"])
            result["dimensions"]["cost"] = str(observed["score"])
            result["facts"]["cost"] = {"profile": profile, **observed}
        except (ValueError, KeyError, FileNotFoundError) as error:
            errors["cost"] = str(error)
    return result


def replay_snapshot(bundle, dimensions, *, root):
    inputs = bundle["huaan_qdii_disclosure_calculation"]["inputs"]
    check_binding(bundle, inputs)
    points = calculate(inputs, root=root)["dimensions"]
    if bundle.get("monthly_disclosure_only") is True:
        for key in ("return", "risk", "execution", "stability", "management", "fee"):
            internal = "cost" if key == "fee" else key
            if inputs["version"] == MONTHLY_EXTENDED_VERSION and points.get(internal) is not None:
                continue
            point = dimensions[key]
            if (
                point.get("candidate") is not None
                or Decimal(str(point["lower"])) != 0
                or Decimal(str(point["upper"])) != 100
            ):
                raise ValueError("QDII_MONTHLY_UNOBSERVED_DIMENSION_BECAME_POINT:" + key)
    for key in ("holdings", "liquidity", "management", "fee"):
        internal = "cost" if key == "fee" else key
        if points.get(internal) is None:
            continue
        if any(
            dimensions[key].get(bound) is None or Decimal(str(dimensions[key][bound])) != Decimal(points[internal])
            for bound in ("lower", "upper", "candidate")
        ):
            raise ValueError("QDII_COMPLETION_DIMENSION_REPLAY:" + key)
        reviewed = bundle["reviewed_document"]["dimensions"][key]
        values = (
            [reviewed.get(bound) for bound in ("lower", "upper", "candidate")]
            if isinstance(reviewed, dict)
            else [reviewed]
        )
        if any(value is None or Decimal(str(value)) != Decimal(points[internal]) for value in values):
            raise ValueError("QDII_COMPLETION_REVIEWED_DOCUMENT_REPLAY:" + key)


def completed_total(bundle, dimensions, weights, *, root):
    """本专用补项和原三年组件均重放后，按既定八维权重合成。"""
    from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS, policy
    from app.services.fund_rating_reviewed_snapshot import KEYS

    replay_snapshot(bundle, dimensions, root=root)
    expected = policy(PASSIVE_BASIS)["weights"]
    if any(Decimal(str(weights[key])) != expected[value] for key, value in KEYS.items()):
        raise ValueError("QDII_COMPLETION_FROZEN_OUTER_WEIGHTS")
    if any(dimensions[key].get("candidate") is None for key in KEYS):
        raise ValueError("QDII_COMPLETION_EIGHT_POINTS_REQUIRED")
    total = sum(Decimal(str(weights[key])) * Decimal(str(dimensions[key]["candidate"])) / 100 for key in KEYS)
    return {
        "score": str(total.quantize(Decimal(".01"))),
        "scope": "EXISTING_THREE_YEAR_COMPONENTS_PLUS_DISCLOSURE_HOLDINGS_LIQUIDITY",
    }
