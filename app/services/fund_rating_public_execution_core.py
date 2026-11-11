"""把已知、重复的公开现金数值偏离接回原25档；不判断违法和责任。"""

import re
from decimal import Decimal as D
from decimal import localcontext

from app.services.fund_rating_active_execution import _assets
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_reviewed_quality import STATES

VERSION = "ACTIVE_PUBLIC_CORE_EXECUTION_REPEATED_DEVIATION_V1"
MONEY = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"


def require(ok, reason):
    if not ok:
        raise ValueError("PUBLIC_CORE_EXECUTION_" + reason)


def cash_interval_state(lower, upper, nav, minimum):
    """仅当整个有据区间位于阈值一侧时作确定判断，跨阈值不能填中点。"""
    lower, upper, nav, minimum = map(D, [lower, upper, nav, minimum])
    require(0 <= lower <= upper and nav > 0 and 0 < minimum < 1, "CASH_BOUND_DOMAIN")
    threshold = nav * minimum
    if upper < threshold:
        return "PROVED_BELOW_MINIMUM"
    if lower >= threshold:
        return "PROVED_AT_OR_ABOVE_MINIMUM"
    return "UNDETERMINED_ACROSS_MINIMUM"


def snapshot(source, source_hash, *, legal_name, manager_name, codes):
    pages = source.pages(source_hash)
    text = "\n".join(pages)
    flat = normalized_text(text)
    require(normalized_text(legal_name) in flat[:5000] and manager_name in flat[:5000], "REPORT_PRODUCT_IDENTITY")
    require("下属分级基金的交易代码" + "".join(codes) in flat[:15000], "SHARE_COLUMN_BINDING")
    match = re.search(r"期末基金资产净值(" + MONEY + r")(" + MONEY + r")", flat)
    require(match is not None and len(codes) == 2, "ALL_SHARE_NAV_REQUIRED")
    nav = sum(D(value.replace(",", "")) for value in match.groups())
    require(nav > 0, "NAV_POSITIVE")
    date_match = re.search(r"2026年(6月30日|9月30日)", flat[:3000])
    require(date_match is not None, "SNAPSHOT_PERIOD")
    period = "2026-06-30" if date_match[1] == "6月30日" else "2026-09-30"
    # 同一核心金额从完整组合表重新读；不接受调用者塞入75/25或本金数值。
    assets, total = _assets(text)
    cash_upper = D(assets["银行存款和结算备付金合计"]["amount_cny"])
    stock = D(assets["权益投资"]["amount_cny"])
    fixed = D(assets["固定收益投资"]["amount_cny"])
    require(sum(D(row["amount_cny"]) for row in assets.values()) == total, "ASSET_TABLE_RECONCILIATION")
    markers = list(re.finditer(r"\d\.[45](?:报告)?期末按债券品种分类的债券投资组合", flat))
    require(bool(markers), "BOND_CLASSIFICATION")
    start = markers[-1].start()
    tail = flat[start:]
    stop = re.search(r"\d\.[56](?:报告)?期末按公允价值", tail)
    require(stop is not None, "BOND_CLASSIFICATION_END")
    bond_text = tail[: stop.start()]
    require("国家债券--" in bond_text and "其他--" in bond_text, "ZERO_GOVERNMENT_SCOPE_NOT_CLOSED")
    policy = re.search(r"其中:政策性金融债(" + MONEY + r")", bond_text)
    require(policy is not None and D(policy[1].replace(",", "")) == fixed, "POLICY_BOND_TOTAL_RECONCILIATION")
    # “未持有其他信用债”这类宽泛句子不能代替明确的零国债与全部政策性债金额。
    for label in ["基金投资", "贵金属投资", "金融衍生品投资", "买入返售金融资产"]:
        require(D(assets[label]["amount_cny"]) == 0, "MANDATE_ASSET_TYPE_REVIEW_REQUIRED")
    with localcontext() as context:
        context.prec = 60
        return {
            "source_hash": source_hash,
            "period": period,
            "all_share_codes": codes,
            "nav": str(nav),
            "share_navs": [value.replace(",", "") for value in match.groups()],
            "total_assets": str(total),
            "stock_amount": str(stock),
            "stock_nav_ratio": str(stock / nav),
            "stock_total_ratio": str(stock / total),
            "qualified_cash_lower": "0",
            "qualified_cash_upper": str(cash_upper),
            "qualified_cash_upper_nav": str(cash_upper / nav),
            "government_bonds": "0",
            "policy_financial_bonds": str(fixed),
            "cash_scope": "银行存款和结算备付金合计只作为合格现金上界，绝不冒充真实可付款本金。",
            "bond_quote": bond_text,
            "mandate_public_types_verified": True,
        }


def calculate(inputs, source):
    review = inputs["execution_public_observations"]
    require(review.get("version") == VERSION, "VERSION")
    require(
        not any(key in review for key in ["score", "state", "allocation_state", "mandate_state"]),
        "EXTERNAL_SCORE_OR_STATE",
    )
    require(
        review.get("known_special_exemptions") == [] and review.get("material_risk_review_pending") is False,
        "ACTUAL_EXCEPTION_OR_MATERIAL_REVIEW_PENDING",
    )
    contract = normalized_text("\n".join(source.pages(inputs["target_refs"]["prospectus"])))
    require(inputs["legal_name"] in contract and inputs["manager_name"] in contract, "CONTRACT_IDENTITY")
    required = (
        "(2)扣除股指期货合约需缴纳的交易保证金后,现金(不包括结算备付金、存出保证金、应收申购款等)"
        "或者到期日在一年以内的政府债券不低于基金资产净值的5%"
    )
    require(required in contract, "CASH_CONTRACT_DENOMINATOR_OR_SCOPE")
    require(
        re.search(r"除上述第(?:\(2\)、\(6\)、\(8\)、\(14\)|2\)、6\)、8\)、14\))项外.{0,180}10个交易日", contract)
        is not None,
        "CASH_GENERAL_GRACE_EXCLUSION",
    )
    lower, upper = map(D, [inputs["execution"]["stock_min"], inputs["execution"]["stock_max"]])
    require(
        "股票资产、存托凭证占基金资产的比例为60%-95%" in contract and (lower, upper) == (D(".6"), D(".95")),
        "STOCK_BAND_PROFILE",
    )
    require(review.get("share_codes") == ["090019", "006038"], "PROFILE_SHARE_COLUMNS")
    require(
        inputs["share_class"] in ["A", "C"]
        and inputs["fund_code"] == review["share_codes"][["A", "C"].index(inputs["share_class"])],
        "FUND_SHARE_IDENTITY",
    )
    hashes = review["snapshot_hashes"]
    require(len(hashes) == 2 and len(set(hashes)) == 2, "DISTINCT_PERIOD_ORIGINALS")
    rows = [
        snapshot(
            source, h, legal_name=inputs["legal_name"], manager_name=inputs["manager_name"], codes=review["share_codes"]
        )
        for h in hashes
    ]
    require(
        len({row["period"] for row in rows}) == 2 and max(row["period"] for row in rows) == inputs["holding_date"],
        "DISTINCT_LATEST_PERIODS",
    )
    current = next(row for row in rows if row["period"] == inputs["holding_date"])
    require(current["source_hash"] == inputs["target_refs"]["half"], "LATEST_REPORT_BINDING")
    for row in rows:
        require(
            cash_interval_state(row["qualified_cash_lower"], row["qualified_cash_upper"], row["nav"], ".05")
            == "PROVED_BELOW_MINIMUM",
            "REPEATED_SHORTFALL_NOT_PROVED",
        )
        require(
            lower <= D(row["stock_total_ratio"]) <= upper and lower <= D(row["stock_nav_ratio"]) <= upper,
            "OTHER_CORE_STOCK_CONSTRAINT_REVIEW_REQUIRED",
        )
    source.claim(inputs["execution"]["strategy"])
    require(inputs["execution"]["actual_review_complete"] is True, "STRATEGY_REVIEW_REQUIRED")
    # 沿用原有25/75档和60/40预算，不按已知结果另拟尺度。
    allocation = D(STATES["MATERIAL_UNRESOLVED"])
    mandate = D(STATES["NORMAL_PUBLIC_EXECUTION"])
    return {
        "version": VERSION,
        "score": str(allocation * D(".6") + mandate * D(".4")),
        "allocation_score": str(allocation),
        "mandate_score": str(mandate),
        "snapshots": rows,
        "finding": (
            "两个不同期末均有可证实的公开现金配置短缺；并非法定违规、责任或损失认定。"
            "细项原因与特殊豁免继续补件。"
        ),
        "current": current,
    }
