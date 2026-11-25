"""外盘月末以本份额官方完整净值原件重放；不采用普通工作日推算。"""

import json
from datetime import date, timedelta
from decimal import Decimal
from urllib.parse import parse_qs, urlsplit

from app.services.fund_rating_direct_index import Originals

CSRC_NAV_PATH = "/fund/disclose/getPublicFundJZInfoMore.do"


def official_month_end(bundle, observed, requested, *, root):
    """复核适用的管理人或证监会实际公开净值序列。"""
    proof = bundle.get("monthly_valuation_proof", {})
    code = bundle.get("fund_code")
    if proof.get("format") == "BOSERA_NAV_YEAR_RANGE_V1":
        return bosera_month_end(bundle, observed, requested, root=root)
    if proof.get("format") == "HUAAN_ISSUER_RETURN_CURVE_V1":
        return huaan_month_end(bundle, observed, requested, root=root)
    if proof.get("format") == "CSRC_EID_SHARE_NAV_V1":
        return csrc_month_end(bundle, observed, requested, root=root)
    if (
        proof.get("format") != "CIFM_PRODUCT_NAV_COMPLETE_V1"
        or proof.get("fund_code") != code
        or proof.get("as_of_date") != str(observed)
        or bundle.get("as_of_date") != str(observed)
        or proof.get("month_end") != str(requested)
        or (requested + timedelta(days=1)).day != 1
        or observed > requested
        or observed.strftime("%Y-%m") != requested.strftime("%Y-%m")
    ):
        raise ValueError("MONTH_END_OFFICIAL_PROOF_SCOPE")
    nav_ref, identity_ref = proof["issuer_nav_source_ref"], proof["issuer_identity_source_ref"]
    if (
        nav_ref.get("url") != "https://ecmob.cifm.com/mall-data-server/product/queryProductNav"
        or identity_ref.get("url") != "https://ecmob.cifm.com/mall-data-server/fundinfo/queryFundDetail"
        or identity_ref.get("params", {}).get("product_code") != code
    ):
        raise ValueError("MONTH_END_ISSUER_SOURCE_IDENTITY")
    src = Originals(
        {"as_of_date": str(date.today()), "sources": {ref["sha256"]: ref for ref in [nav_ref, identity_ref]}},
        root,
        lazy=True,
    )
    nav, identity = src.json(nav_ref["sha256"]), src.json(identity_ref["sha256"])
    info = identity.get("data", {}).get("product_info", {})
    legal = bundle.get("identity", {}).get("legal_name", "")
    if (
        identity.get("code") != 0
        or nav.get("code") != 0
        or info.get("product_code") != code
        or not legal
        or info.get("product_legal_name") != legal
    ):
        raise ValueError("MONTH_END_OFFICIAL_SHARE_IDENTITY")
    product = info["product_id"]
    params = nav_ref.get("params", {})
    if (
        params.get("product_id") != product
        or date.fromisoformat(params["start_date"]) > observed
        or date.fromisoformat(params["end_date"]) <= requested
        or proof.get("sequence")
        != {"path": ["data", "product_nav", product, "list"], "date_field": "nav_date", "identity_field": "product_id"}
    ):
        raise ValueError("MONTH_END_OFFICIAL_QUERY_SCOPE")
    envelope = nav.get("data", {}).get("product_nav", {}).get(product, {})
    rows = envelope.get("list", [])
    if (
        not isinstance(rows, list)
        or not rows
        or envelope.get("pages") != 1
        or envelope.get("pageNum") != 1
        or envelope.get("total") != len(rows)
        or envelope.get("pageSize", 0) < len(rows)
        or any(row.get("product_id") != product for row in rows)
    ):
        raise ValueError("MONTH_END_OFFICIAL_SEQUENCE_INCOMPLETE")
    days = [date.fromisoformat(row["nav_date"]) for row in rows]
    if len(set(days)) != len(days):
        raise ValueError("MONTH_END_OFFICIAL_DUPLICATE")
    before, after = [d for d in days if d <= requested], [d for d in days if d > requested]
    if not before or not after or max(before) != observed or proof.get("next_valuation_date") != str(min(after)):
        raise ValueError("MONTH_END_OFFICIAL_TERMINAL_OR_NEXT")
    return True


def csrc_month_end(bundle, observed, requested, *, root):
    """复核证监会披露的同份额完整跨月净值和下一估值日。"""
    proof = bundle["monthly_valuation_proof"]
    ref = proof["official_nav_source_ref"]
    code = bundle["fund_code"]
    if (
        proof.get("fund_code") != code
        or proof.get("as_of_date") != str(observed)
        or bundle.get("as_of_date") != str(observed)
        or proof.get("month_end") != str(requested)
        or (requested + timedelta(days=1)).day != 1
        or observed.strftime("%Y-%m") != requested.strftime("%Y-%m")
        or observed > requested
        or ref.get("http_status") != 200
    ):
        raise ValueError("MONTH_END_CSRC_PROOF_SCOPE")
    url = urlsplit(ref["url"])
    if (url.scheme not in {"http", "https"} or url.netloc != "eid.csrc.gov.cn"
            or url.path != CSRC_NAV_PATH or url.fragment):
        raise ValueError("MONTH_END_CSRC_SOURCE")
    query = parse_qs(url.query, strict_parsing=True)
    if set(query) != {"aoData"} or len(query["aoData"]) != 1:
        raise ValueError("MONTH_END_CSRC_QUERY")
    fields = json.loads(query["aoData"][0])
    params = {entry["name"]: entry["value"] for entry in fields}
    start, end = date.fromisoformat(params["startDate"]), date.fromisoformat(params["endDate"])
    if (len(params) != len(fields) or params["fundCode"] != code or params["iDisplayStart"] != "0"
            or start > observed or end <= requested):
        raise ValueError("MONTH_END_CSRC_QUERY_SCOPE")
    source = Originals({"as_of_date": str(date.today()), "sources": {ref["sha256"]: ref}}, root, lazy=True)
    raw = source.json(ref["sha256"])
    rows = raw["aaData"]
    if (not isinstance(rows, list) or not rows or int(raw["iTotalDisplayRecords"]) != len(rows)
            or len(rows) > int(params["iDisplayLength"])):
        raise ValueError("MONTH_END_CSRC_SEQUENCE_INCOMPLETE")
    days = []
    for row in rows:
        day, value = date.fromisoformat(row["valuationDate"]), Decimal(str(row["shareNetValue"]))
        if (row["code"] != code or not isinstance(row.get("shortName"), str) or not row["shortName"].strip()
                or not start <= day <= end or not value.is_finite() or value <= 0):
            raise ValueError("MONTH_END_CSRC_ROW")
        days.append(day)
    before, after = [d for d in days if d <= requested], [d for d in days if d > requested]
    if (len(set(days)) != len(days) or not before or not after or max(before) != observed
            or proof.get("next_valuation_date") != str(min(after))):
        raise ValueError("MONTH_END_CSRC_TERMINAL_OR_NEXT")
    return True


def bosera_month_end(bundle, observed, requested, *, root):
    """按博时本C官网实际请求、逐日行和下一估值日闭合月底，拒绝按周末推算。"""
    proof = bundle["monthly_valuation_proof"]
    ref = proof.get("issuer_nav_source_ref", {})
    params = ref.get("params", {})
    if (bundle.get("fund_code") != "018853" or proof.get("fund_code") != "018853"
            or bundle.get("share_class") != "C" or proof.get("as_of_date") != str(observed)
            or bundle.get("as_of_date") != str(observed) or proof.get("month_end") != str(requested)
            or (requested + timedelta(days=1)).day != 1 or observed.strftime("%Y-%m") != requested.strftime("%Y-%m")
            or observed > requested or ref.get("status", ref.get("http_status")) != 200
            or ref.get("url") != "https://www.bosera.com/fund/fundNetValue.json"
            or params.get("fundCode") != "018853" or params.get("netValueType") != "NETVALUEPER"):
        raise ValueError("MONTH_END_BOSERA_PROOF_SCOPE")
    start, end = date.fromisoformat(params["startDate"]), date.fromisoformat(params["endDate"])
    if start > observed or end <= requested:
        raise ValueError("MONTH_END_BOSERA_QUERY_COVERAGE")
    source = Originals({"as_of_date": str(date.today()), "sources": {ref["sha256"]: ref}}, root, lazy=True)
    raw = source.json(ref["sha256"])
    if raw.get("retCode") != "0" or raw.get("data", {}).get("title") != "单位净值":
        raise ValueError("MONTH_END_BOSERA_RESPONSE_IDENTITY")
    days = []
    for row in raw["data"].get("resultList", []):
        if not isinstance(row, dict) or len(row) != 1:
            raise ValueError("MONTH_END_BOSERA_DATE_ROW")
        day, value = next(iter(row.items()))
        instant = date.fromisoformat(day)
        number = Decimal(str(value))
        if not start <= instant <= end or not number.is_finite() or number <= 0:
            raise ValueError("MONTH_END_BOSERA_VALUE_OR_QUERY")
        days.append(instant)
    before, after = [d for d in days if d <= requested], [d for d in days if d > requested]
    if (len(set(days)) != len(days) or not before or not after or max(before) != observed
            or proof.get("next_valuation_date") != str(min(after))):
        raise ValueError("MONTH_END_BOSERA_TERMINAL_OR_NEXT")
    return True


def huaan_month_end(bundle, observed, requested, *, root):
    """只认本C的官网实际曲线请求及跨月观察，不用第三方或普通工作日推算。"""
    proof = bundle["monthly_valuation_proof"]
    ref = proof.get("issuer_nav_source_ref", {})
    params = ref.get("params", {})
    if (bundle.get("fund_code") != "014978" or proof.get("fund_code") != "014978"
            or bundle.get("share_class") != "C" or proof.get("as_of_date") != str(observed)
            or bundle.get("as_of_date") != str(observed) or proof.get("month_end") != str(requested)
            or (requested + timedelta(days=1)).day != 1 or observed.strftime("%Y-%m") != requested.strftime("%Y-%m")
            or observed > requested or ref.get("status_code", ref.get("http_status")) != 200
            or ref.get("url") != "https://www.huaan.com.cn/fund/FundPerformance/fundPerformanceChart.do"
            or params.get("fundcodes") != "014978" or params.get("selecttype") != "ZS"):
        raise ValueError("MONTH_END_HUAAN_PROOF_SCOPE")
    start = date.fromisoformat(params["beginDate"].replace("/", "-"))
    end = date.fromisoformat(params["endDate"].replace("/", "-"))
    if start > observed or end <= requested:
        raise ValueError("MONTH_END_HUAAN_QUERY_COVERAGE")
    source = Originals({"as_of_date": str(date.today()), "sources": {ref["sha256"]: ref}}, root, lazy=True)
    raw = source.json(ref["sha256"])
    if not isinstance(raw, list) or len(raw) != 1 or raw[0].get("titleText") != "收益率走势":
        raise ValueError("MONTH_END_HUAAN_RESPONSE_IDENTITY")
    days = []
    for row in raw[0].get("chartsDatas", []):
        day = date(int(row["year"]), int(row["month"]) + 1, int(row["day"]))
        value = Decimal(str(row["value"]))
        if not start <= day <= end or not value.is_finite() or value <= -100:
            raise ValueError("MONTH_END_HUAAN_VALUE_OR_QUERY")
        days.append(day)
    before, after = [d for d in days if d <= requested], [d for d in days if d > requested]
    if (len(set(days)) != len(days) or not before or not after or max(before) != observed
            or proof.get("next_valuation_date") != str(min(after))):
        raise ValueError("MONTH_END_HUAAN_TERMINAL_OR_NEXT")
    return True
