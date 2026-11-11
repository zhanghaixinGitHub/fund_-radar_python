"""014978持仓/流动性补项：复用已启用披露规则，保留三年业绩组件。"""

import re
from decimal import Decimal, localcontext

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals, _flows
from app.services.fund_rating_market_review import normalized_text

VERSION = "HUAN_QDII_DISCLOSURE_COMPLETION_20261111_V1"
LEGAL = "华安纳斯达克100交易型开放式指数证券投资基金联接基金(QDII)"
TARGET = "华安纳斯达克100交易型开放式指数证券投资基金(QDII)"


def check_binding(bundle, inputs):
    if (
        any(inputs.get(k) != bundle.get(k) for k in ("fund_code", "share_class", "as_of_date", "holding_date"))
        or inputs.get("primary_ref") != bundle["original_report_archive"]["sha256"]
        or inputs.get("legal_name") != bundle["identity"]["legal_name"]
        or inputs.get("manager_name") != bundle["identity"]["manager_name"]
        or bundle.get("family") != "INDEX"
    ):
        raise ValueError("QDII_COMPLETION_BINDING")
    if (
        not bundle.get("huaan_qdii_relative_calculation")
        or not bundle.get("huaan_qdii_benchmark_calculation")
        or bundle.get("evidence_v1_calculation")
        or bundle.get("r5_calculation")
    ):
        raise ValueError("QDII_COMPLETION_EXISTING_REPLAY_REQUIRED")


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
        inputs.get("version") != VERSION
        or inputs.get("fund_code") != "014978"
        or inputs.get("share_class") != "C"
        or inputs.get("as_of_date") != "2026-09-30"
        or inputs.get("holding_date") != "2026-06-30"
    ):
        raise ValueError("QDII_COMPLETION_SCOPE")
    source = Originals(inputs, root, lazy=True)
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
    return {
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


def replay_snapshot(bundle, dimensions, *, root):
    inputs = bundle["huaan_qdii_disclosure_calculation"]["inputs"]
    check_binding(bundle, inputs)
    points = calculate(inputs, root=root)["dimensions"]
    for key in ("holdings", "liquidity"):
        if any(
            dimensions[key].get(bound) is None or Decimal(str(dimensions[key][bound])) != Decimal(points[key])
            for bound in ("lower", "upper", "candidate")
        ):
            raise ValueError("QDII_COMPLETION_DIMENSION_REPLAY:" + key)
        reviewed = bundle["reviewed_document"]["dimensions"][key]
        values = (
            [reviewed.get(bound) for bound in ("lower", "upper", "candidate")]
            if isinstance(reviewed, dict)
            else [reviewed]
        )
        if any(value is None or Decimal(str(value)) != Decimal(points[key]) for value in values):
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
