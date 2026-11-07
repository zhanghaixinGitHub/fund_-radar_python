"""完整持仓、原行业行、真实成交及有原文的人工质量判断编译；不读HHI/K/Q候选。"""

import importlib
import re
from decimal import Decimal as D
from decimal import localcontext

from app.services.fund_quality_v2_asset_structure import capco_publication
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_active_execution import _assets, _stocks
from app.services.fund_rating_direct_index_math import capacity
from app.services.fund_rating_metrics import equity_structure

_source = importlib.import_module((__package__ + "." if __package__ else "") + "fund_rating_thematic_source")
iso, plain, require = _source.iso, _source.plain, _source.require
STRICT_PEERS = ["002450", "001714", "001628", "003053", "001223"]


def ticker(code):
    return code + (".BJ" if code.startswith(("4", "8", "9")) else ".SH" if code.startswith("6") else ".SZ")


def full_book(source, descriptor, code):
    sha = descriptor["report_ref"]
    source.original(sha)
    text = source.body(sha)
    header = plain(text)[:18000]
    require(code in header and "基金" in header, "BOOK_LEGAL_PRODUCT:" + code)
    require(str(report_period(text)[1]) == "2026-06-30", "BOOK_ECONOMIC_PERIOD:" + code)
    rows = _stocks(text, True)
    assets, _ = _assets(text)
    total = D(assets["权益投资"]["amount_cny"])
    require(sum(D(row["amount_cny"]) for row in rows) == total > 0, "BOOK_RECONCILIATION:" + code)
    # 证券代码来自完整表，不把前十当全仓；原表数量仅作原值验证/审计。
    return {"code": code, "source_hash": sha, "positions": rows, "stock_total": str(total), "assets": assets}


def mappings(source, spec, securities):
    registry = source.json(spec["issuer_registry_ref"])["stockList"]
    issuers = {row["code"]: row["orgId"] for row in registry}
    require(len(issuers) == len(registry) and all(issuers.values()), "ISSUER_REGISTRY_SCHEMA")
    classification = source.json(spec["classification_rows_ref"])
    rows = classification["records"]
    pdf = plain(source.body(spec["classification_pdf_ref"]))
    article = capco_publication(
        source.raw[spec["classification_article_ref"]],
        attachment_url=source.sources[spec["classification_pdf_ref"]]["url"],
    )
    source.original(spec["classification_article_ref"])
    require(
        article["period_end"] == "2026-06-30" and article["published_on"] <= source.asof,
        "CLASSIFICATION_ECONOMIC_PERIOD",
    )
    result, audit = {}, []
    for sec in sorted(securities):
        row = rows.get(sec)
        if row:
            raw_row = plain(row["raw_row"])
            require(raw_row in pdf and raw_row.startswith(sec), "CLASSIFICATION_RAW_ROW:" + sec)
            section = row["section_code"]
            require(re.fullmatch("[A-S]", section), "CLASSIFICATION_SECTION")
            after = raw_row.split(section, 1)[1]
            number = re.search(r"\d{2}", after)
            require(number is not None, "CLASSIFICATION_FINE_CODE:" + sec)
            industry = section + number[0]
            require(industry == row["industry_code"], "CLASSIFICATION_DERIVATION_CONFLICT:" + sec)
            issuer = issuers.get(sec)
            require(issuer, "ISSUER_UNMAPPED:" + sec)
            refs = [spec["classification_pdf_ref"], spec["issuer_registry_ref"]]
        else:
            supplement = spec.get("ipo_supplements", {}).get(sec)
            require(supplement, "IPO_CLASSIFICATION_PROOF:" + sec)
            source.claim(supplement)
            body = plain(source.body(supplement["source_hash"]))
            industry, legal = supplement["industry_code"], supplement["legal_name"]
            require(
                re.fullmatch(r"[A-S]\d{2}", industry)
                and sec in body
                and plain(legal) in body
                and industry in plain(supplement["quote"]),
                "IPO_IDENTITY_AND_CLASSIFICATION:" + sec,
            )
            # 新股原表使用同一两位细分类；其真实法定名称可作发行人键，不能填未知orgId。
            issuer = issuers.get(sec) or legal
            refs = [supplement["source_hash"]]
        result[sec] = {"issuer": issuer, "industry": industry}
        audit.append({"security": sec, **result[sec], "source_refs": refs})
    return result, audit


def market_history(source, spec, securities):
    market, all_dates = {}, set()
    for sec in sorted(securities):
        full = ticker(sec)
        require(full in spec["market_refs"], "MARKET_SOURCE_REQUIRED:" + full)
        rows_by_day = {}
        for sha in spec["market_refs"][full]:
            ref = source.original(sha)
            params = (
                ref.get("params_without_credentials")
                or ref.get("request_params_without_credentials")
                or ref.get("params")
                or {}
            )
            requested = str(params.get("ts_code", "")).split(",")
            require(full in requested and len(requested) == len(set(requested)), "MARKET_REQUEST_IDENTITY:" + full)
            for row in source.table(sha):
                require(row["ts_code"] in requested, "MARKET_RESPONSE_REQUEST_CONFLICT")
                if row["ts_code"] != full:
                    continue
                day = iso(row["trade_date"])
                if day > source.asof:
                    continue
                amount = D(str(row["amount"]))
                require(amount.is_finite() and amount >= 0, "MARKET_AMOUNT")
                require(day not in rows_by_day or rows_by_day[day] == amount, "MARKET_CONFLICTING_DUPLICATE")
                rows_by_day[day] = amount
        market[full] = rows_by_day
        all_dates.update(rows_by_day)
    days = sorted(all_dates)[-60:]
    require(
        len(days) == 60 and days == spec["market_dates"] and days[-1] == "2026-09-30", "ACTUAL_LAST60_MARKET_CALENDAR"
    )
    zeros = []
    for full, rows in market.items():
        missing = sorted(set(days) - rows.keys())
        if not missing:
            continue
        proof = spec.get("suspension_proofs", {}).get(full)
        require(proof, "MISSING_TURNOVER_NOT_ZERO:" + full)
        source.claim(proof)
        start, resume = iso(proof["suspended_from"]), iso(proof["resume_on"])
        body = plain(source.body(proof["source_hash"]))
        require(full[:6] in body[:12000] and "停牌" in body and "复牌" in body, "SUSPENSION_SECURITY_BINDING")
        require(all(start <= day < resume for day in missing), "ZERO_VOLUME_OUTSIDE_PROVEN_SUSPENSION")
        # 两个实际日期必须均在同一份原公告；不凭供应商缺日推定停牌。
        for day in [start, resume]:
            year, month, d = map(int, day.split("-"))
            require(re.search(rf"{year}年0?{month}月0?{d}日", body), "SUSPENSION_DATE_ORIGINAL")
        for day in missing:
            rows[day] = D(0)
        zeros.append({"security": full, "zero_dates": missing, "source_hash": proof["source_hash"]})
    return (
        days,
        {sec: {day: str(value) for day, value in rows.items() if day in days} for sec, rows in market.items()},
        zeros,
    )


def quality(source, spec, target):
    reviewed = source.json(spec["quality_review_ref"])
    require(source.sources[spec["quality_review_ref"]]["archive_kind"] == "DERIVED", "QUALITY_REVIEW_KIND")
    judgments = {row["security"]: row for row in reviewed["judgments"]}
    require(set(judgments) == {p["security"] for p in target["positions"]}, "ALL_QUALITY_POSITIONS_REQUIRED")
    positions, audit = [], []
    for p in target["positions"]:
        sec = p["security"]
        judgment = judgments[sec]
        require(
            judgment["state"] in {"NORMAL_REVIEWED", "MATERIAL_UNRESOLVED"}
            and len(judgment["reason"]) >= 30
            and judgment.get("research_judgment_not_legal_verdict") is True,
            "SOURCED_MANUAL_QUALITY_STATE",
        )
        claims = spec["quality_claims"].get(sec, [])
        require(len(claims) >= 2, "QUALITY_ORIGINAL_CLAIMS_REQUIRED:" + sec)
        periods = set()
        for doc in judgment["sources"]:
            sha = doc["source"]["sha256"]
            ref = source.original(sha)
            require(iso(doc["published_on"]) == iso(ref["published_on"]) <= source.asof, "QUALITY_SOURCE_DATE")
            body = plain(source.body(sha))
            require(sec in body[:15000], "QUALITY_ISSUER_IDENTITY")
            expected = "2025" if doc["period"] == "FY2025" else "2026" if doc["period"] == "H126" else None
            require(expected and expected in body[:2500], "QUALITY_COMPANY_PERIOD")
            periods.add(doc["period"])
        require(periods == {"FY2025", "H126"}, "QUALITY_TWO_PERIODS")
        for claim in claims:
            source.claim(claim)
        if judgment["state"] == "MATERIAL_UNRESOLVED":
            event_claims = spec.get("quality_material_event_claims", {}).get(sec, [])
            require(event_claims, "QUALITY_MATERIAL_EVENT_ORIGINAL:" + sec)
            for claim in event_claims:
                source.claim(claim)
        positions.append(
            {
                "security": ticker(sec),
                "amount": p["amount_cny"],
                "state": judgment["state"],
                "reason": judgment["reason"],
            }
        )
        audit.append(
            {
                "security": sec,
                "state": judgment["state"],
                "manual_review_ref": spec["quality_review_ref"],
                "original_claims": len(claims),
            }
        )
    # 目标H126无债券/其他投资风险资产；不把实质非股票信用风险藏在股票内部预算。
    require(D(target["assets"]["固定收益投资"]["amount_cny"]) == 0, "MATERIAL_NONSTOCK_CREDIT_REQUIRES_REVIEW")
    return {"kind": "asset_quality", "positions": positions, "stock_total": target["stock_total"]}, audit


def compile_structure(inputs, source):
    spec = inputs["structure"]
    require(spec["strict_peers"] == STRICT_PEERS, "STRUCTURE_PEER_SET")
    books = {code: full_book(source, spec["books"][code], code) for code in ["018590"] + STRICT_PEERS}
    securities = {p["security"] for book in books.values() for p in book["positions"]}
    mapping, map_audit = mappings(source, spec, securities)
    days, market, zeros = market_history(source, spec, securities)
    structures, capacities = {}, {}
    with localcontext() as ctx:
        ctx.prec = 70
        for code, book in books.items():
            mapped = [
                {
                    "security": ticker(p["security"]),
                    "issuer": mapping[p["security"]]["issuer"],
                    "industry": mapping[p["security"]]["industry"],
                    "amount": p["amount_cny"],
                }
                for p in book["positions"]
            ]
            structures[code] = equity_structure(mapped)
            positions = [
                {"stock_code": ticker(p["security"]), "market_value_cny": p["amount_cny"]} for p in book["positions"]
            ]
            capacities[code] = capacity(positions, book["stock_total"], days, market)
        nodes = {}
        for name, key in [
            ("issuer_concentration", "issuer_hhi"),
            ("industry_concentration", "industry_hhi"),
            ("market_capacity", "K"),
        ]:
            rows = capacities if name == "market_capacity" else structures
            nodes[name] = {
                "kind": "rank",
                "direction": "lower",
                "target": str(rows["018590"][key]),
                "expected_products": ["CN_PUBLIC_FUND:" + p for p in STRICT_PEERS],
                "peers": [{"product_id": "CN_PUBLIC_FUND:" + p, "value": str(rows[p][key])} for p in STRICT_PEERS],
            }
        nodes["asset_quality"], quality_audit = quality(source, spec, books["018590"])
    return nodes, {
        "books": books,
        "mappings": map_audit,
        "capacity": capacities,
        "structure": structures,
        "zero_turnover": zeros,
        "quality": quality_audit,
    }
