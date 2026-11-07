"""004605持仓及流动性的可移植原件适配候选，仅编译部分维度。

全仓、行业及成交量重新读取原始字节；公司质量保留逐公司人工序档判断，
不把人工判断说成审计结论。模块不联网、不登记、不发布，不接受外部总分。
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal as D
from decimal import localcontext

from app.services.fund_rating_r5 import _calculate_frozen
from app.services.material_content import content_digest, document_pdfium, source_bytes, source_exists, source_size

VERSION = "R5_004605_ACTIVE_PORTFOLIO_ORIGINALS_20261030_1_CANDIDATE"
FIELD = "active_portfolio_components_calculation"
FROZEN_COHORT = "2d12743a08362ccba9cf36fccd15e28697b43d5b06e1e16c22f3ef7e4a7febf3"
PRIMARY = "2ce51f77ed65ae0f62e2aacc025b343998eea47dcbc91b3150c06346b2d1c43e"
PROTOCOL_HASH = "abc11cd7e58357e6be0b9dafd4c36c37a3d4f7d27505094fe22210540c5e1023"


def require(condition, reason):
    if not condition:
        raise ValueError("ACTIVE_PORTFOLIO_" + reason)


def compact(value):
    """只归一版式字符；金额、证券身份及正文内容均不替换。"""
    value = unicodedata.normalize("NFKC", value.replace("\\r", "").replace("\\n", ""))
    return re.sub(r"\s+", "", value).replace("，", ",").replace("：", ":")


class Sources:
    """每次调用核全部已绑定字节；缓存限本次复算，不跨请求沿用结论。"""

    def __init__(self, inputs, root):
        self.root, self.refs = root.resolve(), inputs["sources"]
        self.text_cache, self.page_cache = {}, {}
        require(1 <= len(self.refs) <= 1500, "SOURCE_COUNT")
        for digest, ref in self.refs.items():
            require(digest == ref["sha256"], "SOURCE_IDENTITY")
            self.bytes(digest)
            if ref.get("published_on"):
                require(ref["published_on"] <= inputs["as_of_date"], "SOURCE_AFTER_CUTOFF")

    def bytes(self, digest):
        ref = self.refs[digest]
        path = (self.root / ref["file"]).resolve()
        require(path.is_relative_to(self.root / "sources") and source_exists(path), "SOURCE_PATH")
        require(0 < source_size(path) <= 50000000, "SOURCE_SIZE")
        body = source_bytes(path)
        require(content_digest(body) == digest, "SOURCE_HASH")
        return body

    def json(self, digest):
        return json.loads(self.bytes(digest))

    def pages(self, digest):
        if digest not in self.page_cache:
            raw = self.bytes(digest)
            require(raw.startswith(b"%PDF"), "ACTUAL_PDF_REQUIRED")
            with document_pdfium(raw) as document:
                self.page_cache[digest] = [page.get_textpage().get_text_range() for page in document]
        return self.page_cache[digest]

    def text(self, digest):
        if digest not in self.text_cache:
            raw = self.bytes(digest)
            if raw.startswith(b"%PDF"):
                body = "\n".join(self.pages(digest))
            elif self.refs[digest].get("encoding") == "gb18030":
                body = raw.decode("gb18030")
            else:
                data = json.loads(raw)
                require(data.get("success") == 1 and bool(data.get("data", {}).get("notice_content")), "NOTICE_BODY")
                body = data["data"]["notice_content"]
            self.text_cache[digest] = body
        return self.text_cache[digest]


def check_binding(bundle, inputs):
    """限制为同一份额、时期、原报告及已采用参照；禁止覆盖其他类型组件。"""
    require(
        bundle.get("basis") == "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3"
        and bundle.get("family") == "ACTIVE_EQUITY"
        and inputs.get("version") == VERSION
        and inputs.get("fund_code") == bundle["fund_code"] == "004605"
        and inputs.get("share_class") == bundle["share_class"] == "C"
        and inputs.get("as_of_date") == bundle["as_of_date"] == "2026-09-30"
        and inputs.get("holding_date") == bundle["holding_date"] == "2026-06-30"
        and inputs.get("legal_name") == bundle["identity"]["legal_name"]
        and inputs.get("manager_name") == bundle["identity"]["manager_name"]
        and inputs.get("primary_ref") == bundle["original_report_archive"]["sha256"] == PRIMARY,
        "BUNDLE_IDENTITY",
    )
    require(
        not any(
            bundle.get(k) for k in ["r5_calculation", "passive_components_calculation", "passive_holdings_calculation"]
        ),
        "CONFLICTING_COMPONENT",
    )
    packets = bundle.get("active_performance_calculations", [])
    require(
        bool(packets) and all(p["inputs"]["cohort_ref"] == inputs["cohort_ref"] == FROZEN_COHORT for p in packets),
        "ADOPTED_COHORT",
    )
    require(
        all(p["inputs"]["material_event_review_ref"] == inputs["material_event_review_ref"] for p in packets),
        "ADOPTED_MATERIAL_REVIEW",
    )


def _peers(source, inputs):
    frozen = source.json(inputs["cohort_ref"])
    require(inputs["cohort_ref"] == FROZEN_COHORT, "FROZEN_COHORT_HASH")
    group = frozen["holdings_liquidity"]
    require(group["thresholds"] == {"mean_gap_pp": 5, "max_gap_pp": 10, "domestic_min_pct": 100}, "FROZEN_THRESHOLDS")
    rows = group["rows"]
    peers = sorted(row["code"] for row in rows if row["state"] == "ADMITTED")
    require(len(peers) == len(set(peers)) == 185, "FIXED_PEER_COUNT")
    require([row["code"] for row in rows if "PENDING" in row["state"]] == ["010064"], "PENDING_COHORT_CHANGED")
    proof = source.json(inputs["exclusion_review_ref"])
    digest = proof["source"]["sha256"]
    primary = source.json(digest)
    require(primary.get("success") == 1 and primary["data"]["notice_date"][:10] == "2024-04-22", "EXCLUSION_ORIGINAL")
    require({"010064", "010065"} <= {str(v["stock"]) for v in primary["data"]["security"]}, "EXCLUSION_IDENTITY")
    body = compact(primary["data"]["notice_content"])
    equity = re.search(r"权益投资([\d,]+\.\d{2})", body)
    nav = re.search(r"期末基金资产净值([\d,]+\.\d{2})([\d,]+\.\d{2})", body)
    require(equity is not None and nav is not None, "EXCLUSION_ORIGINAL_AMOUNTS")
    stock = D(equity[1].replace(",", ""))
    total = sum(D(nav[i].replace(",", "")) for i in [1, 2])
    gap = abs(D(frozen["target_quarter_exposure"]["2024-03-31"]) - stock / total * 100)
    require(
        stock == D(proof["equity_CNY"]) and total == D(proof["all_share_NAV_CNY"]) and gap > 10,
        "STRICT_EXPOSURE_EXCLUSION",
    )
    require(sorted(inputs["peer_codes"]) == peers and not {"004604", "004605", "010064"} & set(peers), "PEER_IDENTITY")
    return peers


def _portfolios(source, inputs, peers):
    require(set(inputs["portfolios"]) == set(peers) | {"004605"}, "FULL_PORTFOLIO_COVERAGE")
    portfolios = {}
    for code, report in inputs["portfolios"].items():
        digest = report["source_ref"]
        require(report["period"] == "2026-06-30" and report["published_on"] <= "2026-09-30", "PORTFOLIO_PERIOD")
        raw = source.bytes(digest)
        if raw.startswith(b"%PDF"):
            require(code == "004666", "UNSUPPORTED_PORTFOLIO_PDF")
            text = "\n".join(
                line
                for line in source.text(digest).splitlines()
                if not re.fullmatch(
                    r"长城久嘉创新成长混合\s*2026\s*年中期报告|第\s*\d+\s*页\s*共\s*\d+\s*页", line.strip()
                )
            )
            require(content_digest(text.encode("utf-8")) == report["text_derivation_ref"], "PDF_TEXT_DERIVATION")
            require(
                "2026年6月30日" in compact(text)[:6000] and "送出日期:2026年8月31日" in compact(text)[:6000],
                "PDF_PERIOD_PUBLICATION",
            )
        else:
            notice = json.loads(raw)
            require(
                notice.get("success") == 1 and notice["data"]["notice_date"][:10] == report["published_on"],
                "PORTFOLIO_PUBLICATION",
            )
            require(code in {str(v["stock"]) for v in notice["data"]["security"]}, "PORTFOLIO_LEGAL_IDENTITY")
            text = notice["data"]["notice_content"]
        require(re.search(r"2026年0?6月30日", compact(text)[:10000]) is not None, "PORTFOLIO_HOLDING_DATE")
        try:
            parsed = parse_stock_text(code, text)
        except (AssertionError, ValueError, AttributeError):
            parsed = parse_wrapped_text(code, text)
        require(parsed["full_table_reconciled"] and D(parsed["total_stock_value"]) > 0, "FULL_TABLE_RECONCILIATION")
        portfolios[code] = parsed
    return portfolios


def _holdings(source, inputs, peers, portfolios, nodes):
    """原件逐证券合并发行人及同口径行业；序档仍由明确的逐公司研究判断给出。"""
    meta = inputs["classification"]
    require(meta["published_on"] == "2026-09-30", "CLASSIFICATION_DATE")
    require(
        source.refs[meta["pdf_ref"]]["url"] in source.bytes(meta["article_ref"]).decode("utf-8"),
        "CLASSIFICATION_ATTACHMENT",
    )
    classes = parse_classifier(source.pages(meta["pdf_ref"]))
    require(len(classes) == 5521, "CLASSIFICATION_COMPLETENESS")
    for item in meta["ipo_supplements"]:
        pages = source.pages(item["source_ref"])
        require(
            item["source_type"] == "ISSUER_IPO_DOCUMENT_SAME_GRAIN_SUPPLEMENT_NOT_CAPCO_LIST_ASSIGNMENT",
            "IPO_CLASSIFICATION_PROVENANCE",
        )
        require(
            item["security"] in compact("".join(pages))
            and any(item["industry_code"] in compact(pages[p - 1]) for p in item["reviewed_pdf_pages"]),
            "IPO_CLASSIFICATION_ORIGINAL",
        )
        classes[item["security"]] = item
    issuers = {row["code"]: row for row in source.json(inputs["issuer_identity_ref"])["stockList"]}
    for portfolio in portfolios.values():
        by_issuer, by_industry = defaultdict(D), defaultdict(D)
        total = D(portfolio["total_stock_value"])
        for stock in portfolio["stocks"]:
            amount, security = D(stock["market_value"]), stock["security"]
            if amount == 0:
                continue
            require(amount > 0 and bool(issuers[security]["orgId"]), "ISSUER_IDENTITY")
            by_issuer[issuers[security]["orgId"]] += amount
            by_industry[classes[security]["industry_code"]] += amount
        portfolio["issuer_concentration"] = sum((v / total) ** 2 for v in by_issuer.values())
        portfolio["industry_concentration"] = sum((v / total) ** 2 for v in by_industry.values())
    for name in ["issuer_concentration", "industry_concentration"]:
        _rank_matches(nodes[name], peers, {c: p[name] for c, p in portfolios.items()})
    own = portfolios["004605"]
    amounts = {row["security"]: D(row["market_value"]) for row in own["stocks"] if D(row["market_value"]) > 0}
    quality = nodes["asset_quality"]
    require(
        len(quality["positions"]) == len(amounts) == 50
        and {p["security"] for p in quality["positions"]} == set(amounts),
        "ISSUER_JUDGMENT_COVERAGE",
    )
    require(D(quality["stock_total"]) == D(own["total_stock_value"]), "QUALITY_STOCK_TOTAL")
    validation = source.json(inputs["quality_validation_ref"])
    require(
        validation["covered_securities"] == validation["expected_securities"] == 50 and not validation["gaps"],
        "QUALITY_REVIEW_CLOSED",
    )
    validated = {v["security"]: v for v in validation["rows"]}
    for row in quality["positions"]:
        require(D(row["amount"]) == amounts[row["security"]], "QUALITY_POSITION_AMOUNT")
        require(row["judgment_ref"] == validated[row["security"]]["judgment_sha256"], "QUALITY_JUDGMENT_ARCHIVE")
        q = source.json(row["judgment_ref"])
        require(
            q["security"] == row["security"]
            and q["fund_code"] == "004605"
            and q["as_of"] == "2026-09-30"
            and q["review_closed"] is True
            and not q["remaining"],
            "QUALITY_JUDGMENT_IDENTITY",
        )
        require(
            q["quality_point"] in [50, 75]
            and row["state"] == {50: "BASIC_ROLE_EVIDENCED", 75: "NORMAL_REVIEWED"}[q["quality_point"]]
            and row["reason"] == q["manual_reason"]
            and bool(q["manual_reason"].strip()),
            "QUALITY_MANUAL_STATE",
        )
        require(row["assurance_notes"] == q["source_period_notes"], "QUALITY_ASSURANCE_SCOPE")
        bound = {s["source"]["sha256"] for s in q["sources"].values()} | {
            s["source"]["sha256"] for s in q["additional_originals"]
        }
        require(bound <= set(row["source_refs"]), "QUALITY_ORIGINAL_BINDING")
        for digest in bound:
            require(source.bytes(digest).startswith(b"%PDF"), "QUALITY_ORIGINAL_PDF")
        for image in q.get("manual_scan_transcription", {}).get("images", []):
            source.bytes(image["sha256"])
    return {"portfolios": len(portfolios), "fixed_peers": len(peers), "issuer_judgments": len(amounts)}


def _rank_matches(node, peers, values):
    expected = ["CN_PUBLIC_FUND:" + c for c in peers]
    require(
        node["kind"] == "rank" and node["direction"] == "lower" and node["expected_products"] == expected,
        "FIXED_RANK_UNIVERSE",
    )
    require(
        len(node["peers"]) == len(peers) and {r["product_id"] for r in node["peers"]} == set(expected),
        "FIXED_RANK_COVERAGE",
    )
    require(abs(D(node["target"]) - values["004605"]) < D("1e-55"), "TARGET_OBSERVATION")
    for row in node["peers"]:
        require(abs(D(row["value"]) - values[row["product_id"].split(":")[1]]) < D("1e-55"), "PEER_OBSERVATION")


def _licensed(source, digest):
    raw = source.json(digest)
    require(raw.get("code") == 0 and raw["data"].get("has_more") is False, "MARKET_SUCCESS_AND_PAGINATION")
    fields = raw["data"]["fields"]
    require(bool(fields) and len(fields) == len(set(fields)), "MARKET_FIELDS")
    return [dict(zip(fields, values, strict=True)) for values in raw["data"]["items"]]


def _capacity(source, inputs, peers, portfolios, node):
    """成交额千元转元；停牌补零须有原公告，上市前日期不算零成交。"""
    market_inputs = inputs["market"]
    days = market_inputs["calendar"]
    require(
        len(days) == len(set(days)) == 60
        and days == sorted(days)
        and days[0] == "2026-07-08"
        and days[-1] == "2026-09-30",
        "CAPACITY_WINDOW",
    )
    require(set(market_inputs["daily_refs"]) == set(days), "MARKET_DAY_COVERAGE")
    market = {}
    for day in days:
        rows = _licensed(source, market_inputs["daily_refs"][day])
        require(all(str(r["trade_date"]) == day.replace("-", "") for r in rows), "MARKET_TRADE_DATE")
        market[day] = {r["ts_code"].split(".")[0]: r for r in rows}
        require(len(market[day]) == len(rows) and len(rows) >= 1000, "MARKET_DAY_COMPLETENESS")
    listing = {}
    for digest in market_inputs["listing_refs"]:
        for row in _licensed(source, digest):
            code = row["ts_code"].split(".")[0]
            require(code not in listing or listing[code]["list_date"] == row["list_date"], "LIST_DATE_CONFLICT")
            listing[code] = row
    for digest in market_inputs["bj_recheck_refs"]:
        for row in _licensed(source, digest):
            date = str(row["trade_date"])
            day, code = f"{date[:4]}-{date[4:6]}-{date[6:]}", row["ts_code"].split(".")[0]
            if day not in market:
                continue
            require(
                code not in market[day] or D(str(market[day][code]["amount"])) == D(str(row["amount"])),
                "BJ_AMOUNT_CONFLICT",
            )
            market[day][code] = row
    proofs = source.json(market_inputs["suspension_review_ref"])
    require(proofs["as_of"] == "2026-09-30", "SUSPENSION_CUTOFF")
    zeros = {}
    for item in proofs["securities"]:
        require(item["events"] and item["security"] not in zeros, "SUSPENSION_REVIEW")
        for event in item["events"]:
            require(event["published_on"] <= "2026-09-30", "SUSPENSION_PUBLICATION")
            text = source.pages(event["source"]["sha256"])[event["page"] - 1]
            require(compact(event["clause"]) in compact(text), "SUSPENSION_PRIMARY_CLAUSE")
        zeros[item["security"]] = set(item["verified_suspension_days"])
    values, observation_count = {}, 0
    for code, portfolio in portfolios.items():
        contributions = []
        total = D(portfolio["total_stock_value"])
        for stock in portfolio["stocks"]:
            security, mv = stock["security"], D(stock["market_value"])
            if mv == 0:
                continue
            amounts = []
            for day in days:
                row = market[day].get(security)
                if row is not None and row["amount"] is not None:
                    amount = D(str(row["amount"])) * 1000
                    require(amount.is_finite() and amount >= 0, "MARKET_AMOUNT")
                    require(day not in zeros.get(security, set()) or amount == 0, "SUSPENSION_AMOUNT_CONFLICT")
                    amounts.append(amount)
                elif day in zeros.get(security, set()):
                    amounts.append(D(0))
                elif (
                    security in listing
                    and listing[security].get("list_date")
                    and day.replace("-", "") < listing[security]["list_date"]
                ):
                    continue
                else:
                    raise ValueError("ACTIVE_PORTFOLIO_REAL_AMOUNT_OR_CAUSE_REQUIRED:" + security + ":" + day)
            require(len(amounts) >= 30 and sum(amounts) > 0, "INSUFFICIENT_LISTED_OBSERVATIONS")
            adv = sum(amounts) / len(amounts)
            contributions.append(mv / total * mv / adv)
            observation_count += len(amounts)
        values[code] = sum(contributions)
    _rank_matches(node, peers, values)
    return {
        "market_days": len(days),
        "portfolios": len(values),
        "stock_day_observations": observation_count,
        "suspension_reviews": len(zeros),
    }


def _liquidity_facts(source, inputs, own, nodes):
    """用本基金报告和适用合同重核金额、持有人守恒及支付定义域。"""
    baseline = source.json(inputs["report_review_ref"])
    report = inputs["primary_notice_ref"]
    texts = [compact(source.text(report)), compact(source.text(inputs["primary_ref"]))]
    # PDF两种列阅读顺序可能不同；只为原件中完整摘录增加第二解析器。

    from app.services.material_content import document_reader as PdfReader

    texts.append(compact("".join(p.extract_text() or "" for p in PdfReader(source.bytes(inputs["primary_ref"])).pages)))

    def literal(quote):
        require(bool(quote) and any(compact(quote) in t for t in texts), "REPORT_PRIMARY_QUOTE")

    require(all(v in texts[0] for v in ["004605", "2026年06月30日", "692,719,737.23"]), "PRIMARY_REPORT_SCOPE")
    restriction = baseline["restriction"]
    literal(baseline["restriction_scope_quote"])
    restricted = sum(D(row["amount"]) for row in restriction["rows"])
    require(restricted == D(restriction["restricted_amount"]), "RESTRICTION_SUM")
    for row in restriction["rows"]:
        literal(format(D(row["amount"]), ",.2f"))
    node = nodes["restricted"]
    require(
        node["scope_closed"] is True
        and D(node["restricted_amount"]) == restricted
        and D(node["net_assets"]) == D(own["nav"]),
        "RESTRICTION_FACTS",
    )
    holders = baseline["holders"]
    require(not holders["gaps"], "HOLDERS_GAPS")
    for key in ["structure_quote", "single_holder_quote", "actual_custodian_execution_quote"]:
        literal(holders[key])
    for quote in holders["actual_liquidity_execution_quotes"]:
        literal(quote)
    literal(holders["share_flows"]["quote"])
    for row in holders["share_flows"]["rows"]:
        require(D(row["opening"]) + D(row["subscriptions"]) - D(row["redemptions"]) == D(row["closing"]), "SHARE_FLOW")
    for row in holders["single_holder_rows"]:
        require(D(row["begin"]) + D(row["purchase"]) - D(row["redemption"]) == D(row["end"]), "SINGLE_HOLDER_FLOW")
    require(
        any(D(row["end_percent"]) >= 20 for row in holders["single_holder_rows"] if row["end_percent"] is not None),
        "CONCENTRATED_HOLDER",
    )
    for institutional, percent, personal, _, total in holders["holder_structure_reconciled_rows"]:
        require(
            D(institutional) + D(personal) == D(total)
            and abs(D(institutional) / D(total) * 100 - D(percent)) <= D(".005"),
            "HOLDER_STRUCTURE",
        )
    for mechanism in holders["redemption_mechanism"]:
        body = compact(source.text(mechanism["source"]["sha256"]))
        require(
            compact(inputs["legal_name"]) in body and compact(mechanism["quote"]) in body,
            "APPLICABLE_REDEMPTION_CONTRACT",
        )
    holder_node = nodes["holders"]
    require(
        holder_node["structure_type"] == "CONCENTRATED_OPEN_END"
        and holder_node["confirmed_pressure_effects"] == []
        and D(holder_node["actual_redemption_shares"]) == D(holders["share_flows"]["redemptions"]),
        "HOLDER_OBSERVATION",
    )
    require(
        all(
            holder_node[k] is True
            for k in [
                "structure_closed",
                "share_flow_closed",
                "actual_execution_verified",
                "redemption_mechanism_verified",
            ]
        ),
        "HOLDER_EXECUTION_CLOSED",
    )
    payment = baseline["payment"]
    for key in ["total_liabilities", "demand_deposits_principal"]:
        literal(payment["known_accounting_observations"][key]["original_row"])
    known = nodes["known_payment"]
    require(
        D(known["bank_principal"]) == D(payment["bank_principal"])
        and D(known["all_book_liabilities"]) == D(payment["all_book_liabilities"]),
        "PAYMENT_ORIGINAL_AMOUNTS",
    )
    require(
        known["broker_settlement_principal"] is None
        and known["settlement_liabilities"] is None
        and known["as_of"] == inputs["holding_date"]
        and known["disclosed_restrictions"] == [],
        "PAYMENT_UNKNOWN_NOT_IMPUTED",
    )
    from app.services.fund_rating_payment_domain import payment_domain_score

    score, _ = payment_domain_score(known)
    require(score == 75, "PAYMENT_DOMAIN")


def calculate(inputs, *, root):
    """重算所选维度，完整总分及等级始终为空；不改变其他组件。"""
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "004605"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("primary_ref") == PRIMARY,
        "INPUT_SCOPE",
    )
    require(not any(k in inputs for k in ["score", "dimensions", "grade", "candidate", "point"]), "EXTERNAL_SCORE")
    selected = inputs["selected_dimensions"]
    require(
        # 当前只核准完整流动性。持仓质量的旧人工分档正在逐公司复核，
        # 不能因同一候选模块已含该分支而把未经核准的Q一并带入正式评分。
        bool(selected) and len(selected) == len(set(selected)) and set(selected) == {"liquidity"},
        "SELECTED_DIMENSIONS",
    )
    source = Sources(inputs, root)
    from app.services.fund_rating_active_performance import _require_material_review

    material = _require_material_review(inputs, root)
    with localcontext() as context:
        context.prec = 60
        peers = _peers(source, inputs)
        portfolios = _portfolios(source, inputs, peers)
        packet = deepcopy(inputs["packet"])
        require(
            packet["fund_code"] == "004605"
            and packet["as_of_date"] == "2026-09-30"
            and packet["quality_metrics"]["material_event_pending"] is False
            and packet["quality_metrics"]["target_product_aliases"]
            == ["CN_PUBLIC_FUND:004604", "CN_PUBLIC_FUND:004605"],
            "PACKET_SCOPE",
        )
        nodes, audit = packet["quality_metrics"]["observations"], {}
        expected_nodes = (
            {"issuer_concentration", "industry_concentration", "asset_quality"} if "holdings" in selected else set()
        ) | ({"market_capacity", "restricted", "known_payment", "holders"} if "liquidity" in selected else set())
        require(set(nodes) == expected_nodes, "OBSERVATION_SCOPE")
        if "holdings" in selected:
            audit["holdings"] = _holdings(source, inputs, peers, portfolios, nodes)
        if "liquidity" in selected:
            audit["liquidity"] = _capacity(source, inputs, peers, portfolios, nodes["market_capacity"])
            _liquidity_facts(source, inputs, portfolios["004605"], nodes)
        packet["r4_evidence_registry"] = {
            h: {**ref, "path": str((root / ref["file"]).resolve())} for h, ref in inputs["sources"].items()
        }
        packet["raw_archive"] = [{"sha256": h, "kind": ref["archive_kind"]} for h, ref in inputs["sources"].items()]
        require(inputs["protocol_ref"] == PROTOCOL_HASH, "FROZEN_PROTOCOL")
        protocol = source.json(inputs["protocol_ref"])
        result = _calculate_frozen(packet, protocol, {}, selected_dimensions=set(selected))
    require(
        result["score"] is None and result["grade"] is None and set(result["dimensions"]) == set(selected),
        "COMPONENT_ONLY",
    )
    return {
        "version": VERSION,
        "dimensions": result["dimensions"],
        "subscores": result["subscores"],
        "audit": audit,
        "material_review": material,
        "complete_score": None,
        "grade": None,
        "limits": [
            "公司质量是原件支持的人工序档判断，未作独立财务审计或经验概率校准。",
            "支付结论限本报告日披露金额及用途定义域，不代表未来支付保证。",
        ],
    }


def validate_input_archive(bundle, *, root, archive_reader):
    raw = bundle[FIELD]
    require(archive_reader(raw["input_archive"], root) == raw["inputs"], "INPUT_ARCHIVE_MISMATCH")
    check_binding(bundle, raw["inputs"])


def replay_snapshot(bundle, dimensions, *, root):
    raw = bundle[FIELD]["inputs"]
    check_binding(bundle, raw)
    result = calculate(raw, root=root)
    for key, score in result["dimensions"].items():
        require(
            all(
                dimensions[key][bound] is not None and abs(D(dimensions[key][bound]) - D(score)) <= D("1e-55")
                for bound in ["lower", "upper", "candidate"]
            ),
            "SNAPSHOT_RECALCULATION:" + key,
        )
    return result


def compile_bundle(bundle, request, *, root):
    """只更新所选维度及其输入档案，保留日期、既有组件和其他研究记录。"""
    from app.services.fund_rating_pipeline import archive_json
    from app.services.fund_rating_pipeline_prepare import canonical_source

    bundle = deepcopy(bundle)
    require(FIELD not in bundle, "COMPONENT_ALREADY_PRESENT")
    raw = deepcopy(request["inputs"])
    check_binding(bundle, raw)
    require(request["fund_code"] == raw["fund_code"] and request["as_of_date"] == raw["as_of_date"], "REQUEST_IDENTITY")
    raw["sources"] = {h: canonical_source(root, ref) for h, ref in raw["sources"].items()}
    result = calculate(raw, root=root)
    for key, score in result["dimensions"].items():
        reason = request["explanations"][key].strip()
        require(len(reason) >= 40, "PUBLIC_EXPLANATION")
        bundle["reviewed_document"]["dimensions"][key] = {
            "lower": score,
            "upper": score,
            "candidate": score,
            "reason": reason,
        }
        bundle.setdefault("public_explanations", {})[key] = reason
        bundle.get("public_metrics", {}).pop(key, None)
    bundle.update(research_format="single", research_archive=archive_json(root, bundle["reviewed_document"]))
    bundle[FIELD] = {"inputs": raw, "input_archive": archive_json(root, raw)}
    extras = {ref["sha256"]: ref for ref in bundle.get("extra_archives", [])}
    extras.update({h: {k: ref[k] for k in ["file", "sha256", "kind"]} for h, ref in raw["sources"].items()})
    bundle["extra_archives"] = list(extras.values())
    return bundle, result


def prepare(session, request, *, root):
    """只读标准prepare候选；验证封存快照，返回可审阅清单，不写数据库。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot

    current = current_inputs(session, [request["fund_code"]]).get(request["fund_code"])
    require(
        current is not None
        and current["batch_id"] == request["prior_rating_ref"]
        and str(current["as_of_date"]) == request["as_of_date"],
        "CURRENT_PERIOD_AND_REF",
    )
    bundle = next(m["bundle"] for m in current["input_snapshot"]["members"] if m["fund_code"] == request["fund_code"])
    updated, _ = compile_bundle(bundle, request, root=root)
    validate_sources(updated, datetime.now(UTC), root=root)
    snapshot(updated, root=root)
    return {"basis": BASIS, "entries": [{k: v for k, v in updated.items() if k not in ["basis", "reviewed_document"]}]}


def parse_stock_text(code, text):
    """沿用已核全表解析与版式恢复；独立总额、净值及证券集合不闭合即拒绝。"""
    match = list(re.finditer("(?:所有|全部)股票投资明细", text))
    assert match, "NO_FULL_STOCK_TABLE"
    part = text[match[-1].end() :]
    end = re.search("\\d\\.4\\s*(?:报告期内股票|本期股票)", part)
    assert end, "TABLE_END_MISSING"
    part = part[: end.start()]
    part = re.sub(
        "([\\d,]+\\.\\d)\\s+([\\d]+\\.\\d{2})\\s*\\n\\s*(\\d)(?=\\s|$)", lambda m: m[1] + m[3] + " " + m[2], part
    )
    pattern = (
        "(?:^|\\n)[ \\t]*([\\d,]+)[ \\t]+(\\d{6})[ \\t]+(.+?)[ \\t]+"
        "([\\d,]+(?:\\.\\d+)?)[ \\t]+([\\d,]+\\.\\d{2})[ \\t]+([\\d.]+|-)"
    )
    stocks = []
    for m in re.finditer(pattern, part):
        stocks.append(
            {
                "rank": int(m[1].replace(",", "")),
                "security": m[2],
                "name": re.sub("\\s+", "", m[3]),
                "quantity": m[4],
                "market_value": m[5].replace(",", ""),
                "reported_nav_pct": m[6],
            }
        )
    assert stocks and len({s["security"] for s in stocks}) == len(stocks), ("SECURITY_DUPLICATE", code)
    assert stocks[0]["rank"] == 1 and all((a["rank"] <= b["rank"] for a, b in zip(stocks, stocks[1:], strict=False))), (
        "STOCK_RANKS",
        code,
        len(stocks),
    )
    compact = re.sub("\\s+", "", text)
    asset = compact[compact.rfind("期末基金资产组合情况") :]
    asset = re.sub(
        "([\\d,]+\\.\\d)\\s+([\\d]+\\.\\d{2})\\s*\\n\\s*(\\d)(?=\\s|$)",
        lambda m: m[1] + m[3] + " " + m[2],
        text[text.rfind("期末基金资产组合情况") :],
    )
    stock = re.search("其中[：:]股票\\s+([\\d,]+\\.\\d{2})", asset)
    assert stock
    stock_total = D(stock[1].replace(",", ""))
    assert sum(D(s["market_value"]) for s in stocks) == stock_total, (
        "FULL_STOCK_SUM",
        code,
        len(stocks),
        sum(D(s["market_value"]) for s in stocks),
        stock_total,
    )
    nav_matches = list(re.finditer("(?:净资产合计|所有者权益合计)([\\d,]+\\.\\d{2})", compact))
    assert nav_matches
    nav = D(nav_matches[0][1].replace(",", ""))
    assert 0 < stock_total / nav < D("1.5")
    return {
        "code": code,
        "stocks": stocks,
        "total_stock_value": str(stock_total),
        "nav": str(nav),
        "full_table_reconciled": True,
        "period": "2026-06-30",
    }


def parse_wrapped_text(code, text):
    """沿用已核全表解析与版式恢复；独立总额、净值及证券集合不闭合即拒绝。"""
    compact = re.sub("\\s+", "", text)
    nav = D(re.search("(?:净资产合计|所有者权益合计)([\\d,]+\\.\\d{2})", compact)[1].replace(",", ""))
    head = list(re.finditer("(?:所\\s*有|全\\s*部)\\s*股\\s*票\\s*投\\s*资\\s*明\\s*细", text))[-1]
    part = text[head.end() :]
    stop = re.search("(?:\\d\\.4\\s*)?(?:报告期内|本期)股票投资组合的重大变动", part)
    assert stop, "BOUNDARY"
    part = part[: stop.start()]
    table = text[text.rfind("期末基金资产组合情况") :]
    stock_total = D(re.search("其中[：:]\\s*股票\\s+([\\d,]+\\.\\d{2})", table)[1].replace(",", ""))
    pattern = "(?m)^[ \\t]*(\\d{1,4})[ \\t]+(\\d{6})[ \\t]*(.*)$"
    starts = list(re.finditer(pattern, part))
    rows = []
    edits = []
    for i, m in enumerate(starts):
        chunk = m[3] + part[m.end() : starts[i + 1].start() if i + 1 < len(starts) else len(part)]
        dash = re.search("([\\d,]+)\\s+[-－—]\\s+0\\.00\\s*$", m[3])
        if dash:
            rows.append(
                {
                    "rank": int(m[1]),
                    "security": m[2],
                    "name": m[3].split()[0],
                    "market_value": None,
                    "quantity": dash[1],
                    "reported_nav_pct": "0.00",
                    "original_amount_dash": True,
                    "original_row": chunk,
                }
            )
            continue
        tokens = re.findall("(?<![A-Za-z\\d])(?:\\d[\\d,]*(?:\\.\\d*)?|\\.\\d+)(?![A-Za-z\\d])", chunk)
        if not tokens:
            raise ValueError("EMPTY_NUMBERS")
        choices = []
        for offset in [0]:
            tail = tokens[offset:]
            if len(tail) == 3:
                choices.append((tail[0], tail[1], tail[2], "UNCHANGED"))
            if len(tail) == 4 and re.fullmatch("\\d[\\d,]*", tail[1]) and re.fullmatch("\\.\\d{2}", tail[3]):
                choices.append((tail[0], tail[1] + tail[3], tail[2], "AMOUNT_CENT_FRACTION_AFTER_PERCENT_COLUMN"))
            if len(tail) == 4 and re.fullmatch("\\d[\\d,]*\\.\\d", tail[1]) and re.fullmatch("\\d", tail[3]):
                choices.append((tail[0], tail[1] + tail[3], tail[2], "AMOUNT_LAST_CENT_AFTER_PERCENT_COLUMN"))
            if len(tail) == 4 and re.fullmatch("\\d[\\d,]*\\.", tail[1]) and re.fullmatch("\\d{2}", tail[3]):
                choices.append((tail[0], tail[1] + tail[3], tail[2], "AMOUNT_CENTS_AFTER_PERCENT_COLUMN"))
            if len(tail) == 4 and len(m[1]) == 3 and re.fullmatch("\\d", tail[3]):
                candidate_rank = int(m[1] + tail[3])
                if candidate_rank >= 1000 and re.fullmatch("\\d[\\d,]*\\.\\d{2}", tail[1]):
                    choices.append((tail[0], tail[1], tail[2], "WRAPPED_FOUR_DIGIT_RANK_SUFFIX"))
        valid = []
        for quantity, amount, weight, rule in choices:
            try:
                a = D(amount.replace(",", ""))
                w = D(weight)
                if (
                    re.fullmatch("\\d[\\d,]*\\.\\d{2}", amount)
                    and a >= 0
                    and (0 <= w <= 100)
                    and (abs(a / nav * 100 - w) <= D(".0050000001"))
                ):
                    valid.append((quantity, amount, weight, rule))
            except Exception:
                pass
        assert len(set(valid)) == 1, ("AMBIGUOUS_CELL", m[2], tokens, valid)
        qty, amount, weight, rule = valid[0]
        name = re.split("\\d", m[3], maxsplit=1)[0].strip()
        rows.append(
            {
                "rank": int(m[1]),
                "security": m[2],
                "name": name,
                "market_value": str(D(amount.replace(",", ""))),
                "quantity": qty,
                "reported_nav_pct": weight,
            }
        )
        if rule != "UNCHANGED":
            edits.append({"security": m[2], "rule": rule, "original": chunk, "cells": [qty, amount, weight]})
    assert rows and rows[0]["rank"] == 1 and (len({r["security"] for r in rows}) == len(rows))
    known_total = sum(D(r["market_value"]) for r in rows if r["market_value"] is not None)
    assert known_total == stock_total, ("SUM_MISMATCH", code, len(rows), str(known_total), str(stock_total))
    for r in rows:
        if r["market_value"] is None:
            r.update(market_value="0", zero_proof="完整非负股票表已知金额等于独立股票总额，剩余横杠金额唯一为零")
    return {
        "code": code,
        "stocks": rows,
        "total_stock_value": str(stock_total),
        "nav": str(nav),
        "full_table_reconciled": True,
        "period": "2026-06-30",
    }


def parse_classifier(pages):
    """沿用已核全表解析与版式恢复；独立总额、净值及证券集合不闭合即拒绝。"""
    out = {}
    for page_number, page in enumerate(pages, 1):
        matches = list(re.finditer("(?m)^\\s*(\\d{6})\\s+", page))
        for i, m in enumerate(matches):
            text = re.sub(
                "\\s+", " ", page[m.start() : matches[i + 1].start() if i + 1 < len(matches) else len(page)]
            ).strip()
            c = re.search("^\\d{6}\\s+(.+?)\\s+([A-S])\\s+(.+?)(?:\\s+([A-Z]{2})\\s+(.+?))?\\s+(\\d{2})\\s+(.+)", text)
            assert c, ("CLASSIFICATION_ROW_PARSE", text)
            code = m[1]
            row = {
                "security": code,
                "name": c[1].replace(" ", ""),
                "section_code": c[2],
                "section_name": c[3].replace(" ", ""),
                "secondary_code": c[4],
                "secondary_name": c[5].replace(" ", "") if c[5] else None,
                "industry_code": c[2] + c[6],
                "industry_name": re.sub("\\s+", "", c[7]),
                "pdf_page": page_number,
                "raw_row": text,
            }
            assert code not in out
            out[code] = row
    return out
