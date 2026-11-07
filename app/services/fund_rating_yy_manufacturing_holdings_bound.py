"""018125持仓区间独立候选：原件重建结构，人工资产质量判断明示保留争议。

不依赖研究目录或旧HHI/分数表。未知港股行业与优必选50/75争议按固定权重传播，
不补零、不取中点、不缩小分母；人工判断的语义不伪装成关键词自动评级。
"""

import json
import re
from collections import defaultdict
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal as D
from decimal import localcontext
from functools import lru_cache
from pathlib import Path

from bs4 import BeautifulSoup

from app.services.fund_quality_v2_asset_structure import capco_publication
from app.services.fund_quality_v2_balance_proofs import strict_balance_proof
from app.services.fund_rating_direct_index import Originals, _table
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_market_review import normalized_text as plain
from app.services.fund_rating_pdf import PDF_LOCK
from app.services.fund_rating_r5 import exposure_factor, peer_gap_score
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS, policy
from app.services.fund_rating_yy_manufacturing_liquidity import LEGAL, MANAGER, PRIMARY, require
from app.services.fund_rating_yy_manufacturing_liquidity import check_binding as _known_binding
from app.services.fund_rating_yy_manufacturing_market_liquidity import PEERS, _admission, _books, _HK_primary, _pages
from app.services.material_content import content_digest, document_pdfium, source_bytes

VERSION = "R5_018125_HOLDINGS_BOUND_ORIGINALS_20261030_1"
REGISTRY = "36b7b544ac4cfe6772078b84a38eeb7f53c60e8bd7cacc4a71e2b0a4f05172b4"


def _source(inputs, root):
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "018125"
        and inputs.get("share_class") == "C"
        and inputs.get("basis") == ACTIVE_BASIS
        and inputs.get("legal_name") == LEGAL
        and inputs.get("manager_name") == MANAGER
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("primary_ref") == PRIMARY,
        "YYH_SCOPE",
    )
    require(
        inputs.get("material_event_pending") is None and inputs.get("performance_release_requested") is False,
        "YYH_MATERIAL_UNKNOWN_REQUIRED",
    )
    require(
        not any(k in inputs for k in ["score", "dimensions", "packet", "HHI", "asset_quality_Q"]),
        "YYH_NO_PRECOMPUTED_SCORE",
    )
    c = inputs["capacity"]
    reviews = inputs["manual_asset_quality"]
    require(
        tuple(c["peer_codes"]) == PEERS
        and set(c["reports"]) == {"018125", *PEERS}
        and c["reports"]["018125"] == PRIMARY,
        "YYH_FIXED_PEERS_AND_REPORT",
    )
    require(
        inputs["issuer_registry_ref"] == REGISTRY
        and len(reviews) == 24
        and len({r["security"] for r in reviews}) == 24,
        "YYH_COMPLETE_REVIEW_SCOPE",
    )
    require(set(inputs["HK_legal"]) == {"00179.HK", "02507.HK", "06869.HK", "09880.HK"}, "YYH_HK_LEGAL_COVERAGE")
    required = {
        *c["reports"].values(),
        *(r["source_hash"] for r in c["admission_reviews"]),
        *(h for values in c["exposure_refs"].values() for h in values),
        inputs["classification_pdf_ref"],
        inputs["classification_article_ref"],
        REGISTRY,
        *(r["source_hash"] for r in inputs["IPO_classifications"].values()),
        *(r["source_hash"] for r in inputs["HK_legal"].values()),
        *(r[k] for r in inputs["government_bonds"] for k in ["terms_ref", "listing_ref"]),
    }
    financial = {
        *c["reports"].values(),
        *(r["source_hash"] for r in c["admission_reviews"]),
        inputs["classification_pdf_ref"],
        inputs["classification_article_ref"],
        *(r["source_hash"] for r in inputs["IPO_classifications"].values()),
        *(r["source_hash"] for r in inputs["HK_legal"].values()),
        *(r[k] for r in inputs["government_bonds"] for k in ["terms_ref", "listing_ref"]),
    }
    for review in reviews:
        required.add(review["price_ref"])
        required.update(review["applicable_event_refs"])
        required.update(review.get("financial_refs", {}).values())
        required.update(r["source_hash"] for r in review.get("financial_reports", []))
        required.update(review["catalog"].get("pages", {}).values())
        required.update(r["source_hash"] for r in review["catalog"].get("segments", []))
        financial.update(review["applicable_event_refs"])
        financial.update(review.get("financial_refs", {}).values())
        financial.update(r["source_hash"] for r in review.get("financial_reports", []))
        if review.get("external_visual_audit"):
            required.add(review["external_visual_audit"]["source_hash"])
            financial.add(review["external_visual_audit"]["source_hash"])
    require(set(inputs["sources"]) == required, "YYH_MISSING_OR_UNUSED_SOURCE")
    for h, ref in inputs["sources"].items():
        require(ref.get("sha256") == h and ref.get("archive_kind") == "ORIGINAL", "YYH_ORIGINAL_REQUIRED")
        if h in financial:
            require(
                isinstance(ref.get("published_on"), str)
                and date.fromisoformat(ref["published_on"]) <= date(2026, 9, 30),
                "YYH_FINANCIAL_PUBLICATION_REQUIRED",
            )
        else:
            require(ref.get("historical_lookup_original") is True, "YYH_HISTORICAL_LOOKUP_SCOPE")
    return Originals(inputs, Path(root).resolve())


def _page(source, h, physical):
    """公司长报仅重读需要的实际页，避免为少量原句反复解析整本年报。"""
    require(isinstance(physical, int) and physical >= 1, "YYH_PHYSICAL_PAGE")
    if h in source.pages_cache:
        require(physical <= len(source.pages_cache[h]), "YYH_PAGE_OUT_OF_RANGE")
        return source.pages_cache[h][physical - 1]
    cache = getattr(source, "yy_partial_pages", None)
    if cache is None:
        source.yy_partial_pages = {}
        cache = source.yy_partial_pages
    if (h, physical) not in cache:
        with PDF_LOCK:
            doc = document_pdfium(source.raw[h])
            try:
                require(physical <= len(doc), "YYH_PAGE_OUT_OF_RANGE")
                page = doc[physical - 1]
                textpage = page.get_textpage()
                try:
                    cache[(h, physical)] = textpage.get_text_range()
                finally:
                    textpage.close()
                    page.close()
            finally:
                doc.close()
    return cache[(h, physical)]


def _catalog(source, review):
    """完整目录只保证检索覆盖；事件适用性及分档仍由逐家公司人工审阅负责。"""
    catalog = review["catalog"]
    require(catalog["window"] == ["2025-10-01", "2026-09-30"], "YYH_CATALOG_WINDOW")
    code = review["security"]
    if not code.endswith(".HK"):
        pages = catalog["pages"]
        require({int(p) for p in pages} == set(range(1, (catalog["count"] + 29) // 30 + 1)), "YYH_CATALOG_PAGES")
        ids = []
        for num, h in pages.items():
            params = source.sources[h]["params"]
            require(
                params["pageNum"] == num
                and params["seDate"] == "2025-10-01~2026-09-30"
                and params["stock"].split(",")[0] == code[:6],
                "YYH_CATALOG_QUERY_IDENTITY",
            )
            rows = source.json(h)
            require(isinstance(rows["announcements"], list), "YYH_CATALOG_BODY")
            require(all(r["secCode"] == code[:6] for r in rows["announcements"]), "YYH_CATALOG_ISSUER")
            ids.extend(r["announcementId"] for r in rows["announcements"])
            if int(num) == len(pages):
                require(not rows.get("hasMore"), "YYH_CATALOG_LAST_PAGE")
        require(len(ids) == len(set(ids)) == catalog["count"], "YYH_COMPLETE_CATALOG_COUNT")
    else:
        segments = catalog["segments"]
        expected = (
            [("2025-10-01", "2026-09-30")]
            if code == "00179.HK"
            else [
                ("2025-10-01", "2025-12-31"),
                ("2026-01-01", "2026-03-31"),
                ("2026-04-01", "2026-06-30"),
                ("2026-07-01", "2026-09-30"),
            ]
        )
        require([(r["from"], r["to"]) for r in segments] == expected, "YYH_HK_CATALOG_FULL_WINDOW")
        count = 0
        unique = set()
        for segment in segments:
            h = segment["source_hash"]
            params = source.sources[h]["params"]
            require(
                params["stockId"] == ("322" if code == "00179.HK" else "1000208870")
                and params["from"] == segment["from"].replace("-", "")
                and params["to"] == segment["to"].replace("-", ""),
                "YYH_HK_CATALOG_QUERY_IDENTITY",
            )
            soup = BeautifulSoup(source.raw[h], "html.parser")
            m = re.search(r"Total records found:\s*([\d,]+)", soup.get_text(" ", strip=True))
            require(m is not None, "YYH_HK_CATALOG_OFFICIAL_TOTAL")
            total = int(m[1].replace(",", ""))
            docs = soup.select(".doc-link")
            require(
                len(docs) == total == segment["official_total"] == segment["captured"] and total < 100,
                "YYH_HK_CATALOG_CAPTURE",
            )
            for doc in docs:
                cells = doc.find_parent("tr").select("td")
                a = doc.find("a")
                published = (
                    datetime.strptime(
                        cells[0].get_text(" ", strip=True).replace("Release Time:", "").strip(), "%d/%m/%Y %H:%M"
                    )
                    .date()
                    .isoformat()
                )
                require(
                    cells[1].get_text(" ", strip=True).replace("Stock Code:", "").strip() == code[:5]
                    and segment["from"] <= published <= segment["to"],
                    "YYH_HK_CATALOG_ROW_SCOPE",
                )
                identity = (published, a.get_text(" ", strip=True), a["href"])
                require(identity not in unique, "YYH_HK_DUPLICATE_NOTICE")
                unique.add(identity)
            count += total
        require(count == catalog["count"], "YYH_HK_CATALOG_TOTAL")


def _structure(inputs, source, books):
    """从同日统一大类和法律主体重算HHI；三家港股不强行跨分类映射。"""
    pub = capco_publication(
        source.raw[inputs["classification_article_ref"]],
        attachment_url=source.sources[inputs["classification_pdf_ref"]]["url"],
    )
    require(pub["period_end"] == "2026-06-30" and pub["published_on"] == "2026-09-30", "YYH_CLASSIFICATION_PERIOD")
    registry_rows = source.json(REGISTRY)["stockList"]
    registry = {r["code"]: r for r in registry_rows}
    require(len(registry) == len(registry_rows), "YYH_REGISTRY_DUPLICATE")
    sectors = {}
    for code, row in inputs["classification_rows"].items():
        raw = plain(row["raw_row"])
        page = plain(_page(source, inputs["classification_pdf_ref"], row["pdf_page"]))
        require(
            raw.startswith(code) and raw in page and re.fullmatch("[A-S]\\d{2}", row["industry_code"]) is not None,
            "YYH_CLASSIFICATION_ORIGINAL_ROW",
        )
        suffix = raw.split(row["section_code"], 1)[1]
        fine = re.search(r"\d{2}", suffix)
        require(fine and row["section_code"] + fine[0] == row["industry_code"], "YYH_CLASSIFICATION_FINE_GRAIN")
        sectors[code] = row["industry_code"]
    for code, row in inputs["IPO_classifications"].items():
        require(row["claims"], "YYH_IPO_CLASSIFICATION_CLAIMS")
        require(
            code in plain("\n".join(_pages(source, row["source_hash"]))), "YYH_IPO_ORIGINAL_SECURITY_IDENTITY:" + code
        )
        for claim in row["claims"]:
            page = plain(_page(source, row["source_hash"], claim["physical_page"]))
            quote = plain(claim["quote"])
            require(
                quote in page
                and re.search(r"(行业分类代码|所属行业代码|行业代码).{0,60}" + row["industry_code"], quote),
                "YYH_IPO_SAME_GRAIN:" + code,
            )
        sectors[code] = row["industry_code"]
    legal = inputs["HK_legal"]
    proofs = {}
    for code, node in legal.items():
        text = plain("\n".join(_page(source, node["source_hash"], p) for p in node["physical_pages"]))
        expected = {
            "02507.HK": "HK:CIRRUS_AIRCRAFT_LIMITED_CAYMAN",
            "00179.HK": "HK:JOHNSON_ELECTRIC_HOLDINGS_LIMITED_BERMUDA",
            "09880.HK": "HK:UBTECH_ROBOTICS_CORP_LTD_PRC",
        }
        require(code == "06869.HK" or node["issuer"] == expected[code], "YYH_LEGAL_ENTITY_NOT_CONTROLLER_MERGE")
        if code == "02507.HK":
            require(all(x in text for x in ["2507", "西銳飛機有限公司", "開曼群島"]), "YYH_CIRRUS_LEGAL")
        elif code == "06869.HK":
            require(
                all(x in text for x in ["601869", "06869", "A股", "H股"])
                and node["issuer"] == registry["601869"]["orgId"],
                "YYH_A_H_SAME_ISSUER",
            )
            require(node["industry"] == sectors["601869"], "YYH_A_H_SAME_CLASSIFICATION")
        elif code == "00179.HK":
            require("JohnsonElectricHoldingsLimited" in text, "YYH_JE_LEGAL")
        else:
            require("深圳市優必選科技股份有限公司" in text, "YYH_UB_LEGAL")
        require(code == "06869.HK" or node["industry"] is None, "YYH_UNKNOWN_HK_INDUSTRY_PRESERVED")
        proofs[code] = node
    arithmetic = {}
    for code, book in books.items():
        issuers = defaultdict(lambda: D(0))
        buckets = defaultdict(lambda: D(0))
        unknown = []
        for security, amount in book["holdings"].items():
            if security.endswith(".HK"):
                issuer, sector = legal[security]["issuer"], legal[security]["industry"]
            else:
                short = security[:6]
                require(
                    short in registry and short in sectors and registry[short]["orgId"],
                    "YYH_COMPLETE_A_IDENTITY_AND_INDUSTRY",
                )
                issuer, sector = registry[short]["orgId"], sectors[short]
            issuers[issuer] += amount
            if sector is None:
                unknown.append(amount)
            else:
                buckets[sector] += amount
        total = book["total"]
        require(sum(issuers.values()) == sum(book["holdings"].values()) == total, "YYH_FULL_STOCK_DENOMINATOR")
        h = sum((a / total) ** 2 for a in issuers.values())
        fixed = sum((a / total) ** 2 for a in buckets.values())
        weights = [a / total for a in unknown]
        largest = max(buckets.values(), default=D(0)) / total
        lo = fixed + sum(w * w for w in weights)
        hi = fixed - largest * largest + (largest + sum(weights)) ** 2
        arithmetic[code] = {
            "issuer_HHI": h,
            "industry_HHI_interval": [lo, hi],
            "unmapped_stock_amount_CNY": sum(unknown),
            "full_stock_amount_CNY": total,
            "legal_issuers": len(issuers),
        }
    t = arithmetic["018125"]
    issuer = peer_gap_score(t["issuer_HHI"], [arithmetic[p]["issuer_HHI"] for p in PEERS]) * exposure_factor(
        "issuer_concentration", t["issuer_HHI"]
    )
    low, high = t["industry_HHI_interval"]
    industry_lo = peer_gap_score(high, [arithmetic[p]["industry_HHI_interval"][0] for p in PEERS]) * exposure_factor(
        "industry_concentration", high
    )
    industry_hi = peer_gap_score(low, [arithmetic[p]["industry_HHI_interval"][1] for p in PEERS]) * exposure_factor(
        "industry_concentration", low
    )
    return issuer, [industry_lo, industry_hi], arithmetic


def _quality(inputs, source, target):
    reviews = inputs["manual_asset_quality"]
    require({r["security"] for r in reviews} == set(target["holdings"]), "YYH_FULL_QUALITY_COVERAGE")
    low = high = D(0)
    audit = []
    for review in reviews:
        code = review["security"]
        require(len(review["judgment"]) >= 60, "YYH_MANUAL_Q_JUDGMENT_REQUIRED")
        _catalog(source, review)
        if not code.endswith(".HK"):
            require(
                review["manual_q"] == "75" and review["q_interval"] is None and len(review["facts"]) >= 3,
                "YYH_A_EXPLICIT_MANUAL_JUDGMENT",
            )
            financial = review["financial_refs"]
            require(set(financial) == {"annual", "interim"}, "YYH_NECESSARY_FINANCIALS")
            require(
                source.sources[financial["annual"]].get("economic_report_period") == "2025-12-31"
                and source.sources[financial["interim"]].get("economic_report_period") == "2026-06-30",
                "YYH_FINANCIAL_ECONOMIC_PERIOD",
            )
            claims = review["financial_claims"]
            require(
                {"interim_key_metric_pages", "interim_consolidated_balance_pages"} <= {c["role"] for c in claims},
                "YYH_CURRENT_METRICS_AND_CONSOLIDATED_BALANCE",
            )
            for claim in claims:
                require(
                    claim["source_hash"] in financial.values()
                    and len(plain(claim["quote"])) >= 20
                    and plain(claim["quote"]) in plain(_page(source, claim["source_hash"], claim["physical_page"])),
                    "YYH_FINANCIAL_ORIGINAL_PAGE:" + code + ":" + str(claim["physical_page"]),
                )
            visual = review["external_visual_audit"]
            if visual:
                require(
                    code == "688017.SH"
                    and visual["opinion_reviewed"] is True
                    and visual["actual_opinion_physical_pages"] == [2, 3, 4, 5, 6]
                    and "天衡" in visual["audit_firm"]
                    and "公允反映" in visual["opinion"],
                    "YYH_SCANNED_AUDIT_REVIEW",
                )
                _page(source, visual["source_hash"], 6)
            else:
                audit_text = plain("".join(c["quote"] for c in claims if c["role"] == "actual_annual_audit_pages"))
                require(
                    all(x in audit_text for x in ["我们审计了", "公允反映", "现金流量"]),
                    "YYH_ACTUAL_ANNUAL_AUDIT_REVIEW",
                )
            quotes = _table(source.json(review["price_ref"]))
            found = [r for r in quotes if r["ts_code"] == code and r["trade_date"] == "20260930"]
            require(
                len(found) == 1 and D(str(found[0]["close"])) > 0 and D(str(found[0]["amount"])) > 0,
                "YYH_ACTUAL_VALUATION_QUOTE",
            )
        else:
            reports = review["financial_reports"]
            require(len(reports) == (4 if code == "00179.HK" else 2), "YYH_HK_REPORT_COMPLETENESS")
            require(
                all(r["economic_period"] and r["economic_period"] <= "2026-09-30" for r in reports),
                "YYH_HK_ECONOMIC_PERIOD",
            )
            require(review["manual_visual_pages"], "YYH_HK_REVIEW_PAGES")
            for item in review["manual_visual_pages"]:
                require(item["source_hash"] in {r["source_hash"] for r in reports}, "YYH_HK_REVIEW_REPORT_BINDING")
                for page in item["pages"]:
                    _page(source, item["source_hash"], page)
            if code == "09880.HK":
                require(
                    review["manual_q"] is None and review["q_interval"] == ["50", "75"],
                    "YYH_UB_MATERIALITY_DISPUTE_PRESERVED",
                )
            else:
                require(review["manual_q"] == "75" and review["q_interval"] is None, "YYH_JE_EXPLICIT_MANUAL_Q")
            quote = _HK_primary(source.raw[review["price_ref"]], "2026-09-30")[code[:5]]
            require(quote["shares"] > 0 and quote["amount"] > 0, "YYH_HK_ACTUAL_VALUATION")
        bounds = [D(review["manual_q"])] * 2 if review["manual_q"] is not None else list(map(D, review["q_interval"]))
        require(0 <= bounds[0] <= bounds[1] <= 100, "YYH_Q_BOUNDS")
        value = target["holdings"][code]
        low += value * bounds[0]
        high += value * bounds[1]
        audit.append(
            {
                "security": code,
                "manual_q": review["manual_q"],
                "q_interval": review["q_interval"],
                "actual_stock_amount_CNY": str(value),
                "applicable_event_originals": len(review["applicable_event_refs"]),
                "catalog_count": review["catalog"]["count"],
            }
        )
    return [low / target["total"], high / target["total"]], audit


def _compile(inputs, source):
    with localcontext() as ctx:
        ctx.prec = 65
        admission = _admission(source, inputs["capacity"])
        books = _books(source, inputs["capacity"])
        issuer, industry, structure = _structure(inputs, source, books)
        quality, quality_audit = _quality(inputs, source, books["018125"])
        pages = _pages(source, PRIMARY)
        balance = strict_balance_proof(pages, observed_on="2026-06-30", source_hash=PRIMARY)["point_book"]
        require(
            balance is not None
            and all(D(balance["assets"][k]) == 0 for k in ["衍生金融资产", "买入返售金融资产"])
            and D(balance["fund_amount"]) == 0,
            "YYH_CURRENT_NONSTOCK_SCOPE",
        )
        require(
            "本基金本报告期内未投资股指期货" in plain(pages[45])
            and "本基金本报告期内未投资国债期货" in plain(pages[46]),
            "YYH_NOT_ZERO_BALANCE_INFERRED_NOTIONAL",
        )
        require(
            {r["code"] for r in inputs["government_bonds"]} == {"019792", "019785"}
            and "国家债券169,912,671.731.36" in plain(pages[45]),
            "YYH_GOVERNMENT_NONSTOCK_EXPOSURE",
        )
        for bond in inputs["government_bonds"]:
            text = BeautifulSoup(source.raw[bond["terms_ref"]], "html.parser").get_text(" ", strip=True)
            require(
                all(x in plain(text) for x in ["财政部", "兑付日", bond["maturity"]])
                and bond["code"] in plain(_page(source, bond["listing_ref"], 1)),
                "YYH_NONSTOCK_GOVERNMENT_ORIGINALS",
            )
        terms = policy(ACTIVE_BASIS)["terms"]["holdings"]
        require(
            {k: D(w) for k, w in terms}
            == {"issuer_concentration": D(35), "industry_concentration": D(35), "asset_quality": D(30)},
            "YYH_FIXED_WEIGHTS",
        )
        lo = D(".35") * issuer + D(".35") * industry[0] + D(".30") * quality[0]
        hi = D(".35") * issuer + D(".35") * industry[1] + D(".30") * quality[1]
        require(0 <= lo <= hi <= 100, "YYH_HOLDINGS_BOUNDS")

    def strings(value):
        if isinstance(value, D):
            return str(value)
        if isinstance(value, dict):
            return {k: strings(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strings(v) for v in value]
        return value

    return strings(
        {
            "dimensions": {
                "holdings": {
                    "lower": lo,
                    "upper": hi,
                    "candidate": None,
                    "missing": [],
                    "uncertainty": ["UBTECH_FINANCIAL_MATERIALITY_50_OR_75", "THREE_HK_UNIFIED_INDUSTRY_UNKNOWN"],
                }
            },
            "complete_score": None,
            "grade": None,
            "issuer_score": issuer,
            "industry_score_interval": industry,
            "asset_quality_Q_interval": quality,
            "audit": {
                "peer_admission": admission,
                "structure": structure,
                "manual_quality_source_replays": quality_audit,
                "all_24_stock_amount_coverage": "1",
                "manual_semantics_not_automatic_keyword_ratings": True,
                "material_event_pending": None,
                "nonstock_not_assigned_stock_quality_100": True,
            },
        }
    )


@lru_cache(maxsize=1)
def _cached(serialized, root):
    inputs = json.loads(serialized)
    return _compile(inputs, _source(inputs, root))


def calculate(inputs, *, root):
    """每次先重验原字节；人工判断和区间边界仅作为有原件锚点的显式输入。"""
    _source(inputs, root)
    return deepcopy(_cached(canonical(inputs), str(Path(root).resolve())))


def check_binding(bundle, inputs):
    _known_binding(bundle, inputs)
    require(inputs.get("version") == VERSION, "YYH_VERSION_BINDING")


def verify_registered_original(bundle, ref, *, root):
    """只对本组件绑定且与原ref完全相同的PDF沿50MB上限核验，普通归档不放宽。"""

    inputs = bundle["yy_manufacturing_holdings_bound_calculation"]["inputs"]
    check_binding(bundle, inputs)
    digest = ref.get("sha256")
    require(
        inputs["sources"].get(digest) == ref and ref.get("archive_kind") == "ORIGINAL" and ref.get("kind") == "pdf",
        "YYH_BOUND_ORIGINAL_PDF_REFERENCE",
    )
    relative = Path(ref["file"])
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(
        not relative.is_absolute()
        and path.is_relative_to(root / "sources")
        and relative.as_posix() in {"sources/" + digest + ".pdf", "sources/pipeline/" + digest + ".pdf"},
        "YYH_BOUND_PDF_SOURCE_PATH",
    )
    raw = source_bytes(path)
    require(
        raw.startswith(b"%PDF") and len(raw) <= 50000000 and (content_digest(raw) == digest),
        "YYH_BOUND_PDF_ORIGINAL_HASH",
    )


def replay_snapshot_interval(bundle, dimensions, *, root):
    inputs = bundle["yy_manufacturing_holdings_bound_calculation"]["inputs"]
    check_binding(bundle, inputs)
    expected = calculate(inputs, root=root)["dimensions"]["holdings"]
    actual = dimensions["holdings"]
    declared = bundle["reviewed_document"]["dimensions"]["holdings"]
    require(
        actual.get("candidate") is None
        and all(D(actual[k]) == D(expected[k]) for k in ["lower", "upper"])
        and declared.get("missing") == expected["missing"]
        and declared.get("uncertainty") == expected["uncertainty"],
        "YYH_SNAPSHOT_ORIGINAL_REPLAY",
    )


def prepare(session, request, *, root):
    """只读current预演；全原件与人工分档在独立目录重放，不登记或发布。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json, canonical_source
    from app.services.fund_rating_reviewed_snapshot import BASIS

    raw = deepcopy(request["inputs"])
    root = Path(root).resolve()
    require(request["fund_code"] == "018125" and request["as_of_date"] == raw["as_of_date"], "YYH_REQUEST_SCOPE")
    current = current_inputs(session, ["018125"]).get("018125")
    require(current and str(current["as_of_date"]) == raw["as_of_date"], "YYH_CURRENT_REQUIRED")
    require(
        not request.get("expected_current_ref") or current["batch_id"] == request["expected_current_ref"],
        "YYH_CURRENT_CHANGED",
    )
    bundle = deepcopy(next(m["bundle"] for m in current["input_snapshot"]["members"] if m["fund_code"] == "018125"))
    check_binding(bundle, raw)
    require(bundle.get("basis") == BASIS and not bundle.get("r5_calculation"), "YYH_PARTIAL_SNAPSHOT_REQUIRED")
    for h, ref in raw["sources"].items():
        raw["sources"][h] = canonical_source(root, ref)
    value = calculate(raw, root=root)
    doc = bundle["reviewed_document"]
    before = deepcopy(doc["dimensions"])
    reason = request["explanations"]["holdings"].strip()
    require(len(reason) >= 40, "YYH_BUSINESS_REASON_REQUIRED")
    doc["dimensions"]["holdings"] = {**value["dimensions"]["holdings"], "reason": reason}
    bundle.setdefault("public_explanations", {})["holdings"] = reason
    bundle.get("public_metrics", {}).pop("holdings", None)
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, doc),
        yy_manufacturing_holdings_bound_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    existing = {r["sha256"] for r in bundle.setdefault("extra_archives", [])}
    bundle["extra_archives"].extend(ref for h, ref in raw["sources"].items() if h not in existing)
    require(
        all(doc["dimensions"][k] == v for k, v in before.items() if k != "holdings"), "YYH_OTHER_DIMENSIONS_CHANGED"
    )
    replay_snapshot_interval(bundle, doc["dimensions"], root=root)
    # 大于25MB的德昌原年报须走对应快照候选中的绑定PDF校验；由调用方隔离重放
    # validate_sources和calculate，不能在未安装候选钩子的正式快照函数里假装通过。
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in ["basis", "reviewed_document"]}]}
