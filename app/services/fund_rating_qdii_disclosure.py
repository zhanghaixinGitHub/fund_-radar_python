"""境外主动股票的一年公开证据重放。只读原件，不采集、不发布、不关补件。

保留QDII类型和本份额实际日历。新指标评价公开的一年表现、当期核心合同
配置及集中度；不把旧R5全仓私有细项或跨期权证未知自动判为正常或违规。
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services.fund_rating_active_execution import Originals
from app.services.fund_rating_direct_index import _flows
from app.services.fund_rating_management_inputs import calculate as management_calculate
from app.services.fund_rating_market_review import annual_fee_score, fee_profile
from app.services.fund_rating_public_market import balance
from app.services.fund_rating_r5 import PROTOCOL, signed_gap_score
from app.services.material_content import content_digest, source_bytes

VERSION = "QDII_ACTIVE_DISCLOSURE_ONE_YEAR_20261111_V1"
KEYS = ("return", "risk", "efficiency", "stability", "management", "holdings", "liquidity", "cost")
SUPPORTED = {
    "019449": ("C", "摩根日本精选股票型证券投资基金(QDII)", "摩根基金管理(中国)有限公司", "CIFM_JAPAN"),
    "008764": ("C", "天弘越南市场股票型发起式证券投资基金(QDII)", "天弘基金管理有限公司", "TH_VIETNAM"),
}


def require(value, code):
    if not value:
        raise ValueError("QDII_DISCLOSURE_" + code)


def plain(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", str(value)))


def auxiliary_json(inputs, root, role):
    ref = inputs["auxiliary_archives"][role]
    path = (root / ref["file"]).resolve()
    require(path.is_relative_to(root.resolve() / "sources") and ref.get("archive_kind") == "DERIVED", "AUXILIARY_SCOPE")
    raw = source_bytes(path)
    require(content_digest(raw) == ref["sha256"], "AUXILIARY_HASH")
    return json.loads(raw)


def money(value):
    value = D(str(value).replace(",", ""))
    require(value.is_finite(), "NUMBER")
    return value


@contextmanager
def stage(errors, key):
    try:
        yield
    except (KeyError, ValueError, FileNotFoundError) as e:
        errors[key] = str(e)


def section(text, start, end):
    matches = list(re.finditer(start, text))
    require(matches, "SECTION:" + start)
    tail = text[matches[-1].start() :]
    m = re.search(end, tail[1:])
    require(m, "SECTION_END:" + end)
    return tail[: m.start() + 1]


def primary_binding(report, inputs):
    """报告封面和资产负债表都核实际期末；份额列由原文固定顺序核准。"""
    first = re.search(r"(20\d{2})年(\d{1,2})月(\d{1,2})日", plain(report[:1000]))
    require(first and str(date(*map(int, first.groups()))) == inputs["holding_date"], "PRIMARY_COVER_PERIOD")
    sheet = plain(section(report, r"6\.1\s*资产负债表", r"6\.2\s*利润表"))
    m = re.search(r"本期末(?:上年度末)?(20\d{2})年(\d{1,2})月(\d{1,2})日", sheet)
    require(m and str(date(*map(int, m.groups()))) == inputs["holding_date"], "PRIMARY_BALANCE_PERIOD")
    overview = section(report, r"2\.1\s*基金基本情况", r"2\.2\s*基金产品说明")
    codes = re.findall(r"(?<!\d)0\d{5}(?!\d)", overview)
    expected = (
        ["007280", "007280", "019449"] if inputs["fund_code"] == "019449" else ["008763", "008763", "008764", "022524"]
    )
    require(codes == expected, "PRIMARY_SHARE_COLUMN_CODES")
    require("下属分级基金" in overview and re.search(r"(?:QDII|DII)\)C", plain(overview)), "PRIMARY_SHARE_COLUMN_CLASS")
    return {
        "cover_period": inputs["holding_date"],
        "balance_period": inputs["holding_date"],
        "column_codes": expected[1:],
        "own_share_class": "C",
    }


def month_ends(nav, as_of):
    end = date.fromisoformat(as_of)
    idx = end.year * 12 + end.month - 1
    ends = []
    for n in range(idx - 12, idx + 1):
        prefix = f"{n // 12:04d}-{n % 12 + 1:02d}"
        days = [d for d in nav if d[:7] == prefix and d <= as_of]
        require(days, "MONTH_MISSING:" + prefix)
        ends.append(max(days))
    require(ends[-1] == as_of, "ASOF_NOT_LAST_VALUATION")
    return ends


def own_history(source, inputs):
    if inputs["adapter"] == "CIFM_JAPAN":
        from app.services.fund_rating_public_japan import issuer_history

        h = issuer_history(
            source.json(inputs["provider_ref"]),
            source.json(inputs["issuer_detail_ref"]),
            source.json(inputs["issuer_nav_ref"]),
            date.fromisoformat(inputs["as_of_date"]),
        )
        rows = {r["nav_date"]: r for r in h["rows"]}
        nav = {d: money(rows[d]["adjusted_nav"]) for d in h["dates"]}
        provider = source.json(inputs["provider_ref"])
        year_start = month_ends(nav, inputs["as_of_date"])[0]
        licensed = {r["nav_date"] for r in provider["history"]["rows"] if year_start <= r["nav_date"] <= max(nav)}
        require(licensed == {d for d in nav if year_start <= d}, "NAV_CALENDAR_PROVIDER_DISAGREEMENT")
        require(
            all(money(rows[d]["unit_nav"]) == money(rows[d]["adjusted_nav"]) for d in nav), "UNRECONCILED_DISTRIBUTION"
        )
        return nav, h["issuer_valuation_verification"]
    raw = source.raw[inputs["nav_chart_ref"]].decode("utf-8-sig")
    require(re.search(r'fS_code\s*=\s*["\x27]008764["\x27]', raw), "NAV_CHART_SHARE_IDENTITY")
    m = re.search(r"var\s+Data_netWorthTrend\s*=\s*(\[.*?\]);", raw)
    require(m, "NAV_CHART_ROWS")
    points = json.loads(m[1])
    nav = {}
    for row in points:
        dt = datetime.fromtimestamp(row["x"] / 1000, timezone(timedelta(hours=8)))
        require(dt.hour == 0 and dt.minute == 0, "NAV_CHART_DATE_ZONE")
        day = dt.date().isoformat()
        require(day not in nav, "NAV_DUPLICATE")
        nav[day] = money(row["y"])
        # 分红或折算若出现，必须另接原件复权，不把曲线当成自动复权。
        if "2025-09-01" <= day <= inputs["as_of_date"]:
            require(not row.get("unitMoney"), "NAV_EVENT_REVIEW_REQUIRED")
    snapshot = auxiliary_json(inputs, source.root, "provider_crosscheck")
    require(snapshot.get("schema") == "008764_READ_ONLY_NAV_SNAPSHOT", "NAV_PROVIDER_SCHEMA")
    source_info = snapshot["source_registry"]
    require(source_info["enabled"] and source_info["authorization_verified_at"], "NAV_PROVIDER_AUTHORIZATION")
    db = {r["nav_date"]: r for r in snapshot["rows"]}
    require(len(db) == len(snapshot["rows"]), "NAV_PROVIDER_DUPLICATE")
    nav = {d: v for d, v in nav.items() if "2023-09-28" <= d <= inputs["as_of_date"]}
    require(nav and not snapshot["dividends"], "DIVIDEND_REVIEW_REQUIRED")
    year_start = month_ends(nav, inputs["as_of_date"])[0]
    require(
        {d for d in nav if year_start <= d} == {d for d in db if year_start <= d <= inputs["as_of_date"]},
        "NAV_CALENDAR_PROVIDER_DISAGREEMENT",
    )
    for d, v in nav.items():
        require(d in db and db[d]["fund_code"] == inputs["fund_code"], "NAV_PROVIDER_DATE_OR_SHARE")
        require(v == money(db[d]["unit_nav"]) == money(db[d]["adjusted_nav"]), "NAV_PROVIDER_VALUE_CONFLICT")
    return nav, {
        "currency": "CNY",
        "actual_published_dates": len(nav),
        "all_provider_values_matched": True,
        "late_announcements": [
            {"date": d, "ann_date": db[d]["ann_date"]} for d in nav if db[d]["ann_date"] > inputs["as_of_date"]
        ],
    }


def portfolio(source, inputs, report):
    b = balance([report.replace(":", "：")], require_stock=False)
    if inputs["adapter"] == "CIFM_JAPAN":
        from app.services.fund_rating_public_japan import positions

        stocks, _ = positions([report], b["stock_amount"])
    else:
        # 只用按来源绑定的原表片段重放：逐片核原行、拼代码/金额及整表精确合计。
        cache = auxiliary_json(inputs, source.root, "stock_parse_cache")
        require(
            cache.get("fund_code") == "008764" and cache.get("holding_date") == inputs["holding_date"],
            "STOCK_PARSE_IDENTITY",
        )
        require(cache["source"]["sha256"] == inputs["stock_original_ref"], "STOCK_PARSE_SOURCE_BINDING")
        original = source.body(inputs["stock_original_ref"])
        stocks = []
        for row in cache["rows"]:
            require("".join(f["text"] for f in row["code_fragments"]) == row["isin"], "STOCK_ISIN_FRAGMENTS")
            amount = money("".join(f["text"] for f in row["amount_fragments"]))
            require(amount == money(row["cny"]), "STOCK_AMOUNT_FRAGMENTS")
            require(
                all(
                    f["original_line"] in original and f["text"] in f["original_line"]
                    for f in row["code_fragments"] + row["amount_fragments"]
                ),
                "STOCK_FRAGMENT_NOT_ORIGINAL",
            )
            stocks.append({"stock_code": row["isin"], "market_value_cny": str(amount)})
    require(stocks and len({r["stock_code"] for r in stocks}) == len(stocks), "STOCK_DUPLICATE")
    require(sum(money(r["market_value_cny"]) for r in stocks) == money(b["stock_amount"]), "STOCK_BALANCE_SUM")
    return b, stocks


def _cash_principal(report):
    text = plain(section(report, r"6\.4\.7\.1\s*(?:货币资金|银行存款)", r"6\.4\.7\.2"))
    m = re.search(r"(?:减[:：]应计利息[\d,.]+)?(?:等于[:：]|其中[:：])本金([\d,]+\.\d{2})", text)
    if not m:
        m = re.search(r"(?:本金)([\d,]+\.\d{2})", text)
    require(m, "CASH_PRINCIPAL")
    return money(m[1])


def execution(source, inputs, report, balance):
    legal = plain(source.body(inputs["legal_ref"]))
    require("股票资产占基金资产" in legal, "STOCK_BUDGET_CLAUSE")
    stock = money(balance["stock_amount"])
    nav = money(balance["net_assets"])
    total = money(balance["total_assets"])
    band = re.search(r"股票资产占基金资产的(?:比例(?:为)?)?(\d+)%[—－\-~至](\d+)%", legal)
    if band:
        low, high = money(band[1]) / 100, money(band[2]) / 100
    else:
        minimum = re.search(r"股票资产占基金资产的比例不低于(\d+)%", legal)
        require(minimum, "STOCK_BUDGET_BOUND")
        low, high = money(minimum[1]) / 100, D(1)
    cash = _cash_principal(report)
    require(0 <= cash <= money(balance["assets"]["货币资金"]), "CASH_PRINCIPAL_CURRENT_BALANCE")
    cash_clause = re.search(
        r"投资于现金(?:或者|或)到期日在一年以内的政府债券(?:的比例合计)?不低于基金资产净值的5%[，,]其中现金(?:类资产)?不包括结算备付金[、,，]存出保证金[、,，和]应收申购款等",
        legal,
    )
    require(cash_clause, "CASH_BUDGET_CLAUSE")
    gross_clause = re.search(r"本基金(?:总资产不得超过基金净资产|资产总值不超过基金资产净值)的140%", legal)
    require(gross_clause, "GROSS_BUDGET_CLAUSE")
    region = "日本" if inputs["adapter"] == "CIFM_JAPAN" else "越南"
    region_text = plain(section(report, r"7\.2\s*期末在各个国家", r"7\.3\s*期末按行业"))
    m = re.search(region + r"([\d,]+\.\d{2})([\d.]+)", region_text)
    require(m, "REGION_STOCK_AMOUNT")
    regional = money(m[1])
    require(regional == stock, "REGION_MATCH")
    region_pattern = (
        r"投资于日本上市公司股票的比例不低于非现金基金资产的80%"
        if region == "日本"
        else r"投资于越南主题企业发行的股票、存托凭证等的比例不低于非现金基金资产的80%"
    )
    region_clause = re.search(region_pattern, legal)
    require(region_clause, "REGION_BUDGET_CLAUSE")
    if region == "越南":
        require("非注册在越南但在越南上市" in legal, "VIETNAM_LISTING_THEME_QUALIFICATION")
    require(money(balance["assets"]["衍生金融资产"]) == 0, "CURRENT_DERIVATIVE_REVIEW_REQUIRED")
    derivative_note = plain(section(report, r"7\.9\s*期末按公允价值", r"7\.10\s*期末按公允价值"))
    require("本基金本报告期末未持有金融衍生品" in derivative_note, "CURRENT_FUTURES_MARGIN_REVIEW_REQUIRED")
    # 原件明确期末未持有金融衍生品，才可将当期期货保证金义务判为不适用。
    # 账上存出保证金已独立排除；不能由衍生品公允价值0推断无期货义务。
    # 分母歧义不取有利端：股票同时满足总/净资产口径；地域/总资产为非现金比例保守下界。
    passed = (
        low <= stock / total <= high
        and low <= stock / nav <= high
        and cash / nav >= D(".05")
        and regional / total >= D(".8")
        and total / nav <= D("1.4")
    )
    return (D(75) if passed else None), {
        "stock_total_ratio": str(stock / total),
        "stock_nav_ratio": str(stock / nav),
        "cash_principal_CNY": str(cash),
        "excluded_separate_margin_deposit_CNY": balance["assets"]["存出保证金"],
        "current_futures_margin_basis": derivative_note,
        "cash_nav_ratio": str(cash / nav),
        "regional_non_cash_ratio_proven_lower": str(regional / total),
        "gross_nav_ratio": str(total / nav),
        "contract_quotes": {"cash": cash_clause[0], "region": region_clause[0], "gross": gross_clause[0]},
        "bounds": {
            "stock_min": str(low),
            "stock_max": str(high),
            "cash_min": ".05",
            "region_min": ".8",
            "gross_max": "1.4",
        },
        "core_snapshot_passed": passed,
        "unknown_historical_detailed_derivatives_remain_supplements": True,
    }


def share_flows(report, inputs):
    if inputs["adapter"] != "TH_VIETNAM":
        return _flows(
            [report],
            allow_reversed=True,
            allow_interleaved=True,
            additional_shares=inputs.get("additional_shares", ()),
            allow_period_variant=True,
        )[0]
    text = plain(section(report, r"§9\s*开放式基金份额变动", r"§10\s*重大事件揭示"))
    require("发起(QDII)A发起(QDII)C发起(QDII)D" in text, "FLOW_COLUMNS_ACD")
    fields = {
        "opening": "本报告期期初基金份额总额",
        "subscribed": "本报告期基金总申购份额",
        "redeemed": "减:本报告期基金总赎回份额",
        "closing": "本报告期期末基金份额总额",
    }
    result = [{"share": s} for s in "ACD"]
    token = r"([\d,]+\.\d{2}|-)"
    for key, label in fields.items():
        match = re.search(re.escape(label) + token + token + token, text)
        if not match and key == "redeemed":
            match = re.search(r"减:本报告期基金" + token + token + token + "总赎回份额", text)
        require(match, "FLOW_FIELD:" + key)
        for row, value in zip(result, match.groups(), strict=True):
            row[key] = "0" if value == "-" else str(money(value))
    require("本报告期基金拆分变动份额---" in text, "FLOW_NO_SPLIT")
    for row in result:
        require(
            money(row["opening"]) + money(row["subscribed"]) - money(row["redeemed"]) == money(row["closing"]),
            "FLOW_ALL_CLASS_RECONCILIATION",
        )
    return result


def fees_from_original(source, inputs):
    summary = source.body(inputs["summary_ref"])
    legal = source.body(inputs["legal_ref"])
    if inputs["adapter"] != "TH_VIETNAM":
        return fee_profile(
            summary,
            code=inputs["fund_code"],
            share_class=inputs["share_class"],
            legal_name=inputs["legal_name"],
            supporting_text=legal,
            published_on=source.sources[inputs["summary_ref"]]["published_on"],
        )["fees"]
    text = plain(summary)
    require(
        inputs["fund_code"] in text and plain(inputs["legal_name"]) in text and "(C类份额)" in text[:250],
        "FEE_OWN_C_IDENTITY",
    )
    require("天弘越南市场股票发起(QDII)C不收取申购费用" in text, "FEE_SUBSCRIPTION_EXPLICIT")
    selling = text.split("(一)基金销售相关费用", 1)[1].split("(二)基金运作相关费用", 1)[0]
    require("30天≤N0.00%" in selling, "FEE_365_REDEMPTION")
    rates = re.findall(r"基金运作综合费率\(年化\)([\d.]+)%", text)
    require(len(rates) == 1, "FEE_OPERATING_SINGLE_RATE")
    return {
        "subscription": "0",
        "redemption": "0",
        "operating_cost": str(money(rates[0]) / 100),
        "subscription_mode": "EXTERNAL",
    }


def reference(source, inputs, ends):
    kind = inputs.get("reference", {}).get("kind")
    if kind == "DCDS_ACTIVE_ONE_YEAR_COMPARATOR":
        require(inputs["fund_code"] == "008764" and inputs["adapter"] == "TH_VIETNAM", "REFERENCE_COUNTRY_BINDING")
        ref = inputs["reference"]
        # 这两版是扫描件。对应条款已在原图逐页核读，严格固定原件摘要、
        # PDF实际页号和转录正文；不是将调用者任意一句“无分红”当原件。
        legal_refs = {
            "2025-01-21": "d15fa81fe590ee125efffb6d62708302228a661735b232ef067b185d3c5d85ae",
            "2026-03-24": "00c99e7f449d35bb7b41600e9d9c97d1f2a25baefeeda7a5ab37050dfee09d58",
        }
        require(
            ref["legal_refs"] == legal_refs and all(d in source.raw for d in legal_refs.values()), "DCDS_LEGAL_VERSIONS"
        )
        quote = (
            "Để hạn chế tối đa chi phí phát sinh, Quỹ DCDS sẽ không phân phối lợi nhuận. "
            "Toàn bộ lợi nhuận của Quỹ phát sinh trong quá trình hoạt động sẽ được tích lũy "
            "làm gia tăng Giá trị tài sản ròng của Quỹ."
        )
        reviews = ref["manual_legal_review"]
        require(
            reviews["2025"]["source_ref"] == legal_refs["2025-01-21"]
            and reviews["2025"]["no_distribution_pdf_page"] == 38
            and reviews["2025"]["strategy_pdf_page"] == 45
            and reviews["2026"]["source_ref"] == legal_refs["2026-03-24"]
            and reviews["2026"]["no_distribution_pdf_pages"] == [40, 41]
            and all(plain(r["no_distribution_quote"]) == plain(quote) for r in reviews.values()),
            "DCDS_SCANNED_PAGE_REVIEW",
        )
        decoded = source.raw[ref["native_ref"]].decode("utf-8", errors="replace")
        decoder = json.JSONDecoder()
        candidates = []
        for m in re.finditer(r'\{["\s]', decoded):
            try:
                obj, _ = decoder.raw_decode(decoded[m.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                for row in obj.get("funds", []):
                    if row.get("code") == "DCDS" and row.get("history") and "iso" in row["history"][0]:
                        candidates.append(row)
        require(len(candidates) == 1, "DCDS_NATIVE_SINGLE_PRODUCT")
        product = candidates[0]
        require(
            "chứng khoán năng động DC" in product["name"] and "₫" in product["navText"], "DCDS_NATIVE_IDENTITY_CURRENCY"
        )
        rows = product["history"]
        nav = {r["iso"]: money(r["nav"]) for r in rows}
        require(len(nav) == len(rows), "DCDS_NATIVE_DUPLICATE")
        require(all(d in nav and nav[d] > 0 for d in ends), "DCDS_EXACT_MONTHEND_MISSING")
        fx = {}
        for currency, digest in ref["fx_refs"].items():
            require(currency in ["VND", "CNY"], "DCDS_FX_CURRENCY")
            receipt = source.sources[digest]
            require(
                receipt["url"] == "https://tassidicambio.bancaditalia.it/terzevalute-wf-web/rest/v1.0/dailyTimeSeries"
                and receipt["params"]["baseCurrencyIsoCode"] == currency
                and receipt["params"]["currencyIsoCode"] == "EUR",
                "DCDS_OFFICIAL_FX_IDENTITY",
            )
            data = list(csv.DictReader(source.raw[digest].decode("utf-8-sig").splitlines()))
            rates = {}
            for row in data:
                day = row.get("Reference date (CET)", "")
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day or ""):
                    continue
                require(
                    row["ISO Code"] == currency
                    and row["Rate convention"] == "Foreign currency amount for 1 Euro."
                    and day not in rates,
                    "DCDS_FX_ROW_OR_DUPLICATE",
                )
                rates[day] = money(row["Rate"])
                require(rates[day] > 0, "DCDS_FX_NONPOSITIVE")
            require(all(d in rates for d in ends), "DCDS_SAME_DAY_FX_MISSING")
            fx[currency] = rates
        require(set(fx) == {"VND", "CNY"}, "DCDS_FX_PAIR")
        values = {d: nav[d] * fx["CNY"][d] / fx["VND"][d] for d in ends}
        return values, {
            "kind": kind,
            "native_currency": "VND",
            "comparison_currency": "CNY",
            "legal_product_id": "VN_DCDS_05_GCN_UBCK_20131008",
            "distribution": "2025-01-21版第38页与2026-03-24版第40/41页明示不分配利润，全部累积入NAV",
            "fx": "Banca d Italia同日CNY/EUR除VND/EUR；无日期补齐或小数位修补",
            "month_end_originals": [
                {
                    "date": d,
                    "nav_VND": str(nav[d]),
                    "CNY_per_VND": str(fx["CNY"][d] / fx["VND"][d]),
                    "value_CNY": str(values[d]),
                }
                for d in ends
            ],
            "strategy_difference": (
                "越南本土动态主动股票参照可提高现金，非严格风险匹配同行；本轮国家主动机会集统一参照，不作为风险回撤前置"
            ),
            "scanned_legal_reviews": reviews,
        }
    if kind == "JPAN_ACTIVE_ONE_YEAR_COMPARATOR":
        require(inputs["fund_code"] == "019449" and inputs["adapter"] == "CIFM_JAPAN", "REFERENCE_COUNTRY_BINDING")
        from app.services.fund_rating_japan_return import fx_history, legal_review, peer_product

        peer = inputs["reference"]["original_inputs"]
        require(peer["fund_code"] == "019449" and peer["as_of_date"] == inputs["as_of_date"], "PEER_TARGET_BINDING")
        legal_review(source, peer)
        days, prices, events, convention = peer_product(source, peer)
        fx = fx_history(source, peer)
        require(all(d in prices and d in fx for d in ends), "PEER_MONTHEND_OR_SAME_DAY_FX_MISSING")
        dividends = {r["date"]: r["cash_USD"] for r in events}
        growth = {days[0]: D(1)}
        for a, b in zip(days[:-1], days[1:], strict=True):
            growth[b] = growth[a] * (prices[b] + dividends.get(b, D(0))) / prices[a]
        values = {d: growth[d] * fx[d] for d in ends}
        return values, {
            "kind": kind,
            "native_currency": "USD",
            "comparison_currency": "CNY",
            "legal_product_id": "US_SEC_SERIES:S000081865",
            "reinvestment": convention,
            "fx": "same-date SAFE USD/CNY",
            "new_protocol_use": "一年主动同机会集参照；不是扩展旧R5风险/稳定性组件",
            "calendar": "只用原文存在的同日月末；未造缺日",
        }
    raise ValueError("QDII_DISCLOSURE_REFERENCE_NOT_VERIFIED")


def calculate(inputs: dict, *, root: Path) -> dict:
    require(inputs.get("version") == VERSION and inputs.get("profile") == "QDII_ACTIVE", "VERSION")
    code = inputs.get("fund_code")
    require(code in SUPPORTED, "FUND_SCOPE")
    share, name, manager, adapter = SUPPORTED[code]
    require(
        inputs.get("share_class") == share
        and plain(inputs.get("legal_name")) == plain(name)
        and plain(inputs.get("manager_name")) == plain(manager)
        and inputs.get("adapter") == adapter
        and inputs.get("currency") == "CNY",
        "IDENTITY_SHARE_CURRENCY",
    )
    require(not any(k in inputs for k in ["score", "dimensions", "precomputed_scores"]), "CALLER_SCORE_FORBIDDEN")
    require(inputs["primary_ref"] in inputs["sources"], "PRIMARY_SOURCE")
    source = Originals(inputs, root)
    report = source.body(inputs["primary_ref"])
    require(code in plain(report[:18000]) and plain(name) in plain(report[:18000]), "PRIMARY_IDENTITY")
    require(inputs["holding_date"] == "2026-06-30" and inputs["as_of_date"] == "2026-09-30", "VERIFIED_PERIOD_SCOPE")
    binding = primary_binding(report, inputs)
    dimensions = {k: None for k in KEYS}
    facts = {
        "primary_binding": binding,
        "profile": "QDII_ACTIVE",
        "revision": VERSION,
        "source_sha256": sorted(inputs["sources"]),
        "evidence_errors": {},
        "supplements_automatically_closed": False,
    }
    errors = facts["evidence_errors"]
    blocked = False
    reviews = inputs.get("event_reviews", [])
    require(reviews, "MATERIAL_EVENT_REVIEW_REQUIRED")
    require({"2025-12-31", "2026-06-30"} <= {r["period"] for r in reviews}, "MATERIAL_REVIEW_PERIODS")
    require(
        any(r["period"] == inputs["holding_date"] and r["source_ref"] == inputs["primary_ref"] for r in reviews),
        "MATERIAL_CURRENT_PRODUCT_REPORT",
    )
    for item in reviews:
        original = source.body(item["source_ref"])
        require(plain(item["quote"]) in plain(original), "MATERIAL_QUOTE_BINDING")
        require(
            all(
                term in plain(item["quote"])
                for term in ["涉及基金管理人、基金财产、基金托管业务的诉讼", "基金投资策略的改变", "处罚"]
            ),
            "MATERIAL_REVIEW_RELEVANT_SECTIONS",
        )
        require(
            item["disposition"] in ["NO_CONFIRMED_FUND_BLOCKER", "CONFIRMED_FUND_BLOCKER", "UNKNOWN"],
            "MATERIAL_DISPOSITION",
        )
        require(len(item.get("reason", "")) >= 30, "MATERIAL_REASON_REQUIRED")
        blocked = blocked or item["disposition"] != "NO_CONFIRMED_FUND_BLOCKER"
    facts["event_reviews"] = reviews
    with localcontext() as ctx:
        ctx.prec = 60
        nav = {}
        ends = []
        b = None
        with stage(errors, "history"):
            nav, history_audit = own_history(source, inputs)
            ends = month_ends(nav, inputs["as_of_date"])
            days = sorted(d for d in nav if ends[0] <= d <= ends[-1])
            require(len(days) > 180 and min(nav[d] for d in days) > 0, "YEAR_NAV_COVERAGE")
            peak = nav[days[0]]
            drawdown = D(0)
            for d in days:
                peak = max(peak, nav[d])
                drawdown = max(drawdown, 1 - nav[d] / peak)
            dimensions["risk"] = 100 * (1 - drawdown)
            facts["window"] = {"from": ends[0], "through": ends[-1], "month_ends": ends, "valuation_points": len(days)}
            facts["history"] = history_audit
            facts["maximum_drawdown"] = str(drawdown)
        if ends:
            with stage(errors, "reference"):
                values, proof = reference(source, inputs, ends)
                ret = nav[ends[-1]] / nav[ends[0]] - 1
                refret = values[ends[-1]] / values[ends[0]] - 1
                dimensions["return"] = signed_gap_score(ret - refret, PROTOCOL["parameters"]["annual_return_gap_unit"])
                wins = sum(nav[z] / nav[a] >= values[z] / values[a] for a, z in zip(ends[:-1], ends[1:], strict=True))
                dimensions["stability"] = D(wins * 100) / 12
                facts["reference"] = proof
                facts["fund_return"] = str(ret)
                facts["reference_return"] = str(refret)
                facts["months_at_or_above"] = wins
        with stage(errors, "holdings"):
            b, stocks = portfolio(source, inputs, report)
            top = sorted(stocks, key=lambda r: money(r["market_value_cny"]), reverse=True)[:10]
            ratio = sum(money(r["market_value_cny"]) for r in top) / money(b["net_assets"])
            require(0 <= ratio <= 1, "TOP10_RATIO")
            dimensions["holdings"] = 100 * (1 - ratio)
            facts["top10_nav_ratio"] = str(ratio)
            facts["holdings_count"] = len(stocks)
        if b:
            with stage(errors, "efficiency"):
                dimensions["efficiency"], facts["execution"] = execution(source, inputs, report, b)
                if dimensions["efficiency"] is None:
                    blocked = True
                    errors["efficiency"] = "CORE_CONFIGURATION_NUMERIC_DEVIATION_REVIEW"
        with stage(errors, "management"):
            managed = management_calculate(
                inputs["management_inputs"],
                root=root,
                manager_name=manager,
                as_of=inputs["as_of_date"],
                fund_code=code,
                legal_name=name,
                primary_report_hash=inputs["primary_ref"],
            )
            dimensions["management"] = money(managed["score"]) if managed["score"] is not None else None
        with stage(errors, "liquidity"):
            legal = plain(source.body(inputs["legal_ref"]))
            require("赎回" in legal and "巨额" in legal and re.search(r"T\+10", legal), "REDEMPTION_TERMS")
            flows = share_flows(report, inputs)
            own = next(r for r in flows if r["share"] == share)
            require(
                money(own["opening"]) + money(own["subscribed"]) - money(own["redeemed"]) == money(own["closing"]),
                "SHARE_FLOW_RECONCILIATION",
            )
            trustee = plain(
                section(report, r"5\.1\s*报告期内本基金托管人", r"§?6\s*半年度|§?6\s*中期财务|6\.1\s*资产负债")
            )
            require(
                "未发现" in trustee
                and "损害" in trustee
                and "利益" in trustee
                and ("复核" in trustee or "核查" in trustee or "审查" in trustee),
                "TRUSTEE_EXECUTION",
            )
            dimensions["liquidity"] = D(75) if not blocked else None
            facts["share_flows"] = flows
            facts["redemption_terms"] = "T+10及合同巨额赎回/境外休市等顺延限制；不承诺实际到账"
        with stage(errors, "cost"):
            fees = fees_from_original(source, inputs)
            cost = annual_fee_score(fees)
            dimensions["cost"] = money(cost["score"])
            facts["annual_cost_per_10000"] = cost["cost_cny"]
    return {
        "dimensions": {k: str(v) if v is not None else None for k, v in dimensions.items()},
        "facts": facts,
        "blocked": blocked,
    }
