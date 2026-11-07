"""007950 广谱主动参照的实际原件准入编译器。

复用正式仓位原件解析器，重算完整505产品发现池，不用321/144名单代替准入。
本模块没有研究目录导入或固定分值；边界只用真实股票金额和全部份额净资产修复。
"""

import json
import re
from collections import Counter
from decimal import Decimal as D
from decimal import localcontext

from bs4 import BeautifulSoup

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_active_exposure import replay_exposures
from app.services.fund_rating_active_performance import _mandate_identity, _report_publication
from app.services.fund_rating_market_review import normalized_text

ADMISSION_REF = "b81d21f7e5deead63bd76356eb4a8e537bf8338a8fcdfa0ac9dc9d058107ba52"
EXPOSURE_REF = "1afd08683aa8cadf30858d1d91b19d937e7cb872a0f28440abb5a8bebe4a9d79"
PERIODS = [
    f"{year}-{month}"
    for year in (2023, 2024, 2025, 2026)
    for month in ("03-31", "06-30", "09-30", "12-31")
    if "2023-09-30" <= f"{year}-{month}" <= "2026-06-30"
]


def require(value, error):
    if not value:
        raise ValueError("ACTIVE_BROAD_ADMISSION_" + error)


def binding_core(clause):
    """旧人工摘录少数改写了引导词；只去掉明确无业务含义的引导词。

    资产范围、最低比例、分母和条件原句一字不省，并在审计保留旧摘录与实际核中
    原件核心句。不能用模糊匹配、改百分比或省去适用条件凑成法律约束。
    """
    quote = normalized_text(clause)
    for prefix in ("本基金的配置比例为:", "本基金的投资组合比例为:", "本基金投资组合比例为:"):
        if quote.startswith(prefix):
            return quote[len(prefix) :]
    return quote


def financial_budget(source, ref, code, period, asof):
    """从本期原表求股票/全部份额NAV，逐金额保留公开两位分币半量子。

    季报使用3.1同期全部份额净资产；年/中报直接使用本法律产品资产负债表
    净资产合计，避免将A/C各自历史年末列错拼成当期净资产。
    """
    raw = source.raw[ref]
    if raw.startswith(b"%PDF"):
        text = normalized_text("".join(source.pages(ref)))
    else:
        # 共享归档的kind=ORIGINAL是证据角色，不是文件格式；不能让通用_read把
        # JSON字节当普通归档丢掉正文。按实际JSON原件schema读取并校发行人身份。
        original = json.loads(raw)
        require(original.get("success") == 1 and isinstance(original.get("data"), dict), "FINANCIAL_REPRINT_RESPONSE")
        body = original["data"].get("notice_content")
        require(isinstance(body, str) and len(body) > 1000, "FINANCIAL_ACTUAL_FULL_BODY")
        text = normalized_text(BeautifulSoup(body, "html.parser").get_text("\n", strip=True))
    require(code in text[:22000] and str(report_period(text)[1]) == period, "FINANCIAL_PRODUCT_PERIOD")
    published = source.sources[ref].get("published_on")
    if not raw.startswith(b"%PDF"):
        original = json.loads(raw)
        require(original.get("success") == 1, "FINANCIAL_REPRINT_SUCCESS")
        data = original["data"]
        require(code in {row["stock"] for row in data["security"]}, "FINANCIAL_REPRINT_CODE")
        announced = data["notice_date"][:10]
        require(not published or published == announced, "FINANCIAL_PUBLICATION_CONFLICT")
        published = announced
    require(published and published <= asof, "FINANCIAL_PUBLICATION_BOUNDARY")
    # 名义资料期和实际公开日期分别核验，不把取得日当作公开日。
    y, m, d = map(int, published.split("-"))
    require(f"{y}年{m}月{d}日" in text[:8000], "FINANCIAL_COVER_PUBLICATION")
    start = text.rfind("基金资产组合情况")
    require(start >= 0, "FINANCIAL_ASSET_SECTION")
    stock_match = re.search(r"其中[:：]?股票(?:投资)?([\d,]+\.\d{2})(\d{1,3}\.\d{2})", text[start : start + 2000])
    require(stock_match, "FINANCIAL_STOCK_AMOUNT")
    stock = D(stock_match[1].replace(",", ""))
    if period[5:7] in {"03", "09"}:
        begin = text.rfind("3.1主要财务指标")
        require(begin >= 0, "FINANCIAL_QUARTER_TABLE")
        section = text[begin : begin + 2400]
        first = section.index("期末基金资产净值")
        last = section.index("期末基金份额净值", first)
        quote = section[first:last]
        cells = re.findall(r"[\d,]+\.\d{2}", quote)
        require(1 <= len(cells) <= 2, "FINANCIAL_CURRENT_SHARE_COLUMNS")
        nav = sum(D(value.replace(",", "")) for value in cells)
        radius = D(".005") * len(cells)
        nav_basis = "ACTUAL_QUARTER_ALL_SHARE_NAV"
    else:
        blocks = []
        for match in re.finditer(r"[567]\.1资产负债表", text):
            end = re.search(r"[567]\.2利润表", text[match.end() :])
            if end:
                block = text[match.end() : match.end() + end.start()]
                if "人民币" in block and period[:4] in block:
                    blocks.append(block)
        require(len(blocks) == 1, "FINANCIAL_BALANCE_SCOPE")
        match = re.search(r"(?<!负债和)净资产合计([\d,]+\.\d{2})([\d,]+\.\d{2})", blocks[0])
        require(match, "FINANCIAL_LEGAL_NET_ASSETS")
        nav, radius, quote = D(match[1].replace(",", "")), D(".005"), match[0]
        nav_basis = "ACTUAL_ANNUAL_HALF_LEGAL_PRODUCT_BALANCE_NAV"
    require(stock > 0 and nav > radius, "FINANCIAL_POSITIVE_VALUES")
    totals = [D(value) for value in re.findall("合计" + re.escape(stock_match[1]) + r"(\d{1,3}\.\d{2})", text)]
    with localcontext() as context:
        context.prec = 70
        point = stock / nav * 100
        lower = (stock - D(".005")) / (nav + radius) * 100
        upper = (stock + D(".005")) / (nav - radius) * 100
    require(
        totals and len(set(totals)) == 1 and totals[0] - D(".005") <= point <= totals[0] + D(".005"),
        "FINANCIAL_INDUSTRY_TOTAL_RECONCILIATION",
    )
    return {
        "code": code,
        "period": period,
        "source_ref": ref,
        "published_on": published,
        "stock_cny": str(stock),
        "all_share_nav_cny": str(nav),
        "nav_reporting_radius_cny": str(radius),
        "stock_nav_pct": str(point),
        "stock_nav_pct_lower": str(lower),
        "stock_nav_pct_upper": str(upper),
        "stock_row": stock_match[0],
        "nav_row": quote,
        "nav_basis": nav_basis,
    }


def interval_gap(target, peer):
    """两个原始披露区间的绝对差界，保留依赖的保守界，不缩小量子保名单。"""
    a, b = D(target["stock_nav_pct_lower"]), D(target["stock_nav_pct_upper"])
    x, y = D(peer["stock_nav_pct_lower"]), D(peer["stock_nav_pct_upper"])
    return max(D(0), a - y, x - b), max(abs(a - y), abs(b - x))


def proven_region_exclusions(quarters, gaps, threshold):
    """AND准入中一项已证实失败即可另组；不使用冲突期，也不把别期缺口说成已修复。"""
    excluded_days = {g["day"] for g in gaps if g.get("day")}
    return [
        q["date"] for q in quarters if q["date"] not in excluded_days and D(q["domestic_ratio_upper_pct"]) < threshold
    ]


def compile_admission(inputs, source):
    """从全池原件重编两组；每个无法排除的边界仍待核，绝不默默删基金。"""
    config = inputs["source_admission"]
    require(
        config["admission_ref"] == ADMISSION_REF and config["exposure_ref"] == EXPOSURE_REF,
        "FROZEN_DISCOVERY_REVIEW_VERSION",
    )
    require(config["discovery_scope"] == "EXISTING_1986_SHARE_DIRECTORY_505_LEGAL_PRODUCTS", "EXPLICIT_DISCOVERY_SCOPE")
    require(
        config["thresholds"]
        == {
            "return": {"mean_gap_pp": 10, "max_gap_pp": 20, "domestic_min_pct": 80},
            "stability": {"mean_gap_pp": 5, "max_gap_pp": 10, "domestic_min_pct": 90},
        },
        "FROZEN_THRESHOLDS_CHANGED",
    )

    def read_bytes(_inputs, _root, digest):
        require(digest in source.raw, "UNARCHIVED_SOURCE")
        return source.raw[digest]

    def read_json(_inputs, _root, digest):
        return json.loads(read_bytes(_inputs, _root, digest))

    shared = {**config, "sources": inputs["sources"]}
    exposure, original_audit = replay_exposures(shared, source.root, read_json, read_bytes)
    legal_rows = read_json(shared, source.root, config["admission_ref"])
    require(
        len(legal_rows) == len(exposure) == 505 and len({r["code"] for r in legal_rows}) == 505, "FULL_DISCOVERY_POOL"
    )
    legal = {r["code"]: r for r in legal_rows}
    require({r["code"] for r in exposure} == set(legal), "DISCOVERY_EXPOSURE_IDENTITY_SET")
    definition_audit, derived_quota_notes = [], []
    for item in legal_rows:
        definitions = item.get("primary_theme_definitions", [])
        checked_texts = []
        for definition in definitions:
            digest = definition["source"]["sha256"]
            raw = source.raw[digest]
            if raw.startswith(b"%PDF"):
                text = normalized_text("".join(source.pages(digest)))
            elif raw.lstrip().startswith(b"<"):
                text = normalized_text(BeautifulSoup(raw.decode("utf-8"), "html.parser").get_text("\n", strip=True))
            else:
                original = json.loads(raw)
                require(
                    original.get("success") == 1
                    and item["code"] in {r["stock"] for r in original.get("data", {}).get("security", [])},
                    "LEGAL_DEFINITION_RAW_IDENTITY:" + item["code"],
                )
                body = original.get("data", {}).get("notice_content", "")
                require(len(body) > 1500, "LEGAL_DEFINITION_ACTUAL_BODY:" + item["code"])
                text = normalized_text(BeautifulSoup(body, "html.parser").get_text("\n", strip=True))
            require(
                _report_publication(shared, source.root, definition) <= inputs["as_of_date"],
                "LEGAL_DEFINITION_PUBLICATION:" + item["code"],
            )
            for excerpt in definition.get("definition_excerpts", []):
                quote = normalized_text(excerpt["text"])
                require(quote and quote in text, "LEGAL_DEFINITION_QUOTE:" + item["code"])
            checked_texts.append(text)
            definition_audit.append(
                {
                    "code": item["code"],
                    "source_ref": digest,
                    "quoted_definitions": len(definition.get("definition_excerpts", [])),
                }
            )
        if item["decision"] == "SEPARATE_FIXED_INDUSTRY_OR_BASKET":
            if not definitions:
                # 原审阅有六个有限行业篮子直接据完整定报/招募书的2.2、6.4.1或
                # 实际投资条款确认，另一个“主题定义摘录”列表为空不代表没有原件。
                # 沿原有效证据路径逐段核正文，不新增只准用某个研究字段的要求。
                quoted = 0
                for report in item["current_originals"] + item["window_originals"]:
                    claims = [report["strategy"]] if report.get("strategy") else []
                    claims += report.get("legal_notes", []) + report.get(
                        "contract_ratio_fragments_without_basic_heading", []
                    )
                    if not claims:
                        continue
                    digest = report["source"]["sha256"]
                    raw = source.raw[digest]
                    if raw.startswith(b"%PDF"):
                        text = normalized_text("".join(source.pages(digest)))
                    else:
                        original = json.loads(raw)
                        require(
                            original.get("success") == 1
                            and item["code"] in {r["stock"] for r in original.get("data", {}).get("security", [])},
                            "REPORT_DEFINITION_RAW_IDENTITY:" + item["code"],
                        )
                        text = normalized_text(
                            BeautifulSoup(original["data"]["notice_content"], "html.parser").get_text("\n", strip=True)
                        )
                    require(
                        _report_publication(shared, source.root, report) <= inputs["as_of_date"],
                        "REPORT_DEFINITION_PUBLICATION:" + item["code"],
                    )
                    require(
                        all(normalized_text(claim) in text for claim in claims),
                        "REPORT_DEFINITION_ORIGINAL_QUOTE:" + item["code"],
                    )
                    quoted += len(claims)
                    definition_audit.append(
                        {
                            "code": item["code"],
                            "source_ref": digest,
                            "quoted_definitions": len(claims),
                            "route": "FULL_REPORT_STRATEGY_AND_LEGAL_CLAUSES",
                        }
                    )
                require(quoted, "FIXED_BASKET_ORIGINAL_CLASSIFICATION:" + item["code"])
            else:
                require(
                    any(d.get("definition_excerpts") for d in definitions),
                    "FIXED_BASKET_ORIGINAL_CLASSIFICATION:" + item["code"],
                )
        for clause in item.get("binding_quota_clause", []):
            # 旧字段是人工归纳而非quote，不能给归纳补造逐字原件身份。
            # 真正引文为上面逐段核中的definition_excerpts，分类仍是已冻结的原件审阅。
            derived_quota_notes.append(
                {
                    "code": item["code"],
                    "derived_note": clause,
                    "verbatim_note_or_core_present": any(binding_core(clause) in text for text in checked_texts),
                    "used_as_original_quote": False,
                }
            )
    ranges, strategies, constraints = [], [], []
    for digest in config["contract_refs"]:
        text = source.plain(digest)
        require(inputs["legal_name"] in text, "TARGET_CONTRACT_LEGAL_NAME")
        require(
            source.sources[digest].get("published_on")
            and source.sources[digest]["published_on"] <= inputs["as_of_date"],
            "TARGET_CONTRACT_PUBLICATION",
        )
        start = text.index("二、投资范围")
        end = text.index("三、投资策略", start)
        quote = text[start:end]
        require(
            "国内依法发行上市的A股股票和存托凭证" in quote
            and "股票、存托凭证占基金资产的比例不低于80%" in quote
            and "本基金主要采用量化模型" in text,
            "TARGET_BROAD_ACTIVE_QUANT_OPPORTUNITY",
        )
        ranges.append(quote)
        strategies.append(text[end : end + 2600])
        constraint_start = text.find("四、投资限制", end)
        constraint_end = text.find("五、业绩比较基准", constraint_start) if constraint_start >= 0 else -1
        constraints.append(
            {
                "source_ref": digest,
                "actual_restrictions_section": text[constraint_start:constraint_end]
                if 0 <= constraint_start < constraint_end
                else None,
                "execution_grade_inferred": False,
            }
        )
    require(len(ranges) == 2 and ranges[0] == ranges[1], "TARGET_CONTRACT_OPPORTUNITY_CHANGED")
    target = {}
    strategy_claims = []
    for row in config["target_quarters"]:
        period, digest = row["period"], row["source_hash"]
        require(period not in target, "TARGET_DUPLICATE_QUARTER")
        target[period] = financial_budget(source, digest, "007950", period, inputs["as_of_date"])
        source.claim({"source_hash": digest, "quote": row["strategy_quote"]})
        require(
            inputs["legal_name"] in source.plain(digest)
            and any(term in row["strategy_quote"] for term in ("量化", "策略", "模型")),
            "TARGET_HISTORICAL_STRATEGY",
        )
        strategy_claims.append({"period": period, "source_ref": digest, "quote": row["strategy_quote"]})
    require(sorted(target) == PERIODS, "TARGET_COMPLETE_TWELVE_PERIODS")
    require(config["primary_report_ref"] == inputs["primary_ref"], "CURRENT_PRIMARY_REPORT_BINDING")
    primary = financial_budget(
        source, config["primary_report_ref"], "007950", inputs["holding_date"], inputs["as_of_date"]
    )
    main_period = target[inputs["holding_date"]]
    require(
        primary["stock_cny"] == main_period["stock_cny"]
        and primary["all_share_nav_cny"] == main_period["all_share_nav_cny"],
        "CURRENT_PRIMARY_REPORT_FINANCIAL_RECONCILIATION",
    )
    repairs = {}
    for row in config["precision_repairs"]:
        key = (row["code"], row["period"])
        require(key not in repairs and row["code"] in legal, "PRECISION_REPAIR_KEY")
        repairs[key] = financial_budget(source, row["source_ref"], row["code"], row["period"], inputs["as_of_date"])
    decisions, admitted = [], {"return": [], "stability": []}
    with localcontext() as context:
        context.prec = 70
        for product in exposure:
            code = product["code"]
            original_legal = legal[code]
            qs = product["quarters"]
            gaps = []
            conflicted_days = {g["day"] for g in product["gaps"] if g.get("day")}
            for q in qs:
                if q["date"] in conflicted_days and (code, q["date"]) not in repairs:
                    continue
                if (code, q["date"]) in repairs:
                    peer = repairs[(code, q["date"])]
                    require(
                        abs(D(peer["stock_nav_pct"]) - D(q["equity_nav_pct"]))
                        <= D(q["rounding_radius_pp"]) + D(".005"),
                        "REPAIR_ORIGINAL_EXPOSURE_CONFLICT",
                    )
                else:
                    point, radius = D(q["equity_nav_pct"]), D(q["rounding_radius_pp"])
                    peer = {"stock_nav_pct_lower": str(point - radius), "stock_nav_pct_upper": str(point + radius)}
                gaps.append(interval_gap(target[q["date"]], peer))
            result = {}
            for kind, limits in config["thresholds"].items():
                mean, maximum, region = limits["mean_gap_pp"], limits["max_gap_pp"], limits["domestic_min_pct"]
                decision = original_legal["decision"]
                region_exclusions = proven_region_exclusions(qs, product["gaps"], region)
                if code in {"007950", "001917"}:
                    state = "TARGET_LEGAL_PRODUCT_EXCLUDED"
                elif decision in {
                    "SEPARATE_FIXED_INDUSTRY_OR_BASKET",
                    "SEPARATE_INDEX_CONVERSION",
                    "SEPARATE_QDII_MECHANISM",
                }:
                    state = decision
                elif region_exclusions:
                    state = "SEPARATE_ACTUAL_REGION_EXPOSURE"
                elif max((a for a, _ in gaps), default=D(0)) > maximum or sum(a for a, _ in gaps) / 12 > mean:
                    state = "SEPARATE_OBSERVED_EXPOSURE"
                elif len(qs) != 12 or product["gaps"]:
                    state = "MISSING_OR_CONFLICTED_ORIGINAL_EXPOSURE"
                elif max(b for _, b in gaps) > maximum or sum(b for _, b in gaps) / 12 > mean:
                    state = "PRECISION_BOUNDARY_PENDING"
                elif not product["region_status_by_cutoff"][str(region)].startswith("REGION_PASS"):
                    state = product["region_status_by_cutoff"][str(region)]
                else:
                    require(
                        decision == "COMPARABLE_MANDATE_EVIDENCE_AVAILABLE",
                        "NEW_LEGAL_OPPORTUNITY_NOT_REVIEWED:" + code,
                    )
                    state = "ADMITTED"
                    admitted[kind].append(code)
                result[kind] = state
            decisions.append(
                {
                    "code": code,
                    "original_legal_decision": original_legal["decision"],
                    "decisions": result,
                    "unresolved_original_gaps_preserved": product["gaps"],
                    "usable_budget_quarters": len(gaps),
                    "independent_region_exclusion_dates": {
                        str(t): proven_region_exclusions(qs, product["gaps"], t) for t in (80, 90)
                    },
                    "mean_gap_lower_pp": str(sum(a for a, _ in gaps) / 12),
                    "mean_gap_upper_pp": str(sum(b for _, b in gaps) / 12) if len(gaps) == 12 else None,
                    "maximum_gap_lower_pp": str(max((a for a, _ in gaps), default=0)),
                    "maximum_gap_upper_pp": str(max((b for _, b in gaps), default=0)) if len(gaps) == 12 else None,
                }
            )
    require(
        not any(
            state.endswith("PENDING") or state.startswith("MISSING")
            for row in decisions
            for state in row["decisions"].values()
        ),
        "OPEN_ADMISSION_DEPENDENCY",
    )
    # 逐个合法产品的实际合同复述/期间报告核其身份和公开边界；不能把研究摘要冒充原件。
    # 原共享报告判定属于既定机会集合人工审阅，本次不凭名称或数值结果另挑组。
    legal_audit = []
    for code in sorted(set(admitted["return"]) | set(admitted["stability"])):
        item = legal[code]
        product = {"fund_code": code, "legal_name": item["legal_name"]}
        reports = item["current_originals"] + item["window_originals"]
        require(reports and code in item["share_codes"], "PEER_REPORTS_AND_SHARE_IDENTITY")
        for report in reports:
            require(_report_publication(shared, source.root, report) <= inputs["as_of_date"], "PEER_FUTURE_MANDATE")
            require(_mandate_identity(shared, source.root, report, product), "PEER_MANDATE_ORIGINAL_IDENTITY:" + code)
        legal_audit.append(
            {
                "code": code,
                "legal_name": item["legal_name"],
                "share_codes": item["share_codes"],
                "report_source_refs": [r["source"]["sha256"] for r in reports],
            }
        )
    return {
        "return_codes": sorted(admitted["return"]),
        "stability_codes": sorted(admitted["stability"]),
        "target_quarter_financials": target,
        "current_primary_report_financials": primary,
        "precision_repair_financials": list(repairs.values()),
        "target_contract_ranges": ranges,
        "target_contract_strategy_sections": strategies,
        "target_contract_execution_constraints_for_separate_review": constraints,
        "target_historical_strategy_claims": strategy_claims,
        "discovery_products": len(legal),
        "discovery_scope": config["discovery_scope"],
        "full_market_census_claimed": False,
        "exposure_original_replay": original_audit,
        "admission_decisions": decisions,
        "source_legal_decision_counts": dict(Counter(r["decision"] for r in legal_rows)),
        "peer_legal_originals": legal_audit,
        "legal_definition_originals": definition_audit,
        "derived_quota_notes_not_original_quotes": derived_quota_notes,
        "no_score_used_for_admission": True,
    }
