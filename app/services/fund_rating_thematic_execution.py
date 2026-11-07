"""体育文化合同逐公开截面与原公司业务判断编译；不从八数字导入正常75。"""

import importlib
import re
from decimal import Decimal as D
from decimal import localcontext

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_active_execution import MONEY, _assets, _catalog, _section, _share_binding, _stocks

_source = importlib.import_module((__package__ + "." if __package__ else "") + "fund_rating_thematic_source")
iso, plain, require = _source.iso, _source.plain, _source.require


def company_period(body):
    head = plain(body)[:3000]
    annual = re.search(r"(20\d{2})年年度报告", head)
    half = re.search(r"(20\d{2})年(?:半年度|中期)报告", head)
    require(bool(annual) != bool(half), "COMPANY_ACTUAL_REPORT_PERIOD")
    return (annual[1] + "-12-31") if annual else (half[1] + "-06-30")


def verify_business(source, proof, security, period):
    require(
        proof["security"] == security
        and proof["period_end"] == period
        and proof.get("human_source_judgment") is True
        and proof.get("not_current_industry_backfilled") is True,
        "BUSINESS_REVIEW_BINDING",
    )
    sha = proof["source_hash"]
    source.original(sha)
    body = source.body(sha)
    require(security in plain(body)[:20000], "BUSINESS_COMPANY_IDENTITY")
    source.claim({"source_hash": sha, "quote": proof["original_quote"]["quote"]})
    require(
        len(plain(proof["original_quote"]["quote"])) >= 100 and len(proof["qualification_reason"]) >= 30,
        "BUSINESS_SCOPE_REASON",
    )
    actual = company_period(body)
    require(actual == proof["company_report_period"], "COMPANY_PERIOD_NOT_CALLER_LABEL")
    bridge = proof.get("historical_activity_bridge")
    if actual != period:
        require(
            bridge
            and iso(bridge["economic_period_start"]) <= period <= iso(bridge["economic_period_end"])
            and len(bridge["scope_reason"]) >= 20,
            "HISTORICAL_ACTIVITY_BRIDGE_REQUIRED:" + security + "|" + period,
        )
        source.original(bridge["source_hash"])
        source.claim(bridge)
        require(security in plain(source.body(bridge["source_hash"]))[:20000], "BUSINESS_BRIDGE_COMPANY")
        require(len(plain(bridge["quote"])) >= 40, "BUSINESS_BRIDGE_ORIGINAL_SCOPE")
    quote = plain(proof["original_quote"]["quote"])
    if security == "002315":
        require(all(term in quote for term in ["文笔天天网", "整合广告", "为厂商提供"]), "ACTUAL_EXTERNAL_ADVERTISING")
    if security == "688787":
        require(
            "京西智谷数字人平台及配音平台项目" in quote and "2,000余个数字人形象" in quote, "ACTUAL_CULTURAL_PRODUCTION"
        )
    return {
        "security": security,
        "company_report_period": actual,
        "fund_period": period,
        "source_hash": sha,
        "historical_bridge": bridge,
        "category": proof["category"],
        "qualification_reason": proof["qualification_reason"],
    }


def compile_execution(inputs, source):
    spec = {**inputs["execution"], "sources": source.sources}
    _share_binding(source, spec)
    catalog = _catalog(source, spec)
    contract = plain(source.body(spec["contract_ref"]))
    require(spec["legal_name"] in contract and spec["manager_name"] in contract, "CONTRACT_LEGAL_IDENTITY")
    require(
        "股票资产占基金资产的0%-95%" in contract
        and "体育、文化主题的上市公司股票和债券不低于非现金基金资产的80%" in contract,
        "CONTRACT_BANDS",
    )
    require(
        "不低于基金资产净值5%的现金或到期日在一年以内的政府债券" in contract
        and "现金不包括结算备付金、存出保证金、应收申购款" in contract,
        "CONTRACT_QUALIFIED_CASH",
    )
    require(
        "本基金持有一家公司发行的证券,其市值不超过基金资产净值的10%" in contract, "CONTRACT_COMPANY_SECURITIES_LIMIT"
    )
    leverage = re.findall(r"本基金的基金资产总值不得超过基金资产净值的(\d+(?:\.\d+)?)%", contract)
    require(leverage and len(set(leverage)) == 1 and D(leverage[0]) == 140, "CONTRACT_LEVERAGE")
    registry = {row["code"]: row["orgId"] for row in source.json(spec["issuer_registry_ref"])["stockList"]}
    decision = source.json(inputs["critical_theme_decision_ref"])
    require(
        decision["fund_code"] == "018590"
        and decision["contract_source"] == spec["contract_ref"]
        and decision["changes_threshold_formula_weights_or_window"] is False,
        "SOURCE_BASED_THEME_DECISION",
    )
    for row in decision["accepted"]:
        proof = spec["theme_eligibility_proofs"].get(row["security"] + "|" + row["period"])
        require(proof and proof["source_hash"] == row["source_sha256"], "CRITICAL_THEME_DECISION_SCOPE")
    rows, themes, used = [], [], set()
    with localcontext() as ctx:
        ctx.prec = 70
        for role in ["quarters", "full_reports"]:
            for period, sha in sorted(spec[role].items()):
                full = role == "full_reports"
                text = source.body(sha)
                ptext = plain(text)
                entry = catalog[role][period]
                require(
                    spec["fund_code"] in ptext[:18000]
                    and spec["legal_name"] in ptext[:18000]
                    and str(report_period(text)[1]) == period,
                    "FUND_REPORT_IDENTITY_OR_PERIOD",
                )
                require(source.sources[sha]["published_on"] == entry["PUBLISHDATEDesc"][:10], "FUND_REPORT_PUBLICATION")
                if source.sources[sha]["format"] == "json":
                    require(source.json(sha)["data"]["art_code"] == entry["ID"], "FUND_REPORT_ARTICLE_BINDING")
                assets, total = _assets(text)
                stocks = _stocks(text, full)
                stock = D(assets["权益投资"]["amount_cny"])
                require(
                    all(
                        assets[name]["explicit_dash_zero"]
                        for name in ["基金投资", "贵金属投资", "金融衍生品投资", "买入返售金融资产"]
                    ),
                    "OTHER_PUBLIC_ASSET_CLASS_REQUIRES_ADAPTER",
                )
                compliance = _section(text, r"4\.2\s*管理人对报告期内本基金运作遵规守信情况的说明", r"4\.3\s*")
                cp = plain(compliance)
                require(
                    all(
                        term in cp
                        for term in [
                            "本基金投资组合符合有关法规及基金合同的约定",
                            "本基金无违法、违规行为",
                            "无损害基金份额持有人利益的行为",
                        ]
                    ),
                    "ACTUAL_MANAGER_EXECUTION",
                )
                require(
                    not any(term in cp for term in ["但是", "除外", "尚未", "未完成"]),
                    "KNOWN_PUBLIC_EXECUTION_CONTRADICTION",
                )
                if full:
                    balance = _section(text, r"[67]\.1\s*资产负债表", r"[67]\.2\s*利润表")
                    nav_match = re.search(r"净资产合计\s+(" + MONEY + r")", balance)
                    require(nav_match, "FULL_NAV")
                    nav = D(nav_match[1].replace(",", ""))
                    require(sum(D(p["amount_cny"]) for p in stocks) == stock, "FULL_STOCK_BOOK")
                    note = _section(text, r"[67]\.4\.7\.1\s*(?:货币资金|银行存款)", r"[67]\.4\.7\.2\s*交易性金融资产")
                    cash = re.search(r"等于[：:]本金\s+(" + MONEY + r")", note)
                    require(cash, "QUALIFIED_BANK_CASH_ORIGINAL")
                    bank = D(cash[1].replace(",", ""))
                    require(bank / nav >= D(".05"), "CASH_FLOOR_REVIEW_REQUIRED")
                    trustee = _section(text, r"5\.2\s*托管人对报告期内本基金投资运作", r"5\.3\s*托管人")
                    require(
                        "未发现基金管理人有损害基金份额持有人利益的行为" in plain(trustee), "ACTUAL_TRUSTEE_EXECUTION"
                    )
                    require(
                        "基金投资策略未发生改变" in ptext and "未持有港股通投资股票" in ptext,
                        "PUBLIC_STRATEGY_OR_REGION",
                    )
                    require(
                        D(assets["固定收益投资"]["amount_cny"]) == 0 and "未持有债券" in ptext,
                        "MATERIAL_NONSTOCK_SECURITIES",
                    )
                    for term in [
                        "未持有资产支持证券",
                        "未持有贵金属",
                        "未持有权证",
                        "未投资股指期货",
                        "未投资国债期货",
                    ]:
                        require(term in ptext, "PUBLIC_SCOPE:" + term)
                    eligible, excluded = [], []
                    for position in stocks:
                        key = position["security"] + "|" + period
                        proof = spec["theme_eligibility_proofs"].get(key)
                        if proof is None:
                            excluded.append(position)
                            continue
                        verified = verify_business(source, proof, position["security"], period)
                        eligible.append({**position, "business_review": verified})
                        used.add(key)
                    denominator = total - bank
                    amount = sum(D(p["amount_cny"]) for p in eligible)
                    require(
                        denominator > 0 and amount / denominator >= D(".8"), "THEME_CONSERVATIVE_LOWER_BOUND:" + period
                    )
                    themes.append(
                        {
                            "period": period,
                            "source_hash": sha,
                            "eligible": eligible,
                            "not_included_not_means_ineligible": excluded,
                            "noncash_upper": str(denominator),
                            "theme_lower": str(amount),
                            "ratio_lower_pct": str(amount / denominator * 100),
                            "margin_cny": str(amount - denominator * D(".8")),
                        }
                    )
                else:
                    match = re.search(r"4\.期末基金资产净值\s+((?:" + MONEY + r"\s+){1,8}" + MONEY + r")", text)
                    require(match, "QUARTER_NAV")
                    values = re.findall(MONEY, match[1])
                    require(
                        len(values) == len(spec["share_binding"]["column_codes"]) == 2 and len(stocks) == 10,
                        "QUARTER_SHARE_COLUMNS",
                    )
                    nav = sum(D(v.replace(",", "")) for v in values)
                    bank, trustee = None, None
                    bond = D(assets["固定收益投资"]["amount_cny"])
                    if bond:
                        table = _section(text, r"5\.4\s*报告期末按债券品种分类", r"5\.6\s*报告期末按公允价值")
                        government = re.search(r"1\s+国家债券\s+(" + MONEY + r")", table)
                        require(
                            government
                            and D(government[1].replace(",", "")) == bond
                            and "019694" in table
                            and re.search(r"23\s*国债\s*01", table),
                            "COMPANY_BOND_OVERLAP_REQUIRES_ADAPTER",
                        )
                require(
                    0 < nav <= total and stock / total <= D(".95") and stock / nav <= D(".95"), "STOCK_DENOMINATORS"
                )
                require(total / nav * 100 <= D(leverage[0]), "ACTUAL_LEVERAGE_LIMIT")
                issuers = {}
                for p in stocks:
                    sec = p["security"]
                    require(
                        re.fullmatch(r"(?:00|30|60|68)\d{4}", sec) and sec in registry,
                        "PUBLIC_STOCK_INSTRUMENT_OR_ISSUER",
                    )
                    issuers[registry[sec]] = issuers.get(registry[sec], D(0)) + D(p["amount_cny"])
                require(max(issuers.values()) / nav <= D(".1"), "KNOWN_COMPANY_SECURITIES_COUNTEREXAMPLE")
                rows.append(
                    {
                        "period": period,
                        "kind": "FULL" if full else "QUARTER",
                        "source_hash": sha,
                        "nav_cny": str(nav),
                        "total_assets_cny": str(total),
                        "stock_amount_cny": str(stock),
                        "bank_principal": str(bank) if bank is not None else None,
                        "leverage_pct": str(total / nav * 100),
                        "compliance_quote": compliance,
                        "trustee_quote": trustee,
                        "quarter_theme_not_invented": not full,
                        "daily_compliance_certified": False,
                    }
                )
        require(used == set(spec["theme_eligibility_proofs"]), "BUSINESS_PROOF_POSITION_CLOSURE")
        for row in [r for r in rows if r["kind"] == "FULL"]:
            quarter = next(q for q in rows if q["period"] == row["period"] and q["kind"] == "QUARTER")
            require(
                all(row[k] == quarter[k] for k in ["nav_cny", "total_assets_cny", "stock_amount_cny"]),
                "SAME_DATE_REPORT_CONFLICT",
            )
    nodes = {
        name: {"kind": "state", "state": "NORMAL_PUBLIC_EXECUTION", "reason": reason}
        for name, reason in [
            (
                "allocation",
                "逐期原件核股票两分母、完整披露现金、主题80%保守下限及140%杠杆；季报未知本金或主题分子不填值。",
            ),
            (
                "mandate",
                "逐完整持仓和实际已公开非股票范围、同公司证券限额及当期履职核准；不认证内部备选库、全管理组合或逐日交易。",
            ),
        ]
    }
    return nodes, {"report_ledger": rows, "theme_lower_bound_ledger": themes, "catalog_coverage": catalog}
