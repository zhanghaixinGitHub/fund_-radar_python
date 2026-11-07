"""017960两版60%-95%合同的R5公开执行适配候选；只读原件，不登记、不发布。

正常75档限于公开可复核截面及管理/托管实际陈述，不认证日内账簿、全管理人额度。
全部银行存款本金的合同现金口径与支付维度的用途口径分别核定，禁止借此扩大支付本金池。
"""

import re
from collections import defaultdict
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from bs4 import BeautifulSoup

from app.services.fund_quality_v2_restrictions import full_asset_restriction_scope, restricted_securities
from app.services.fund_rating_active_execution import MONEY, Originals, _plain, _section
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import content_digest

VERSION = "R5_CMF_017960_EXECUTION_PUBLIC_ORIGINALS_20261029_2"
LEGAL = "招商趋势领航混合型证券投资基金"
PERIODS = [
    "2023-09-30",
    "2023-12-31",
    "2024-03-31",
    "2024-06-30",
    "2024-09-30",
    "2024-12-31",
    "2025-03-31",
    "2025-06-30",
    "2025-09-30",
    "2025-12-31",
    "2026-03-31",
    "2026-06-30",
]
FULL = [x for x in PERIODS if x[5:7] in {"06", "12"}]


def require(condition, code):
    if not condition:
        raise ValueError("CMF_EXECUTION_" + code)


def money(value):
    return D(value.replace(",", ""))


def assets(text):
    """兼容同一物理行中连续列出的空类别，仍按原表序号与金额守恒闭合。"""
    table = _section(text, r"\d\.1\s*(?:报告)?期末基金资产组合情况", r"\d\.2\s*报告期末按行业")
    labels = [
        "权益投资",
        "基金投资",
        "固定收益投资",
        "贵金属投资",
        "金融衍生品投资",
        "买入返售金融资产",
        "银行存款和结算备付金合计",
        "其他(?:各项)?资产",
    ]
    rows = {}
    for rank, label in enumerate(labels, 1):
        matches = list(
            re.finditer(r"(?<!\d)" + str(rank) + r"\s+" + label + r"\s+(" + MONEY + r"|-)\s+([\d.]+|-)", table)
        )
        require(len(matches) == 1, "ASSET_CATEGORY:" + label)
        m = matches[0]
        rows[label] = {
            "amount": D(0) if m[1] == "-" else money(m[1]),
            "explicit_dash_zero": m[1] == "-",
            "original_row": m[0],
        }
    ms = list(re.finditer(r"(?<!\d)9\s+合计\s+(" + MONEY + r")\s+100\.00", table))
    require(len(ms) == 1, "ASSET_TOTAL")
    total = money(ms[0][1])
    require(sum(x["amount"] for x in rows.values()) == total > 0, "ASSET_SUM")
    return rows, total, table


def security_rows(table):
    pattern = r"(?:^|\n)\s*(\d+)\s+(\d{5,6})\s+(.+?)\s+([\d,]+)\s+(" + MONEY + r")\s+([\d.]+)"
    return [
        {
            "rank": int(m[1]),
            "code": m[2],
            "name": re.sub(r"\s+", "", m[3]),
            "quantity": m[4].replace(",", ""),
            "amount": money(m[5]),
            "original_row": m[0].strip(),
        }
        for m in re.finditer(pattern, table, re.S)
    ]


def bank_principal(note, label, stop):
    start = note.find(label)
    end = note.find(stop, start + len(label))
    require(0 <= start < end, "BANK_NOTE_CATEGORY:" + label)
    text = note[start:end]
    m = re.search(r"等于\s*[：:]\s*本金\s*(" + MONEY + r"|-)", text)
    require(m is not None, "BANK_PRINCIPAL:" + label)
    return {
        "category": label,
        "principal": D(0) if m[1] == "-" else money(m[1]),
        "explicit_dash_zero": m[1] == "-",
        "original_text": text,
    }


def stock_restriction_bound(body, stocks, bond_total, nav):
    """年报7.4.12与中报6.4.12是同一标准披露项目，仅作有记录的章节标识别名。"""
    edits = []

    def alias(m):
        to = "6.4." + m[1]
        edits.append({"offset": m.start(), "from": m[0], "to": to})
        return to

    parser_view = re.sub(r"(?<![\d.])7\.4\.(12|13)(?=[.\s\u4e00-\u9fff])", alias, body)
    positions = [
        {
            "stock_code": x["code"] + ".SOURCE",
            "stock_name": x["name"],
            "quantity_shares": x["quantity"],
            "market_value_cny": str(x["amount"]),
        }
        for x in stocks
    ]
    result = restricted_securities([parser_view], positions, str(bond_total), scope="STOCKS_ONLY")
    scope = full_asset_restriction_scope(body)
    require(scope is not None, "OTHER_RESTRICTION_SCOPE")
    upper = D(result["restricted_amount"]) + bond_total
    require(upper / nav <= D(".15"), "ACTIVE_RESTRICTED_CAP_BOUND")
    return {
        "all_restricted_stock_cny": result["restricted_amount"],
        "all_bonds_assumed_restricted_cny": str(bond_total),
        "upper_cny": str(upper),
        "upper_nav_pct": str(upper / nav * 100),
        "actual_active_restricted_cny": None,
        "scope_quote": scope[0],
        "stock_rows": result["rows"],
        "section_identifier_aliases_only": edits,
        "original_text_sha256": content_digest(body.encode()),
        "parser_view_sha256": content_digest(parser_view.encode()),
        "not_liquidity_exact_amount": True,
    }


def event_review_identity(inputs):
    """事项状态须为显式、来源绑定的本产品判断；缺失、未知或待核均不能进入正常档。"""
    review = inputs.get("material_event_review")
    require(isinstance(review, dict), "EVENT_REVIEW_REQUIRED")
    require(review.get("schema") == "CMF_017960_PUBLIC_EXECUTION_EVENT_REVIEW_V1", "EVENT_REVIEW_SCHEMA")
    require(
        review.get("fund_code") == inputs["fund_code"]
        and review.get("legal_name") == LEGAL
        and review.get("manager_name") == inputs["manager_name"]
        and review.get("share_aliases") == ["017960", "017961"]
        and review.get("as_of_date") == inputs["as_of_date"]
        and review.get("scope") == "PUBLISHED_PRODUCT_CONTRACT_EXECUTION",
        "EVENT_REVIEW_SCOPE",
    )
    pending = review.get("fund_specific_material_execution_issue_pending")
    require(type(pending) is bool, "EVENT_REVIEW_PENDING_UNKNOWN")
    require(not pending, "MATERIAL_EVENT_PENDING")
    return review


def replay_event_review(review, inputs, source, rows):
    """重读18期履职原文与本基金2025年报已知监管事项，不能以自述验收清除治理风险。"""
    expected = {(r["period"], r["full"], r["source_hash"]): r for r in rows}
    supplied = review.get("reviewed_periods")
    require(isinstance(supplied, list) and len(supplied) == 18, "EVENT_REVIEW_PERIOD_CLOSURE")
    keys = [(r.get("period"), r.get("full"), r.get("source_hash")) for r in supplied]
    require(len(set(keys)) == 18 and set(keys) == set(expected), "EVENT_REVIEW_SOURCE_CLOSURE")
    for key, row in zip(keys, supplied, strict=True):
        actual = expected[key]
        require(
            row.get("actual_compliance_text") == actual["actual_compliance_text"]
            and row.get("trustee_actual_text") == actual.get("trustee_actual_text"),
            "EVENT_REVIEW_ORIGINAL_QUOTE",
        )
    event = review.get("known_manager_supervisory_event")
    require(isinstance(event, dict), "KNOWN_MANAGER_EVENT_REQUIRED")
    require(
        event.get("source_hash") == inputs["full_report_texts"]["2025-12-31"]
        and event.get("source_period") == "2025-12-31",
        "KNOWN_MANAGER_EVENT_SOURCE",
    )
    h = event["source_hash"]
    body = _plain(source.body(h))
    quote = event.get("original_quote_normalized")
    require(isinstance(quote, str) and len(quote) > 500 and quote in body, "KNOWN_MANAGER_EVENT_ORIGINAL_QUOTE")
    require(
        all(s in quote for s in ["2025年10月27日", "深圳监管局", "责令改正", "监管谈话", "合规内控"]),
        "KNOWN_MANAGER_EVENT_FACTS",
    )
    require(
        event.get("source_disclosed_on") == source.sources[h]["published_on"]
        and event.get("event_date") == "2025-10-27",
        "KNOWN_MANAGER_EVENT_DATE",
    )
    require(
        event.get("subject") == "MANAGER_AND_RELATED_STAFF"
        and event.get("classification") == "GOVERNANCE_REMEDIATION_OBSERVATION_NOT_SOURCE_PROVEN_PRODUCT_MANDATE_BREACH"
        and event.get("no_independent_regulatory_acceptance_claim") is True
        and len(event.get("scope_judgment", "")) >= 100,
        "KNOWN_MANAGER_EVENT_APPLICABILITY_REVIEW",
    )
    return {
        "source_bound_review": review,
        "public_periods_replayed": 18,
        "known_manager_event_ref": h,
        "fund_specific_material_execution_issue_pending": review["fund_specific_material_execution_issue_pending"],
        "manager_governance_independently_cleared": None,
    }


def calculate(inputs: dict, *, root: Path) -> dict:
    """从实际两版合同、18份原PDF、六份原全文响应及国债条款重算两个执行子项。"""
    require(inputs.get("version") == VERSION, "VERSION")
    require(
        not any(k in inputs for k in ["score", "state", "dimensions", "allocation_state", "mandate_state"]),
        "NO_BARE_SCORE_OR_STATE",
    )
    require(
        inputs.get("fund_code") == "017960" and inputs.get("share_class") == "A" and inputs.get("legal_name") == LEGAL,
        "PRODUCT",
    )
    require(inputs.get("manager_name") == "招商基金管理有限公司", "MANAGER")
    require(inputs.get("as_of_date") == "2026-09-30" and inputs.get("holding_date") == "2026-06-30", "WINDOW")
    review = event_review_identity(inputs)
    require(
        set(inputs["quarters"]) == set(PERIODS)
        and set(inputs["full_reports"]) == set(FULL)
        and set(inputs["full_report_texts"]) == set(FULL),
        "NECESSARY_PERIODS",
    )
    require(inputs.get("primary_ref") == inputs["full_reports"]["2026-06-30"], "PRIMARY_REPORT")
    source = Originals(inputs, root)
    used = set()
    contract_clauses = []
    require([x["effective_from"] for x in inputs["contracts"]] == ["2023-07-10", "2026-06-01"], "CONTRACT_VERSIONS")
    for item in inputs["contracts"]:
        h = item["source_hash"]
        t = _plain(source.body(h))
        used.add(h)
        require(
            source.sources[h].get("published_on") and source.sources[h]["published_on"] <= item["effective_from"],
            "CONTRACT_PUBLICATION",
        )
        require(LEGAL in t and "招商基金管理有限公司" in t, "CONTRACT_IDENTITY")
        scope = t[t.rfind("二、投资范围") : t.find("三、投资策略", t.rfind("二、投资范围"))]
        limits = t[t.rfind("四、投资限制") : t.find("五、业绩比较基准", t.rfind("四、投资限制"))]
        require(
            all(
                x in scope
                for x in [
                    "60%-95%",
                    "50%",
                    "5%",
                    "现金不包括结算备付金、存出保证金、应收申购款",
                    "港股通标的股票",
                    "银行存款",
                    "债券回购",
                ]
            ),
            "ACTUAL_ALLOCATION_CLAUSE",
        )
        require(
            "同一家公司在境内和香港同时上市的A股和H股合并计算" in limits and "其市值不超过基金资产净值的10%" in limits,
            "ALL_SECURITIES_ISSUER_CLAUSE",
        )
        require("140%" in limits and "95%" in limits and "15%" in limits, "ADDITIONAL_LIMITS")
        contract_clauses.append(
            {
                "source_hash": h,
                "effective_from": item["effective_from"],
                "investment_scope_original": scope,
                "investment_limits_original": limits,
            }
        )
    fee = _plain(source.body(inputs["fee_amendment_ref"]))
    used.add(inputs["fee_amendment_ref"])
    require("2023年7月10日" in fee and LEGAL in fee, "2023_EFFECTIVE_NOTICE")
    change = _plain(source.body(inputs["benchmark_amendment_ref"]))
    used.add(inputs["benchmark_amendment_ref"])
    require("2026年6月1日" in change and LEGAL in change, "2026_EFFECTIVE_NOTICE")
    custody = _plain(source.body(inputs["custody_ref"]))
    used.add(inputs["custody_ref"])
    require(
        LEGAL in custody
        and "证券资金账户" in custody
        and "现金资产的记录" in custody
        and "第三方存管账户即托管账户" in custody,
        "CUSTODY_CASH_ACCOUNT_SCOPE",
    )
    # 交易所本通知实际只有一页，不能套基金长报告的1000字下限。
    # 原始字节仍由Originals统一核SHA；此角色另核完整标题、三条正文与发布方落款。
    from app.services.fund_rating_pdf import text_pages

    bridge_ref = inputs["government_bridge_ref"]
    bridge_source = source.sources[bridge_ref]
    require(
        bridge_source["kind"] == "pdf"
        and bridge_source["url"].startswith("https://static.sse.com.cn/disclosure/bond/announcement/"),
        "SSE_ORIGINAL_ROLE",
    )
    bridge_pages = text_pages(root / bridge_source["file"])
    bridge = _plain("\n".join(bridge_pages))
    used.add(bridge_ref)
    require(
        all(
            x in bridge
            for x in [
                "关于2025年记账式附息(八期)国债上市交易的通知",
                "一、本期债券于2025年4月17日",
                "二、本期债券上市交易的现券证券名称",
                "三、本期债券其他具体要素内容见发行公告",
                "上海证券交易所",
                "二〇二五年四月十四日",
            ]
        ),
        "COMPLETE_SHORT_SSE_NOTICE",
    )
    require(all(x in bridge for x in ["2025年记账式附息(八期)国债", "25国债08", "019773"]), "GOVERNMENT_CODE_BRIDGE")
    h = inputs["government_terms_ref"]
    term_source = source.sources[h]
    require(
        term_source["kind"] == "html"
        and term_source["url"] == "https://www.chinamoney.com.cn/chinese/ltss1/20250513/3113229.html",
        "CFETS_ORIGINAL_ROLE",
    )
    # 原发行要素表同为短公告，按整表字段及落款核完整性，保留原HTML而不补字。
    terms = _plain(BeautifulSoup(source.raw[h], "html.parser").get_text(" ", strip=True))
    used.add(h)
    require(
        all(
            x in terms
            for x in [
                "2025年记账式附息(八期)国债",
                "证券代码250008",
                "债券期限1年",
                "起息日2025-04-15",
                "兑付日2026-04-15",
            ]
        ),
        "GOVERNMENT_MATURITY",
    )
    require(
        "发行人中华人民共和国财政部" in terms and "中国外汇交易中心2025年5月13日" in terms,
        "COMPLETE_CFETS_ISSUANCE_TABLE",
    )
    rows = []
    for role, full in [("quarters", False), ("full_reports", True)]:
        for period, h in sorted(inputs[role].items()):
            text = source.body(h)
            t = _plain(text)
            used.add(h)
            require(bool(source.sources[h].get("published_on")), "REPORT_PUBLICATION")
            require(LEGAL in t[:18000] and "017960" in t[:18000] and "017961" in t[:18000], "REPORT_IDENTITY")
            y, m, d = map(int, period.split("-"))
            require(re.search(rf"{y}年0?{m}月0?{d}日", t[:18000]) is not None, "REPORT_PERIOD")
            categories, total, asset_table = assets(text)
            stock = categories["权益投资"]["amount"]
            fixed = categories["固定收益投资"]["amount"]
            require(
                all(categories[k]["explicit_dash_zero"] for k in ["基金投资", "贵金属投资", "金融衍生品投资"]),
                "OTHER_TOOLS_REVIEW",
            )
            heading = "所有股票投资明细" if full else "前十名股票投资明细"
            table = _section(
                text,
                r"\d\.3(?:\.1)?\s*(?:报告)?期末按公允价值占基金资产净值比例大小排序的" + heading,
                r"\d\.4\s*(?:报告期内股票|报告期末按债券)",
            )
            stocks = security_rows(table)
            ranks = sorted({x["rank"] for x in stocks})
            require(stocks and ranks == list(range(1, len(ranks) + 1)), "COMPLETE_ORIGINAL_STOCK_RANK_GROUPS")
            require(
                all(re.fullmatch(r"\d{5,6}", x["code"]) for x in stocks)
                and len({x["code"] for x in stocks}) == len(stocks),
                "ACTUAL_STOCK_TABLE_CODES",
            )
            require(
                all(x in t for x in ["未持有资产支持证券", "未持有贵金属", "未持有权证"]),
                "OTHER_SECURITY_TYPES_ACTUAL_ABSENCE",
            )
            if full:
                balance = _section(text, r"[67]\.1\s*资产负债表", r"[67]\.2\s*利润表")
                v = re.search(r"(?:净资产合计|所有者权益合计)\s+(" + MONEY + r")", balance)
                require(v is not None, "FULL_NET_ASSETS")
                nav = money(v[1])
                require(sum(x["amount"] for x in stocks) == stock, "FULL_STOCK_VALUE_SUM")
            else:
                v = re.search(r"4\.\s*期末基金资产净值\s+(" + MONEY + r")\s+(" + MONEY + r")", text)
                require(v is not None, "QUARTER_TWO_SHARE_NAV")
                nav = money(v[1]) + money(v[2])
                require(len(ranks) == 10, "TEN_ISSUER_GROUPS")
            require(
                nav > 0 and D(60) <= stock / total * 100 <= D(95) and D(60) <= stock / nav * 100 <= D(95),
                "STOCK_BAND_BOTH_DENOMINATORS",
            )
            require(nav <= total <= nav * D("1.4"), "ASSETS_NAV_CAP")
            a = _plain(asset_table)
            hk = re.search(r"港股金额人民币([\d,]+\.\d{2})元", a)
            # 2024Q1资产表没有港股附注，但实际港股行业表完整存在，不能把附注缺省当0。
            hk_section = _section(
                text, r"\d\.2\.2\s*报告期末按行业分类的港股通(?:投资)?股票投资组合", r"\d\.3\s*(?:报告)?期末按公允价值"
            )
            hk_sum = re.search(r"合计\s+(" + MONEY + r")\s+[\d.]+", hk_section)
            if hk_sum:
                hk_amount = money(hk_sum[1])
            else:
                require("未持有港股通投资股票" in _plain(hk_section), "HK_NOT_FROM_MISSING")
                hk_amount = D(0)
            if hk:
                require(money(hk[1]) == hk_amount, "HK_NOTE_INDUSTRY_CONFLICT")
            domestic_section = _section(
                text, r"\d\.2\.1\s*报告期末按行业分类的境内股票投资组合", r"\d\.2\.2\s*报告期末按行业"
            )
            domestic_sum = re.search(r"合计\s+(" + MONEY + r")\s+[\d.]+", domestic_section)
            require(
                domestic_sum is not None and money(domestic_sum[1]) + hk_amount == stock, "ALL_MARKET_STOCK_AMOUNT_SUM"
            )
            require(D(0) <= hk_amount <= stock * D(".5"), "HK_STOCK_CAP")
            if categories["买入返售金融资产"]["amount"] > 0:
                buyout_zero = re.search(r"买断式回购的买入返售金融资产[-－—]{2}", a)
                # 2024年报该空子项的标签跨页、横杠插在“金融/资产”之间；财务附注
                # 另有当期及上年末均无买断式逆回购取得债券的完整声明，可独立证明持券0。
                actual_no_bond = re.search(r"本基金本报告期末(?:及上年度末)?无买断式逆回购交易中取得的债券", t)
                require(
                    buyout_zero is not None or (full and actual_no_bond is not None),
                    "BUYOUT_REPO_NOT_ZERO_BY_ABSENCE:" + period,
                )
            else:
                # 全部买入返售明确无项目时，其买断式子集也唯一为0，无须另索空子表。
                require(categories["买入返售金融资产"]["explicit_dash_zero"], "ALL_REPO_ACTUAL_ZERO")
            national = D(0)
            short_government = D(0)
            bond_rows = []
            if fixed > 0:
                bond_table = _section(
                    text, r"\d\.\d\s*(?:报告)?期末按债券品种分类的债券投资组合", r"\d\.\d\s*(?:报告)?期末按公允价值"
                )
                b = re.search(r"(?<!\d)1\s+国家债券\s+(" + MONEY + r"|-)\s+([\d.]+|-)", bond_table)
                require(b is not None, "SOVEREIGN_CATEGORY")
                national = D(0) if b[1] == "-" else money(b[1])
                require(D(0) <= national <= fixed, "GOVERNMENT_AMOUNT")
                btable = _section(
                    text,
                    r"\d\.\d\s*(?:报告)?期末按公允价值占基金资产净值比例大小(?:排序|排名)的前五名债券投资明细",
                    r"\d\.\d\s*(?:报告)?期末按公允价值",
                )
                bond_rows = security_rows(btable)
                require(
                    bool(bond_rows) and sum(x["amount"] for x in bond_rows) == fixed,
                    "FULL_BONDS_CLOSED_BY_TOP_TABLE_AMOUNT_SUM",
                )
                # 只按实际代码及原发行条款排除019773；不为其余国债猜剩余期限。
                for bond in bond_rows:
                    if bond["code"] == "019773":
                        require("25国债08" in bond["name"], "BOND_IDENTITY")
                        now = date.fromisoformat(period)
                        maturity = date(2026, 4, 15)
                        require(now < maturity <= now.replace(year=now.year + 1), "GOVERNMENT_WITHIN_ONE_YEAR")
                        short_government += bond["amount"]
                require(short_government <= national, "SHORT_GOVERNMENT_UPPER")
            issuer_groups = defaultdict(list)
            for x in stocks:
                issuer_groups[x["rank"]].append(x)
            largest = max(sum(x["amount"] for x in group) for group in issuer_groups.values())
            # 给任一股票发行人同时分配全部非国债固定收益，仍是全证券集中度的保守上界。
            issuer_upper = largest + fixed - national
            require(issuer_upper <= nav * D(".1"), "ALL_SECURITIES_ISSUER_BOUND")
            securities_upper = stock + fixed - short_government
            require(securities_upper <= nav * D(".95"), "SECURITIES95_BOUND")
            compliance = _section(text, r"4\.2\s*管理人对报告期内本基金运作遵规守信情况的说明", r"4\.3\s*")
            cp = _plain(compliance)
            require(
                "基金运作整体合法合规" in cp and "基金的投资范围以及投资运作符合有关法律法规及基金合同的规定" in cp,
                "ACTUAL_PERIOD_COMPLIANCE",
            )
            applicable = [x for x in inputs["contracts"] if x["effective_from"] <= period][-1]
            row = {
                "period": period,
                "full": full,
                "source_hash": h,
                "applicable_contract_source_hash": applicable["source_hash"],
                "net_assets_cny": str(nav),
                "stock_total_assets_pct": str(stock / total * 100),
                "stock_nav_pct": str(stock / nav * 100),
                "hk_stock_pct": str(hk_amount / stock * 100),
                "assets_nav_pct": str(total / nav * 100),
                "all_stock_rows": len(stocks),
                "original_rank_issuer_groups": len(ranks),
                "all_securities_one_issuer_upper_nav_pct": str(issuer_upper / nav * 100),
                "securities95_upper_nav_pct": str(securities_upper / nav * 100),
                "short_government_019773_cny": str(short_government),
                "actual_compliance_text": compliance,
                "quarter_cash_not_observed": not full,
            }
            row["reported_issuer_groups"] = [
                {
                    "reported_rank": rank,
                    "source_security_rows": group,
                    "combined_stock_value_cny": str(sum(x["amount"] for x in group)),
                }
                for rank, group in sorted(issuer_groups.items())
            ]
            row["all_reported_bond_rows"] = bond_rows
            row["issuer_bound_basis"] = (
                "按原报告合并排名组归集同公司A/H，并将全部非国债固定收益分配给任一最大股票组作全证券上界；完整表逐行守恒，季度表保留原十个排名组。"
            )
            if full:
                note = _section(text, r"[67]\.4\.7\.1\s*(?:货币资金|银行存款)", r"[67]\.4\.7\.2\s*交易性金融资产")
                parts = [
                    bank_principal(note, "活期存款", "定期存款"),
                    bank_principal(note, "定期存款", "其他存款"),
                    bank_principal(note, "其他存款", "合计"),
                ]
                require(parts[1]["explicit_dash_zero"], "TERM_DEPOSIT_CASH_NOT_ASSUMED")
                require("本基金的银行存款存放" in t and "基金托管人" in t, "ALL_BANK_CUSTODY_STATEMENT")
                require(
                    re.search(r"本基金本报告期末(?:及上年度末)?无衍生金融工具", t) is not None,
                    "NO_CURRENT_DERIVATIVE_MARGIN",
                )
                bank = sum(x["principal"] for x in parts)
                require(bank / nav >= D(".05"), "CASH_PRINCIPAL_FLOOR")
                trustee = _section(text, r"5\.2\s*托管人对报告期内本基金投资运作", r"5\.3\s*托管人")
                tp = _plain(trustee)
                require(
                    ("不存在损害基金份额持有人利益的行为" in tp and "严格按照基金合同的规定进行" in tp)
                    or "未发现基金管理人有损害基金份额持有人利益的行为" in tp,
                    "TRUSTEE_ACTUAL_PERIOD_REVIEW",
                )
                j = inputs["full_report_texts"][period]
                body = source.body(j)
                used.add(j)
                require(LEGAL in _plain(body) and "017960" in body, "FULL_TEXT_SOURCE_IDENTITY")
                restrictions = stock_restriction_bound(body, stocks, fixed, nav)
                row.update(
                    bank_principal_components=[{**x, "principal": str(x["principal"])} for x in parts],
                    contract_cash_principal_cny=str(bank),
                    contract_cash_principal_nav_pct=str(bank / nav * 100),
                    cash_interpretation="报告内货币资金项全部银行存款本金，剔除应计利息；定期存款明确无项目，其他存款与结算备付金/保证金/应收申购分别列示，实际托管/第三方存管现金账户条款及全资产受限说明共同核读。此处不扩大支付用途本金池。",
                    trustee_actual_text=trustee,
                    restricted_cap_upper_bound=restrictions,
                )
            rows.append(row)
    require(len(rows) == 18 and sum(r["all_stock_rows"] for r in rows if r["full"]) == 613, "REPORT_SET_CLOSURE")
    quarter = {r["period"]: r for r in rows if not r["full"]}
    for r in rows:
        if r["full"]:
            require(
                r["net_assets_cny"] == quarter[r["period"]]["net_assets_cny"]
                and r["stock_nav_pct"] == quarter[r["period"]]["stock_nav_pct"],
                "SAME_DATE_REPORT_CONFLICT",
            )
    event_audit = replay_event_review(review, inputs, source, rows)
    require(used == set(source.sources), "UNUSED_OR_MISSING_ORIGINALS")
    refs = sorted(used)
    observations = {
        key: {"kind": "state", "state": "NORMAL_PUBLIC_EXECUTION", "source_refs": refs, "reason": reason}
        for key, reason in {
            "allocation": (
                "两版合同、12季配置及六期全仓现金逐项核对；A/H合并与非国债证券上界、"
                "已核短国债和受限15%上界闭合，未见公开证据中的非豁免配置偏离。"
            ),
            "mandate": (
                "18份原始公开资产组合、六期613条完整股票、实际债券/回购类别与合同允许工具逐项配对，"
                "并核管理人和托管人实际履职陈述。"
            ),
        }.items()
    }
    packet = {
        "fund_code": "017960",
        "basis": ACTIVE_BASIS,
        "as_of_date": inputs["as_of_date"],
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "target_product_aliases": ["017960", "017961"],
            "material_event_pending": event_audit["fund_specific_material_execution_issue_pending"],
            "observations": observations,
        },
        "r4_evidence_registry": {h: {**r, "path": str(root.resolve() / r["file"])} for h, r in source.sources.items()},
        "quality_public": {"dataDates": {"holdings": "2026-06-30"}},
        "raw_archive": [{"sha256": h, "kind": "ORIGINAL"} for h in refs],
    }
    with localcontext() as context:
        context.prec = 70
        result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"efficiency"})
    require(result["score"] is None and result["grade"] is None, "NO_COMPLETE_TOTAL")
    return {
        "version": VERSION,
        "fund_code": "017960",
        "score": result["dimensions"]["efficiency"],
        "dimensions": result["dimensions"],
        "subscores": result["subscores"],
        "packet": packet,
        "anchors": {},
        "observations": observations,
        "snapshots": rows,
        "actual_contract_clauses": contract_clauses,
        "source_refs": refs,
        "material_event_review_audit": event_audit,
        "normal_public_anchor_not_daily_audit": True,
        "not_independently_certified": [
            "逐日/日内所有限制",
            "管理人全部基金和组合持有额度",
            "非公开申购申报及交易对手/关联审批账簿",
            "历史回购的上一日基金净值分母；当前截面比率仅作解释，履约陈述另存",
        ],
        "complete_score": None,
        "grade": None,
        "requires_coordinator_integration_and_release": True,
    }
