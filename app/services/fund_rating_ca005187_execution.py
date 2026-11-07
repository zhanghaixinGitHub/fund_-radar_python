"""005187仅执行维的标准原件组件V2；不含稳定性、不采集、不提交数据库。

六期全公司证券合并核约及十二季公开观察逐次由Originals重建。正常公开履约75
沿既有60/40规则；管理人全组合/逐日限制明确未知，不冒充全部条款独立审计。
"""

import json
import re
import unicodedata
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS

VERSION = "R5_CA005187_PUBLIC_EXECUTION_ORIGINALS_V2_20261030"
FIELD = "ca005187_execution_calculation"
MASTER = "005186"
CODE = "005187"
LEGAL = "长安鑫兴灵活配置混合型证券投资基金"
MANAGER = "长安基金管理有限公司"
MATERIAL_REVIEW_SHA = "c46b39940e6445a57e7419b31acfb74d3606f388a761e97888852d6d218f4395"
MONEY = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
FUNDAMENTAL = {"FUND_REPORT", "FUND_PROSPECTUS", "FUND_NOTICE", "ISSUER_EVENT", "FUND_CONTRACT", "BOND_IDENTITY"}
RESEARCH = {"FUND_CATALOG", "MATERIAL_REVIEW", "PUBLIC_EXECUTION_SCOPE_DECISION"}
LIMITATIONS = [
    "仅评价近三年实际已公开配置截面，不推断每日合规或未来投资效果。",
    "管理人全部组合的发行量/流通量限额未独立重建；每日交易和发行申报限制未由期末持仓证明，事实保留未知。",
    "正常公开履约档不是监管审计结论，未宣称21条合同限制全部独立过关。",
    "本包不计算稳定性；合同参照93.05仅历史诊断。",
]
SCOPE_DECISION_SHA = "c1246a1c32830e32bef37207ec8a831db542090fe099b2d905f9ce3fb6ec9aea"


def _require(ok, code):
    if not ok:
        raise ValueError("CA005187_" + code)


def _plain(value):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _iso(value):
    _require(isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value), "DATE_REQUIRED")
    return date.fromisoformat(value).isoformat()


def _money(s):
    value = D(s.replace(",", ""))
    _require(value.is_finite() and value >= 0, "NONNEGATIVE_MONEY")
    return value


def _json(source, h):
    return json.loads(source.raw[h], parse_float=D)


def _text(source, h):
    ref = source.sources[h]
    if ref["kind"] == "pdf":
        return "\n".join(source.pages(h))
    raw = _json(source, h)
    _require(
        isinstance(raw, dict)
        and raw.get("success") == 1
        and isinstance(raw.get("data", {}).get("notice_content"), str),
        "NOTICE_ORIGINAL_REQUIRED",
    )
    return raw["data"]["notice_content"]


def _claim(source, item):
    _require(
        item.get("source_hash") in source.raw and isinstance(item.get("quote"), str) and item["quote"],
        "EVENT_CLAIM_REQUIRED",
    )
    _require(_plain(item["quote"]) in _plain(_text(source, item["source_hash"])), "EVENT_QUOTE_MISMATCH")


def _sources(inputs, root):
    """只接受相对root/sources；逐次重验原字节、公开日和父级适用判决原件。"""
    allowed = {
        "version",
        "fund_code",
        "basis",
        "share_class",
        "legal_name",
        "manager_name",
        "as_of_date",
        "holding_date",
        "reviewed_on",
        "primary_ref",
        "full_reports",
        "quarter_reports",
        "prospectus_ref",
        "execution_contract_refs",
        "issuer_identity_refs",
        "material_review_ref",
        "material_event_pending",
        "late_catalog_refs",
        "public_execution_scope_decision_ref",
        "sources",
    }
    _require(
        not any(key in inputs for key in ["score", "dimensions", "observations", "quality_metrics"]), "RAW_INPUTS_ONLY"
    )
    _require(
        set(inputs) == allowed
        and inputs.get("version") == VERSION
        and inputs.get("fund_code") == CODE
        and inputs.get("share_class") == "C"
        and inputs.get("basis") == ACTIVE_BASIS,
        "EXECUTION_INPUT_SCOPE",
    )
    _require(
        inputs.get("legal_name") == LEGAL
        and inputs.get("manager_name") == MANAGER
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30",
        "IDENTITY_PERIOD",
    )
    reviewed = _iso(inputs.get("reviewed_on"))
    _require(reviewed >= "2026-10-30", "REVIEW_DATE_REQUIRED")
    _require(isinstance(inputs.get("sources"), dict) and len(inputs["sources"]) == 38, "EXECUTION_SOURCE_COUNT")
    for h, ref in inputs["sources"].items():
        _require(re.fullmatch(r"[0-9a-f]{64}", h) and ref.get("sha256") == h, "SOURCE_DIGEST_SCHEMA")
        file = Path(ref["file"])
        _require(not file.is_absolute() and file.parts[0] == "sources", "RELATIVE_SOURCE_REQUIRED")
        _require(not ref.get("path") or ref["path"] == ref["file"], "ABSOLUTE_SOURCE_PATH_FORBIDDEN")
        role = ref.get("role")
        _require(role in FUNDAMENTAL | RESEARCH, "SOURCE_ROLE_REQUIRED")
        if role in FUNDAMENTAL:
            _require(_iso(ref.get("published_on")) <= inputs["as_of_date"], "FUNDAMENTAL_AFTER_CUTOFF")
        elif ref.get("published_on"):
            _require(_iso(ref["published_on"]) <= reviewed, "RESEARCH_AFTER_REVIEW")
    source = Originals({**inputs, "as_of_date": reviewed}, Path(root))
    _require(
        inputs["public_execution_scope_decision_ref"] == SCOPE_DECISION_SHA, "PUBLIC_EXECUTION_SCOPE_DECISION_REQUIRED"
    )
    decision = _json(source, SCOPE_DECISION_SHA)
    _require(
        decision.get("fund_code") == CODE
        and decision.get("decision") == "REVIEW_NORMAL_PUBLIC_EXECUTION_WITH_EXPLICIT_LIMITATIONS"
        and decision.get("formal_score_publication_authorized_by_this_file") is False,
        "PUBLIC_EXECUTION_SCOPE_DECISION_BINDING",
    )
    return source


def _notice(source, h):
    raw = _json(source, h)
    _require(raw.get("success") == 1, "NOTICE_RESPONSE")
    data = raw["data"]
    _require(data["notice_date"][:10] == source.sources[h]["published_on"], "NOTICE_PUBLICATION_CONFLICT")
    _require(source.sources[h].get("params", {}).get("art_code") == data["art_code"], "NOTICE_REQUEST_BINDING")
    return data


def _fund_report(source, h, period):
    data = _notice(source, h)
    text = data["notice_content"]
    plain = _plain(text)
    _require(
        {MASTER, CODE} <= {str(r["stock"]) for r in data.get("security", [])} and LEGAL in plain and MANAGER in plain,
        "REPORT_WRONG_LEGAL_SHARE",
    )
    master = re.search(r"基金主代码(\d{6})", plain)
    _require(master and master[1] == MASTER, "REPORT_MASTER_CODE")
    begin, end = report_period(text)
    _require(str(end) == period and str(begin) <= period, "REPORT_PERIOD_MISMATCH")
    return text, plain


def _slice(plain, start_pattern, end_pattern):
    matches = list(re.finditer(start_pattern, plain))
    _require(matches, "SECTION_START")
    start = matches[-1].start()
    end = re.search(end_pattern, plain[matches[-1].end() :])
    _require(end, "SECTION_END")
    return plain[start : matches[-1].end() + end.start()]


def _public_stock_and_cash_observations(inputs, source):
    reports = inputs["full_reports"]
    quarters = inputs["quarter_reports"]
    expected_full = ["2023-12-31", "2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31", "2026-06-30"]
    expected_quarters = [
        f"{y}-{m}-{day}"
        for y in range(2023, 2027)
        for m, day in [("03", "31"), ("06", "30"), ("09", "30"), ("12", "31")]
        if "2023-09-30" <= f"{y}-{m}-{day}" <= "2026-06-30"
    ]
    _require(list(reports) == expected_full and list(quarters) == expected_quarters, "REPORT_WINDOW_CLOSURE")
    _require(reports["2026-06-30"] == inputs["primary_ref"], "PRIMARY_FULL_REPORT")
    _, prospectus = _fund_report_like_prospectus(source, inputs["prospectus_ref"])
    full_audit = []
    quarter_audit = []
    for period, h in reports.items():
        text, s = _fund_report(source, h, period)
        _require(
            re.search(r"股票投资占基金资产的比例为0[-－—～至]95%", s)
            and "其中现金不包括结算备付金、存出保证金、应收申购款" in s,
            "ACTUAL_CONTRACT_0_95_CASH_EXCLUSIONS",
        )
        _require(re.search(r"不低于基金资产净值[的]*5%", s), "CASH_CONTRACT_5")
        balance = _slice(s, r"[67]\.1资产负债表", r"[67]\.2(?:利润表|收益表)")
        navm = re.search(r"(?:净资产合计|所有者权益合计)(" + MONEY + ")", balance)
        assetm = re.search(r"资产总计(" + MONEY + ")", balance)
        _require(navm and assetm, "BALANCE_TOTALS")
        nav, assets = _money(navm[1]), _money(assetm[1])
        _require(nav > 0 and assets / nav <= D("1.4"), "ASSET_LEVERAGE_PUBLIC_CONTRACT")
        note = _slice(s, r"[67]\.4\.7\.1货币资金", r"[67]\.4\.7\.2交易性金融资产")
        bankm = re.search(r"活期存款[\d,.]+等于[:：]本金(" + MONEY + ")", note)
        _require(bankm, "BANK_PRINCIPAL_NOT_INTEREST")
        bank = _money(bankm[1])
        _require(bank / nav >= D(".05"), "QUALIFIED_BANK_LOWER_BOUND")
        table = _slice(s, r"[78]\.1期末基金资产组合情况", r"[78]\.2(?:报告期末|期末)按行业")
        stockm = re.search(r"其中[:：]股票(" + MONEY + r"|--)", table)
        _require(stockm, "FULL_STOCK_TOTAL")
        stock = D(0) if stockm[1] == "--" else _money(stockm[1])
        _require(stock / assets <= D(".95"), "FULL_STOCK_TOTAL_ASSET_LIMIT")
        positions = []
        if stock:
            heading = list(re.finditer(r"(?m)^\s*[78]\.3\s*(?:期末|报告期末)[^\n]*所有股票投资明细", text))
            _require(heading, "COMPLETE_STOCK_HEADER")
            start = heading[-1].end()
            end = re.search(r"(?m)^\s*[78]\.4\s*报告期", text[start:])
            _require(end, "COMPLETE_STOCK_END")
            segment = text[start : start + end.start()]
            for m in re.finditer(
                r"(?m)^\s*(\d+)\s+(\d{6})\s+([^\d\n]+?)\s+([\d,]+(?:\.\d+)?)\s+(" + MONEY + r")\s+([\d.]+)\s*$", segment
            ):
                positions.append(
                    {
                        "sequence": int(m[1]),
                        "security": m[2],
                        "name": m[3].strip(),
                        "quantity": str(_money(m[4]) if "." in m[4] else D(m[4].replace(",", ""))),
                        "value_CNY": str(_money(m[5])),
                    }
                )
            _require(
                positions
                and [r["sequence"] for r in positions] == list(range(1, len(positions) + 1))
                and sum(D(r["value_CNY"]) for r in positions) == stock,
                "COMPLETE_STOCK_RECONCILIATION",
            )
            _require(
                all(r["security"][0] in "036" and D(r["value_CNY"]) / nav <= D(".10") for r in positions),
                "DOMESTIC_STOCK_SINGLE_ISSUER_LIMIT",
            )
        else:
            _require("未持有股票" in s or "未持有任何股票" in s, "ZERO_STOCK_EXPLICIT_ORIGINAL")
        for category in ["基金投资", "资产支持证券", "贵金属投资", "金融衍生品投资"]:
            _require(re.search(re.escape(category) + r"--", table), "UNREVIEWED_ASSET_CLASS:" + category)
        _require(
            "本基金运作合法合规" in s and "无损害基金份额持有人利益的行为" in s, "ACTUAL_PUBLIC_EXECUTION_STATEMENT"
        )
        full_audit.append(
            {
                "period": period,
                "source_hash": h,
                "NAV_CNY": str(nav),
                "total_assets_CNY": str(assets),
                "stock_CNY": str(stock),
                "bank_principal_CNY": str(bank),
                "qualified_cash_lower_bound_ratio": str(bank / nav),
                "stock_total_assets_ratio": str(stock / assets),
                "complete_positions": positions,
                "contract_paired": True,
            }
        )
    for period, h in quarters.items():
        text, s = _fund_report(source, h, period)
        table = _slice(s, r"5\.1(?:报告)?期末基金资产组合情况", r"5\.2(?:报告期末|期末)按行业")
        totalm = re.search(r"9合计(" + MONEY + ")", table)
        stockm = re.search(r"其中[:：]股票(" + MONEY + r"|--)", table)
        navm = re.search(r"期末基金资产净值\s+(" + MONEY + r")\s+(" + MONEY + ")", text)
        _require(totalm and stockm and navm, "QUARTER_RAW_TOTALS")
        total = _money(totalm[1])
        stock = D(0) if stockm[1] == "--" else _money(stockm[1])
        nav = _money(navm[1]) + _money(navm[2])
        _require(total > 0 and nav > 0 and stock / total <= D(".95"), "QUARTER_STOCK_TOTAL_ASSET_LIMIT")
        if stock == 0:
            _require("未持有股票" in s or "未持有任何股票" in s, "QUARTER_ZERO_STOCK_NOT_ASSUMED")
        _require("本基金运作合法合规" in s, "QUARTER_PUBLIC_EXECUTION")
        quarter_audit.append(
            {
                "period": period,
                "source_hash": h,
                "stock_CNY": str(stock),
                "total_assets_CNY": str(total),
                "legal_product_NAV_CNY": str(nav),
                "stock_NAV_ratio": str(stock / nav),
                "cash_observation": None,
            }
        )
    refs = list(dict.fromkeys([*reports.values(), *quarters.values(), inputs["prospectus_ref"]]))
    reason = (
        "原公开合同与12季、6期全文配置逐项匹配；现金用已披露普通银行本金下界核验，"
        "不含备付金、利息或应收。只评价公开截面，不推断逐日账簿合规或未来收益。"
    )
    nodes = {
        k: {"kind": "state", "state": "NORMAL_PUBLIC_EXECUTION", "reason": reason, "source_refs": refs}
        for k in ["allocation", "mandate"]
    }
    return nodes, {"full_reports": full_audit, "quarters": quarter_audit, "quarter_cash_not_default_zero": True}


def _aggregate_company_securities(rows, nav):
    """同一法律发行人跨股票、信用债、转债及其他公司证券合并；政府债单列。

    rows的发行人必须先与实际原件绑定，不能靠简称猜测，也不能把公司债改标国债。
    此处只核本基金NAV10%，不把它冒充管理人全部组合的发行量限制。
    """
    groups = {}
    government = D(0)
    _require(nav > 0, "ISSUER_NAV")
    for row in rows:
        value = D(row["value_CNY"])
        _require(value.is_finite() and value >= 0, "ISSUER_SECURITY_VALUE")
        if row["kind"] == "GOVERNMENT_BOND":
            government += value
            continue
        _require(
            row.get("issuer") and row["kind"] in {"STOCK", "CREDIT_BOND", "CONVERTIBLE", "OTHER_COMPANY_SECURITY"},
            "COMPANY_ISSUER_BINDING_REQUIRED",
        )
        node = groups.setdefault(row["issuer"], {"value": D(0), "securities": []})
        node["value"] += value
        node["securities"].append(row)
    issuers = [
        {
            "issuer": key,
            "value_CNY": str(row["value"]),
            "ratio_to_NAV": str(row["value"] / nav),
            "within_10_percent": row["value"] <= nav * D(".10"),
            "securities": row["securities"],
        }
        for key, row in groups.items()
    ]
    return {
        "issuers": issuers,
        "government_bond_CNY_excluded_from_company_rule": str(government),
        "all_company_issuers_within_10_percent": all(row["within_10_percent"] for row in issuers),
    }


def _line_section(text, number, next_number):
    matches = list(re.finditer(r"(?m)^\s*[78]\." + str(number) + r"\s*[^\n]*", text))
    _require(matches, "BOND_SECTION_START")
    tail = text[matches[-1].end() :]
    end = re.search(r"(?m)^\s*[78]\." + str(next_number) + r"\s*", tail)
    _require(end, "BOND_SECTION_END")
    return text[matches[-1].start() : matches[-1].end() + end.start()]


def _combined_issuer_limits(inputs, source, base):
    """按债券品种金额守恒证明全部公司债已识别；前五之外余额只准保留真实国债桶。"""
    _require(
        isinstance(inputs.get("issuer_identity_refs"), dict) and inputs["issuer_identity_refs"],
        "ISSUER_IDENTITY_ORIGINALS_REQUIRED",
    )
    text = _text(source, base["source_hash"])
    classes = _plain(_line_section(text, 5, 6))
    top = _line_section(text, 6, 7)
    if "本基金本报告期末未持有债券" in classes:
        total = government = corporate = D(0)
        bonds = []
    else:
        categories = {}
        for category in [
            "国家债券",
            "央行票据",
            "金融债券",
            "企业债券",
            "企业短期融资券",
            "中期票据",
            "可转债(可交换债)",
            "同业存单",
            "其他",
        ]:
            m = re.search(re.escape(category) + r"(" + MONEY + r"|-)(?:[\d.]+|-)", classes)
            _require(m, "BOND_CATEGORY:" + category)
            categories[category] = D(0) if m[1] == "-" else _money(m[1])
        m = re.search(r"10合计(" + MONEY + r")", classes)
        _require(m, "BOND_CLASS_TOTAL")
        total = _money(m[1])
        _require(sum(categories.values()) == total, "BOND_CATEGORY_SUM")
        government = categories["国家债券"]
        corporate = total - government
        bonds = []
        for line in top.splitlines():
            cols = line.split()
            if (
                len(cols) == 6
                and cols[0].isdigit()
                and re.fullmatch(r"\d{6}", cols[1])
                and re.fullmatch(MONEY, cols[4])
            ):
                bonds.append(
                    {"sequence": int(cols[0]), "security": cols[1], "name": cols[2], "value_CNY": str(_money(cols[4]))}
                )
        _require(
            bonds
            and [r["sequence"] for r in bonds] == list(range(1, len(bonds) + 1))
            and sum(D(r["value_CNY"]) for r in bonds) <= total,
            "BOND_TOP_TABLE_CLOSURE",
        )
    rows = [
        {
            "kind": "STOCK",
            "security": r["security"],
            "issuer": "CN_A_SHARE_ISSUER:" + r["security"],
            "value_CNY": r["value_CNY"],
        }
        for r in base["complete_positions"]
    ]
    # 本基金六期事实为“信用债时股票为0，其余仅国债/无债”。若未来两类并存，
    # 必须新增真实股票-债券法律发行人连接，不能用证券代码不同来假设发行人不同。
    _require(not corporate or D(base["stock_CNY"]) == 0, "CROSS_ASSET_LEGAL_ISSUER_MAPPING_REQUIRED")
    credit = []
    listed_government = D(0)
    for bond in bonds:
        if "国债" in bond["name"]:
            listed_government += D(bond["value_CNY"])
            continue
        ref = inputs["issuer_identity_refs"].get(bond["security"])
        _require(ref and ref["source_ref"] in source.raw, "UNBOUND_COMPANY_BOND")
        identity_pages = source.pages(ref["source_ref"])
        _require(
            ref["short_name"] == bond["name"]
            and any(
                all(_plain(term) in _plain(page) for term in [bond["security"], ref["short_name"], ref["legal_name"]])
                for page in identity_pages
            ),
            "BOND_LEGAL_ISSUER_ORIGINAL_MISMATCH",
        )
        credit.append(
            {**bond, "kind": "CREDIT_BOND", "issuer": ref["legal_name"], "identity_source_ref": ref["source_ref"]}
        )
    _require(
        sum(D(r["value_CNY"]) for r in credit) == corporate and listed_government <= government,
        "COMPANY_BOND_AMOUNT_CLOSURE_REQUIRED",
    )
    rows += credit
    rows.append({"kind": "GOVERNMENT_BOND", "issuer": "中华人民共和国政府", "value_CNY": str(government)})
    result = _aggregate_company_securities(rows, D(base["NAV_CNY"]))
    _require(result["all_company_issuers_within_10_percent"], "COMBINED_COMPANY_SECURITY_LIMIT_EXCEEDED")
    return {
        **result,
        "company_bond_CNY": str(corporate),
        "government_bond_CNY": str(government),
        "government_bond_unlisted_residual_CNY": str(government - listed_government),
        "company_bond_complete_by_category_amount_conservation": True,
    }


def _position_constraint_review(source, base):
    """核真实期货披露、受限估值和回购融资；不从零公允价值推零名义敞口。"""
    raw = _text(source, base["source_hash"])
    s = _plain(raw)
    balance = _slice(s, r"[67]\.1资产负债表", r"[67]\.2(?:利润表|收益表)")
    m = re.search(r"卖出回购金融资产款(" + MONEY + r"|-)", balance)
    _require(m, "REPO_FINANCING_REQUIRED")
    repo = D(0) if m[1] == "-" else _money(m[1])
    _require(repo <= D(base["NAV_CNY"]) * D(".4"), "REPO_FINANCING_40_PERCENT")
    restricted = _slice(s, r"[67]\.4\.12\.1因认购", r"[67]\.4\.12\.3期末债券")
    _require(
        re.search(r"未持有因认购新发或增发证券[^。]*流通受限证券", restricted), "NEW_ISSUE_RESTRICTION_REVIEW_REQUIRED"
    )
    if "未持有暂时停牌等流通受限股票" in restricted:
        restricted_value = D(0)
        positions = []
    else:
        matches = list(re.finditer(r"[67]\.4\.12\.2", raw))
        _require(matches, "RESTRICTED_TABLE_REQUIRED")
        tail = raw[matches[-1].end() :]
        end = re.search(r"[67]\.4\.12\.3", tail)
        _require(end, "RESTRICTED_TABLE_END")
        positions = []
        for m in re.finditer(r"([\d,]+)\s+(" + MONEY + r")\s+(" + MONEY + r")\s*-", tail[: end.start()]):
            quantity = D(m[1].replace(",", ""))
            value = _money(m[3])
            stock_matches = [
                r for r in base["complete_positions"] if D(r["quantity"]) == quantity and D(r["value_CNY"]) == value
            ]
            _require(len(stock_matches) == 1, "RESTRICTED_STOCK_BOOK_RECONCILIATION")
            positions.append(
                {"security": stock_matches[0]["security"], "quantity": str(quantity), "value_CNY": str(value)}
            )
        _require(positions, "RESTRICTED_VALUE_UNKNOWN")
        restricted_value = sum(D(row["value_CNY"]) for row in positions)
    _require(restricted_value <= D(base["NAV_CNY"]) * D(".15"), "RESTRICTED_15_PERCENT")
    futures = _slice(s, r"[78]\.10本基金投资股指期货", r"[78]\.12投资组合报告附注")
    _require(
        re.search(r"未(?:进行|持有)[^。]{0,15}股指期货", futures)
        and re.search(r"未(?:进行|持有)[^。]{0,15}国债期货", futures),
        "FUTURES_NOMINAL_EXPOSURE_UNRESOLVED",
    )
    _require("未持有资产支持证券" in s and "未持有权证" in s, "OTHER_SECURITIES_SCOPE_UNRESOLVED")
    return {
        "restricted_CNY": str(restricted_value),
        "restricted_positions": positions,
        "repo_financing_CNY": str(repo),
        "explicit_futures_absence_quote": futures,
        "manager_all_portfolios_independently_reconstructed": False,
        "daily_trading_constraints_independently_reconstructed": False,
    }


def _execution(inputs, source):
    """执行观察先完成六期全公司证券合并及其余公开持仓限制；不再仅核单股。"""
    _require(
        inputs.get("execution_contract_refs") and inputs.get("issuer_identity_refs"),
        "EXECUTION_CONTRACT_AND_ISSUER_ORIGINALS_REQUIRED",
    )
    nodes, audit = _public_stock_and_cash_observations(inputs, source)
    for base in audit["full_reports"]:
        applicable = max((day for day in inputs["execution_contract_refs"] if day <= base["period"]), default=None)
        _require(applicable, "HISTORICAL_CONTRACT_REQUIRED")
        h = inputs["execution_contract_refs"][applicable]
        notice = _notice(source, h)
        s = _plain(notice["notice_content"])
        _require(
            LEGAL in s
            and MANAGER in s
            and {MASTER, CODE} <= {str(r["stock"]) for r in notice.get("security", [])}
            and re.search(r"持有一家公司发行的证券[^。]{0,30}10%", s),
            "COMPANY_SECURITY_CONTRACT_BINDING",
        )
        base["contract_source_hash"] = h
        base["merged_company_securities"] = _combined_issuer_limits(inputs, source, base)
        base["other_position_constraints"] = _position_constraint_review(source, base)
    refs = list(inputs["execution_contract_refs"].values()) + [
        r["source_ref"] for r in inputs["issuer_identity_refs"].values()
    ]
    for node in nodes.values():
        node["source_refs"] = list(dict.fromkeys(node["source_refs"] + refs))
    audit["full_contract_daily_manager_scope_not_claimed"] = True
    return nodes, audit


def _fund_report_like_prospectus(source, h):
    data = _notice(source, h)
    s = _plain(data["notice_content"])
    # 招募正文只列主代码时，使用同一公告的A/C证券列表与正文C类条款绑定；
    # 不额外要求005187必须在招募正文中出现，也不借其他法律产品的合同。
    _require(
        LEGAL in s
        and MANAGER in s
        and MASTER in s
        and "C类基金份额" in s
        and {MASTER, CODE} <= {str(r["stock"]) for r in data.get("security", [])}
        and "招募说明书" in data["notice_title"],
        "PROSPECTUS_IDENTITY",
    )
    _require(re.search(r"股票投资占基金资产的比例为0[-－—～至]95%", s), "PROSPECTUS_0_95")
    return data, s


def _material(inputs, source):
    _require(
        inputs.get("material_event_pending") is False and inputs.get("material_review_ref") == MATERIAL_REVIEW_SHA,
        "MATERIAL_REVIEW_UNKNOWN_OR_UNAPPROVED",
    )
    review = _json(source, MATERIAL_REVIEW_SHA)
    _require(
        review.get("version") == "CA005187_EXECUTION_MATERIAL_REVIEW_V2"
        and review.get("fund_code") == CODE
        and review.get("master_code") == MASTER
        and review.get("primary_ref") == inputs["primary_ref"]
        and review.get("legal_name") == LEGAL
        and review.get("manager_name") == MANAGER
        and review.get("as_of_date") == inputs["as_of_date"]
        and review.get("review_closed") is True,
        "MATERIAL_REVIEW_BINDING",
    )
    _require(
        review.get("scope") == "EXECUTION_ONLY"
        and review.get("general_governance_cleared") is False
        and review.get("material_event_pending") is False,
        "MATERIAL_REVIEW_SCOPE",
    )
    _require(
        review["full_report_hashes"] == list(inputs["full_reports"].values())
        and review["quarter_report_hashes"] == list(inputs["quarter_reports"].values()),
        "MATERIAL_REPORT_SET_CHANGED",
    )
    _require(isinstance(review.get("events"), list) and len(review["events"]) >= 13, "PER_EVENT_REVIEW_REQUIRED")
    ids = set()
    temporary = {}
    for event in review["events"]:
        _require(
            event.get("id") not in ids
            and event.get("claims")
            and event.get("disposition")
            and event.get("unresolved_for_selected_components") is False,
            "UNRESOLVED_OR_UNSOURCED_EVENT",
        )
        ids.add(event["id"])
        for claim in event["claims"]:
            _claim(source, claim)
        if event.get("category") == "TEMPORARY_ANNOUNCEMENT":
            temporary[event["article_id"]] = event["claims"][0]["source_hash"]
    rows = []
    covered = False
    for i, h in enumerate(inputs["late_catalog_refs"], 1):
        ref = source.sources[h]
        raw = _json(source, h)
        _require(
            ref["params"] == {"fundcode": CODE, "pageIndex": i, "pageSize": 100, "type": 0}
            and raw.get("ErrCode") == 0
            and raw.get("PageIndex") == i,
            "CATALOG_REQUEST_BINDING",
        )
        _require(isinstance(raw.get("Data"), list), "CATALOG_ROWS")
        rows += raw["Data"]
        covered |= any(r["PUBLISHDATEDesc"] <= "2026-06-30" for r in raw["Data"]) or len(rows) >= raw["TotalCount"]
    _require(covered, "CATALOG_NOT_CLOSED")
    periodics = {
        _json(source, h)["data"]["art_code"]
        for h in [*inputs["full_reports"].values(), *inputs["quarter_reports"].values()]
    }
    selected = {
        r["ID"]: r
        for r in rows
        if "2026-06-30" < r["PUBLISHDATEDesc"] <= inputs["as_of_date"] and r["ID"] not in periodics
    }
    _require(set(temporary) == set(selected), "UNREVIEWED_TEMPORARY_EVENT")
    for article, h in temporary.items():
        notice = _notice(source, h)
        _require(
            notice["art_code"] == article
            and notice["notice_title"] == selected[article]["TITLE"]
            and notice["notice_date"][:10] == selected[article]["PUBLISHDATEDesc"],
            "EVENT_CATALOG_BODY_CONFLICT",
        )
    expected = (
        set(inputs["full_reports"].values())
        | set(inputs["quarter_reports"].values())
        | set(review["supporting_hashes"])
        | {
            inputs[k]
            for k in ["primary_ref", "prospectus_ref", "material_review_ref", "public_execution_scope_decision_ref"]
        }
        | set(inputs["late_catalog_refs"])
        | set(inputs["execution_contract_refs"].values())
        | {row["source_ref"] for row in inputs["issuer_identity_refs"].values()}
    )
    _require(set(source.sources) == expected, "UNDECLARED_SOURCE_SET")
    return {
        "review_hash": MATERIAL_REVIEW_SHA,
        "events_reviewed": len(review["events"]),
        "late_temporary_notices_reviewed": len(temporary),
        "governance_state_preserved": "REMEDIATION_DISCLOSED",
        "general_governance_cleared": False,
        "material_event_pending_for_selected_components": False,
    }


def calculate(inputs, *, root):
    """真实原件只编译执行两子项，标准R5选efficiency；其余七维和总分不生成。"""
    source = _sources(inputs, root)
    material = _material(inputs, source)
    with localcontext() as ctx:
        ctx.prec = 75
        observations, audit = _execution(inputs, source)
    # 单只中小企业私募债的真实合同上限为10%NAV。任一单只金额不大于同公司
    # 全证券合计；六期后者已核<=10%，因此无需猜每只信用债是否属于SME私募。
    for row in audit["full_reports"]:
        merged = row["merged_company_securities"]
        credit = [
            D(s["value_CNY"])
            for issuer in merged["issuers"]
            for s in issuer["securities"]
            if s["kind"] == "CREDIT_BOND"
        ]
        upper = max(credit, default=D(0))
        _require(upper <= D(row["NAV_CNY"]) * D(".10"), "SINGLE_PRIVATE_BOND_UPPER_BOUND")
        row["single_SME_private_bond_upper_bound_CNY"] = str(upper)
        row["single_SME_private_bond_contract_limit_ratio"] = ".10"
        row["SME_classification_not_guessed"] = True
    registry = {h: {**ref, "path": str(Path(root) / ref["file"])} for h, ref in source.sources.items()}
    packet = {
        "fund_code": CODE,
        "basis": ACTIVE_BASIS,
        "as_of_date": inputs["as_of_date"],
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "target_product_aliases": ["CN_PUBLIC_FUND:" + MASTER, "CN_PUBLIC_FUND:" + CODE],
            "material_event_pending": material["material_event_pending_for_selected_components"],
            "observations": observations,
        },
        "r4_evidence_registry": registry,
        "raw_archive": [
            {"kind": "RESEARCH_NOTE" if ref["role"] in RESEARCH else "ORIGINAL", "sha256": h}
            for h, ref in registry.items()
        ],
        "quality_public": {"dataDates": {"holdings": inputs["holding_date"]}},
    }
    replay = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"efficiency"})
    _require(
        set(replay["dimensions"]) == {"efficiency"} and all(replay[k] is None for k in ["score", "raw_score", "grade"]),
        "EXECUTION_PARTIAL_SCOPE",
    )
    return {
        "dimensions": replay["dimensions"],
        "observations": observations,
        "complete_score": None,
        "grade": None,
        "stability_score": None,
        "limitations": LIMITATIONS,
        "audit": {
            **audit,
            "material_events": material,
            "source_hashes_revalidated": len(source.raw),
            "public_execution_scope_decision_ref": SCOPE_DECISION_SHA,
        },
        "standard_replay": replay,
    }


def check_binding(bundle, inputs):
    """绑定最新本C份额、本期间、本法律产品、管理人及真实主报告；不接受A类。"""
    _require(
        bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == inputs.get("fund_code") == CODE
        and bundle.get("share_class") == inputs.get("share_class") == "C"
        and str(bundle.get("as_of_date")) == inputs.get("as_of_date")
        and str(bundle.get("holding_date")) == inputs.get("holding_date")
        and bundle["identity"]["legal_name"] == inputs.get("legal_name") == LEGAL
        and bundle["identity"]["manager_name"] == inputs.get("manager_name") == MANAGER
        and bundle["original_report_archive"]["sha256"] == inputs.get("primary_ref"),
        "CURRENT_BINDING",
    )


def validate_snapshot_component(bundle, *, root):
    """快照始终固定重算执行维；不能通过删作用域或改文档分数绕过原件。"""
    from app.services.fund_rating_reviewed_snapshot import _read

    field = bundle[FIELD]
    inputs = field["inputs"]
    check_binding(bundle, inputs)
    _require(_read(field["input_archive"], Path(root)) == inputs, "INPUT_ARCHIVE_MISMATCH")
    result = calculate(inputs, root=root)
    point = D(result["dimensions"]["efficiency"])
    node = bundle["reviewed_document"]["dimensions"]["execution"]
    _require(
        all(
            node.get(key) is not None and abs(D(str(node[key])) - point) <= D("1e-55")
            for key in ["lower", "upper", "candidate"]
        ),
        "SNAPSHOT_DIMENSION_RECALCULATION:execution",
    )
    return result


def prepare(session, request, *, root):
    """只读最新current，仅追加执行维；不改其他七维、历史或资料有效期。"""
    from datetime import UTC

    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_pipeline_prepare import archive_json
    from app.services.fund_rating_reviewed_snapshot import BASIS, validate_sources
    from app.services.fund_rating_reviewed_snapshot import calculate as snapshot_calculate

    _require(request.get("fund_code") == CODE and request.get("as_of_date") == "2026-09-30", "PREPARE_REQUEST_SCOPE")
    previous = current_inputs(session, [CODE]).get(CODE)
    _require(previous and str(previous["as_of_date"]) == request["as_of_date"], "LATEST_CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == CODE))
    inputs = deepcopy(request["inputs"])
    check_binding(bundle, inputs)
    _require(
        bundle.get("basis") == BASIS
        and not bundle.get("r5_calculation")
        and not bundle.get("ca005187_components_calculation"),
        "PUBLIC_EXECUTION_PARTIAL_BASE_REQUIRED",
    )
    result = calculate(inputs, root=root)
    document = bundle["reviewed_document"]
    old = document["dimensions"]["execution"] or {}
    point = result["dimensions"]["efficiency"]
    if old.get("candidate") is not None:
        _require(
            all(old.get(key) is not None and D(str(old[key])) == D(point) for key in ["lower", "upper", "candidate"]),
            "EXISTING_COMPONENT_POINT_CONFLICT",
        )
        raise ValueError("CA005187_COMPONENT_ALREADY_CURRENT")
    explanation = request.get("explanation", "").strip()
    _require(len(explanation) >= 40, "BUSINESS_EXPLANATION_REQUIRED")
    document["dimensions"]["execution"] = {"lower": point, "upper": point, "candidate": point, "reason": explanation}
    bundle.setdefault("public_explanations", {})["execution"] = explanation
    bundle.get("public_metrics", {}).pop("execution", None)
    bundle.update(research_format="single", research_archive=archive_json(Path(root), document))
    bundle[FIELD] = {"inputs": inputs, "input_archive": archive_json(Path(root), inputs), "limitations": LIMITATIONS}
    known = {ref["sha256"]: ref for ref in bundle.get("extra_archives", [])}
    known.update({h: {key: ref[key] for key in ["file", "sha256", "kind"]} for h, ref in inputs["sources"].items()})
    bundle["extra_archives"] = list(known.values())
    validate_sources(bundle, datetime.now(UTC), root=Path(root))
    validate_snapshot_component(bundle, root=root)
    snapshot_calculate(bundle, root=Path(root))
    return {
        "basis": BASIS,
        "entries": [{key: value for key, value in bundle.items() if key not in ["basis", "reviewed_document"]}],
    }
