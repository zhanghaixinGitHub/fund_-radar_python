"""004605完整合同公开执行候选：原件编译、条款绑定与独立快照回放。"""

from __future__ import annotations

import calendar
import json
import re
import unicodedata
from copy import deepcopy
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path

from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import content_digest, source_bytes, source_exists

VERSION = "R5_004605_EXECUTION_FULL_CONTRACT_CUSTODY_20261030_5_CANDIDATE"
FIELD = "active_004605_execution_calculation"
SECURITIES_ADAPTER = "CN_004605_FULL_CONTRACT_CUSTODY_PUBLIC_EXECUTION_V3"
REVIEW_SHA = "58934b47e24a9b5e677d9fb9b278679f24fd6fbdd8c668846c81932d04fe7522"
CONTRACT_SHA = "5ca870ac61bdca8098d8b83b36c6fad39c4e3e560616a4794465dd72fb996400"
PRIMARY = "2ce51f77ed65ae0f62e2aacc025b343998eea47dcbc91b3150c06346b2d1c43e"
MONEY = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
D = Decimal


def _reject(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError("R5_ACTIVE_EXECUTION_" + code)


def _plain(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def _amount(text: str) -> Decimal:
    value = D(text.replace(",", ""))
    _reject(value.is_finite() and value >= 0, "INVALID_AMOUNT")
    return value


class Originals:
    """每次重放核原始字节、来源目录和日期；不接受派生状态当作原件。"""

    def __init__(self, inputs: dict, root: Path):
        self.root = root.resolve()
        self.sources = inputs["sources"]
        _reject(1 <= len(self.sources) <= 300, "SOURCE_COUNT")
        self.raw: dict[str, bytes] = {}
        self.cache: dict[str, str] = {}
        for digest, ref in self.sources.items():
            path = (self.root / ref["file"]).resolve()
            _reject(path.is_relative_to(self.root / "sources") and source_exists(path), "SOURCE_PATH")
            raw = source_bytes(path)
            _reject(len(raw) <= 50000000 and digest == ref["sha256"] == content_digest(raw), "SOURCE_HASH")
            _reject(ref.get("archive_kind", "ORIGINAL") == "ORIGINAL", "DERIVED_SOURCE")
            if ref.get("published_on"):
                _reject(ref["published_on"] <= inputs["as_of_date"], "SOURCE_AFTER_CUTOFF")
            self.raw[digest] = raw

    def json(self, digest: str):
        return json.loads(self.raw[digest])

    def body(self, digest: str) -> str:
        if digest in self.cache:
            return self.cache[digest]
        ref = self.sources[digest]
        if ref["kind"] == "pdf":
            from app.services.fund_rating_pdf import text_pages

            text = "\n".join(text_pages(self.root / ref["file"]))
        elif ref["kind"] == "json":
            data = self.json(digest)
            _reject(
                data.get("success") == 1 and isinstance(data.get("data", {}).get("notice_content"), str), "NOTICE_JSON"
            )
            text = data["data"]["notice_content"]
        elif ref["kind"] in {"text", "html", "htm"}:
            encoding = ref.get("encoding", "utf-8")
            _reject(encoding in {"utf-8", "utf-8-sig", "gb18030"}, "ENCODING")
            text = self.raw[digest].decode(encoding, errors="strict")
            if ref["kind"] != "text":
                from bs4 import BeautifulSoup

                text = BeautifulSoup(text, "html.parser").get_text("\n", strip=True)
        else:
            raise ValueError("R5_ACTIVE_EXECUTION_SOURCE_FORMAT")
        _reject(len(text) >= 1000, "BODY_INCOMPLETE")
        self.cache[digest] = text
        return text


def _section(text: str, pattern: str, end: str) -> str:
    matches = [m for m in re.finditer(pattern, text) if not re.match(r"\s*(?:\.{3,}|…)", text[m.end() :])]
    _reject(bool(matches), "SECTION_MISSING:" + pattern)
    match = matches[-1]
    tail = text[match.end() :]
    stop = re.search(end, tail)
    _reject(stop is not None, "SECTION_BOUNDARY:" + end)
    return text[match.start() : match.end() + stop.start()]


def _quarter_end(year: int, quarter: int) -> date:
    month = quarter * 3
    return date(year, month, calendar.monthrange(year, month)[1])


def _catalog(source: Originals, inputs: dict) -> dict:
    """由真实公告目录推导必要截面，不能由调用者删除某个季度后仍通过。"""
    from app.integrations.public_fund_reports import report_period

    cutoff = date.fromisoformat(inputs["as_of_date"])
    window_start = date(
        cutoff.year - 3, cutoff.month, min(cutoff.day, calendar.monthrange(cutoff.year - 3, cutoff.month)[1])
    )
    first = max(
        _quarter_end(year, quarter)
        for year in range(window_start.year - 1, window_start.year + 1)
        for quarter in range(1, 5)
        if _quarter_end(year, quarter) <= window_start
    )
    pages, total, records = [], None, []
    for digest in inputs["catalog_refs"]:
        raw = source.json(digest)
        _reject(raw.get("ErrCode") == 0 and isinstance(raw.get("Data"), list), "CATALOG_FORMAT")
        _reject(total in {None, raw["TotalCount"]}, "CATALOG_TOTAL_CHANGED")
        total = raw["TotalCount"]
        pages.append(raw["PageIndex"])
        records.extend(raw["Data"])
    _reject(sorted(pages) == list(range(1, len(pages) + 1)), "CATALOG_PAGES_MISSING")
    ids = [row["ID"] for row in records]
    _reject(len(set(ids)) == len(ids), "CATALOG_DUPLICATE")
    _reject(all(row["FUNDCODE"] == inputs["fund_code"] for row in records), "CATALOG_WRONG_SHARE")
    oldest = min(row["PUBLISHDATEDesc"][:10] for row in records)
    _reject(len(records) == total or oldest < str(first), "CATALOG_WINDOW_NOT_CLOSED")
    expected: dict[str, dict[str, dict]] = {"quarters": {}, "full_reports": {}}
    excluded = []
    for row in records:
        period = report_period(row["TITLE"])
        if not period or any(word in row["TITLE"] for word in ["提示", "摘要"]):
            continue
        end, kind = period
        if not str(first) <= end <= str(cutoff):
            continue
        if row["PUBLISHDATEDesc"][:10] > str(cutoff):
            excluded.append({"period": end, "title": row["TITLE"], "published_on": row["PUBLISHDATEDesc"][:10]})
            continue
        role = "quarters" if kind == "QUARTER" else "full_reports" if kind in {"ANNUAL", "HALF"} else None
        if role:
            _reject(end not in expected[role], "CATALOG_PERIOD_CONFLICT")
            expected[role][end] = row
    _reject(bool(expected["quarters"]) and bool(expected["full_reports"]), "CATALOG_EMPTY")
    last = max(expected["quarters"])
    sequence = [
        str(_quarter_end(year, quarter))
        for year in range(first.year, cutoff.year + 1)
        for quarter in range(1, 5)
        if str(first) <= str(_quarter_end(year, quarter)) <= last
    ]
    _reject(sorted(expected["quarters"]) == sequence, "CATALOG_QUARTERS_NOT_CONTIGUOUS")
    full_sequence = [period for period in sequence if period[5:7] in {"06", "12"} and period <= inputs["holding_date"]]
    _reject(sorted(expected["full_reports"]) == full_sequence, "CATALOG_FULL_REPORTS_NOT_CONTIGUOUS")
    _reject(
        set(inputs["quarters"]) == set(expected["quarters"])
        and set(inputs["full_reports"]) == set(expected["full_reports"]),
        "NECESSARY_PERIODS_MISSING",
    )
    expected["excluded_after_cutoff"] = excluded
    expected["first_boundary_anchor"] = str(first)
    return expected


def _share_binding(source: Originals, inputs: dict) -> None:
    """原表按列绑定份额名称和代码，避免同一报告出现多份额时串用身份。"""
    primary = _plain(source.body(inputs["primary_ref"]))[:18000]
    binding = inputs["share_binding"]
    names, codes = binding["column_names"], binding["column_codes"]
    _reject(len(names) == len(codes) >= 1 and len(set(codes)) == len(codes), "SHARE_COLUMNS")
    _reject("下属分级基金的基金简称" + "".join(map(_plain, names)) in primary, "SHARE_NAMES_NOT_IN_ORIGINAL")
    _reject("下属分级基金的交易代码" + "".join(codes) in primary, "SHARE_CODES_NOT_IN_ORIGINAL")
    matching = [i for i, name in enumerate(names) if _plain(name).endswith(inputs["share_class"])]
    _reject(len(matching) == 1 and codes[matching[0]] == inputs["fund_code"], "WRONG_SHARE")


def _issuer_securities_upper(stocks: list[dict], bond_amount: Decimal, nav: Decimal) -> Decimal:
    """全部债券保守并入最大股票发行人；无须虚构债券发行人关系也能验证10%上界。"""
    _reject(bool(stocks) and nav > 0 and bond_amount >= 0, "ISSUER_SECURITIES_INPUT")
    maximum = max(D(row["amount_cny"]) for row in stocks)
    _reject(maximum >= 0, "NEGATIVE_STOCK_AMOUNT")
    upper = (maximum + bond_amount) / nav * 100
    _reject(upper <= 10, "ALL_SECURITIES_ISSUER_LIMIT_REVIEW_REQUIRED")
    return upper


def _flex_contract(source: Originals, inputs: dict) -> dict:
    """证券合并限额的独立分派；仅核真实条款，不为缺失的非股票比例制造上下限。"""
    item = inputs["securities_contract"]
    text = _plain(source.body(item["source_ref"]))
    _reject(_plain(inputs["legal_name"]) in text and _plain(inputs["manager_name"]) in text, "CONTRACT_IDENTITY")
    # 原件只给出版本月份，没有确切公告/生效日。本适配保留这一精度，并要求
    # 六份当期完整报告各自复述相同投资范围和比例；不填一个虚假的月初生效日。
    _reject(item.get("document_month") == "2022-09" and "二零二二年九月" in text, "CONTRACT_VERSION_MONTH")
    required = {
        "equity": "本基金股票及存托凭证投资占基金资产的比例为0%-95%",
        "cash": "不低于基金资产净值5%的现金或者到期日在一年以内的政府债券",
        "cash_exclusions": "现金不包括结算备付金、存出保证金、应收申购款",
        "issuer": "本基金持有一家公司发行的证券,其市值不超过基金资产净值的10%",
        "leverage": "基金资产总值不得超过基金资产净值的140%",
    }
    for key, clause in required.items():
        _reject(clause in text, "FLEX_CONTRACT_CLAUSE:" + key)
    _reject(
        all(
            word in text[text.find("投资范围") :] for word in ["国内依法发行上市的股票", "存托凭证", "债券", "银行存款"]
        ),
        "FLEX_CONTRACT_ALLOWED_ASSETS",
    )
    return {
        "source_ref": item["source_ref"],
        "document_month": item["document_month"],
        "exact_publication_or_effective_day_not_inferred": True,
        "clauses": required,
    }


def _flex_assets(text: str) -> tuple[dict, Decimal]:
    """兼容没有单列基金投资的八行表；缺列保持None，横杠由全表非负金额守恒校验。"""
    table = _section(text, r"期末基金资产组合情况", r"(?:\d\.2\s*)?报告期末按行业")
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
    rows, ranks = {}, []
    for label in labels:
        match = re.search(r"(?:^|\n)\s*(\d+)\s+" + label + r"\s+(" + MONEY + r"|[-—－])\s+([\d.]+|[-—－])", table)
        if match is None and label == "基金投资":
            rows[label] = {"amount_cny": None, "explicit_dash_zero": False, "absent_in_original_layout": True}
            continue
        _reject(match is not None, "FLEX_ASSET_ROW:" + label)
        ranks.append(int(match[1]))
        dash = match[2] in {"-", "—", "－"}
        rows[label] = {
            "amount_cny": "0" if dash else str(_amount(match[2])),
            "explicit_dash_zero": dash,
            "source_row": match[0].strip(),
        }
    _reject(ranks == list(range(1, len(ranks) + 1)), "FLEX_ASSET_LAYOUT_SEQUENCE")
    total = re.search(r"(?:^|\n)\s*(\d+)\s+合计\s+(" + MONEY + r")\s+100\.00", table)
    _reject(total is not None and int(total[1]) == len(ranks) + 1, "FLEX_ASSET_TOTAL")
    amount = _amount(total[2])
    _reject(
        sum(D(row["amount_cny"]) for row in rows.values() if row["amount_cny"] is not None) == amount > 0,
        "FLEX_ASSET_RECONCILIATION",
    )
    for label, subtotal_label in [("权益投资", "股票"), ("固定收益投资", "债券")]:
        subtotal = re.search(r"其中[：:]\s*" + subtotal_label + r"\s+(" + MONEY + r"|[-—－])", table)
        _reject(subtotal is not None, "FLEX_INVESTMENT_SUBTOTAL:" + label)
        value = D(0) if subtotal[1] in {"-", "—", "－"} else _amount(subtotal[1])
        _reject(value == D(rows[label]["amount_cny"]), "FLEX_STOCK_OR_BOND_SUBTOTAL_CONFLICT")
    return rows, amount


def _flex_stocks(text: str, full: bool) -> list[dict]:
    """只在完整/前十表的明确标题与下一节之间解析，兼容原件标题中文字间换行。"""
    heading = r"\s*".join("所有股票投资明细" if full else "前十名股票投资明细")
    table = _section(text, heading, r"(?:\d\.4\s*)?(?:报告期内股票|报告期末按债券)")
    # 公开转载会把六位代码与中文名称相连；证券代码固定六位，不靠空格猜边界。
    pattern = r"(?:^|\n)\s*(\d+)\s+(\d{6})\s*(.+?)\s+([\d,]+)\s+(" + MONEY + r")\s+([\d.]+)"
    rows = [
        {"rank": int(m[1]), "security": m[2], "name": m[3].strip(), "amount_cny": str(_amount(m[5]))}
        for m in re.finditer(pattern, table)
    ]
    _reject(bool(rows) and [r["rank"] for r in rows] == list(range(1, len(rows) + 1)), "STOCK_RANK_CLOSURE")
    _reject(len({r["security"] for r in rows}) == len(rows), "DUPLICATE_STOCK")
    return rows


def _flex_report(source: Originals, inputs: dict, digest: str, entry: dict, period: str, full: bool) -> dict:
    """按真实18个披露截面核资产、发行人保守上界和本金，未知日间账户保持未知。"""
    from app.services.fund_quality_v2_disclosures import report_period

    text = unicodedata.normalize("NFKC", source.body(digest))
    plain = _plain(text)
    _reject(inputs["fund_code"] in plain[:18000] and _plain(inputs["legal_name"]) in plain[:18000], "REPORT_IDENTITY")
    _reject(str(report_period(text)[1]) == period, "REPORT_PERIOD")
    article = source.json(digest)["data"]
    _reject(
        article.get("art_code") == entry["ID"]
        and _plain(article.get("notice_title", "")) == _plain(entry["TITLE"])
        and article["notice_date"][:10] == source.sources[digest].get("published_on") == entry["PUBLISHDATEDesc"][:10],
        "FLEX_NOTICE_CATALOG_BINDING",
    )
    assets, total = _flex_assets(text)
    stock = D(assets["权益投资"]["amount_cny"])
    bonds = D(assets["固定收益投资"]["amount_cny"])
    stocks = _flex_stocks(text, full)
    _reject(all(re.fullmatch(r"(?:0|3|6)\d{5}", s["security"]) for s in stocks), "FLEX_DOMESTIC_STOCK_SCOPE")
    _reject(
        all(assets[name]["explicit_dash_zero"] for name in ["贵金属投资", "金融衍生品投资", "买入返售金融资产"])
        and (assets["基金投资"]["amount_cny"] is None or assets["基金投资"]["explicit_dash_zero"]),
        "FLEX_OTHER_ASSET_CLASS_REVIEW_REQUIRED",
    )
    compliance = _section(
        text,
        r"(?m)^\s*(?:4\.2\s*)?管理人对报告期内本基金运作遵规守信情况的说明",
        r"(?m)^\s*(?:4\.3\s*|(?:管理人对报告期内)?公平交易)",
    )
    _reject("基金投资组合符合有关法规及基金合同的规定" in _plain(compliance), "FLEX_ACTUAL_COMPLIANCE")
    _reject(
        not any(w in _plain(compliance) for w in ["但是", "除外", "尚未", "未完成"]), "FLEX_COMPLIANCE_CONTRADICTION"
    )
    row = {
        "period": period,
        "source_ref": digest,
        "kind": "FULL" if full else "QUARTER",
        "stock_amount_cny": str(stock),
        "total_assets_cny": str(total),
        "stocks": stocks,
        "manager_quote": compliance,
        "quarter_cash_principal_cny": None,
        "daily_breach_state": None,
        "manager_wide_breach_state": None,
        "quarter_unobserved_cash_not_backfilled": True,
        "fund_investment_absent_original_layout_not_imputed": assets["基金投资"]["amount_cny"] is None,
    }
    if full:
        balance = _section(text, r"(?m)^\s*(?:[67]\.1\s*)?资产负债表\s*$", r"(?m)^\s*(?:[67]\.2\s*)?利润表\s*$")
        nav_row = re.search(r"净资产合计\s+(" + MONEY + r")", balance)
        _reject(nav_row is not None, "FLEX_BALANCE_NAV")
        nav = _amount(nav_row[1])
        _reject(sum(D(s["amount_cny"]) for s in stocks) == stock, "FULL_STOCK_RECONCILIATION")
        legal = _section(text, r"(?m)^\s*(?:[67]\.4\.1\s*)?基金基本情况\s*$", r"(?:[67]\.4\.2\s*|会计报表的编制基础)")
        legal_plain = _plain(legal)
        _reject(
            all(
                clause in legal_plain
                for clause in [
                    "股票及存托凭证投资占基金资产的比例为0%-95%",
                    "不低于基金资产净值的5%",
                    "现金不包括结算备付金、存出保证金、应收申购款",
                    "国内依法发行上市的股票",
                    "债券",
                    "银行存款",
                ]
            ),
            "FLEX_REPORT_ACTUAL_CONTRACT_SCOPE",
        )
        cash_note = _section(
            text,
            r"(?m)^\s*(?:[67]\.4\.7\.1\s*)?(?:货币资金|银行存款)\s*$",
            r"(?m)^\s*(?:[67]\.4\.7\.2\s*)?交易性金融资产\s*$",
        )
        cash_row = re.search(r"等于[：:]本金\s+(" + MONEY + r")", cash_note)
        _reject(cash_row is not None, "QUALIFIED_CASH_PRINCIPAL")
        cash = _amount(cash_row[1])
        # 本金单独满足5%即可，不将备付金、保证金、利息、应收或不明国债期限加进去。
        _reject(cash / nav * 100 >= 5, "CASH_CONSTRAINT_REVIEW_REQUIRED")
        trustee = _section(
            text,
            r"(?m)^\s*(?:5\.2\s*)?托管人对报告期内本基金投资运作",
            r"(?:5\.3\s*)?托管人对本(?:年度|中期)报告中财务信息",
        )
        _reject("未发现基金管理人有损害基金份额持有人利益的行为" in _plain(trustee), "TRUSTEE_ACTUAL_REVIEW")
        bond_table = _section(
            text,
            r"(?:[78]\.5\s*)?期末按债券品种分类的债券投资组合",
            r"(?:[78]\.6\s*)?期末按公允价值",
        )
        bond_rows = [
            {"rank": int(m[1]), "category": _plain(m[2]), "amount_cny": "0" if m[3] == "-" else str(_amount(m[3]))}
            for m in re.finditer(r"(?:^|\n)\s*([1-9])\s+([^\n]+?)\s+(" + MONEY + r"|-)\s+([\d.]+|-)", bond_table)
        ]
        if bonds == 0:
            # 无债券期间原文只披露明确的“未持有”句子，没有九行空表，不能伪造九行0。
            _reject(not bond_rows and "本基金本报告期末未持有债券" in _plain(bond_table), "ACTUAL_NO_BONDS_STATEMENT")
        else:
            _reject([b["rank"] for b in bond_rows] == list(range(1, 10)), "BOND_CATEGORIES_INCOMPLETE")
            _reject(sum(D(b["amount_cny"]) for b in bond_rows) == bonds, "BOND_TOTAL")
        permitted = {
            "国家债券",
            "央行票据",
            "金融债券",
            "企业债券",
            "企业短期融资券",
            "中期票据",
            "可转债(可交换债)",
            "同业存单",
            "其他",
        }
        _reject(all(b["category"] in permitted for b in bond_rows), "BOND_SCOPE_REQUIRES_ADAPTER")
        if bond_rows:
            _reject(next(b for b in bond_rows if b["category"] == "其他")["amount_cny"] == "0", "UNKNOWN_OTHER_BONDS")
        row.update(
            qualified_cash_principal_cny=str(cash),
            qualified_cash_lower_pct=str(cash / nav * 100),
            cash_note=cash_note,
            trustee_quote=trustee,
            actual_report_scope_quote=legal,
            bond_categories=bond_rows,
        )
    else:
        nav_row = re.search(r"4[.、]?\s*期末基金资产净值\s+((?:" + MONEY + r"\s*)+)", text)
        _reject(nav_row is not None, "QUARTER_NAV_COLUMNS")
        columns = re.findall(MONEY, nav_row[1])
        _reject(len(columns) == len(inputs["share_binding"]["column_codes"]), "QUARTER_SHARE_COUNT")
        nav = sum(_amount(v) for v in columns)
        _reject(len(stocks) == 10, "QUARTER_TOP10")
    _reject(
        nav > 0 and nav <= total and 0 <= stock / total * 100 <= 95 and stock / nav * 100 <= 95,
        "STOCK_DENOMINATOR_AMBIGUITY_REVIEW_REQUIRED",
    )
    _reject(total / nav * 100 <= 140, "LEVERAGE_CONSTRAINT_REVIEW_REQUIRED")
    # 合同17(2)限制全部有价证券；将所有债券都纳入分子作保守上界，
    # 不把可能合格的短政府债先扣掉，也不以股票单独低于95%代替此项。
    securities_upper = (stock + bonds) / nav * 100
    _reject(securities_upper <= 95, "ALL_SECURITIES_95_CONSTRAINT_REVIEW_REQUIRED")
    upper = _issuer_securities_upper(stocks, bonds, nav)
    row.update(
        nav_cny=str(nav),
        stock_total_assets_pct=str(stock / total * 100),
        stock_nav_pct=str(stock / nav * 100),
        leverage_pct=str(total / nav * 100),
        all_securities_nav_upper_pct=str(securities_upper),
        issuer_all_bonds_conservative_upper_pct=str(upper),
    )
    return row


def _read_derived(inputs, root, digest):
    """派生核查单只作引用图；每次核字节，不以其中的分数或通过标记驱动评分。"""
    reference = inputs["sources"][digest]
    path = (root / reference["file"]).resolve()
    _reject(path.is_relative_to(root.resolve() / "sources"), "DERIVED_PATH")
    raw = source_bytes(path)
    _reject(len(raw) <= 524288 and content_digest(raw) == digest == reference["sha256"], "DERIVED_HASH")
    _reject(reference["archive_kind"] == "DERIVED", "DERIVED_ROLE")
    return json.loads(raw)


def check_binding(bundle, inputs):
    """只准同份额同披露日接入执行；既有表现和流动性组件按原包保留。"""
    _reject(
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
        "BUNDLE_BINDING",
    )
    _reject(
        not any(
            bundle.get(k)
            for k in [
                "r5_calculation",
                "passive_components_calculation",
                "passive_execution_calculation",
                "active_execution_calculation",
            ]
        ),
        "CONFLICTING_COMPONENT",
    )
    packets = bundle.get("active_performance_calculations", [])
    _reject(
        bool(packets)
        and all(p["inputs"]["material_event_review_ref"] == inputs["material_event_review_ref"] for p in packets),
        "ADOPTED_MATERIAL_REVIEW",
    )


def _restriction_endpoint(source, inputs, review):
    """从真实单价和数量重算22笔新股限售，并核停牌、回购和出借原文。"""
    digest = inputs["full_reports"][inputs["holding_date"]]
    text = source.body(digest)
    section = text[text.index("6.4.12.1.1") : text.index("6.4.12.2")]
    lots = []
    for m in re.finditer(r"(?m)^\s*(\d{6})[^\n]*新股限售\s+(" + MONEY + r")\s+(" + MONEY + r")\s+([\d,]+)", section):
        price, quantity = _amount(m[3]), D(m[4].replace(",", ""))
        lots.append(
            {
                "security": m[1],
                "unit_price_cny": str(price),
                "quantity": str(quantity),
                "market_value_cny": str(price * quantity),
                "original_row": m[0],
            }
        )
    _reject(len(lots) == 22 and len({r["security"] for r in lots}) == 22, "RESTRICTED_LOTS_SCOPE")
    amount = sum(D(r["market_value_cny"]) for r in lots)
    stated = re.search(MONEY, review["contract_combination_clauses"][19]["public_evidence"])
    _reject(stated is not None and amount == _amount(stated[0]), "RESTRICTED_REVIEW_AMOUNT")
    tail = _plain(text[text.index("6.4.12.2") : text.index("6.4.13 金融工具")])
    quotes = [
        "本基金本报告期末未持有暂时停牌等流通受限股票",
        "本基金无因从事银行间市场债券正回购交易形成的卖出回购证券款余额",
        "本基金无因从事交易所市场债券正回购交易形成的卖出回购证券款余额",
        "本基金本报告期末未持有参与转融通证券出借业务的证券",
    ]
    _reject(all(q in tail for q in quotes), "OTHER_RESTRICTIONS_REVIEW_REQUIRED")
    return {"source_ref": digest, "lots": lots, "amount_cny": str(amount), "other_endpoint_absence_quotes": quotes}


def _full_contract_scope(source, inputs, rows):
    """将23条及禁止行为核查单绑定到合同原文、18截面和最新受限资产实算。"""
    digest = inputs.get("contract_scope_review_ref")
    _reject(digest == REVIEW_SHA, "FULL_CONTRACT_REVIEW_REQUIRED")
    review = _read_derived(inputs, source.root, digest)
    _reject(review["fund_code"] == inputs["fund_code"] and review["as_of"] == inputs["as_of_date"], "REVIEW_IDENTITY")
    _reject(
        inputs["securities_contract"]["source_ref"] == review["contract_source_ref"] == CONTRACT_SHA, "REVIEW_CONTRACT"
    )
    profile = _flex_contract(source, inputs)
    contract = source.body(CONTRACT_SHA)
    start = contract.index("四、投资限制")
    end = contract.index("2、禁止行为", start)
    body = contract[start:end].split("除第（2）")[0]
    marks = list(re.finditer(r"（(\d+)）", body))
    actual = {
        int(m[1]): body[m.start() : marks[i + 1].start() if i + 1 < len(marks) else None] for i, m in enumerate(marks)
    }
    clauses = review["contract_combination_clauses"]
    _reject(
        sorted(actual) == list(range(1, 24)) and [r["clause"] for r in clauses] == list(range(1, 24)), "ALL_23_CLAUSES"
    )
    for row in clauses:
        _reject(_plain(row["actual_contract_text"]) == _plain(actual[row["clause"]]), "CLAUSE_QUOTE_BINDING")
    prohibition = contract[end : contract.index("五、", end)]
    _reject(
        _plain(prohibition) == _plain(review["prohibitions_and_related_transactions_actual_text"]),
        "PROHIBITION_BINDING",
    )
    _reject([int(m[1]) for m in re.finditer(r"（(\d+)）", prohibition)] == list(range(1, 7)), "ALL_SIX_PROHIBITIONS")
    _reject(all(str(n) + "）" in actual[17] for n in range(1, 6)), "FIVE_FUTURES_SUBCLAUSES")
    _reject(_plain(review["futures_five_subclauses_actual_text_retained"]) == _plain(actual[17]), "FUTURES_SCOPE")
    _reject(
        _plain(review["external_causes_exceptions_actual_text"]) == _plain(contract[start:end][len(body) :]),
        "EXEMPTION_SCOPE",
    )
    keys = {
        "period",
        "kind",
        "source_ref",
        "stock_total_assets_pct",
        "stock_nav_pct",
        "leverage_pct",
        "issuer_all_bonds_conservative_upper_pct",
        "qualified_cash_lower_pct",
        "quarter_cash_principal_cny",
    }
    observed = [{k: v for k, v in r.items() if k in keys} for r in rows]
    _reject(observed == review["public_period_observations"], "ALL_18_PERIOD_REVIEW_BINDING")
    full = {r["period"]: r for r in rows if r["kind"] == "FULL"}
    _reject(len(review["manager_and_trustee_original_quotes"]) == len(full) == 6, "SIX_FULL_QUOTES")
    for item in review["manager_and_trustee_original_quotes"]:
        row = full[item["period"]]
        _reject(
            item["source_ref"] == row["source_ref"]
            and _plain(item["manager"]) == _plain(row["manager_quote"])
            and _plain(item["trustee"]) == _plain(row["trustee_quote"]),
            "ACTUAL_MANAGER_TRUSTEE_QUOTES",
        )
    absence = review["endpoint_absence_original_quotes"]
    expected = {(role, period, h) for role in ["quarters", "full_reports"] for period, h in inputs[role].items()}
    _reject(
        len(absence) == 18 and {(r["role"], r["period"], r["source_ref"]) for r in absence} == expected, "ABSENCE_SCOPE"
    )
    for item in absence:
        text = _plain(source.body(item["source_ref"]))
        _reject(
            set(item["quotes"])
            == ({"warrants", "ABS", "futures"} if item["role"] == "quarters" else {"warrants", "ABS"}),
            "ABSENCE_ROLES",
        )
        for name, quote in item["quotes"].items():
            normalized = _plain(quote)
            _reject(bool(normalized) and normalized in text, "ABSENCE_QUOTE_BINDING:" + name)
            _reject("未" in normalized, "ABSENCE_NOT_A_POSITIVE_HOLDING")
    restricted = _restriction_endpoint(source, inputs, review)
    nav = D(full[inputs["holding_date"]]["nav_cny"])
    _reject(D(restricted["amount_cny"]) / nav * 100 <= 15, "RESTRICTED_15_LIMIT")
    # True仅用于有原件支撑的期末数值；每日交易及全管理人事实仍为None。
    numeric = {1, 2, 3, 6, 9, 10, 11, 13, 16, 17, 18, 20, 22}
    profile.update(
        full_contract_review_ref=digest,
        clauses=[
            {
                "clause": r["clause"],
                "actual_text": actual[r["clause"]],
                "public_scope": r["reviewed_scope"],
                "unobserved_boundary": r["unobserved_boundary"],
                "observed_endpoint_result": True if r["clause"] in numeric else None,
                "daily_or_manager_wide_result": None,
            }
            for r in clauses
        ],
        prohibition_and_related_transaction_original=prohibition,
        restricted_current_endpoint=restricted,
        unobserved_facts_do_not_become_no_breach=True,
        review_candidate_point_not_used=True,
    )
    return profile


def calculate(inputs, *, root):
    """重核原始字节及完整条款，按既有60/40公开执行锚点计算；不输出完整总分。"""
    root = Path(root).resolve()
    _reject(
        inputs.get("version") == VERSION
        and inputs.get("adapter") == SECURITIES_ADAPTER
        and inputs.get("fund_code") == "004605"
        and inputs.get("share_class") == "C"
        and inputs.get("as_of_date") == "2026-09-30"
        and inputs.get("holding_date") == "2026-06-30"
        and inputs.get("primary_ref") == PRIMARY
        and inputs.get("selected_dimensions") == ["execution"],
        "INPUT_SCOPE",
    )
    _reject(
        not any(k in inputs for k in ["score", "candidate", "state", "grade", "dimensions", "precomputed_scores"]),
        "EXTERNAL_SCORE_OR_STATE",
    )
    derived = {h for h, r in inputs["sources"].items() if r.get("archive_kind") == "DERIVED"}
    _reject(derived == {REVIEW_SHA, inputs["material_event_review_ref"], CUSTODY_REVIEW_SHA}, "DERIVED_REGISTRY_SCOPE")
    _read_derived(inputs, root, inputs["material_event_review_ref"])
    _reject(
        inputs.get("contract_scope_review_ref") == REVIEW_SHA
        and inputs["securities_contract"]["source_ref"] == CONTRACT_SHA,
        "FULL_CONTRACT_REVIEW_REQUIRED",
    )
    _read_derived(inputs, root, REVIEW_SHA)
    _reject(
        inputs.get("custody_agreement", {}).get("source_ref") == CUSTODY_SHA
        and inputs["custody_agreement"].get("supplement_review_ref") == CUSTODY_REVIEW_SHA,
        "CUSTODY_BINDING_REQUIRED",
    )
    _read_derived(inputs, root, CUSTODY_REVIEW_SHA)
    source = Originals({**inputs, "sources": {h: r for h, r in inputs["sources"].items() if h not in derived}}, root)
    from app.services.fund_rating_active_performance import _require_material_review

    material = _require_material_review(inputs, root)
    _share_binding(source, inputs)
    catalog = _catalog(source, inputs)
    with localcontext() as context:
        context.prec = 60
        rows = [
            _flex_report(source, inputs, h, catalog[role][period], period, role == "full_reports")
            for role in ["quarters", "full_reports"]
            for period, h in sorted(inputs[role].items())
        ]
        quarters = {r["period"]: r for r in rows if r["kind"] == "QUARTER"}
        for row in rows:
            if row["kind"] == "FULL":
                q = quarters[row["period"]]
                _reject(
                    all(row[k] == q[k] for k in ["stock_amount_cny", "total_assets_cny", "nav_cny"]),
                    "FULL_QUARTER_MATCH",
                )
        _reject(
            all("本基金本报告期末未投资股指期货" in _plain(source.body(r["source_ref"])) for r in quarters.values()),
            "ACTUAL_FUTURES_END_SCOPE",
        )
        profile = _full_contract_scope(source, inputs, rows)
        profile["custody_supplement"] = _custody_scope(source, inputs, rows)
        _reject(profile["full_contract_review_ref"] == REVIEW_SHA, "FULL_CONTRACT_NOT_COMPILED")
        observations = {
            key: {
                "kind": "state",
                "state": "NORMAL_PUBLIC_EXECUTION",
                "source_refs": sorted(inputs["sources"]),
                "reason": "已核完整合同适用范围及实际公开截面；每日交易及全管理人限制保留未知，不据托管声明计算偏离0。",
            }
            for key in ["allocation", "mandate"]
        }
        packet = {
            "fund_code": inputs["fund_code"],
            "basis": ACTIVE_BASIS,
            "as_of_date": inputs["as_of_date"],
            "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
            "quality_metrics": {
                "target_product_aliases": inputs["share_binding"]["column_codes"],
                "material_event_pending": material["pending"],
                "observations": observations,
            },
            "r4_evidence_registry": {
                h: {**r, "path": str((root / r["file"]).resolve())} for h, r in inputs["sources"].items()
            },
            "quality_public": {"dataDates": {"holdings": inputs["holding_date"]}},
            "raw_archive": [{"sha256": h, "kind": r["archive_kind"]} for h, r in inputs["sources"].items()],
        }
        result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"efficiency"})
    _reject(result["score"] is None and result["grade"] is None, "COMPONENT_ONLY")
    return {
        "version": VERSION,
        "dimensions": {"execution": result["dimensions"]["efficiency"]},
        "score": result["dimensions"]["efficiency"],
        "report_ledger": rows,
        "contract_profile": profile,
        "material_event_review": material,
        "catalog_coverage": catalog,
        "subscores": result["subscores"],
        "complete_score": None,
        "grade": None,
    }


def validate_input_archive(bundle, *, root, archive_reader):
    """回放前核封存输入档案与内嵌原件清单一致，防止只改包内输入。"""
    item = bundle[FIELD]
    _reject(archive_reader(item["input_archive"], root) == item["inputs"], "INPUT_ARCHIVE_MISMATCH")
    check_binding(bundle, item["inputs"])


def replay_snapshot(bundle, dimensions, *, root):
    """独立重算执行并比对三界；不调用已完成的185参照流动性重放。"""
    from app.services.fund_rating_reviewed_snapshot import _read

    validate_input_archive(bundle, root=root, archive_reader=_read)
    result = calculate(bundle[FIELD]["inputs"], root=root)
    _reject(
        all(
            dimensions["execution"][bound] is not None
            and abs(D(dimensions["execution"][bound]) - D(result["score"])) <= D("1e-55")
            for bound in ["lower", "upper", "candidate"]
        ),
        "SNAPSHOT_RECALCULATION",
    )
    return result


def compile_bundle(bundle, request, *, root):
    """只更新执行研究记录和专属输入；拒绝已有执行组件或错误份额。"""
    from app.services.fund_rating_pipeline import archive_json
    from app.services.fund_rating_pipeline_prepare import canonical_source

    bundle = deepcopy(bundle)
    _reject(FIELD not in bundle, "COMPONENT_ALREADY_PRESENT")
    raw = deepcopy(request["inputs"])
    check_binding(bundle, raw)
    _reject(request["fund_code"] == raw["fund_code"] and request["as_of_date"] == raw["as_of_date"], "REQUEST_IDENTITY")
    raw["sources"] = {h: canonical_source(root, r) for h, r in raw["sources"].items()}
    result = calculate(raw, root=root)
    reason = request["explanation"].strip()
    _reject(len(reason) >= 40, "PUBLIC_EXPLANATION")
    bundle["reviewed_document"]["dimensions"]["execution"] = {
        "lower": result["score"],
        "upper": result["score"],
        "candidate": result["score"],
        "reason": reason,
    }
    bundle.setdefault("public_explanations", {})["execution"] = reason
    bundle.get("public_metrics", {}).pop("execution", None)
    bundle.update(research_format="single", research_archive=archive_json(root, bundle["reviewed_document"]))
    bundle[FIELD] = {"inputs": raw, "input_archive": archive_json(root, raw)}
    extras = {r["sha256"]: r for r in bundle.get("extra_archives", [])}
    extras.update({h: {k: r[k] for k in ["file", "sha256", "kind"]} for h, r in raw["sources"].items()})
    bundle["extra_archives"] = list(extras.values())
    return bundle, result


def prepare_dossier(session, request, *, root):
    """从真实当前包编译只读候选；本入口不注册、发布或重跑既有流动性。"""
    from app.services.fund_rating_pipeline import current_inputs
    from app.services.fund_rating_reviewed_snapshot import BASIS

    current = current_inputs(session, [request["fund_code"]]).get(request["fund_code"])
    _reject(
        current is not None
        and current["batch_id"] == request["prior_rating_ref"]
        and str(current["as_of_date"]) == request["as_of_date"],
        "CURRENT_REF_OR_PERIOD",
    )
    base = next(m["bundle"] for m in current["input_snapshot"]["members"] if m["fund_code"] == request["fund_code"])
    bundle, _ = compile_bundle(base, request, root=root)
    replay_snapshot(bundle, bundle["reviewed_document"]["dimensions"], root=root)
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in ["basis", "reviewed_document"]}]}


prepare = prepare_dossier

CUSTODY_SHA = "88d9cfcd1951c2c34e8019e7994a86288bde4c32b1bc39c9ad4b2ea38ebef600"
CUSTODY_REVIEW_SHA = "d8a197e167a2cb19fbfe2cc5e76a9697c7096fc652c3676e60ffc91d842d7111"


def _check_custody_bounds(total_upper, single_upper, nav):
    """上界超过比例只表示须读更细原件，不把保守上界判成已发生违规。"""
    _reject(all(v.is_finite() and v >= 0 for v in [total_upper, single_upper, nav]) and nav > 0, "CUSTODY_BOUND_AMOUNT")
    _reject(total_upper / nav * 100 <= 20, "CUSTODY_TOTAL20_BOUND_REQUIRES_REVIEW")
    _reject(single_upper / nav * 100 <= 10, "CUSTODY_SINGLE10_BOUND_REQUIRES_REVIEW")
    _reject(total_upper / nav * 100 <= 15, "CUSTODY_ACTIVE15_UPPER_REQUIRES_REVIEW")


def _restricted_period_bounds(source, inputs, rows):
    """以名单对应的全仓股票加全部债券覆盖总受限额；不是估计实际受限金额。"""
    results = []
    for row in rows:
        if row["kind"] != "FULL":
            continue
        text = source.body(row["source_ref"])
        head = re.search(r"[67]\.4\.12\.1\s*因认购", text)
        _reject(head is not None, "CUSTODY_RESTRICTED_NOTE_HEAD")
        stop = re.search(r"[67]\.4\.12\.2\s*期末", text[head.end() :])
        _reject(stop is not None, "CUSTODY_RESTRICTED_NOTE_END")
        end = head.end() + stop.start()
        section = text[head.start() : end]
        markers = list(re.finditer(r"(?m)^\s*(\d{6})\s+([^\W\d_]+)", section))
        codes = [m[1] for m in markers]
        if markers:
            categories = re.findall(r"受限证券类别[：:]\s*([^\n]+)", section)
            _reject(categories and {_plain(c) for c in categories} == {"股票"}, "RESTRICTED_CATEGORY_REVIEW")
            for i, marker in enumerate(markers):
                chunk = _plain(section[marker.start() : markers[i + 1].start() if i + 1 < len(markers) else None])
                _reject("新股认购" in chunk or ("新股限" in chunk and "售" in chunk), "RESTRICTED_ROW_TYPE")
        else:
            _reject(
                "本基金无因认购新发/增发证券而于期末持有的流通受限证券" in _plain(section),
                "RESTRICTED_ABSENCE_NOT_EXPLICIT",
            )
        tail_end = re.search(r"[67]\.4\.13\s*金融工具", text[end:])
        _reject(tail_end is not None, "RESTRICTED_OTHER_SCOPE_END")
        tail = text[end : end + tail_end.start()]
        quotes = [
            "本基金本报告期末未持有暂时停牌等流通受限股票",
            "本基金无因从事银行间市场债券正回购交易形成的卖出回购证券款余额",
            "本基金无因从事交易所市场债券正回购交易形成的卖出回购证券款余额",
            "本基金本报告期末未持有参与转融通证券出借业务的证券",
        ]
        _reject(all(q in _plain(tail) for q in quotes), "RESTRICTED_OTHER_EXPOSURE_REVIEW")
        positions = {r["security"]: D(r["amount_cny"]) for r in row["stocks"]}
        _reject(set(codes) <= positions.keys(), "RESTRICTED_SECURITY_NOT_IN_FULL_HOLDINGS")
        bonds = sum(D(r["amount_cny"]) for r in row["bond_categories"])
        stock_upper = sum(positions[c] for c in set(codes))
        all_upper = stock_upper + bonds
        single_upper = max(positions.values()) + bonds
        nav = D(row["nav_cny"])
        total_pct, single_pct = all_upper / nav * 100, single_upper / nav * 100
        _check_custody_bounds(all_upper, single_upper, nav)
        _reject(
            abs(single_pct - D(row["issuer_all_bonds_conservative_upper_pct"])) <= D("1e-55"),
            "CUSTODY_ISSUER_BOUND_MATCH",
        )
        results.append(
            {
                "period": row["period"],
                "source_ref": row["source_ref"],
                "nav_cny": str(nav),
                "restricted_note_original": section,
                "other_restrictions_original": tail,
                "restricted_row_codes_with_duplicates": codes,
                "restricted_security_count": len(set(codes)),
                "restricted_stock_whole_position_upper_cny": str(stock_upper),
                "all_bonds_conservative_addition_cny": str(bonds),
                "all_restricted_securities_upper_cny": str(all_upper),
                "all_restricted_securities_upper_pct": str(total_pct),
                "single_restricted_security_upper_cny": str(single_upper),
                "single_restricted_security_upper_pct": str(single_pct),
                "actual_restricted_total_cny": None,
                "daily_or_private_account_observation": None,
                "bound_method": "受限名单各证券全部股票持仓去重汇总加全部债券；单证券用最大股票持仓加全部债券。",
                "bound_not_actual_restricted_value": True,
            }
        )
    _reject(
        len(results) == 6 and {r["period"] for r in results} == set(inputs["full_reports"]), "CUSTODY_SIX_FULL_PERIODS"
    )
    return results


def _custody_scope(source, inputs, rows):
    """核原始托管协议和补充核查单，重算六期20/10及主动15上界。"""
    supplement = _read_derived(inputs, source.root, CUSTODY_REVIEW_SHA)
    article = source.json(CUSTODY_SHA)["data"]
    _reject(
        article["art_code"] == "AN202209261578669546"
        and article["notice_date"][:10] == source.sources[CUSTODY_SHA]["published_on"] == "2022-09-26"
        and _plain(article["notice_title"]) == _plain(inputs["legal_name"] + "托管协议"),
        "CUSTODY_ARTICLE_IDENTITY",
    )
    body = source.body(CUSTODY_SHA)
    _reject(
        all(
            _plain(w) in _plain(body[:2000])
            for w in [inputs["legal_name"], inputs["manager_name"], "中国建设银行股份有限公司", "二零二二年九月"]
        ),
        "CUSTODY_LEGAL_PARTIES_AND_MONTH",
    )
    clauses = {}
    for number in [18, 20]:
        found = list(re.finditer(r"(?m)^\s*" + str(number) + r"\.[\s\S]*?(?=^\s*" + str(number + 1) + r"\.)", body))
        _reject(len(found) == 1, "CUSTODY_CLAUSE_LOCATOR")
        clauses[str(number)] = found[0][0].strip()
    normalized = _plain(clauses["18"])
    _reject(
        "所有流通受限证券" in normalized
        and "净值的20%" in normalized
        and "同一流通受限证券" in normalized
        and "净值的10%" in normalized,
        "CUSTODY_20_10_ACTUAL_CLAUSES",
    )
    _reject(
        "主动投资于流动性受限资产" in _plain(clauses["20"]) and "净值的15%" in _plain(clauses["20"]),
        "CUSTODY_ACTIVE_15_ACTUAL_CLAUSE",
    )
    matched = []
    for digest in inputs["catalog_refs"]:
        for item in source.json(digest)["Data"]:
            day = item["PUBLISHDATEDesc"][:10]
            if "托管协议" not in item["TITLE"] or day > inputs["as_of_date"]:
                continue
            _reject(day <= "2022-09-26", "NEWER_CUSTODY_VERSION_REQUIRES_REVIEW")
            if item["ID"] == article["art_code"]:
                _reject(
                    item["FUNDCODE"] == inputs["fund_code"]
                    and day == "2022-09-26"
                    and _plain(item["TITLE"]) == _plain(article["notice_title"]),
                    "CUSTODY_CATALOG_BINDING",
                )
                matched.append({"source_ref": digest, "article_id": item["ID"], "published_on": day})
    _reject(len(matched) == 1, "CUSTODY_CATALOG_NOT_UNIQUE")
    bounds = _restricted_period_bounds(source, inputs, rows)
    _reject(
        supplement["fund_code"] == inputs["fund_code"]
        and supplement["as_of"] == inputs["as_of_date"]
        and supplement["original_23_review_ref"] == REVIEW_SHA
        and supplement["custody_original_ref"] == CUSTODY_SHA
        and supplement["actual_custody_clauses"] == clauses
        and supplement["six_full_period_conservative_bounds"] == bounds,
        "CUSTODY_SUPPLEMENT_ORIGINAL_BINDING",
    )
    return {
        "source_ref": CUSTODY_SHA,
        "supplement_review_ref": CUSTODY_REVIEW_SHA,
        "notice_published_on": "2022-09-26",
        "catalog_binding": matched,
        "actual_custody_clauses": clauses,
        "six_full_period_conservative_bounds": bounds,
        "unobserved_daily_or_manager_wide_result": None,
        "total_20_single_10_and_active_15_endpoint_bounds_replayed": True,
    }
