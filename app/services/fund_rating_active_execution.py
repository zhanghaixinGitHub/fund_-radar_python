"""主动基金R5目标执行的原件复算；只读，不采集、不登记、不发布。

适配分派按已核合同结构选择解析器。当前仅支持普通境内主动灵活配置、
股票区间及非股票区间、5%现金下限的公开截面；其他结构显式拒收，不能传入
裸state或分数开通。正常档仍为75，配置60%/范围40%，不增加日账簿门槛。
"""

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

VERSION = "R5_ACTIVE_EXECUTION_ORIGINALS_20261029_1"
ADAPTER = "CN_ACTIVE_FLEX_STOCK_BAND_CASH5_PUBLIC_SNAPSHOTS_V1"
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


def _contract_profile(source: Originals, inputs: dict, item: dict) -> dict:
    """保守结构分派：范围阈值须从真实合同句子取得，不能预填数值改变契约。"""
    _reject(item.get("adapter") == ADAPTER, "UNSUPPORTED_ADAPTER")
    _reject(
        source.sources[item["source_hash"]].get("published_on") is not None
        and source.sources[item["source_hash"]]["published_on"] <= item["effective_from"],
        "CONTRACT_EFFECTIVE_DATE_SOURCE",
    )
    text = _plain(source.body(item["source_hash"]))
    _reject(_plain(inputs["legal_name"]) in text and _plain(inputs["manager_name"]) in text, "CONTRACT_IDENTITY")
    _reject("基金合同" in text and "灵活配置混合" in _plain(inputs["legal_name"]), "CONTRACT_PRODUCT_TYPE")
    # 本分派器的单一主体约束仅支持原文明确的上市公司股票10%条款。
    # “一家公司发行的证券”须合并债券等敞口，不能复用股票检查冒充已核。
    _reject(
        re.search(r"本基金持有一家上市公司的股票[，,]其市值不超过基金资产净值的10%", text) is not None,
        "SINGLE_ISSUER_SCOPE_REQUIRES_ADAPTER",
    )
    clause = _plain(item["allocation_clause"])
    _reject(clause in text and len(clause) >= 60, "CONTRACT_QUOTE")
    stock = re.search(r"股票(?:\(含存托凭证\))?资产占基金资产的(\d+(?:\.\d+)?)%[-—至](\d+(?:\.\d+)?)%", clause)
    nonstock = re.search(r"(?:其他证券品种|其他金融工具)占基金资产的(\d+(?:\.\d+)?)%[-—至](\d+(?:\.\d+)?)%", clause)
    cash = re.search(r"不低于基金资产净值的(\d+(?:\.\d+)?)%", clause)
    _reject(stock is not None and nonstock is not None and cash is not None, "CONTRACT_STRUCTURE_NOT_SUPPORTED")
    lower, upper = map(D, stock.groups())
    other_lower, other_upper = map(D, nonstock.groups())
    _reject(0 <= lower <= upper <= 95 and 0 <= other_lower <= other_upper <= 100 and D(cash[1]) == 5, "CONTRACT_BOUNDS")
    _reject(
        "债券" in clause and "现金" in clause and "结算备付金" in clause and "应收申购款" in clause,
        "QUALIFIED_CASH_CLAUSE",
    )
    # 当前分派器只核广谱主动配置；主题最低配比、港股强制配比等另加已核适配，不能跳过。
    investment = text[text.find("投资范围") :]
    _reject(
        not re.search(
            r"投资于[^。;]{0,100}(?:主题|行业|产业|区域)[^。;]{0,60}(?:不低于|不得低于|至少)\d+%", investment
        ),
        "BINDING_OPPORTUNITY_SET_REQUIRES_ADAPTER",
    )
    return {
        "source_hash": item["source_hash"],
        "effective_from": item["effective_from"],
        "effective_to": item.get("effective_to"),
        "stock_min": lower,
        "stock_max": upper,
        "nonstock_min": other_lower,
        "nonstock_max": other_upper,
        "cash_min_nav": D(5),
        "actual_clause": item["allocation_clause"],
    }


def _assets(text: str) -> tuple[dict, Decimal]:
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
        match = re.search(r"(?:^|\n)\s*" + str(rank) + r"\s+" + label + r"\s+(" + MONEY + r"|-)\s+([\d.]+|-)", table)
        _reject(match is not None, "ASSET_ROW:" + label)
        rows[label] = {
            "amount_cny": "0" if match[1] == "-" else str(_amount(match[1])),
            "explicit_dash_zero": match[1] == "-",
            "source_row": match[0].strip(),
        }
    total = re.search(r"(?:^|\n)\s*9\s+合计\s+(" + MONEY + r")\s+100\.00", table)
    _reject(total is not None, "ASSET_TOTAL")
    amount = _amount(total[1])
    _reject(sum(D(row["amount_cny"]) for row in rows.values()) == amount > 0, "ASSET_RECONCILIATION")
    return rows, amount


def _stocks(text: str, full: bool) -> list[dict]:
    heading = "所有股票投资明细" if full else "前十名股票投资明细"
    table = _section(
        text,
        r"\d\.3(?:\.1)?\s*(?:报告)?期末按公允价值占基金资产净值比例大小排序的" + heading,
        r"\d\.4\s*(?:报告期内股票|报告期末按债券)",
    )
    pattern = r"(?:^|\n)\s*(\d+)\s+(\d{6})\s+(.+?)\s+([\d,]+)\s+(" + MONEY + r")\s+([\d.]+)"
    rows = [
        {"rank": int(m[1]), "security": m[2], "name": m[3].strip(), "amount_cny": str(_amount(m[5]))}
        for m in re.finditer(pattern, table)
    ]
    _reject(bool(rows) and [r["rank"] for r in rows] == list(range(1, len(rows) + 1)), "STOCK_RANK_CLOSURE")
    _reject(len(set(r["security"] for r in rows)) == len(rows), "DUPLICATE_STOCK")
    return rows


def _report(source: Originals, inputs: dict, digest: str, entry: dict, period: str, full: bool, profile: dict) -> dict:
    from app.services.fund_quality_v2_disclosures import report_period

    text = source.body(digest)
    plain = _plain(text)
    _reject(inputs["fund_code"] in plain[:18000] and _plain(inputs["legal_name"]) in plain[:18000], "REPORT_IDENTITY")
    _reject(str(report_period(text)[1]) == period, "REPORT_PERIOD")
    ref = source.sources[digest]
    _reject(ref.get("published_on") == entry["PUBLISHDATEDesc"][:10], "REPORT_PUBLICATION_DATE")
    if ref["kind"] == "json":
        article = source.json(digest)["data"]
        _reject(
            article.get("art_code") == entry["ID"]
            and _plain(article.get("notice_title", "")) == _plain(entry["TITLE"]),
            "NOTICE_CATALOG_BINDING",
        )
    assets, total = _assets(text)
    stock = D(assets["权益投资"]["amount_cny"])
    stocks = _stocks(text, full)
    # 已公开的季度资产类别也须纳入范围核验；非零的新工具进入专用适配，
    # 不能只看半年末为零就忽略季度中已披露的其他投资类别。
    _reject(
        all(
            assets[name]["explicit_dash_zero"]
            for name in (
                "基金投资",
                "贵金属投资",
                "金融衍生品投资",
                "买入返售金融资产",
            )
        ),
        "QUARTER_OTHER_ASSET_CLASSES_REQUIRES_ADAPTER",
    )
    compliance = _section(text, r"4\.2\s*管理人对报告期内本基金运作遵规守信情况的说明", r"4\.3\s*")
    compliance_plain = _plain(compliance)
    _reject(
        "基金投资管理符合有关法规和基金合同的规定" in compliance_plain
        and "无违法违规、未履行基金合同承诺或损害基金份额持有人利益的行为" in compliance_plain,
        "ACTUAL_COMPLIANCE_STATEMENT",
    )
    _reject(
        not any(word in compliance_plain for word in ["但是", "除外", "尚未", "未完成"]),
        "COMPLIANCE_CONTRADICTION_REVIEW_REQUIRED",
    )
    row = {
        "period": period,
        "source_hash": digest,
        "contract_source_hash": profile["source_hash"],
        "kind": "FULL" if full else "QUARTER",
        "stock_amount_cny": str(stock),
        "total_assets_cny": str(total),
        "stocks": stocks,
        "compliance_quote": compliance,
        "quarter_cash_principal": None,
        "daily_compliance_certified": False,
    }
    if full:
        balance = _section(text, r"[67]\.1\s*资产负债表", r"[67]\.2\s*利润表")
        nav_match = re.search(r"净资产合计\s+(" + MONEY + r")", balance)
        _reject(nav_match is not None, "NAV_ROW")
        nav = _amount(nav_match[1])
        _reject(sum(D(s["amount_cny"]) for s in stocks) == stock, "FULL_STOCK_RECONCILIATION")
        note = _section(text, r"[67]\.4\.7\.1\s*(?:货币资金|银行存款)", r"[67]\.4\.7\.2\s*交易性金融资产")
        principal = re.search(r"等于[：:]本金\s+(" + MONEY + r")", note)
        _reject(principal is not None, "QUALIFIED_CASH_PRINCIPAL")
        cash = _amount(principal[1])
        trustee = _section(text, r"5\.2\s*托管人对报告期内本基金投资运作", r"5\.3\s*托管人")
        _reject("未发现基金管理人有损害基金份额持有人利益的行为" in _plain(trustee), "TRUSTEE_ACTUAL_REVIEW")
        _reject(re.search(r"投资策略未发生(?:重大)?改变", plain) is not None, "STRATEGY_CONTINUITY")
        _reject("报告期末按行业分类的境内股票投资组合" in plain and "未持有港股通投资股票" in plain, "DOMESTIC_SCOPE")
        _reject(
            all(re.fullmatch(r"(?:00|30|60|68)\d{4}", s["security"]) for s in stocks), "SECURITY_SCOPE_REQUIRES_ADAPTER"
        )
        _reject(
            all(
                assets[name]["explicit_dash_zero"]
                for name in ["基金投资", "贵金属投资", "金融衍生品投资", "买入返售金融资产"]
            ),
            "OTHER_ASSET_CLASSES_REQUIRES_ADAPTER",
        )
        for phrase in [
            "未持有资产支持证券",
            "未持有贵金属",
            "未持有权证",
            "股指期货不属于本基金的投资范围",
            "国债期货不属于本基金的投资范围",
        ]:
            _reject(phrase in plain, "INSTRUMENT_SCOPE:" + phrase)
        bond_table = _section(text, r"[78]\.5\s*期末按债券品种分类的债券投资组合", r"[78]\.6\s*期末按公允价值")
        bond_rows = [
            {"rank": int(m[1]), "category": m[2].strip(), "amount_cny": "0" if m[3] == "-" else str(_amount(m[3]))}
            for m in re.finditer(r"(?:^|\n)\s*([1-9])\s+([^\n]+?)\s+(" + MONEY + r"|-)\s+([\d.]+|-)", bond_table)
        ]
        _reject([r["rank"] for r in bond_rows] == list(range(1, 10)), "BOND_CATEGORIES_INCOMPLETE")
        _reject(sum(D(r["amount_cny"]) for r in bond_rows) == D(assets["固定收益投资"]["amount_cny"]), "BOND_TOTAL")
        _reject(
            all(
                r["category"]
                in {
                    "国家债券",
                    "央行票据",
                    "金融债券",
                    "企业债券",
                    "企业短期融资券",
                    "中期票据",
                    "可转债（可交换债）",
                    "同业存单",
                    "其他",
                }
                for r in bond_rows
            )
            and next(r for r in bond_rows if r["category"] == "其他")["amount_cny"] == "0",
            "BOND_SCOPE_REQUIRES_ADAPTER",
        )
        eligible_nonstock = cash + D(assets["固定收益投资"]["amount_cny"])
        _reject(
            eligible_nonstock / total * 100 >= profile["nonstock_min"]
            and (total - stock) / total * 100 <= profile["nonstock_max"],
            "NONSTOCK_CONSTRAINT_REVIEW_REQUIRED",
        )
        _reject(cash / nav * 100 >= profile["cash_min_nav"], "CASH_CONSTRAINT_REVIEW_REQUIRED")
        row.update(
            qualified_cash_principal_cny=str(cash),
            cash_note=note,
            trustee_quote=trustee,
            eligible_nonstock_lower_bound_pct=str(eligible_nonstock / total * 100),
            bond_categories=bond_rows,
            outside_scope_confirmed_cny="0",
            internal_stock_pool_certified=False,
        )
    else:
        nav_match = re.search(r"4\.期末基金资产净值\s+((?:" + MONEY + r"\s+){1,8}" + MONEY + r")", text)
        _reject(nav_match is not None, "QUARTER_NAV_COLUMNS")
        values = re.findall(MONEY, nav_match[1])
        _reject(len(values) == len(inputs["share_binding"]["column_codes"]), "QUARTER_SHARE_COUNT")
        nav = sum(_amount(v) for v in values)
        _reject(len(stocks) == 10, "QUARTER_TOP10")
    _reject(
        nav > 0 and profile["stock_min"] <= stock / total * 100 <= profile["stock_max"],
        "STOCK_CONSTRAINT_REVIEW_REQUIRED",
    )
    # “基金资产”未在本适配中额外指定总资产/净资产口径；只有两种相容
    # 财务口径都落在原合同区间内才确认正常，避免自行选有利分母。
    _reject(
        nav <= total and profile["stock_min"] <= stock / nav * 100 <= profile["stock_max"],
        "STOCK_DENOMINATOR_AMBIGUITY_REVIEW_REQUIRED",
    )
    if full:
        _reject(
            (total - stock) / nav * 100 <= profile["nonstock_max"], "NONSTOCK_DENOMINATOR_AMBIGUITY_REVIEW_REQUIRED"
        )
    _reject(max(D(s["amount_cny"]) / nav * 100 for s in stocks) <= 10, "ISSUER_LIMIT_REVIEW_REQUIRED")
    row.update(
        nav_cny=str(nav),
        stock_total_assets_pct=str(stock / total * 100),
        stock_nav_pct=str(stock / nav * 100),
        single_stock_max_nav_pct=str(max(D(s["amount_cny"]) / nav * 100 for s in stocks)),
    )
    return row


def calculate(inputs: dict, *, root: Path) -> dict:
    """显式原件进入，按原60/40公式返回唯一正常执行点值；不返回总分或等级。"""
    if inputs.get("version") == "R5_BD500_STOCK80_CASH5_FULL_CONSTRAINT_PUBLIC_EXECUTION_20261030_2":
        # 独立核历史完整约束及期货名义持仓，不以旧版三条约束候选代替完整适配。
        from app.services.fund_rating_bd500_execution import calculate as calculate_bd500_execution

        return calculate_bd500_execution(inputs, root=root)
    if inputs.get("version") == "R5_CMF_017960_EXECUTION_PUBLIC_ORIGINALS_20261029_2":
        # 两版60%-95%合同及来源绑定事项使用专用编译器；旧V1或其他基金
        # 继续原分派，不能以该入口绕过004237已知公开发行人限额冲突。
        from app.services.fund_rating_cmf_execution import calculate as calculate_cmf_execution

        return calculate_cmf_execution(inputs, root=root)
    _reject(inputs.get("version") == VERSION, "VERSION")
    # 004237原2023年末股票与同发行人转债合计已形成公开限额冲突。
    # 旧V1只核股票限额，不能用一般履职声明覆盖该事实；实际豁免/调整
    # 仍未核明。没有任何已发布V1批次，此处阻止误用旧候选返回正常75。
    _reject(inputs.get("fund_code") != "004237", "KNOWN_PUBLIC_ISSUER_CONFLICT_REQUIRES_NEW_ADAPTER")
    _reject(
        not any(k in inputs for k in ["score", "state", "allocation_state", "mandate_state", "dimensions"]),
        "EXTERNAL_SCORE_OR_STATE",
    )
    _reject(re.fullmatch(r"\d{6}", inputs["fund_code"]) is not None, "FUND_CODE")
    _reject(inputs["share_class"] in {"A", "B", "C", "D", "E", "I", "Y"}, "SHARE_CLASS")
    source = Originals(inputs, root)
    _share_binding(source, inputs)
    primary = _plain(source.body(inputs["primary_ref"]))
    _reject(_plain(inputs["legal_name"]) in primary and _plain(inputs["manager_name"]) in primary, "PRIMARY_IDENTITY")
    catalog = _catalog(source, inputs)
    profiles = [_contract_profile(source, inputs, item) for item in inputs["contracts"]]
    _reject(bool(profiles), "CONTRACT_MISSING")
    rows = []
    for role in ["quarters", "full_reports"]:
        for period, digest in sorted(inputs[role].items()):
            applicable = [
                p
                for p in profiles
                if p["effective_from"] <= period and (p["effective_to"] is None or period <= p["effective_to"])
            ]
            _reject(len(applicable) == 1, "CONTRACT_PERIOD_COVERAGE")
            rows.append(
                _report(source, inputs, digest, catalog[role][period], period, role == "full_reports", applicable[0])
            )
    latest = max(inputs["full_reports"])
    _reject(latest == inputs["holding_date"], "HOLDING_DATE")
    from app.services.fund_quality_v2_disclosures import report_period

    _reject(str(report_period(source.body(inputs["primary_ref"]))[1]) == latest, "PRIMARY_PERIOD")
    quarter = {r["period"]: r for r in rows if r["kind"] == "QUARTER"}
    for row in [r for r in rows if r["kind"] == "FULL"]:
        same = quarter.get(row["period"])
        _reject(
            same is not None and all(row[k] == same[k] for k in ["nav_cny", "stock_amount_cny", "total_assets_cny"]),
            "SAME_DATE_REPORT_CONFLICT",
        )
    refs = sorted(source.sources)
    observations = {
        "allocation": {
            "kind": "state",
            "state": "NORMAL_PUBLIC_EXECUTION",
            "reason": "实际公开季度、完整报告、合格现金及适用合同逐项核对，未见已证非豁免配置偏离。",
            "source_refs": refs,
        },
        "mandate": {
            "kind": "state",
            "state": "NORMAL_PUBLIC_EXECUTION",
            "reason": "全部已公开完整股票和债券类别均与适用范围配对，并有逐期管理人及托管人实际履职正文支持。",
            "source_refs": refs,
        },
    }
    packet = {
        "fund_code": inputs["fund_code"],
        "basis": ACTIVE_BASIS,
        "as_of_date": inputs["as_of_date"],
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "target_product_aliases": inputs["share_binding"]["column_codes"],
            "material_event_pending": False,
            "observations": observations,
        },
        "r4_evidence_registry": {
            h: {**source.sources[h], "path": str((root / source.sources[h]["file"]).resolve())} for h in refs
        },
        "quality_public": {"dataDates": {"holdings": latest}},
        "raw_archive": [{"sha256": h, "kind": "ORIGINAL"} for h in refs],
    }
    with localcontext() as context:
        context.prec = 70
        result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"efficiency"})
    _reject(result["score"] is None and result["grade"] is None, "COMPONENT_ONLY")
    return {
        "version": VERSION,
        "score": result["dimensions"]["efficiency"],
        "dimensions": result["dimensions"],
        "subscores": result["subscores"],
        "packet": packet,
        "anchors": {},
        "report_ledger": rows,
        "catalog_coverage": catalog,
        "contract_profiles": profiles,
        "complete_score": None,
        "grade": None,
        "limits": [
            "仅评价真实公开披露频率和范围，不认证逐日交易、全部管理组合限制或内部备选库。",
            "季度没有单列的现金本金保持未知，不借邻期本金或结算备付金填零。",
            "发生实际偏离、豁免或无法识别的新合同结构时拒绝正常档，转入相应原件复核。",
        ],
    }


def prepare(session, request: dict, *, root: Path) -> dict:
    """标准prepare入口：仅编译同一期部分包，保留其他维度、日期和历史。"""
    from app.services.fund_rating_pipeline import archive_json, current_inputs
    from app.services.fund_rating_reviewed_snapshot import BASIS

    code = request["fund_code"]
    previous = current_inputs(session, [code]).get(code)
    _reject(previous is not None and str(previous["as_of_date"]) == request["as_of_date"], "CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    _reject(
        bundle.get("basis") == BASIS
        and bundle.get("family") == "ACTIVE_EQUITY"
        and not any(bundle.get(k) for k in ["r5_calculation", "passive_components_calculation"]),
        "PARTIAL_ACTIVE_BUNDLE_REQUIRED",
    )
    raw = deepcopy(request["inputs"])
    _reject(
        raw["fund_code"] == code
        and raw["as_of_date"] == request["as_of_date"]
        and raw["share_class"] == bundle["share_class"]
        and raw["legal_name"] == bundle["identity"]["legal_name"]
        and raw["manager_name"] == bundle["identity"]["manager_name"]
        and raw["holding_date"] == bundle["holding_date"]
        and raw["primary_ref"] == bundle["original_report_archive"]["sha256"],
        "PREPARE_IDENTITY_CONFLICT",
    )
    explanation = request.get("explanation", "").strip()
    _reject(len(explanation) >= 40, "PUBLIC_EXPLANATION_REQUIRED")
    from app.services.fund_rating_pipeline_prepare import canonical_source

    raw["sources"] = {digest: canonical_source(root, ref) for digest, ref in raw["sources"].items()}
    calculated = calculate(raw, root=root)
    document = bundle["reviewed_document"]
    document["dimensions"]["execution"] = {
        "lower": calculated["score"],
        "upper": calculated["score"],
        "candidate": calculated["score"],
        "reason": explanation,
    }
    bundle.update(
        research_format="single",
        research_archive=archive_json(root, document),
        active_execution_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    known_archives = {ref["sha256"]: ref for ref in bundle.get("extra_archives", [])}
    known_archives.update(
        {digest: {key: ref[key] for key in ("file", "sha256", "kind")} for digest, ref in raw["sources"].items()}
    )
    bundle["extra_archives"] = list(known_archives.values())
    bundle.setdefault("public_explanations", {})["execution"] = explanation
    bundle.get("public_metrics", {}).pop("execution", None)
    from datetime import UTC, datetime

    from app.services.fund_rating_reviewed_snapshot import calculate as calculate_snapshot
    from app.services.fund_rating_reviewed_snapshot import validate_sources

    validate_sources(bundle, datetime.now(UTC), root=root)
    calculate_snapshot(bundle, root=root)
    return {"basis": BASIS, "entries": [{k: v for k, v in bundle.items() if k not in {"basis", "reviewed_document"}}]}
