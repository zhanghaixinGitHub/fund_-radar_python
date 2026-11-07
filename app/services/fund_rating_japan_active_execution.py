"""019449日本主动股票执行的原件适配候选；不采集、不登记、不发布。

按真实公告目录闭合12季和6份全报，重读三版相同投资限制；同一日本股票
固有名代码的全部类别合并。其他证券类别非零时拒收，不能只检查股票10%。
只评价公开截面，未公开日交易、管理人全组合及境外借贷细项明确留未知。
"""

from __future__ import annotations

import inspect
import re
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services.fund_rating_active_execution import MONEY, Originals, _catalog, _plain, _section
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import content_digest

VERSION = "R5_JAPAN_ACTIVE_EXECUTION_ORIGINALS_20261030_1"


def require(condition, code):
    if not condition:
        raise ValueError("R5_JAPAN_EXECUTION_" + code)


def amount(value):
    number = D(value.replace(",", ""))
    require(number.is_finite() and number >= 0, "INVALID_AMOUNT")
    return number


def first_amount(text, pattern):
    match = re.search(pattern, text)
    require(match is not None, "AMOUNT_ROW:" + pattern)
    return amount(match[1])


def zero_or_amount(text, label):
    match = re.search(r"(?:^|\n)\s*" + label + r"\s+(" + MONEY + r"|-)\s+", text)
    require(match is not None, "BALANCE_ROW:" + label)
    return D(0) if match[1] == "-" else amount(match[1])


def share_binding(source, inputs):
    """按原表两列的名称与代码顺序绑定；标签在PDF中折行时不误拼进业务值。"""
    names = inputs["share_binding"]["column_names"]
    codes = inputs["share_binding"]["column_codes"]
    require(len(names) == len(codes) == 2 and len(set(codes)) == 2, "SHARE_COLUMNS")
    primary = _plain(source.body(inputs["primary_ref"]))
    table = _section(primary, r"2\.1基金基本情况", r"2\.2基金产品说明")
    pattern = (
        r"下属分级基金的基(?:金简称)?"
        + re.escape("".join(map(_plain, names)))
        + r"(?:金简称)?下属分级基金的交(?:易代码)?"
        + re.escape("".join(codes))
        + r"(?:易代码)?"
    )
    match = re.search(pattern, table)
    require(match is not None, "SHARE_NAMES_CODES_NOT_IN_ORIGINAL_COLUMNS")
    matched = [i for i, name in enumerate(names) if _plain(name).endswith(inputs["share_class"])]
    require(len(matched) == 1 and codes[matched[0]] == inputs["fund_code"], "WRONG_SHARE")
    return {"original_column_binding": match[0], "target_column": matched[0], "code": inputs["fund_code"]}


def assets(text):
    table = _section(text, r"\d\.1\s*(?:报告)?期末基金资产组合情况", r"\d\.2\s*(?:报告)?期末")
    labels = [
        "权益投资",
        "基金投资",
        "固定收益投资",
        "金融衍生品投资",
        "买入返售金融资产",
        "货币市场工具",
        "银行存款和结算备付金合计",
        "其他各项资产",
    ]
    rows = {}
    for rank, label in enumerate(labels, 1):
        printed_label = "其他(?:各项)?资产" if label == "其他各项资产" else label
        match = re.search(
            r"(?:^|\n)\s*" + str(rank) + r"\s+" + printed_label + r"\s+(" + MONEY + r"|-)\s+([\d.]+|-)", table
        )
        require(match is not None, "ASSET_ROW:" + label)
        rows[label] = {
            "amount_cny": "0" if match[1] == "-" else str(amount(match[1])),
            "explicit_dash_zero": match[1] == "-",
            "source_row": match[0].strip(),
        }
    total = first_amount(table, r"(?:^|\n)\s*9\s+合计\s+(" + MONEY + r")\s+100\.00")
    require(sum(D(row["amount_cny"]) for row in rows.values()) == total > 0, "ASSET_TOTAL")
    common = first_amount(_plain(table), r"普通股(" + MONEY + r")")
    require(common == D(rows["权益投资"]["amount_cny"]), "NON_COMMON_EQUITY")
    # 普通股金额与权益资产总额精确相等，已闭合其他非负权益类别总额为零；
    # 旧报告对子类别名称的换行/简称差异不再额外成为正常档门槛。
    for label in labels[1:6]:
        require(rows[label]["explicit_dash_zero"], "NONZERO_OTHER_SECURITY_REQUIRES_ADAPTER:" + label)
    return rows, total


def stocks(text, full):
    table = _section(
        text, r"\d\.4(?:\.1)?\s*(?:报告)?期末按公允价值占基金资产净值比例大小排序的", r"\d\.5\s*(?:报告期内|报告期末)"
    )
    # 列内折行可能把“日本交”放在证券代码上一行；金额后缀和国家列定位，不能依赖固定行数。
    pattern = r"(?<![A-Z0-9])([1-9][0-9A-Z]{3,4})\s+([^0-9]{0,80}?)日本\s+([\d,]+)\s+(" + MONEY + r")\s+([\d.]+)"
    rows = [
        {
            "security": m[1],
            "issuer_key": "JPX_STOCK_ENTITY:" + m[1][:4],
            "quantity_shares": str(amount(m[3] + ".00")),
            "amount_cny": str(amount(m[4])),
            "country": "日本",
            "source_excerpt": m[0],
        }
        for m in re.finditer(pattern, table)
    ]
    tails = re.findall(r"日本\s+[\d,]+\s+" + MONEY + r"\s+[\d.]+", table)
    if not full and not rows and not tails:
        # 2024Q1窄列PDF将代码、金额拆成多行，先按原字符拼接；千分组约束
        # 分隔数量和金额，禁止用股数×价格补账面价值。第9行跨页需明确重接原片段。
        compact = _plain(table)
        quantity = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
        simple = (
            r"([1-9][0-9A-Z]{3})(?:日本证券交易所?|日本交易所)日本(" + quantity + r")(" + MONEY + r")([\d]+\.\d{2})"
        )
        for m in re.finditer(simple, compact):
            rows.append(
                {
                    "security": m[1],
                    "issuer_key": "JPX_STOCK_ENTITY:" + m[1],
                    "quantity_shares": str(amount(m[2] + ".00")),
                    "amount_cny": str(amount(m[3])),
                    "country": "日本",
                    "source_excerpt": m[0],
                    "format": "WRAPPED_COLUMNS",
                }
            )
        bridge = (
            r"9HOYACORPHoya株式(\d{2})日日(" + quantity + r")(\d{1,3}(?:,\d{3})+)(\d+\.\d{2})"
            r"摩根日本精选股票型证券投资基金\(QDII\)2024年第1季度报告13会社(\d{2})本证券交易所本\.(\d{2})"
        )
        m = re.search(bridge, compact)
        require(m is not None, "CROSS_PAGE_STOCK_FRAGMENT")
        code = m[1] + m[5]
        rows.append(
            {
                "security": code,
                "issuer_key": "JPX_STOCK_ENTITY:" + code,
                "quantity_shares": str(amount(m[2] + ".00")),
                "amount_cny": str(amount(m[3] + "." + m[6])),
                "country": "日本",
                "source_excerpt": m[0],
                "format": "EXPLICIT_CROSS_PAGE_FRAGMENTS",
                "published_NAV_percent": m[4],
            }
        )
        require(len(rows) == 10, "WRAPPED_QUARTER_TOP10")
        tails = rows
    require(rows and len(rows) == len(tails), "STOCK_ROWS_NOT_CLOSED")
    require(len({r["security"] for r in rows}) == len(rows), "DUPLICATE_SECURITY_ROW")
    return rows


def merge_issuers(rows, nav):
    """普通股/类别股按JPX四位固有名代码归并，不把类别末位当成另一家公司。"""
    merged = {}
    for row in rows:
        code = row["security"]
        require(re.fullmatch(r"[1-9][0-9A-Z]{3,4}", code) is not None, "SECURITY_CODE")
        key = "JPX_STOCK_ENTITY:" + code[:4]
        item = merged.setdefault(key, {"issuer_key": key, "securities": [], "amount_cny": D(0)})
        item["securities"].append(code)
        item["amount_cny"] += D(row["amount_cny"])
    return [
        {
            "issuer_key": x["issuer_key"],
            "securities": x["securities"],
            "amount_cny": str(x["amount_cny"]),
            "ratio_to_NAV": str(x["amount_cny"] / nav),
        }
        for x in merged.values()
    ]


def profiles(source, inputs):
    output = []
    for ref in inputs["contracts"]:
        digest = ref["source_hash"]
        text = _plain(source.body(digest))
        require(_plain(inputs["legal_name"]) in text and _plain(inputs["manager_name"]) in text, "CONTRACT_IDENTITY")
        restrictions = _section(text, r"四、投资限制", r"五、")
        scope = _section(text, r"二、投资范围", r"三、投资策略")
        require("在日本证券市场以及在其他证券市场交易的日本公司股票" in scope, "JAPAN_SCOPE")
        band = re.search(r"投资于股票资产的比例占基金资产的(\d+)%-([\d]+)%", restrictions)
        require(band is not None, "STOCK_BAND")
        japan = re.search(r"投资于日本上市公司股票的比例不低于非现金基金资产的(\d+)%", restrictions)
        require(japan is not None, "JAPAN_BUDGET")
        checks = {
            "stock_min": D(band[1]) / 100,
            "stock_max": D(band[2]) / 100,
            "japan_min": D(japan[1]) / 100,
        }
        # 每个数值同时绑定完整原句，避免适配器在合同变化后继续使用旧阈值。
        required = [
            "投资于日本上市公司股票的比例不低于非现金基金资产的80%",
            "现金或到期日在一年以内的政府债券的比例合计不低于基金资产净值的5%",
            "现金类资产不包括结算备付金、存出保证金、应收申购款等",
            "本基金总资产不得超过基金净资产的140%",
            "本基金持有同一家银行的存款不得超过基金资产净值的20%",
            "本基金持有同一机构(政府、国际金融组织除外)发行的证券市值不得超过基金资产净值的10%",
            "借入现金的比例不得超过基金资产净值的10%",
            "基金持有非流动性资产市值不得超过基金净值的10%",
            "本基金主动投资于流动性受限资产的市值合计不得超过本基金资产净值的15%",
        ]
        for quote in required:
            require(quote in restrictions, "CONTRACT_LIMIT_CHANGED:" + quote)
        output.append(
            {
                "source_hash": digest,
                "published_on": source.sources[digest]["published_on"],
                "restriction_sha256": content_digest(restrictions.encode()),
                "restrictions": restrictions,
                "investment_scope": scope,
                "bounds": checks,
            }
        )
    require(len(output) == 3 and len({x["restriction_sha256"] for x in output}) == 1, "CONTRACT_CHANGE_REQUIRES_REVIEW")
    require(min(x["published_on"] for x in output) <= "2023-09-30", "INITIAL_CONTRACT")
    return output


def material_review(source, inputs):
    """将已人工读过的逐期事件复核绑定原段，再由真实明确披露推导阻断标志。

    审计机构变更单列已知事项并核本基金实际审计意见，不能把变更本身当成
    执行违规，也不能把空事件列表或调用者传来的False当作无重大事项证据。
    """
    reviews = inputs.get("material_event_reviews", [])
    require(
        len(reviews) == 6 and {r["period"] for r in reviews} == set(inputs["full_reports"]), "MATERIAL_REVIEW_PERIODS"
    )
    checked = []
    blocking = []
    for review in reviews:
        period = review["period"]
        digest = inputs["full_reports"][period]
        require(review["source_ref"] == digest, "MATERIAL_REVIEW_SOURCE_BINDING")
        text = source.body(digest)
        actual = _section(text, r"\d+\.3\s*涉及基金管理人", r"\d+\.7\s*基金租用")
        require(_plain(review["quote"]) == _plain(actual), "MATERIAL_REVIEW_QUOTE")
        plain = _plain(actual)
        facts = {
            "fund_litigation_explicit_absence": "本报告期内无涉及基金管理人、基金财产、基金托管业务的诉讼" in plain,
            "strategy_continuity_explicit": "本报告期内无基金投资策略的改变" in plain,
            "manager_action_explicit_absence": bool(re.search(r"管理人未受(?:稽查|调查)或处罚", plain)),
            "trustee_action_scope_explicit_absence": bool(
                re.search(r"托管人未受稽查或处罚|本基金托管人的基金托管业务履职不涉及受调查或处罚等情况", plain)
            ),
        }
        changed = "本基金改聘" in plain
        known = review.get("known_items", [])
        if changed:
            require(
                len(known) == 1 and known[0]["kind"] == "AUDITOR_CHANGE" and _plain(known[0]["quote"]) in plain,
                "KNOWN_AUDITOR_CHANGE_NOT_REVIEWED",
            )
        else:
            require(not known, "UNBOUND_KNOWN_ITEM")
        annual = period.endswith("-12-31")
        if annual:
            # 三个年度都读实际意见，不以未改聘或表头词代替；也不跨年借意见。
            audit = _section(text, r"6\.1\s*审计(?:意见|报告基本信息)", r"§7\s*年度")
            opinion_match = re.search(r"我们认为[，,]后附的财务报表.*?公允反映了.*?情况[。.]", _plain(audit))
            opinion = opinion_match[0] if opinion_match else ""
            facts["annual_actual_audit_opinion_reviewed"] = bool(
                opinion
                and "在所有重大方面" in opinion
                and period[:4] + "年12月31日" in opinion
                and not any(x in opinion for x in ["除", "保留", "否定", "无法表示"])
            )
        pending = (review.get("disposition") != "NO_BLOCKING_FUND_OPERATION_ISSUE") or not all(facts.values())
        row = {
            "period": period,
            "source_ref": digest,
            "quote": actual,
            "facts": facts,
            "known_items": known,
            "review_reason": review.get("reason"),
            "material_event_pending": pending,
        }
        if annual:
            row["actual_audit_opinion_quote"] = opinion
        checked.append(row)
        if pending:
            blocking.append(row)
    return {
        "source_bound_reviews": checked,
        "blocking_events": blocking,
        "material_event_pending": bool(blocking),
        "scope": "本基金公开运作与已知事件的来源绑定复核；不等于独立监管无处罚检索。",
    }


def report(source, inputs, digest, entry, period, full, profile):
    text = source.body(digest)
    plain = _plain(text)
    require("019449" in plain[:18000] and _plain(inputs["legal_name"]) in plain[:18000], "REPORT_IDENTITY")
    year, month, day = period.split("-")
    require(f"{year}年{int(month)}月{int(day)}日" in plain[:1500], "REPORT_PERIOD")
    require(source.sources[digest]["published_on"] == entry["PUBLISHDATEDesc"][:10], "PUBLICATION_BINDING")
    if source.sources[digest]["kind"] == "json":
        article = source.json(digest)["data"]
        require(
            article["art_code"] == entry["ID"] and _plain(article["notice_title"]) == _plain(entry["TITLE"]),
            "NOTICE_BINDING",
        )
    a, total = assets(text)
    stock = D(a["权益投资"]["amount_cny"])
    country = _plain(_section(text, r"\d\.2\s*(?:报告)?期末在各个国家", r"\d\.3\s*(?:报告)?期末按行业"))
    japan = first_amount(country, r"日本(" + MONEY + r")")
    require(japan == stock, "COUNTRY_SCOPE_NOT_CLOSED")
    positions = stocks(text, full)
    manager = _section(
        text,
        r"\d\.\d\s*(?:管理人对)?报告期内(?:本)?基金运作遵规守信情况(?:的)?说明",
        r"\d\.\d\s*(?:管理人对报告期内)?公平交易",
    )
    require(
        "基金合同" in _plain(manager) and any(x in _plain(manager) for x in ["遵守", "遵循"]),
        "MANAGER_PUBLIC_STATEMENT",
    )
    require(not any(x in _plain(manager) for x in ["但是", "尚未完成", "违反基金合同"]), "KNOWN_STATEMENT_CONFLICT")
    if full:
        balance = _section(text, r"[67]\.1\s*资产负债表", r"[67]\.2\s*利润表")
        nav = first_amount(balance, r"(?:净资产合计|所有者权益合计)\s+(" + MONEY + r")")
        require(first_amount(balance, r"资产总计\s+(" + MONEY + r")") == total, "BALANCE_TOTAL")
        require(first_amount(balance, r"负债合计\s+(" + MONEY + r")") == total - nav, "LIABILITY_NAV_RECONCILIATION")
        require(sum(D(r["amount_cny"]) for r in positions) == stock, "FULL_SECURITIES_TOTAL")
    else:
        match = re.search(r"期末基金资产净值\s*(" + MONEY + r")\s+(" + MONEY + r")", text)
        require(match is not None and len(positions) == 10, "QUARTER_NAV_OR_TOP10")
        nav = amount(match[1]) + amount(match[2])
    lower, upper = profile["bounds"]["stock_min"], profile["bounds"]["stock_max"]
    require(nav > 0 and nav <= total, "ASSETS_NAV")
    require(lower <= stock / total <= upper and lower <= stock / nav <= upper, "STOCK_DENOMINATOR_BOUNDS")
    require(japan / total >= profile["bounds"]["japan_min"], "JAPAN_NONCASH_LOWER_BOUND")
    # 全部负债是借入现金的保守上界；不把应付赎回直接称为借款。
    liabilities = total - nav
    require(liabilities / nav <= D(".10"), "BORROW_CASH_UPPER_BOUND")
    require(total / nav <= D("1.40"), "TOTAL_LEVERAGE")
    bank_upper = D(a["银行存款和结算备付金合计"]["amount_cny"])
    require(bank_upper / nav <= D(".20"), "SINGLE_BANK_CONSERVATIVE_BOUND")
    issuers = merge_issuers(positions, nav)
    require(max(D(x["ratio_to_NAV"]) for x in issuers) <= D(".10"), "MERGED_ISSUER_LIMIT")
    row = {
        "period": period,
        "kind": "FULL" if full else "QUARTER",
        "source_hash": digest,
        "assets": a,
        "total_assets_cny": str(total),
        "nav_cny": str(nav),
        "stock_amount_cny": str(stock),
        "stock_total_assets_ratio": str(stock / total),
        "stock_NAV_ratio": str(stock / nav),
        "japan_non_cash_assets_lower_bound": str(japan / total),
        "borrowing_upper_bound_all_liabilities_cny": str(liabilities),
        "borrowing_upper_bound_NAV_ratio": str(liabilities / nav),
        "total_assets_NAV_ratio": str(total / nav),
        "all_bank_and_settlement_upper_bound_NAV_ratio": str(bank_upper / nav),
        "securities": positions,
        "issuer_aggregates": issuers,
        "issuer_scope": "ALL_SECURITIES" if full else "PUBLISHED_TOP10_ONLY",
        "other_security_classes_explicit_zero": True,
        "outside_scope_confirmed_cny": "0" if full else None,
        "manager_statement": manager,
        "contract_source_hash": profile["source_hash"],
        "quarter_cash_principal": None,
        "daily_compliance_certified": False,
    }
    if full:
        note = _section(text, r"[67]\.4\.7\.1\s*(?:货币资金|银行存款)", r"[67]\.4\.7\.2\s*交易性金融资产")
        cash = first_amount(note, r"等于[：:]\s*本金\s+(" + MONEY + r")")
        money = zero_or_amount(balance, r"(?:货币资金|银行存款)(?:\s+[\d.]+)?")
        settlement = zero_or_amount(balance, "结算备付金")
        require(cash <= money and money + settlement == bank_upper, "CASH_BOOK_MONEY_RECONCILIATION")
        margin = zero_or_amount(balance, "存出保证金")
        lower_cash = max(D(0), cash - margin)
        require(lower_cash / nav >= D(".05"), "ELIGIBLE_CASH_AFTER_CONSERVATIVE_MARGIN")
        require(
            zero_or_amount(balance, "短期借款") == 0 and zero_or_amount(balance, "卖出回购金融资产款") == 0,
            "BORROW_OR_REPO_BALANCE",
        )
        require(
            zero_or_amount(balance, r"衍生金融资产(?:\s+[\d.]+)?") == 0
            and zero_or_amount(balance, r"衍生金融负债(?:\s+[\d.]+)?") == 0,
            "DERIVATIVE_BALANCES",
        )
        restrictions = _section(text, r"[67]\.4\.12\s*期末[（(]", r"[67]\.4\.13\s*金融工具风险")
        require(len(re.findall(r"(?m)^\s*无[。.]?\s*$", restrictions)) == 5, "RESTRICTED_FULL_NOTES")
        for phrase in [
            "本基金本报告期末未持有债券",
            "本基金本报告期末未持有资产支持证券",
            "本基金本报告期末未持有金融衍生品",
            "本基金本报告期末未持有基金",
        ]:
            require(phrase in plain, "INSTRUMENT_SCOPE:" + phrase)
        require("本报告期内无基金投资策略的改变" in plain, "STRATEGY_CONTINUITY")
        trustee = _section(text, r"5\.2\s*托管人对报告期内本基金投资运作", r"5\.3\s*托管人")
        require("未发现" in _plain(trustee) and "损害基金份额持有人利益" in _plain(trustee), "TRUSTEE_STATEMENT")
        custody = re.search(r"本基金的银行存款分别由基金托管人.*?保管[，,]按适用利率计息[。.]", plain)
        require(custody is not None, "BANK_CUSTODY_SCOPE")
        row.update(
            qualified_cash_principal_cny=str(cash),
            margin_upper_bound_cny=str(margin),
            eligible_cash_conservative_lower_bound_cny=str(lower_cash),
            eligible_cash_lower_bound_NAV_ratio=str(lower_cash / nav),
            cash_note=note,
            restricted_note=restrictions,
            bank_custody_quote=custody[0],
            trustee_quote=trustee,
            restricted_securities_cny="0",
            domestic_ABS_and_credit_cny="0",
        )
    return row


def calculate(inputs: dict, *, root: Path) -> dict:
    require(
        inputs.get("version") == VERSION
        and inputs.get("fund_code") == "019449"
        and inputs.get("as_of_date") == "2026-09-30",
        "VERSION_OR_SCOPE",
    )
    require(
        not any(k in inputs for k in ["score", "state", "allocation_state", "mandate_state", "dimensions"]),
        "EXTERNAL_SCORE_OR_STATE",
    )
    source = Originals(inputs, root)
    binding = share_binding(source, inputs)
    catalog = _catalog(source, inputs)
    contract_profiles = profiles(source, inputs)
    events = material_review(source, inputs)
    require(not events["material_event_pending"], "SOURCE_BOUND_MATERIAL_EVENT_REVIEW_PENDING")
    csrc = _plain(source.body(inputs["authority_refs"]["CSRC_JAPAN_MOU"]))
    require("日期:2019-06-28" in csrc and "1997.3.18日本日本大藏省谅解备忘录东京" in csrc, "JAPAN_MOU_ORIGINAL")
    coding = _plain(source.body(inputs["authority_refs"]["JPX_ISSUER_CODE_RULE"]))
    require(
        "2022年5月31日" in coding and "4けたの株式固有名コード" in coding and "種類株式" in coding,
        "JPX_ISSUER_KEY_ORIGINAL",
    )
    rows = []
    with localcontext() as context:
        context.prec = 70
        for role in ["quarters", "full_reports"]:
            for period, digest in sorted(inputs[role].items()):
                available = [p for p in contract_profiles if p["published_on"] <= period]
                require(available, "CONTRACT_DISCLOSURE_COVERAGE")
                profile = max(available, key=lambda p: p["published_on"])
                try:
                    rows.append(
                        report(source, inputs, digest, catalog[role][period], period, role == "full_reports", profile)
                    )
                except ValueError as exc:
                    raise ValueError("R5_JAPAN_EXECUTION_REPORT:" + period + ":" + str(exc)) from exc
        quarters = {r["period"]: r for r in rows if r["kind"] == "QUARTER"}
        for full in [r for r in rows if r["kind"] == "FULL"]:
            same = quarters[full["period"]]
            require(
                all(full[k] == same[k] for k in ["total_assets_cny", "nav_cny", "stock_amount_cny"]),
                "SAME_DATE_REPORT_CONFLICT",
            )
        require(
            inputs["holding_date"] == max(inputs["full_reports"])
            and inputs["full_reports"][inputs["holding_date"]] == inputs["primary_ref"],
            "PRIMARY_PERIOD",
        )
        refs = sorted(source.sources)
        observations = {
            name: {
                "kind": "state",
                "state": "NORMAL_PUBLIC_EXECUTION",
                "reason": (
                    "12季公开资产、6期完整证券同发行人合并、合格现金和真实合同约束已核，"
                    "保留未公开日交易与管理人全组合未知。"
                ),
                "source_refs": refs,
            }
            for name in ["allocation", "mandate"]
        }
        packet = {
            "fund_code": "019449",
            "basis": ACTIVE_BASIS,
            "as_of_date": inputs["as_of_date"],
            "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
            "quality_metrics": {
                "target_product_aliases": inputs["share_binding"]["column_codes"],
                "material_event_pending": events["material_event_pending"],
                "observations": observations,
            },
            "r4_evidence_registry": {
                h: {**source.sources[h], "path": str((root / source.sources[h]["file"]).resolve())} for h in refs
            },
            "quality_public": {"dataDates": {"holdings": inputs["holding_date"]}},
            "raw_archive": [{"sha256": h, "kind": "ORIGINAL"} for h in refs],
        }
        anchors = {"material_events": {"019449": events["blocking_events"]}}
        scored = _calculate_frozen(packet, PROTOCOL, anchors, selected_dimensions={"efficiency"})
    require(scored["score"] is None and scored["grade"] is None, "NO_TOTAL")
    return {
        "version": VERSION,
        "score": scored["dimensions"]["efficiency"],
        "dimensions": scored["dimensions"],
        "subscores": scored["subscores"],
        "packet": packet,
        "anchors": anchors,
        "report_ledger": rows,
        "catalog_coverage": catalog,
        "material_event_review": events,
        "contract_profiles": contract_profiles,
        "share_binding": binding,
        "complete_score": None,
        "grade": None,
        "contract_effective_date_boundary": (
            "三份公开合同完整投资限制相同，按公开顺序配对；未把发布日期冒充未经原公告核定的法律生效日。"
        ),
        "explicit_unknowns": [
            "管理人全部基金/组合的跨产品持仓与投票权比例。",
            "未公开逐日交易、禁止行为内部审批及控制/影响发行人目的的独立审计。",
            "境外证券借贷的未单列余额、逐日担保及返还细项；转融通无出借仅证明其明示范围。",
            "季度未拆分银行本金与结算备付金，季度合格现金不填零；仅六份全文核实。",
        ],
        "limits": [
            "评价仅限36个月实际已公开配置截面和完整披露范围，非全部合同条款逐日独立合规审计。",
            "未知类别只使用有据保守上界唯一核限，不猜分类；新增非零证券工具、已知偏离或原合同变化会拒收。",
        ],
    }


def prepare(session, request: dict, *, root: Path) -> dict:
    """仅编译本基金QDII部分包，其他维度原样保留；不登记、不发布。

    必须先接入snapshot的原件重放分派，禁止旧snapshot未识别新字段时把
    研究点值当成已重放。工作窗口仅在独占暂存区验证该最小分派补丁。
    """
    from copy import deepcopy
    from datetime import UTC, datetime

    from app.services import fund_rating_reviewed_snapshot as snapshot
    from app.services.fund_rating_pipeline import archive_json, current_inputs
    from app.services.fund_rating_pipeline_prepare import canonical_source

    require(
        "japan_active_execution_calculation" in inspect.getsource(snapshot.calculate)
        and "japan_active_execution_calculation" in inspect.getsource(snapshot.validate_sources),
        "SNAPSHOT_ORIGINAL_REPLAY_HOOK_REQUIRED",
    )
    code = request["fund_code"]
    previous = current_inputs(session, [code]).get(code)
    require(previous is not None and str(previous["as_of_date"]) == request["as_of_date"], "CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    require(
        bundle.get("basis") == snapshot.BASIS
        and bundle.get("family") == "QDII_EQUITY"
        and not any(
            bundle.get(k) for k in ["r5_calculation", "passive_components_calculation", "active_execution_calculation"]
        ),
        "PARTIAL_JAPAN_QDII_BUNDLE_REQUIRED",
    )
    raw = deepcopy(request["inputs"])
    require(
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
    require(len(explanation) >= 40, "PUBLIC_EXPLANATION_REQUIRED")
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
        japan_active_execution_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
    )
    archives = {ref["sha256"]: ref for ref in bundle.get("extra_archives", [])}
    archives.update({digest: {k: ref[k] for k in ["file", "sha256", "kind"]} for digest, ref in raw["sources"].items()})
    bundle["extra_archives"] = list(archives.values())
    bundle.setdefault("public_explanations", {})["execution"] = explanation
    bundle.get("public_metrics", {}).pop("execution", None)
    snapshot.validate_sources(bundle, datetime.now(UTC), root=root)
    snapshot.calculate(bundle, root=root)
    return {
        "basis": snapshot.BASIS,
        "entries": [{k: v for k, v in bundle.items() if k not in ["basis", "reviewed_document"]}],
    }


def check_binding(bundle: dict, inputs: dict) -> None:
    """把本执行原件输入绑定到同一QDII产品、C份额、报告和研究时点。

    输入归档与字典的逐字相等由snapshot公共入口另核；本函数不登记、
    不发布，也不接受将其他类型的执行结果附到本基金的评分包。
    """
    calculation = bundle.get("japan_active_execution_calculation")
    require(
        isinstance(calculation, dict) and calculation.get("inputs") == inputs,
        "SNAPSHOT_TYPED_INPUT_BINDING",
    )
    require(
        bundle.get("family") == "QDII_EQUITY"
        and bundle.get("fund_code") == inputs.get("fund_code") == "019449"
        and bundle.get("share_class") == inputs.get("share_class") == "C"
        and bundle.get("as_of_date") == inputs.get("as_of_date") == "2026-09-30"
        and bundle.get("holding_date") == inputs.get("holding_date") == "2026-06-30",
        "SNAPSHOT_PRODUCT_SHARE_DATES",
    )
    identity = bundle.get("identity", {})
    require(
        bool(identity.get("legal_name"))
        and bool(identity.get("manager_name"))
        and inputs.get("legal_name") == identity["legal_name"]
        and inputs.get("manager_name") == identity["manager_name"],
        "SNAPSHOT_LEGAL_IDENTITY",
    )
    primary = bundle.get("original_report_archive", {}).get("sha256")
    require(
        isinstance(primary, str)
        and len(primary) == 64
        and inputs.get("primary_ref") == primary
        and inputs.get("full_reports", {}).get(inputs["holding_date"]) == primary,
        "SNAPSHOT_PRIMARY_REPORT",
    )
    require(
        not any(
            bundle.get(key)
            for key in (
                "r5_calculation",
                "active_execution_calculation",
                "passive_components_calculation",
                "passive_execution_calculation",
            )
        ),
        "SNAPSHOT_CONFLICTING_EXECUTION",
    )


def replay_snapshot(bundle: dict, dimensions: dict, *, root: Path) -> dict:
    """真实重编原件并核执行lower/upper/candidate三界，不以旧预览数字替代。

    只核本执行维度；其余维度、完整总分和等级由既有公共流程处理。
    对无候选点、非有限数或与实际冻结R5结果不一致的任一界明确拒收。
    """
    calculation = bundle.get("japan_active_execution_calculation")
    require(isinstance(calculation, dict) and isinstance(calculation.get("inputs"), dict), "SNAPSHOT_RAW_REQUIRED")
    inputs = calculation["inputs"]
    check_binding(bundle, inputs)
    calculated = calculate(inputs, root=Path(root))
    point = D(str(calculated["score"]))
    actual = dimensions.get("execution")
    require(isinstance(actual, dict) and point.is_finite(), "SNAPSHOT_EXECUTION_DIMENSION")
    for bound in ("lower", "upper", "candidate"):
        value = actual.get(bound)
        require(value is not None, "SNAPSHOT_EXECUTION_BOUND_MISSING:" + bound)
        try:
            number = D(str(value))
        except ArithmeticError as exc:
            raise ValueError("R5_JAPAN_EXECUTION_SNAPSHOT_INVALID_BOUND:" + bound) from exc
        require(number.is_finite() and number == point, "SNAPSHOT_EXECUTION_RECALCULATION:" + bound)
    return calculated
