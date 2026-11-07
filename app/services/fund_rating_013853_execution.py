"""013853公开执行原件适配候选：A/H全证券同发行人、现金排除项及总杠杆。

当前只验已冻结的013853、2026-09-30三年窗口，拒绝直接传分数/状态。
所有观察逐次从归档原件产生；不依赖worker探查金额、旧审阅结论或私有日账簿。
执行正常档仅限已公开截面，prepare/snapshot均重放同一原件输入；不登记或发布。
"""

from __future__ import annotations

import inspect
import re
from collections import defaultdict
from datetime import date
from decimal import Decimal, localcontext
from decimal import Decimal as D
from pathlib import Path

from app.services.fund_quality_v2_accounting import accounting_observations
from app.services.fund_quality_v2_balance_proofs import strict_balance_values
from app.services.fund_quality_v2_restrictions import full_asset_restriction_scope
from app.services.fund_quality_v2_restrictions import restricted_securities as restricted_check
from app.services.fund_quality_v2_stock_book import _stock_parser_view, periodic_financial_view
from app.services.fund_rating_active_execution import Originals, _plain, _section, _share_binding
from app.services.fund_rating_direct_index import _liabilities
from app.services.fund_rating_public_stock_tables import _ROW, cells, parse
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS

VERSION = "R5_ACTIVE_EXECUTION_AH_013853_PUBLIC_ORIGINALS_20261030_2"
MONEY = r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d{2}"
QUARTERS = {
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
}
FULL = {"2023-12-31", "2024-06-30", "2024-12-31", "2025-06-30", "2025-12-31", "2026-06-30"}


def need(condition, code):
    """遇真实冲突或缺项拒绝正常档，不把解析失败改写成违约或低分。"""
    if not condition:
        raise ValueError("R5_AH_EXECUTION_" + code)


def amount(value):
    value = D(value.replace(",", ""))
    need(value.is_finite() and value >= 0, "AMOUNT")
    return value


def top_ten(text):
    """保留原报告同一排名的A/H行，共同归一发行人，不逐证券误验10%。"""
    matches = list(re.finditer(r"(?m)^5\.3\.1\s*报告期末.*前十名股票投资明细", text))
    need(len(matches) == 1, "TOP10_TITLE")
    body = text[matches[0].end() :]
    stop = re.search(r"5\.4\s*报告期末按债券品种分类", body)
    need(stop is not None, "TOP10_BOUNDARY")
    lines = body[: stop.start()].splitlines()
    starts = [i for i, line in enumerate(lines) if _ROW.match(line)]
    rows = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        rank, code, first = _ROW.match(lines[start]).groups()
        rest = []
        for line in lines[start + 1 : end]:
            if line.strip().startswith("注："):
                break
            if line.strip():
                rest.append(line.strip())
        name, values = cells([first, *rest])
        rows.append(
            {"rank": int(rank), "code": code, "name": name, "amount": str(amount(values[1])), "quantity": values[0]}
        )
    need({r["rank"] for r in rows} == set(range(1, 11)), "TOP10_RANK_CLOSURE")
    return rows


def contracts(source, inputs):
    """按真实两期合同核全部已用于评分的约束，日期及原文均须闭合。"""
    output = []
    need(len(inputs["contracts"]) == 2, "CONTRACT_COUNT")
    for item in inputs["contracts"]:
        digest = item["source_hash"]
        text = _plain(source.body(digest))
        need(inputs["legal_name"] in text and inputs["manager_name"] in text, "CONTRACT_IDENTITY")
        for clause in [
            "本基金投资于股票、存托凭证的比例为基金资产的60%-95%",
            "港股通标的股票投资比例不超过本基金股票资产的50%",
            "不低于基金资产净值5%的现金或到期日在一年以内的政府债券",
            "现金不包括结算备付金、存出保证金、应收申购款等",
            "基金总资产不得超过基金净资产的140%",
        ]:
            need(clause in text, "ACTUAL_CONTRACT_CLAUSE:" + clause)
        original_restrictions = _section(text, r"四、投资限制", r"五、业绩比较基准")
        # 原PDF把重复合同页眉插进港股限制句；只移除明确的本基金合同页眉。
        header = inputs["legal_name"] + "基金合同"
        restrictions = original_restrictions.replace(header, "")
        exact_limits = [
            "本基金投资于股票、存托凭证的比例为基金资产的60%-95%",
            "港股通标的股票投资比例不超过本基金股票资产的50%",
            "现金不包括结算备付金、存出保证金、应收申购款等",
            "本基金每个交易日日终在扣除股指期货合约、国债期货合约需缴纳的交易保证金后,应当保持不低于基金资产净值5%的现金或到期日在一年以内的政府债券",
            "本基金持有一家公司发行的证券(同一家公司在内地和香港同时上市的A+H股合计计算),其市值不超过基金资产净值的10%",
            "本基金主动投资于流动性受限资产的市值合计不得超过基金资产净值的15%",
            "基金总资产不得超过基金净资产的140%",
            "本基金参与融资的,在任何交易日日终持有的融资买入股票与其他有价证券市值之和,不得超过基金资产净值的95%",
        ]
        for clause in exact_limits:
            need(clause in restrictions, "ACTUAL_RESTRICTION_CLAUSE:" + clause)
        need(re.search(r"本基金持有一家公司发行的证券.*?不超过基金资产净值的10%", text), "ALL_SECURITY_ISSUER_CLAUSE")
        need(source.sources[digest]["published_on"] <= item["effective_from"], "CONTRACT_EFFECTIVE_DATE")
        start = text.index("二、投资范围")
        end = text.index("三、投资策略", start)
        scope = text[start:end]
        need("港股通" in scope and "债券" in scope and "股票" in scope, "INVESTMENT_SCOPE")
        output.append(
            {
                **item,
                "scope_quote": scope,
                "restriction_quote": original_restrictions,
                "restriction_parser_view": restrictions,
                "contract_limit_clauses": exact_limits,
                "removed_exact_repeated_contract_headers": original_restrictions.count(header),
            }
        )
    return output


def public_execution_review(source, inputs):
    """核公开执行审阅的身份、完整事项章节及明确判断，不默认无重大事项。

    原文只证事实，执行影响来自显式审阅；公司人事变更或发行人监管披露
    不自动等于基金执行失败。未决/确认失败状态交给原冻结门禁拒收。
    """
    review = inputs.get("public_execution_review")
    need(isinstance(review, dict), "PUBLIC_REVIEW_REQUIRED")
    need(all(review.get(k) == inputs[k] for k in ["fund_code", "as_of_date", "holding_date"]), "PUBLIC_REVIEW_IDENTITY")
    need(
        "material_event_pending" in review and review["material_event_pending"] in [True, False, None],
        "EXPLICIT_EVENT_STATUS",
    )
    reports = review.get("reports")
    need(isinstance(reports, list) and len(reports) == 18, "PUBLIC_REVIEW_COVERAGE")
    actual = {(r["period_end"], r["kind"], r["source_hash"]) for r in inputs["reports"]}
    need(
        {(r.get("period"), r.get("kind"), r.get("source_hash")) for r in reports} == actual, "PUBLIC_REVIEW_SOURCE_SET"
    )
    patterns = {
        "TOP10_ISSUER_DISCLOSURE": r"(5\.11\.1本基金投资的前十名证券的发行主体.+?)(?=5\.11\.2)",
        "OTHER_IMPORTANT_QUARTER": r"(8\.2影响投资者决策的其他重要信息.+?)(?=§?9备查文件目录)",
        "MANAGER_EVENT_DISCLOSURE": r"((?:10|11)\.6\.1管理人.+?)(?=(?:10|11)\.6\.2)",
    }
    states = []
    for row in reports:
        scopes = (
            {"TOP10_ISSUER_DISCLOSURE", "OTHER_IMPORTANT_QUARTER"}
            if row["kind"] == "QUARTER"
            else {"MANAGER_EVENT_DISCLOSURE"}
        )
        events = row.get("events", [])
        need(len(events) == len(scopes) and {e.get("scope") for e in events} == scopes, "PUBLIC_REVIEW_EVENT_SCOPES")
        body = _plain(source.body(row["source_hash"]))
        for event in events:
            matches = re.findall(patterns[event["scope"]], body)
            need(len(matches) == 1 and _plain(event.get("quote", "")) == matches[0], "PUBLIC_REVIEW_ACTUAL_FULL_QUOTE")
            state = event.get("execution_impact")
            need(
                state in {"NO_EXECUTION_FAILURE_IDENTIFIED", "UNRESOLVED", "EXECUTION_FAILURE_CONFIRMED"},
                "PUBLIC_REVIEW_EXPLICIT_JUDGMENT",
            )
            need(isinstance(event.get("reason"), str) and bool(event["reason"].strip()), "PUBLIC_REVIEW_REASON")
            states.append(state)
    pending = True if "EXECUTION_FAILURE_CONFIRMED" in states else None if "UNRESOLVED" in states else False
    need(review["material_event_pending"] is pending, "EVENT_STATUS_RECONCILIATION")
    return {
        **review,
        "derived_material_event_pending": pending,
        "actual_full_sections_bound": len(states),
        "all_issuer_quality_clearance_claimed": False,
    }


def _calculate(inputs, root):
    need(inputs.get("version") == VERSION, "VERSION")
    need(inputs["fund_code"] == "013853" and inputs["as_of_date"] == "2026-09-30", "BOUNDED_SCOPE")
    need(inputs["legal_name"] == "大成匠心卓越三年持有期混合型证券投资基金", "LEGAL_NAME")
    need(inputs["holding_date"] == "2026-06-30", "HOLDING_DATE")
    need(
        not any(k in inputs for k in ["score", "state", "dimensions", "allocation_state", "mandate_state"]),
        "NO_EXTERNAL_STATE",
    )
    source = Originals(inputs, root)
    _share_binding(source, inputs)
    from app.services.fund_quality_v2_disclosures import report_period

    primary_text = source.body(inputs["primary_ref"])
    need(str(report_period(primary_text)[1]) == inputs["holding_date"], "PRIMARY_ACTUAL_PERIOD")
    need(inputs["manager_name"] in _plain(primary_text), "PRIMARY_MANAGER_IDENTITY")
    profiles = contracts(source, inputs)
    need({r["period_end"] for r in inputs["reports"] if r["kind"] == "QUARTER"} == QUARTERS, "TWELVE_QUARTERS")
    need({r["period_end"] for r in inputs["reports"] if r["kind"] == "FULL"} == FULL, "SIX_FULL_REPORTS")
    need(len(inputs["reports"]) == 18, "DUPLICATE_REPORT")
    event_review = public_execution_review(source, inputs)
    prospectus = _plain(source.body(inputs["historical_prospectus_ref"]))
    regulation = _plain(source.body(inputs["client_settlement_regulation_ref"]))
    need(inputs["legal_name"] in prospectus, "ACCOUNT_LEGAL_IDENTITY")
    need(
        "与基金托管人开立的托管账户建立第三方存管关系" in prospectus
        and "不为证券交易资金账户另行开立银行托管账户以外的其他银行账户" in prospectus,
        "ACTUAL_ACCOUNT_OWNERSHIP",
    )
    need(
        "客户交易结算资金专用存款账户和清算备付金账户" in regulation
        and "客户交易结算资金只能用于客户的证券交易结算和客户提款" in regulation,
        "CLIENT_CASH_NOT_SETTLEMENT_RESERVE",
    )
    rows = []
    for item in inputs["reports"]:
        digest, period = item["source_hash"], item["period_end"]
        text = source.body(digest)
        plain = _plain(text)
        data = source.json(digest)["data"]
        need(data["art_code"] == item["article_id"] and data["notice_title"] == item["title"], "REPORT_NOTICE_BINDING")
        need(
            data["notice_date"][:10] == source.sources[digest]["published_on"] <= inputs["as_of_date"],
            "ACTUAL_PUBLICATION_DATE",
        )
        need(inputs["legal_name"] in plain[:18000] and inputs["fund_code"] in plain[:18000], "REPORT_IDENTITY")
        if item["kind"] == "QUARTER":
            match = re.search(r"(20\d{2})年第(\d)季度", item["title"])
            need(match is not None, "ACTUAL_QUARTER_TITLE")
            year, quarter = map(int, match.groups())
            end = f"{year}-{quarter * 3:02d}-" + ("30" if quarter in [2, 3] else "31")
            need(period == end, "ACTUAL_QUARTER_PERIOD")
        else:
            from app.services.fund_quality_v2_disclosures import report_period

            need(str(report_period(text)[1]) == period, "ACTUAL_FULL_PERIOD")
        applicable = [
            p
            for p in profiles
            if p["effective_from"] <= period and (p.get("effective_to") is None or period <= p["effective_to"])
        ]
        need(len(applicable) == 1, "CONTRACT_PERIOD_COVERAGE")
        compliance = list(
            re.finditer(
                r"管理人对报告期内本基金运作遵规守信情况的说明(.+?)(?:\d\.\d(?:管理人对报告期内)?公平交易)", plain
            )
        )
        need(bool(compliance) and "无损害基金份额持有人利益的行为" in compliance[-1][1], "MANAGER_ACTUAL_STATEMENT")
        need(not any(x in compliance[-1][1] for x in ["但是", "除外", "尚未", "未完成"]), "KNOWN_CONFLICT")
        row = {
            "period": period,
            "kind": item["kind"],
            "source_hash": digest,
            "contract_source_hash": applicable[0]["source_hash"],
            "compliance_quote": compliance[-1][0],
            "quarter_cash_principal": None,
            "daily_compliance_certified": False,
        }
        if item["kind"] == "QUARTER":
            block = text[re.search(r"(?m)^5\.1\s*报告期末基金资产组合情况", text).end() :]
            block = block[: re.search(r"(?m)^5\.2\s", block).start()]

            def cell(label):
                # 本轮立即调用且不外传闭包；保持冻结原算法，不在后续迭代延迟读取。
                values = re.findall(r"(?m)^\s*(?:\d+\s+)?" + label + r"\s+(" + MONEY + r")\s+", block)  # noqa: B023
                need(len(values) == 1, "QUARTER_ASSET_CELL:" + label)
                return amount(values[0])

            stock, total, combined_cash, other = (
                cell(k) for k in ["权益投资", "合计", "银行存款和结算备付金合计", "其他资产"]
            )
            # 资产表各类为非负总额，三项已精确耗尽合计，其余列不可能有正余额；
            # 这不把季度混合“存款+备付金”当作可用现金本金。
            need(stock + combined_cash + other == total, "QUARTER_FULL_ASSET_RECONCILIATION")
            nav_match = re.search(r"(?m)^\s*4\.期末基金资产净值\s+([^\r\n]+)", text)
            if nav_match is None:
                nav_match = re.search(r"(?m)^\s*4\.期末基金资产[ \t]+([^\r\n]+)\r?\n[ \t]*净值[ \t]*\r?$", text)
            need(nav_match is not None, "QUARTER_NAV_ROW")
            nav_cells = re.findall(MONEY, nav_match[1])
            need(len(nav_cells) in [2, 3], "QUARTER_ALL_SHARES_NAV")
            nav = sum(map(amount, nav_cells), D(0))
            positions = top_ten(text)
            by_issuer = defaultdict(D)
            for position in positions:
                by_issuer[position["rank"]] += D(position["amount"])
            hk_matches = re.findall(r"港股公允价值为\s*(" + MONEY + r")\s*元", block)
            need(len(hk_matches) == 1, "QUARTER_HK_TOTAL")
            hk = amount(hk_matches[0])
            row.update(
                stock_rows=positions, all_asset_residual="0", AH_merged_ranking_note="合并计算公允价值参与排序" in plain
            )
        else:
            view, _ = periodic_financial_view([text])
            book = strict_balance_values(view, observed_on=period)["point_book"]
            need(bool(book), "FULL_POINT_BOOK")
            parsed = parse(view, book["stock_amount"], book["net_assets"], require_industry=False)
            stock, total, nav = D(book["stock_amount"]), D(book["total_assets"]), D(book["net_assets"])
            positions = parsed["holdings"]
            need(sum(D(p["market_value_cny"]) for p in positions) == stock, "FULL_STOCK_TOTAL")
            need("合并计算公允价值参与排序" in plain, "FULL_AH_ISSUER_BINDING")
            by_issuer = defaultdict(D)
            for position in positions:
                for original in position["reported_rows"]:
                    by_issuer[original["rank"]] += amount(original["amount"])
            need(D(book["bond_amount"]) == D(book["fund_amount"]) == 0, "OTHER_SECURITIES_SCOPE")
            need(
                D(book["assets"]["交易性金融资产"]) == stock and D(book["assets"]["衍生金融资产"]) == 0,
                "FULL_ASSET_CLASSES",
            )
            need(re.search(r"期末基金持有的期货合约情况无[。.]", plain), "ACTUAL_FUTURES_BALANCE")
            liability = _liabilities(view, book["liabilities"])
            financing_names = ["短期借款", "交易性金融负债", "衍生金融负债", "卖出回购金融资产款"]
            need(all(D(liability["point_values"][k]) == 0 for k in financing_names), "PUBLIC_FINANCING_BALANCE")
            normalized, _ = _stock_parser_view(view, positions)
            kwargs = {"stock_row_parser": parse_stock_valuation_rows} if period in {"2024-06-30", "2025-06-30"} else {}
            restrictions = restricted_check(normalized, positions, book["bond_amount"], **kwargs)
            scope = full_asset_restriction_scope(text)
            need(scope is not None, "FULL_RESTRICTED_ASSET_SCOPE")
            restricted_ratio = D(restrictions["restricted_amount"]) / nav
            # 全部受限资产是主动受限资产的保守上界，不猜每笔属于主动或被动。
            need(restricted_ratio <= D(".15"), "RESTRICTED_ASSET_15_PERCENT")
            row.update(
                public_liability_proof=liability,
                restricted_asset_proof=restrictions,
                public_limit_ledger={
                    "restricted_all_assets_NAV_ratio": str(restricted_ratio),
                    "active_restricted_15pct_conservative_upper_passed": True,
                    "other_assets_scope_original_quote": scope[0],
                    "financing_point_values": {k: liability["point_values"][k] for k in financing_names},
                    "unpublished_scope": ["管理人全部私有组合的合并交易", "非公开逐日成交及融资账簿"],
                    "not_certified_daily_or_manager_wide": True,
                },
            )
            trustee = list(
                re.finditer(
                    r"托管人对报告期内本基金投资运作遵规守信、净值计算、利润分配等情况的说明(.+?)(?:\d\.\d托管人对本)",
                    plain,
                )
            )
            need(bool(trustee) and "根据" in trustee[-1][1], "TRUSTEE_ACTUAL_STATEMENT")
            hk_matches = re.findall(r"港股公允价值为\s*(" + MONEY + r")\s*元", text)
            need(len(hk_matches) == 1, "FULL_HK_TOTAL")
            hk = amount(hk_matches[0])
            account = accounting_observations(view, date.fromisoformat(period))
            bank = D(account["observations"]["demand_deposits_principal"]["value"])
            cash = bank
            cash_proof = {"ordinary_bank_principal": str(bank), "broker_customer_deposit_principal": "0"}
            if bank / nav < D(".05"):
                # 只在实际唯一未决截面按原表的本期本金重建；明确排除另列的结算备付金。
                need(period == "2023-12-31", "NEW_CASH_GAP_REQUIRES_REVIEW")
                matches = re.findall(
                    r"(?m)^\s*其他存款\s+("
                    + MONEY
                    + r")\s+("
                    + MONEY
                    + r")\s*\n\s*等于[：:]本金\s+("
                    + MONEY
                    + r")\s+("
                    + MONEY
                    + r")",
                    text,
                )
                need(len(matches) == 1, "BROKER_PRINCIPAL_UNIQUE_ORIGINAL_CELL")
                broker = amount(matches[0][2])
                need("其他存款为存放在证券经纪商专用存款账户的证券交易结算资金" in plain, "ACTUAL_CLIENT_DEPOSIT_NOTE")
                need("本报告期内,本基金未发生重大流动性风险事件" in plain, "KNOWN_LIQUIDITY_CONFLICT")
                need(
                    re.search(r"除在附注.*?期末本基金持有的流通受限证券.*?其余均能及时变现", plain),
                    "ACTUAL_RESTRICTION_SCOPE",
                )
                need(D(book["assets"]["结算备付金"]) != broker, "NO_RESERVE_RECLASSIFICATION")
                cash += broker
                cash_proof.update(
                    broker_customer_deposit_principal=str(broker),
                    excluded_settlement_reserves=str(book["assets"]["结算备付金"]),
                    account_qualification_is_legal_and_accounting_inference=True,
                )
            need(cash / nav >= D(".05"), "QUALIFIED_CASH_BOUND")
            row.update(
                full_stock_rows=positions,
                point_book=book,
                qualified_cash_principal=str(cash),
                cash_qualification=cash_proof,
                accounting_proof=account,
                trustee_quote=trustee[-1][0],
            )
        maximum = max(by_issuer.values())
        need(
            nav > 0 and D(".60") <= stock / total <= D(".95") and D(".60") <= stock / nav <= D(".95"),
            "STOCK_BAND_BOTH_DENOMINATORS",
        )
        need(
            hk / stock <= D(".50") and total / nav <= D("1.40") and maximum / nav <= D(".10"),
            "HK_LEVERAGE_ALL_ISSUER_LIMITS",
        )
        row.update(
            stock_amount=str(stock),
            total_assets=str(total),
            net_assets=str(nav),
            HK_stock_amount=str(hk),
            maximum_issuer_amount=str(maximum),
            ratios={
                "stock_total": str(stock / total),
                "stock_nav": str(stock / nav),
                "HK_stock": str(hk / stock),
                "gross_leverage": str(total / nav),
                "maximum_issuer_nav": str(maximum / nav),
            },
        )
        rows.append(row)
    quarters = {r["period"]: r for r in rows if r["kind"] == "QUARTER"}
    for row in [r for r in rows if r["kind"] == "FULL"]:
        need(
            all(
                row[k] == quarters[row["period"]][k]
                for k in ["stock_amount", "total_assets", "net_assets", "HK_stock_amount"]
            ),
            "SAME_DATE_REPORT_CONFLICT",
        )
    for row in quarters.values():
        need(row["AH_merged_ranking_note"] or row["period"] in FULL, "QUARTER_AH_ISSUER_PAIRING")
    refs = sorted(source.sources)
    observations = {
        name: {"kind": "state", "state": "NORMAL_PUBLIC_EXECUTION", "reason": reason, "source_refs": refs}
        for name, reason in [
            ("allocation", "12个实际公开季度、6份完整报告与两期原合同核60%-95%、合格现金、港股比例和总杠杆。"),
            (
                "mandate",
                "完整股票按原报告A/H同发行人合并，其他证券类别与全资产守恒闭合，并与逐期实际管理及托管履职核对。",
            ),
        ]
    }
    packet = {
        "fund_code": inputs["fund_code"],
        "basis": ACTIVE_BASIS,
        "as_of_date": inputs["as_of_date"],
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "target_product_aliases": inputs["share_binding"]["column_codes"],
            "material_event_pending": event_review["derived_material_event_pending"],
            "observations": observations,
        },
        "r4_evidence_registry": {
            h: {**source.sources[h], "path": str((root / source.sources[h]["file"]).resolve())} for h in refs
        },
        "quality_public": {"dataDates": {"holdings": inputs["holding_date"]}},
        "raw_archive": [{"sha256": h, "kind": "ORIGINAL"} for h in refs],
    }
    result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"efficiency"})
    need(result["score"] is None and result["grade"] is None, "NO_WHOLE_SCORE")
    return {
        "version": VERSION,
        "score": result["dimensions"]["efficiency"],
        "dimensions": result["dimensions"],
        "subscores": result["subscores"],
        "packet": packet,
        "anchors": {},
        "report_ledger": rows,
        "contract_profiles": profiles,
        "public_execution_review": event_review,
        "complete_score": None,
        "grade": None,
        "limits": [
            "只核36个月已公开截面，不认证未公开逐日交易、所有私有组合或内部股票备选库。",
            "季度混合现金与备付金保持未知，未借邻期数值填充。",
            "客户结算专户本金资格为原表、账户权属和监管提款规则的公开解释，保留复核，不承诺未来支取安全。",
        ],
    }


def calculate(inputs, *, root):
    """为prepare和snapshot提供同一纯原件重放入口，统一精度且无正式写入。"""
    with localcontext() as context:
        context.prec = 70
        return _calculate(inputs, root)


# 原transfer parser逐函数内嵌，消除worker目录依赖；原数值/成本未知口径不变。
QUANTITY = r"(?:\d{1,3}(?:,\d{3})+|\d+)"
FINANCIAL = re.compile(
    rf"(?<![\d,.])({MONEY}|[-－—])\s+({MONEY})\s+({QUANTITY})\s+"
    rf"({MONEY}|[-－—])\s+({MONEY})(?=\s|$)"
)


def decimal(value):
    return Decimal(value.replace(",", ""))


def parse_stock_valuation_rows(stock_part, positions):
    """恢复整张股票表，拒绝缺行、重复金融片段、金额冲突及超完整持仓。"""
    compact = re.sub(r"\s+", "", stock_part)
    for label in ["金额单位：人民币元", "证券", "代码", "认购", "价格", "估值", "数量", "成本总额", "股"]:
        if label not in compact:
            raise ValueError("REVIEW_013853_TRANSFER_HEADER:" + label)
    by_code = {p["stock_code"].split(".")[0]: p for p in positions}
    blocks = re.split(r"(?m)(?=^[ \t]*\d{6}[ \t]+)", stock_part)[1:]
    if not 1 <= len(blocks) <= 1000:
        raise ValueError("REVIEW_013853_TRANSFER_ROW_COUNT")
    rows = []
    totals = defaultdict(lambda: [Decimal(0), Decimal(0)])
    for block in blocks:
        code = re.match(r"\s*(\d{6})", block)[1]
        # 脚注不能被拼进金融列，保留未经改写的整行供人工复核。
        financial_block = re.split(r"(?m)^\s*(?:注[：:]|[56]\.4\.12\.)", block, maxsplit=1)[0]
        found = list(FINANCIAL.finditer(financial_block))
        if code not in by_code or len(found) != 1:
            raise ValueError("REVIEW_013853_TRANSFER_FINANCIAL_UNIQUE:" + code)
        match = found[0]
        issue_cell, price_cell, quantity_cell, cost_cell, fair_cell = match.groups()
        quantity, price, fair = map(decimal, [quantity_cell, price_cell, fair_cell])
        if quantity <= 0 or price < 0 or fair < 0 or quantity * price != fair:
            raise ValueError("REVIEW_013853_TRANSFER_VALUATION_CONFLICT:" + code)
        missing_cost = issue_cell in "-－—" or cost_cell in "-－—"
        if missing_cost:
            remarks = financial_block[match.end() :].lstrip()
            if issue_cell not in "-－—" or cost_cell not in "-－—" or not remarks.startswith("转增"):
                raise ValueError("REVIEW_013853_TRANSFER_COST_SCOPE:" + code)
            cost, issue = None, None
        else:
            issue, cost = map(decimal, [issue_cell, cost_cell])
            if issue * quantity != cost:
                raise ValueError("REVIEW_013853_TRANSFER_COST_CONFLICT:" + code)
        totals[code][0] += quantity
        totals[code][1] += fair
        rows.append(
            {
                "code": code,
                "quantity": str(quantity),
                "amount": str(fair),
                "cost": None if cost is None else str(cost),
                "issue_price": None if issue is None else str(issue),
                "valuation_price": str(price),
                "reported_cost_cell": cost_cell,
                "reported_issue_price_cell": issue_cell,
                "cost_unknown_not_zero": missing_cost,
                "valuation_formula_checked": True,
                "original_row": financial_block,
            }
        )
    for code, (quantity, fair) in totals.items():
        holding = by_code[code]
        if quantity > decimal(holding["quantity_shares"]) or fair > decimal(holding["market_value_cny"]):
            raise ValueError("REVIEW_013853_TRANSFER_COMPLETE_HOLDING_BOUND:" + code)
    return rows


def prepare(session, request: dict, *, root: Path) -> dict:
    """编译本基金执行分项的标准请求，保留其余维度和实际核验时间。

    只准备归档，不登记或发布；公共快照必须识别本typed原件输入后才可运行。
    份额长期持有情景的费用不适用状态仍由原评分包处理，不在本入口更改。
    """
    from copy import deepcopy
    from datetime import UTC, datetime

    from app.services import fund_rating_reviewed_snapshot as snapshot
    from app.services.fund_rating_pipeline import archive_json, current_inputs
    from app.services.fund_rating_pipeline_prepare import canonical_source

    need(
        "fund_013853_execution_calculation" in inspect.getsource(snapshot.calculate)
        and "fund_013853_execution_calculation" in inspect.getsource(snapshot.validate_sources),
        "SNAPSHOT_ORIGINAL_REPLAY_HOOK_REQUIRED",
    )
    code = request["fund_code"]
    previous = current_inputs(session, [code]).get(code)
    need(previous is not None and str(previous["as_of_date"]) == request["as_of_date"], "CURRENT_PERIOD_REQUIRED")
    bundle = deepcopy(next(m["bundle"] for m in previous["input_snapshot"]["members"] if m["fund_code"] == code))
    need(bundle.get("basis") == snapshot.BASIS, "REVIEWED_BUNDLE_REQUIRED")
    raw = deepcopy(request["inputs"])
    bundle["fund_013853_execution_calculation"] = {"inputs": raw}
    check_binding(bundle, raw)
    explanation = request.get("explanation", "").strip()
    need(len(explanation) >= 40, "PUBLIC_EXPLANATION_REQUIRED")
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
        fund_013853_execution_calculation={"inputs": raw, "input_archive": archive_json(root, raw)},
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
    """将原件与本法律产品、A份额、持仓日及研究截止日逐项绑定。

    原件输入归档与字典相等由公共快照另核。primary是独立的原报告PDF，
    18个期间原文另有原公告JSON，不能错误要求两种原始文件SHA相同。
    已补执行分项的快照仍可复算，不限制此前已有几个评分点。
    """
    calculation = bundle.get("fund_013853_execution_calculation")
    need(
        isinstance(calculation, dict) and calculation.get("inputs") == inputs,
        "SNAPSHOT_TYPED_INPUT_BINDING",
    )
    need(
        bundle.get("family") == "ACTIVE_EQUITY"
        and bundle.get("fund_code") == inputs.get("fund_code") == "013853"
        and bundle.get("share_class") == inputs.get("share_class") == "A"
        and bundle.get("as_of_date") == inputs.get("as_of_date") == "2026-09-30"
        and bundle.get("holding_date") == inputs.get("holding_date") == "2026-06-30",
        "SNAPSHOT_PRODUCT_SHARE_DATES",
    )
    identity = bundle.get("identity", {})
    need(
        identity.get("legal_name") == inputs.get("legal_name") == "大成匠心卓越三年持有期混合型证券投资基金"
        and identity.get("manager_name") == inputs.get("manager_name") == "大成基金管理有限公司",
        "SNAPSHOT_LEGAL_IDENTITY",
    )
    primary = bundle.get("original_report_archive", {}).get("sha256")
    need(
        isinstance(primary, str)
        and re.fullmatch(r"[0-9a-f]{64}", primary)
        and inputs.get("primary_ref") == primary
        and primary in inputs.get("sources", {}),
        "SNAPSHOT_PRIMARY_REPORT",
    )
    need(
        not any(
            bundle.get(key)
            for key in (
                "r5_calculation",
                "active_execution_calculation",
                "passive_components_calculation",
                "passive_execution_calculation",
                "japan_active_execution_calculation",
            )
        ),
        "SNAPSHOT_CONFLICTING_EXECUTION",
    )


def replay_snapshot(bundle: dict, dimensions: dict, *, root: Path) -> dict:
    """从归档23原件重编，并核execution三界；只处理本维，不修改原评分包。"""
    calculation = bundle.get("fund_013853_execution_calculation")
    need(isinstance(calculation, dict) and isinstance(calculation.get("inputs"), dict), "SNAPSHOT_RAW_REQUIRED")
    inputs = calculation["inputs"]
    check_binding(bundle, inputs)
    calculated = calculate(inputs, root=Path(root))
    point = D(str(calculated["score"]))
    actual = dimensions.get("execution")
    need(isinstance(actual, dict) and point.is_finite(), "SNAPSHOT_EXECUTION_DIMENSION")
    for bound in ("lower", "upper", "candidate"):
        value = actual.get(bound)
        need(value is not None, "SNAPSHOT_EXECUTION_BOUND_MISSING:" + bound)
        try:
            number = D(str(value))
        except ArithmeticError as exc:
            raise ValueError("R5_AH_EXECUTION_SNAPSHOT_INVALID_BOUND:" + bound) from exc
        need(number.is_finite() and number == point, "SNAPSHOT_EXECUTION_RECALCULATION:" + bound)
    return calculated
