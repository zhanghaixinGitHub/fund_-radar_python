"""境内黄金联接一年披露口径：逐产品原合同、黄金经济敞口及公开申赎。"""

import re
from datetime import date
from decimal import Decimal, localcontext

from app.services.direction_1d_protocol import calendar
from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals, _flows
from app.services.fund_rating_evidence_preview import management, performance
from app.services.fund_rating_market_review import annual_fee_score, fee_profile, normalized_text
from app.services.fund_rating_reviewed_snapshot import _read

VERSION = "GOLD_FEEDER_ONE_YEAR_DISCLOSURE_20261111_V2"
PROTOCOL_SHA256 = "e44789d8b54dbdd396466f0ec25e4554846a53e237de6580f7de1432b9450d64"
KEYS = ("return", "risk", "efficiency", "stability", "management", "holdings", "liquidity", "cost")
D = Decimal


def plain(value):
    result = normalized_text(value)
    result = re.sub(r"AU99\.?99", "Au99.99", result, flags=re.I)
    return result.replace("AU(T+D)", "Au(T+D)")


def money(value):
    return D(value.replace(",", ""))


def report(source, digest, code, legal, holding):
    pages = source.pages(digest)
    body = plain("\n".join(pages))
    if (
        code not in body[:18000]
        or plain(legal) not in body[:18000]
        or str(report_period("\n".join(pages))[1]) != holding
    ):
        raise ValueError("GOLD_DISCLOSURE_REPORT_BINDING")
    return pages, body


def first_money(body, label):
    m = re.search(re.escape(label) + r"(?:6\.4\.7\.\d+)?([\d,]+\.\d{2})", body)
    if not m:
        raise ValueError("GOLD_DISCLOSURE_BALANCE_ROW:" + label)
    return money(m[1])


def budget(source, digest, legal):
    body = plain("\n".join(source.pages(digest)))
    if plain(legal) not in body[:18000]:
        raise ValueError("GOLD_DISCLOSURE_CONTRACT_IDENTITY")
    own = re.search(r"2\.2基金产品说明(.+?)2\.2\.1目标基金产品说明", body)
    scope = own[1] if own else body
    m = re.search(
        r"业绩比较基准(?:为[:：]?)?(?:上海黄金交易所Au99\.99现货实盘合约(?:价格)?|国内黄金现货价格)收益率[×*](\d+(?:\.\d+)?)%[+＋]人民币活期存款税后利率[×*](\d+(?:\.\d+)?)%",
        scope,
    )
    if not m:
        raise ValueError("GOLD_DISCLOSURE_CONTRACT_GOLD_CASH")
    w = tuple(D(x) / 100 for x in m.groups())
    if sum(w) != 1 or min(w) < 0:
        raise ValueError("GOLD_DISCLOSURE_CONTRACT_BUDGET")
    return w, m[0]


def balance(body):
    result = {
        k: first_money(body, label)
        for k, label in [("net_assets", "净资产合计"), ("total_assets", "资产总计"), ("liabilities", "负债合计")]
    }
    if result["total_assets"] - result["liabilities"] != result["net_assets"] or result["net_assets"] <= 0:
        raise ValueError("GOLD_DISCLOSURE_BALANCE_RECONCILIATION")
    return result


def metal(body):
    amount = first_money(body, "贵金属投资")
    # 只核完整披露的唯一Au99.99行，不能把“前五名”自动视为全量。
    blocks = list(
        re.finditer(
            r"7\.[89]报告期末按公允价值占基金资产净值比例大小排序的前五名贵金属投资明细(.+?)7\.(?:9|10)期末", body
        )
    )
    if not blocks:
        blocks = list(
            re.finditer(
                r"7\.[89]报告期末按公允价值占基金资产净值比例大小排序的前五名贵金属投资明细(.+?)7\.(?:9|10)(?:报告期末|期末)",
                body,
            )
        )
    if not blocks:
        raise ValueError("GOLD_DISCLOSURE_COMPLETE_METAL_TABLE")
    block = blocks[-1][1]
    literal = format(amount, ",.2f")
    # 无“仅持有”原句时，以该唯一黄金明细金额等于全部贵金属账面余额闭合范围。
    if block.count("Au99.99") != 2 or literal not in block:
        raise ValueError("GOLD_DISCLOSURE_SINGLE_AU9999_BALANCE")
    return amount


def derivative(body):
    m = re.search(r"6\.4\.7\.3\.2期末基金持有的黄金衍生品情况(.+?)6\.4\.7\.4买入返售", body)
    if not m:
        raise ValueError("GOLD_DISCLOSURE_DERIVATIVE_SECTION")
    block = m[1]
    if "Au(T+D)" not in block:
        if not any(x in block for x in ["本基金本报告期末未持有", "无余额", "无。"]):
            raise ValueError("GOLD_DISCLOSURE_DERIVATIVE_UNKNOWN")
        return {"quantity": D(0), "market_value": D(0), "pnl": D(0), "explicit_sign": True, "quote": block}
    row = re.search(r"Au\(T\+D\)(?:Au\(T\+D\)|黄金延期合约)(-?[\d,]+\.\d{2})([\d,]+\.\d{2})(-?[\d,]+\.\d{2})", block)
    if not row:
        raise ValueError("GOLD_DISCLOSURE_DERIVATIVE_ROW")
    q, value, pnl = map(money, row.groups())
    if value <= 0 or q == 0 or "净额0.00" not in block:
        raise ValueError("GOLD_DISCLOSURE_DERIVATIVE_NETTING")
    return {
        "quantity": q,
        "market_value": value,
        "pnl": pnl,
        "explicit_sign": "买入持仓量以正数表示,卖出持仓量以负数表示" in block,
        "quote": block,
    }


def sensitivity(body):
    m = re.search(r"6\.4\.13\.4\.3\.2其他价格风险的敏感性分析(.+?)6\.4\.14公允价值", body)
    if not m:
        raise ValueError("GOLD_DISCLOSURE_SENSITIVITY_SECTION")
    up = re.search(r"业绩比较基准上升5%([\d,]+\.\d{2})", m[1])
    down = re.search(r"业绩比较基准下降5%(-[\d,]+\.\d{2})", m[1])
    if not up or not down or money(up[1]) != -money(down[1]):
        raise ValueError("GOLD_DISCLOSURE_SENSITIVITY_SYMMETRY")
    return money(up[1])


def signed_derivative(row, base, shock):
    if row["market_value"] == 0:
        return D(0), "EXPLICIT_NONE"
    if row["explicit_sign"]:
        return (
            row["market_value"] if row["quantity"] > 0 else -row["market_value"]
        ), "ORIGINAL_EXPLICIT_SIGN_CONVENTION"
    choices = [
        sign * row["market_value"]
        for sign in (1, -1)
        if abs((base + sign * row["market_value"]) * D(".05") - shock) <= D(".01")
    ]
    if len(choices) != 1:
        raise ValueError("GOLD_DISCLOSURE_DERIVATIVE_DIRECTION_UNKNOWN")
    return choices[0], "UNIQUE_EXACT_REPORTED_FIVE_PERCENT_SHOCK"


def holdings(source, inputs):
    own_pages, own = report(
        source, inputs["primary_ref"], inputs["fund_code"], inputs["report_legal_name"], inputs["holding_date"]
    )
    target = inputs["target"]
    if budget(source, inputs["primary_ref"], inputs["report_legal_name"])[0] != tuple(map(D, inputs["weights"])):
        raise ValueError("GOLD_DISCLOSURE_EFFECTIVE_BUDGET")
    target_pages, etf = report(
        source, target["source_hash"], target["fund_code"], target["legal_name"], inputs["holding_date"]
    )
    relation = re.search(r"2\.1\.1目标基金基本情况(.+?)2\.2基金产品说明", own)
    if not relation or target["fund_code"] not in relation[1] or plain(target["legal_name"]) not in relation[1]:
        raise ValueError("GOLD_DISCLOSURE_TARGET_RELATION")
    own_bal, etf_bal = balance(own), balance(etf)
    fund = first_money(own, "基金投资")
    raw = "\n".join(own_pages).rsplit("7.2 期末投资目标基金明细", 1)[1].split("7.3 ", 1)[0]
    if re.findall(r"(?m)^\s*([1-9])\s+", raw) != ["1"]:
        raise ValueError("GOLD_DISCLOSURE_SOLE_TARGET_TABLE")
    compact = plain(raw)
    literal = format(fund, ",.2f")
    amount_ok = literal in compact
    if not amount_ok:
        for split in range(max(1, len(literal) - 7), len(literal) - 1):
            if re.search(re.escape(literal[:split]) + r".{0,180}?" + re.escape(literal[split:]), compact):
                amount_ok = True
                break
    if not amount_ok:
        raise ValueError("GOLD_DISCLOSURE_TARGET_AMOUNT")
    if not re.search(r"基金投资-[-－—]?", etf):
        raise ValueError("GOLD_DISCLOSURE_TARGET_FURTHER_FUND_UNKNOWN")
    own_metal, etf_metal = metal(own), metal(etf)
    own_deriv, etf_deriv = derivative(own), derivative(etf)
    # 明示符号约定本身足够核方向；不把统计敏感性估计硬当精确点位delta。
    own_shock = None if own_deriv["explicit_sign"] else sensitivity(own)
    etf_shock = None if etf_deriv["explicit_sign"] else sensitivity(etf)
    etf_d, etf_mode = signed_derivative(etf_deriv, etf_metal, etf_shock)
    own_d, own_mode = signed_derivative(own_deriv, fund + own_metal, own_shock)
    etf_gold = etf_metal + etf_d
    actual_gold = own_metal + own_d + fund / etf_bal["net_assets"] * etf_gold
    g = actual_gold / own_bal["net_assets"]
    r = 1 - g
    wg, wc = map(D, inputs["weights"])
    score = max(D(0), 100 * (1 - (abs(g - wg) + abs(r - wc)) / 2))
    return score, {
        "target_fund": target["fund_code"],
        "target_holding_cny": str(fund),
        "target_holding_nav_fraction": str(fund / own_bal["net_assets"]),
        "own_net_assets": str(own_bal["net_assets"]),
        "target_net_assets": str(etf_bal["net_assets"]),
        "own_spot_gold_cny": str(own_metal),
        "own_derivative_market_notional_cny": str(own_d),
        "target_spot_gold_cny": str(etf_metal),
        "target_derivative_market_notional_cny": str(etf_d),
        "own_derivative_direction_proof": own_mode,
        "target_derivative_direction_proof": etf_mode,
        "own_derivative_reported_quantity": str(own_deriv["quantity"]),
        "target_derivative_reported_quantity": str(etf_deriv["quantity"]),
        "reported_quantity_unit": "原报告未明示单位的不加手/克标签，敞口直接使用报告人民币当前市值。",
        "economic_gold_exposure_fraction": str(g),
        "net_non_gold_residual_budget_fraction": str(r),
        "cash_budget_is_not_free_payment_cash": True,
        "report_zero_fair_value_not_zero_notional": True,
        "source_sha256": sorted(source.raw),
    }


def history(source, inputs):
    raw = source.json(inputs["nav_ref"])
    if (
        raw.get("fund_code") != inputs["fund_code"]
        or raw.get("share_class") != inputs["share_class"]
        or raw.get("source_code") != "TUSHARE_PRO_FUND"
        or raw.get("authorization_verified") is not True
    ):
        raise ValueError("GOLD_DISCLOSURE_NAV_SHARE_AUTHORIZATION")
    div = raw["dividend_query"]
    if raw.get("dividend_query_params") != {"ts_code": inputs["fund_code"] + ".OF"}:
        raise ValueError("GOLD_DISCLOSURE_UNFILTERED_SHARE_DIVIDEND_QUERY")
    if div.get("code") != 0 or div["data"]["items"]:
        raise ValueError("GOLD_DISCLOSURE_DIVIDEND_RECONSTRUCTION_REQUIRED")
    rows = raw["rows"]
    days = [r["nav_date"] for r in rows]
    if raw.get("original_nav_ref"):
        original = source.json(raw["original_nav_ref"])
        if original.get("code") != 0:
            raise ValueError("GOLD_DISCLOSURE_ORIGINAL_NAV_RESPONSE")
        values = {}
        for row in original["data"]["items"]:
            item = dict(zip(original["data"]["fields"], row, strict=True))
            if item["ts_code"] != inputs["fund_code"] + ".OF":
                raise ValueError("GOLD_DISCLOSURE_ORIGINAL_NAV_SHARE")
            values[str(date.fromisoformat(item["nav_date"]))] = item
        for row in rows:
            item = values[row["nav_date"]]
            if any(
                D(str(item[a])) != D(row[b])
                for a, b in (("unit_nav", "unit_nav"), ("accum_nav", "accumulated_nav"), ("adj_nav", "adjusted_nav"))
            ):
                raise ValueError("GOLD_DISCLOSURE_ORIGINAL_NAV_VALUE")
    sessions, calhash = calendar()
    expected = [str(d) for d in sessions if inputs["window_start"] <= str(d) <= inputs["as_of_date"]]
    if days != expected or len(days) < 200 or days[-1] != inputs["as_of_date"]:
        raise ValueError("GOLD_DISCLOSURE_VALUATION_CALENDAR")
    if any(
        not row["ann_date"]
        or row["ann_date"] < row["nav_date"]
        or D(row["unit_nav"]) != D(row["accumulated_nav"])
        or D(row["adjusted_nav"]) != D(row["unit_nav"])
        for row in rows
    ):
        raise ValueError("GOLD_DISCLOSURE_NAV_ADJUSTMENT")
    gold = source.json(inputs["gold_ref"])
    meta = source.sources[inputs["gold_ref"]]
    if meta.get("url") != "https://www.sge.com.cn/graph/Dailyhq" or meta.get("params") != {"instid": "Au99.99"}:
        raise ValueError("GOLD_DISCLOSURE_OFFICIAL_PRICE_IDENTITY")
    levels = {r[0]: D(str(r[2])) for r in gold["time"]}
    prov = source.json(inputs["gold_crosscheck_ref"])
    if prov.get("code") != 0 or prov["data"]["fields"] != ["ts_code", "trade_date", "close"]:
        raise ValueError("GOLD_DISCLOSURE_SECONDARY_PRICE_SCHEMA")
    cross = {str(date.fromisoformat(r[1])): D(str(r[2])) for r in prov["data"]["items"] if r[0] == "Au99.99"}
    if any(d not in levels or d not in cross or levels[d] != cross[d] or levels[d] <= 0 for d in days):
        raise ValueError("GOLD_DISCLOSURE_PRICE_CONFLICT_OR_GAP")
    if "活期存款0.350.35" not in source.plain(inputs["cash_rate_ref"]):
        raise ValueError("GOLD_DISCLOSURE_CASH_RATE")
    own_w, quote = budget(source, inputs["primary_ref"], inputs["report_legal_name"])
    if tuple(map(D, inputs["weights"])) != own_w:
        raise ValueError("GOLD_DISCLOSURE_EFFECTIVE_BUDGET")
    if len(inputs["contract_refs"]) != 2 or len(set(inputs["contract_refs"])) != 2:
        raise ValueError("GOLD_DISCLOSURE_ANNUAL_AND_CURRENT_CONTRACTS")
    periods = set()
    for digest in inputs["contract_refs"]:
        w, _ = budget(source, digest, inputs["report_legal_name"])
        periods.add(str(report_period("\n".join(source.pages(digest)))[1]))
        if w != own_w:
            raise ValueError("GOLD_DISCLOSURE_CHANGED_CONTRACT_WITHIN_WINDOW")
    if periods != {"2025-12-31", inputs["holding_date"]}:
        raise ValueError("GOLD_DISCLOSURE_CONTRACT_PERIOD_COVERAGE")
    if "国内黄金现货价格" in quote:
        definition = source.plain(inputs["gold_definition_ref"])
        if (
            "目前国内黄金现货价格收益率以AU99.99合约价格为基础进行计算" not in definition
            and "目前国内黄金现货价格收益率以Au99.99合约价格为基础进行计算" not in definition
        ):
            raise ValueError("GOLD_DISCLOSURE_EXACT_AU9999_DEFINITION")
    wg, wc = own_w
    # 用同一中报公布的三/六/十二个月基准回报交叉核对真实黄金收盘口径。
    # 两位百分数只允许其原公布半个最小刻度，不用人为放宽容差。
    report_body = plain("\n".join(source.pages(inputs["primary_ref"])))
    benchmark_checks = []
    for label, start in (("过去三个月", "2026-03-31"), ("过去六个月", "2025-12-31"), ("过去一年", "2025-06-30")):
        matched = re.search(re.escape(label) + r"(-?\d+(?:\.\d+)?)%(-?\d+(?:\.\d+)?)%(-?\d+(?:\.\d+)?)%", report_body)
        if not matched:
            raise ValueError("GOLD_DISCLOSURE_REPORTED_BENCHMARK_ROW")
        check_days = [str(d) for d in sessions if start <= str(d) <= inputs["holding_date"]]
        value = D(1)
        for a, b in zip(check_days[:-1], check_days[1:], strict=True):
            if a not in levels or b not in levels:
                raise ValueError("GOLD_DISCLOSURE_BENCHMARK_CHECK_PRICE_GAP")
            gap = (date.fromisoformat(b) - date.fromisoformat(a)).days
            value *= 1 + wg * (levels[b] / levels[a] - 1) + wc * D(".0035") * gap / 365
        calculated = (value - 1) * 100
        reported = D(matched[3])
        if abs(calculated - reported) > D(".005"):
            raise ValueError("GOLD_DISCLOSURE_REPORTED_BENCHMARK_CONFLICT:" + label)
        benchmark_checks.append(
            {
                "label": label,
                "from": start,
                "through": inputs["holding_date"],
                "reported_percent": str(reported),
                "calculated_percent": str(calculated),
                "precision_percent": ".01",
            }
        )
    bench = [D(1)]
    for a, b in zip(days[:-1], days[1:], strict=True):
        gap = (date.fromisoformat(b) - date.fromisoformat(a)).days
        bench.append(bench[-1] * (1 + wg * (levels[b] / levels[a] - 1) + wc * D(".0035") * gap / 365))
    ends = [max(d for d in days if d[:7] == month) for month in sorted({d[:7] for d in days})]
    hist = {
        "dates": days,
        "month_ends": ends,
        "adjusted_nav": [r["adjusted_nav"] for r in rows],
        "benchmark_index": list(map(str, bench)),
    }
    points, facts = performance(hist)
    facts.update(
        month_ends=ends, calendar_hash=calhash, contract_quote=quote, reported_benchmark_crosschecks=benchmark_checks
    )
    return points, facts


def liquidity(source, inputs):
    pages, own = report(
        source, inputs["primary_ref"], inputs["fund_code"], inputs["report_legal_name"], inputs["holding_date"]
    )
    _, target = report(
        source,
        inputs["target"]["source_hash"],
        inputs["target"]["fund_code"],
        inputs["target"]["legal_name"],
        inputs["holding_date"],
    )
    try:
        flows, proof = _flows(
            pages,
            allow_reversed=True,
            allow_interleaved=True,
            allow_period_variant=True,
            additional_shares=tuple(inputs.get("additional_shares", [])),
        )
    except ValueError:
        # 华安A/C/I原报告以共享前缀和末行份额标签列示；独立小节逐字段核列。
        block = own.rsplit("开放式基金份额变动", 1)[1].split("重大事件揭示", 1)[0]
        order = ["A", "C", "I"]
        if inputs.get("additional_shares") != ["I"] or not all("华安黄金ETF联接" + s in block for s in order):
            raise
        names = {
            "opening": "本报告期期初基金份额总额",
            "subscribed": "本报告期基金总申购份额",
            "redeemed": "本报告期基金总赎回份额",
            "split": "本报告期基金拆分变动份额",
            "closing": "本报告期期末基金份额总额",
        }
        parsed = {s: {"share": s} for s in order}
        for field, label in names.items():
            m = re.search(re.escape(label) + r"([\d,]+\.\d{2}|-)([\d,]+\.\d{2}|-)([\d,]+\.\d{2}|-)", block)
            if not m:
                raise ValueError("GOLD_DISCLOSURE_SHARE_FLOW_ROW:" + field) from None
            for share, value in zip(order, m.groups(), strict=True):
                parsed[share][field] = None if value == "-" else str(money(value))
        flows = list(parsed.values())
        proof = {}
        for row in flows:
            if D(row["opening"]) + D(row["subscribed"]) - D(row["redeemed"]) + D(row["split"] or 0) != D(
                row["closing"]
            ):
                raise ValueError("GOLD_DISCLOSURE_SHARE_FLOW_RECONCILIATION") from None
    if "在基金合同中设计了巨额赎回条款" not in own:
        raise ValueError("GOLD_DISCLOSURE_OPEN_END_REDEMPTION")
    td_terms = source.plain(inputs["td_contract_ref"])
    td_start = td_terms.find("合约代码Au(T+D)")
    td_section = td_terms[td_start : td_start + 1600] if td_start >= 0 else ""
    if "当日无负债结算制度" not in td_section or "交易单位1000克/手" not in td_section:
        raise ValueError("GOLD_DISCLOSURE_TD_SETTLEMENT_MECHANISM")
    # 目标ETF采用自身交易型开放式申赎机制，不硬套普通联接的巨额赎回条款。
    tblock = target.rsplit("开放式基金份额变动", 1)[1].split("重大事件揭示", 1)[0]
    target_flows = {}
    for label in [
        "本报告期期初基金份额总额",
        "本报告期基金总申购份额",
        "本报告期基金总赎回份额",
        "本报告期基金拆分变动份额",
        "本报告期期末基金份额总额",
    ]:
        m = re.search(re.escape(label) + r"([\d,]+\.\d{2}|-)", tblock)
        if not m:
            raise ValueError("GOLD_DISCLOSURE_TARGET_ETF_FLOW")
        target_flows[label] = D(0) if m[1] == "-" else money(m[1])
    if (
        target_flows["本报告期期初基金份额总额"]
        + target_flows["本报告期基金总申购份额"]
        - target_flows["本报告期基金总赎回份额"]
        + target_flows["本报告期基金拆分变动份额"]
        != target_flows["本报告期期末基金份额总额"]
    ):
        raise ValueError("GOLD_DISCLOSURE_TARGET_ETF_FLOW_RECONCILIATION")
    if "交易型开放式" not in target:
        raise ValueError("GOLD_DISCLOSURE_TARGET_ETF_REDEMPTION")
    for body in (own, target):
        if "净额0.00" not in body or "保证金" not in body:
            raise ValueError("GOLD_DISCLOSURE_LIQUIDITY_MECHANISM")
        if not all(
            term in body for term in ["报告期", "托管人", "复核", "未发现基金管理人有损害基金份额持有人利益的行为"]
        ):
            raise ValueError("GOLD_DISCLOSURE_TRUSTEE_EXECUTION")
    if inputs["risk_review"].get("actual_operation_reviewed") is not True:
        raise ValueError("GOLD_DISCLOSURE_ACTUAL_OPERATION_REVIEW_REQUIRED")
    return D(75), {
        "share_flows": flows,
        "flow_proof": proof,
        "target_share_flows": {k: str(v) for k, v in target_flows.items()},
        "standard": "公开申赎、有效合同、托管及黄金保证金机制；不保证赎回到账，不要求未披露的逐笔付款本金。",
    }


def management_replay(source, inputs, root):
    if inputs.get("management_inputs"):
        return management(inputs, root, inputs["primary_ref"])
    # 独立样本复用既有同源审阅：分值必须由原事实档位重算，并核原研究档案。
    reviewed = inputs["reviewed_management_facts"]
    doc = _read(reviewed["research_archive"], root)
    if doc.get("code") != inputs["fund_code"]:
        raise ValueError("GOLD_DISCLOSURE_REVIEWED_MANAGEMENT_PRODUCT")
    original = doc["dimensions"]["management"]
    point = original["candidate"] if isinstance(original, dict) else original
    values = {
        "BASIC_ROLE_EVIDENCED": D(50),
        "CONTINUITY_EVIDENCED": D(75),
        "REMEDIATION_DISCLOSED": D(50),
        "NORMAL_REVIEWED": D(75),
    }
    for claim in reviewed["claims"]:
        source.claim(claim)
    if any(s not in values for s in reviewed["states"]):
        raise ValueError("GOLD_DISCLOSURE_MANAGEMENT_STATE")
    result = sum(
        values[state] * weight for state, weight in zip(reviewed["states"], (D(".4"), D(".2"), D(".4")), strict=True)
    )
    if result != D(point):
        raise ValueError("GOLD_DISCLOSURE_MANAGEMENT_FACT_REPLAY")
    return result


def calculate(inputs, *, root):
    if (
        inputs.get("version") != VERSION
        or inputs.get("profile") != "GOLD"
        or inputs.get("fund_code") not in {"021740", "000216"}
    ):
        raise ValueError("GOLD_DISCLOSURE_PROFILE_SCOPE")
    for field in [
        "fund_code",
        "share_class",
        "as_of_date",
        "holding_date",
        "legal_name",
        "manager_name",
        "primary_ref",
        "sources",
    ]:
        if not inputs.get(field):
            raise ValueError("GOLD_DISCLOSURE_REQUIRED_BINDING:" + field)
    if (
        inputs["as_of_date"] != "2026-09-30"
        or inputs["holding_date"] != "2026-06-30"
        or inputs["window_start"] != "2025-09-30"
    ):
        raise ValueError("GOLD_DISCLOSURE_FROZEN_WINDOW")
    expected_share = {"021740": "C", "000216": "A"}[inputs["fund_code"]]
    if inputs["share_class"] != expected_share:
        raise ValueError("GOLD_DISCLOSURE_SHARE_BINDING")
    src = Originals(inputs, root, lazy=True)
    if inputs.get("protocol_ref") != PROTOCOL_SHA256 or src.json(PROTOCOL_SHA256).get("version") != VERSION:
        raise ValueError("GOLD_DISCLOSURE_FROZEN_PROTOCOL")
    expected_legal = {
        "021740": "前海开源黄金交易型开放式证券投资基金联接基金",
        "000216": "华安易富黄金交易型开放式证券投资基金联接基金",
    }[inputs["fund_code"]]
    if inputs["report_legal_name"] != expected_legal:
        raise ValueError("GOLD_DISCLOSURE_FULL_LEGAL_NAME")
    pages, body = report(
        src, inputs["primary_ref"], inputs["fund_code"], inputs["report_legal_name"], inputs["holding_date"]
    )
    if plain(inputs["manager_name"]) not in body[:18000]:
        raise ValueError("GOLD_DISCLOSURE_MANAGER_BINDING")
    # 同一产品短名仅作旧目录身份；完整法律全名和代码仍由同一原报告核准。
    if plain(inputs["legal_name"]) not in body[:18000]:
        raise ValueError("GOLD_DISCLOSURE_LEGAL_BINDING")
    risk = inputs["risk_review"]
    if (
        risk.get("material_event_pending") is not False
        or risk.get("confirmed_redemption_or_valuation_failure") is not False
    ):
        return {"dimensions": {k: None for k in KEYS}, "facts": {"risk_review": risk}, "blocked": True}
    scores = {k: None for k in KEYS}
    facts = {
        "profile": "GOLD",
        "protocol": VERSION,
        "errors": {},
        "preserved_supplements": [
            "三年同行与滚动表现",
            "全仓逐证券成交承接",
            "逐笔可支付本金",
            "详细黄金租赁/保证金费用",
        ],
    }
    with localcontext() as context:
        context.prec = 60
        for stage in ("history", "holdings", "liquidity", "management", "cost"):
            try:
                if stage == "history":
                    points, details = history(src, inputs)
                    scores.update(points)
                    facts["window"] = details
                elif stage == "holdings":
                    scores["holdings"], facts["holdings"] = holdings(src, inputs)
                elif stage == "liquidity":
                    scores["liquidity"], facts["liquidity"] = liquidity(src, inputs)
                elif stage == "management":
                    scores["management"] = management_replay(src, inputs, root)
                else:
                    summary = "\n".join(src.pages(inputs["summary_ref"]))
                    if inputs["fund_code"] == "021740":
                        # 保留A/C真实表头，只去除紧邻费表标签的冒号，兼容同一原概要版式。
                        summary = re.sub(r"(前海开源黄金ETF联接[AC])[:：](?=基金运作综合费率)", r"\1", plain(summary))
                    fees = fee_profile(
                        summary,
                        code=inputs["fund_code"],
                        share_class=inputs["share_class"],
                        legal_name=inputs["legal_name"],
                        supporting_text="\n".join(src.pages(inputs["prospectus_ref"]))
                        if inputs.get("prospectus_ref")
                        else "",
                        published_on=src.sources[inputs["summary_ref"]]["published_on"],
                    )
                    value = annual_fee_score(fees["fees"])
                    scores["cost"] = D(value["score"])
                    facts["annual_cost_per_10000"] = value["cost_cny"]
            except (ValueError, KeyError, FileNotFoundError) as error:
                facts["errors"][stage] = str(error)
    facts["source_sha256"] = sorted(set(src.raw) | set(inputs.get("management_inputs", {}).get("sources", {})))
    return {
        "dimensions": {k: str(v) if v is not None else None for k, v in scores.items()},
        "facts": facts,
        "blocked": False,
    }
