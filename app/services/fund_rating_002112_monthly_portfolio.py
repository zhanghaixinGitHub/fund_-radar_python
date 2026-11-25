"""002112本期全仓/容量原件重放；不沿旧33池，不把新Q3前十仓当全仓。"""

import re
from collections import defaultdict
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from app.services.fund_rating_active_portfolio_components import parse_classifier, parse_stock_text, parse_wrapped_text
from app.services.fund_rating_monthly_r5 import rolling_window
from app.services.fund_rating_thematic_source import RawArchive, iso, plain, require

VERSION = "R5_002112_MONTHLY_PORTFOLIO_ORIGINALS_V1"


def _full_day_table(text, security, start, resume):
    """只认明确停复牌表头下、本只A股精确起止日期行，不借用转债行。"""
    if not resume or "证券代码证券简称停复牌类型停牌起始日停牌期间停牌终止日复牌日" not in text:
        return False
    begin = date.fromisoformat(start)
    end = date.fromisoformat(resume)
    formats = [
        (f"{begin.year}/{begin.month}/{begin.day}", f"{end.year}/{end.month}/{end.day}"),
        (f"{begin.year}/{begin.month:02}/{begin.day:02}", f"{end.year}/{end.month:02}/{end.day:02}"),
    ]
    return any(re.search(re.escape(security) + r"[^0-9]{1,30}A股复牌" + re.escape(a + b)
                         + r"(?=$|[^0-9/]|[0-9]{6}[^0-9/])", text) for a, b in formats)


def _sse_suspension_status(source, digest, security, start, through):
    ref = source.original(digest)
    url = urlsplit(ref.get("url", ""))
    query = parse_qs(url.query)
    day = through.replace("-", "")
    require(
        url.scheme == "https" and url.netloc == "query.sse.com.cn" and url.path == "/commonSoaQuery.do"
        and query.get("sqlId") == ["GW_PL_JYTS_TFPXX"] and query.get("productCode") == [security]
        and query.get("startStopDate") == query.get("endStopDate") == [day]
        and through <= source.asof,
        "002112_SUSPENSION_EXCHANGE_REQUEST",
    )
    result = source.json(digest)
    require(isinstance(result, dict), "002112_SUSPENSION_EXCHANGE_RESPONSE")
    rows = result.get("result")
    page = result.get("pageHelp")
    require(
        isinstance(rows, list) and len(rows) == 1
        and isinstance(page, dict) and page.get("total") == 1
        and not result.get("actionErrors")
        and isinstance(rows[0], dict)
        and rows[0].get("productCode") == security and rows[0].get("controlType") == "TR"
        and rows[0].get("type") == "LXTP"
        and rows[0].get("startStopDate") == start.replace("-", "")
        and rows[0].get("endStopDate") == "" and rows[0].get("stopTime") == "",
        "002112_SUSPENSION_EXCHANGE_RESPONSE",
    )
    return True


def suspension_days(source, security, intervals, calendar_days):
    """只认原件明确的全日停牌安排；预计时长或API缺行不能变成0成交。"""
    covered = set()
    for interval in intervals:
        start = iso(interval["start_date"])
        claims = interval["claims"]
        roles = {}
        for claim in claims:
            ref = source.original(claim["source_hash"])
            source.claim(claim)
            text = plain(claim["quote"])
            require(security in plain(source.body(claim["source_hash"]))[:20000], "002112_SUSPENSION_SECURITY")
            day = iso(claim["date"])
            parsed = date.fromisoformat(day)
            forms = [day, day.replace("-", ""), f"{parsed.year}年{parsed.month}月{parsed.day}日",
                     f"{parsed.year}年{parsed.month:02}月{parsed.day:02}日"]
            dated = any(form in text for form in forms)
            if claim["role"] == "SUSPENSION_STATUS" and any(word in text for word in ["截至本公告披露日", "截至本公告披露之日", "截至目前"]):
                dated = dated or ref.get("published_on") == day
            require(dated, "002112_SUSPENSION_LITERAL_DATE")
            roles.setdefault(claim["role"], []).append((day, text))
        starts = [text for day, text in roles.get("SUSPENSION_START", []) if day == start and "停牌" in text
                  and not any(w in text for w in ["盘中", "临时停牌"])
                  and (any(w in text for w in ["开市", "全日", "全天", "停牌一天", "停牌1天", "停牌一个交易日", "停牌1个交易日"])
                       or re.search(r"自.{0,32}日起(?:继续)?停牌", text)
                       or _full_day_table(text, security, start, interval.get("resume_date")))]
        require(bool(starts), "002112_SUSPENSION_FULL_DAY_START")
        if interval.get("one_day"):
            require(any(any(w in text for w in ["停牌一天", "停牌1天", "停牌一个交易日", "停牌1个交易日"])
                        for text in starts), "002112_SUSPENSION_ONE_DAY_LITERAL")
            covered.update(day for day in calendar_days if day == start)
        elif interval.get("resume_date"):
            resume = iso(interval["resume_date"])
            require(resume > start and any(day == resume and "复牌" in text
                    and not any(word in text for word in ["预计复牌", "拟复牌", "尚不复牌"])
                    for day, text in roles.get("RESUMPTION", [])), "002112_SUSPENSION_ACTUAL_RESUMPTION")
            covered.update(day for day in calendar_days if start <= day < resume)
        else:
            through = iso(interval["through_date"])
            sse_status = interval.get("sse_status_ref")
            quoted = any(
                day == through and any(word in text for word in ["继续停牌", "仍处于停牌", "仍处停牌", "停牌状态"])
                for day, text in roles.get("SUSPENSION_STATUS", [])
            )
            require(
                through >= start and (quoted or (
                    sse_status and _sse_suspension_status(source, sse_status, security, start, through)
                )),
                "002112_SUSPENSION_ACTUAL_THROUGH_DATE",
            )
            covered.update(day for day in calendar_days if start <= day <= through)
    return covered


def _book(source, code, spec, holding_date):
    from app.services.fund_quality_v2_disclosures import report_period

    source.original(spec["source_ref"])
    body = source.body(spec["source_ref"])
    header = plain(body[:22000])
    require(plain(spec["legal_name"]) in header and any(c in header for c in spec["share_codes"]), "002112_BOOK_IDENTITY")
    require(str(report_period(body)[1]) == spec["period"] == holding_date
            and spec["published_on"] == source.sources[spec["source_ref"]]["published_on"] <= source.asof,
            "002112_BOOK_PERIOD")
    try:
        book = parse_stock_text(code, body)
    except (AssertionError, ValueError, AttributeError):
        book = parse_wrapped_text(code, body)
    # 旧版通用表解析器的period只是固定展示字段；经济期末由本报告实际正文核定。
    book["period"] = holding_date
    return book


def concentration(stocks, issuers, industries):
    """合并同一法律发行人及相同行业，分母为全部股票金额，不选前十或再归一。"""
    issuer_amount, industry_amount = defaultdict(lambda: D(0)), defaultdict(lambda: D(0))
    total = sum(D(row["market_value"]) for row in stocks)
    require(total > 0, "002112_BOOK_POSITIVE_TOTAL")
    for row in stocks:
        # 原完整表明确披露市值0的退市残余股份仍留在原表；经济敞口为零，
        # 对任何发行人/行业归属贡献均为0，不把缺分类冒充缺金额或删正额资产。
        if D(row["market_value"]) == 0:
            continue
        code = row["security"]
        require(code in issuers and code in industries, "002112_BOOK_CLASSIFICATION_MISSING:" + code)
        issuer_amount[issuers[code]] += D(row["market_value"])
        industry_amount[industries[code]] += D(row["market_value"])
    return [sum((value / total) ** 2 for value in groups.values()) for groups in [issuer_amount, industry_amount]]


def _quality(source, spec, own):
    from app.services.fund_rating_reviewed_quality import STATES

    review = source.json(spec["asset_review_ref"])
    amounts = {row["security"]: row["market_value"] for row in own["stocks"]}
    require(review["fund_code"] == "002112" and review["as_of_date"] == source.asof,
            "002112_QUALITY_REVIEW_PERIOD")
    records = {row["code"]: row for row in review["records"]}
    require(set(records) == set(amounts) and len(records) == len(review["records"]), "002112_QUALITY_ALL_HOLDINGS")
    positions = []
    for key, required_date in [("valuation_ref", source.asof), ("report_valuation_ref", own["period"])]:
        rows = {row["ts_code"].split(".")[0]: row for row in source.provider_rows(spec[key], ["ts_code", "trade_date", "close"])}
        require(all(code in rows and iso(rows[code]["trade_date"]) == required_date
                    and D(str(rows[code]["close"])) > 0 for code in amounts), "002112_QUALITY_ACTUAL_PRICE_PERIOD")
    for code, row in records.items():
        require(row["state"] in STATES and bool(row["reason"].strip()) and row.get("review_closed") is True,
                "002112_QUALITY_ACTUAL_JUDGMENT")
        roles = {claim["role"] for claim in row["claims"]}
        require({"ANNUAL_AUDIT", "INTERIM_FINANCIAL"} <= roles, "002112_QUALITY_FINANCIAL_SOURCE_ROLES")
        if spec["asset_company_inputs"][code].get("current_q3_ref"):
            require("Q3_FINANCIAL" in roles, "002112_QUALITY_CURRENT_QUARTER_REQUIRED")
        for claim in row["claims"]:
            original = source.original(claim["source_hash"])
            require(original.get("published_on", source.asof) <= source.asof, "002112_QUALITY_FUNDAMENTAL_TIME_BOUNDARY")
            expected_role_ref = dict(ANNUAL_AUDIT="annual_ref", INTERIM_FINANCIAL="interim_ref", Q3_FINANCIAL="current_q3_ref")
            if claim["role"] in expected_role_ref:
                require(claim["source_hash"] == spec["asset_company_inputs"][code][expected_role_ref[claim["role"]]],
                        "002112_QUALITY_FINANCIAL_IDENTITY")
            if claim["role"] == "ANNUAL_AUDIT":
                require("财务报表" in plain(claim["quote"]) and "我们认为" in plain(claim["quote"]),
                        "002112_QUALITY_FINANCIAL_NOT_INTERNAL_OR_FUNDS_AUDIT")
            source.claim(claim)
        positions.append(dict(security=code, amount=str(amounts[code]), state=row["state"], reason=row["reason"]))
    return dict(kind="asset_quality", positions=positions, stock_total=own["total_stock_value"])


def _target_liquidity(source, inputs):
    from app.services.fund_rating_payment_domain import compile_payment_from_primary
    from app.services.fund_rating_active_execution import MONEY, _section

    payment_inputs = dict(inputs, share_binding=dict(column_names=["德邦鑫星价值A", "德邦鑫星价值C"], column_codes=["001412", "002112"]))
    payment = compile_payment_from_primary(payment_inputs, root=source.root)
    body = source.body(inputs["primary_ref"])
    text = plain(body)
    flow = _section(body, r"§?\d{1,2}\s*开放式基金份额变动", r"§?\d{1,2}\s*重大事件")
    rows = {}
    labels = dict(opening="本报告期期初基金份额总额", purchases="本报告期基金总申购份额",
                  redemptions="本报告期基金总赎回份额", ending="本报告期期末基金份额总额")
    for key, label in labels.items():
        found = re.search(label + r"\s+(" + MONEY + r")\s+(" + MONEY + r")", flow)
        require(found is not None, "002112_HOLDER_FLOW_ROWS")
        rows[key] = [D(v.replace(",", "")) for v in found.groups()]
    require(all(rows["opening"][n] + rows["purchases"][n] - rows["redemptions"][n] == rows["ending"][n]
                for n in range(2)), "002112_HOLDER_SHARE_CONSERVATION")
    require("本报告期内无单一投资者持有基金份额比例达到或超过20%的情况" in text
            and "本报告期内,本基金未发生重大流动性风险事件" in text
            and "托管人未发现损害基金持有人利益的行为" in text, "002112_HOLDER_EXECUTION_SOURCE")
    prospectus = plain(source.body(inputs["fee_prospectus_ref"]))
    require(plain(inputs["legal_name"]) in prospectus[:20000] and "巨额赎回" in prospectus
            and "开放日" in prospectus, "002112_HOLDER_REDEMPTION_CONTRACT")
    nav = re.search(r"净资产合计\s+(" + MONEY + r")", body)
    require(nav is not None, "002112_LIQUIDITY_NAV")
    holders = dict(kind="holder_execution_r5", structure_type="DISPERSED", structure_closed=True,
        share_flow_closed=True, actual_execution_verified=True, redemption_mechanism_verified=True,
        actual_redemption_shares=str(sum(rows["redemptions"])), confirmed_pressure_effects=[],
        disclosure_notes="实际本期完整报告份额守恒、分散结构、巨额赎回条款及托管人履职；不保证未来赎回安全。")
    return dict(known_payment=payment["node"], restricted=dict(kind="restricted", restricted_amount="0",
                net_assets=nav[1].replace(",", ""), scope_closed=True), holders=holders)


def _capacity(source, inputs, codes, books):
    spec = inputs["portfolio_originals"]
    market = spec["market"]
    days, _ = rolling_window([iso(r["cal_date"]) for r in source.provider_rows(inputs["calendar_ref"],
                            ["exchange", "cal_date", "is_open"]) if r["exchange"] == "SSE" and r["is_open"] == 1], inputs["as_of_date"])
    expected = days[-60:]
    require([iso(d) for d in market["days"]] == expected and set(market["daily_refs"]) == set(expected),
            "002112_CAPACITY_EXACT_60_DAY_WINDOW")
    needed = {r["security"] for code in ["002112", *codes] for r in books[code]["stocks"] if D(r["market_value"]) > 0}
    amounts = {code: {} for code in needed}
    for day in expected:
        for row in source.provider_rows(market["daily_refs"][day], ["ts_code", "trade_date", "amount"]):
            require(iso(row["trade_date"]) == day, "002112_CAPACITY_MARKET_DAY")
            code = row["ts_code"].split(".")[0]
            if code in needed:
                value = D(str(row["amount"])) * 1000
                require(value.is_finite() and value >= 0 and day not in amounts[code], "002112_CAPACITY_ACTUAL_AMOUNT")
                amounts[code][day] = value
    suspension_audit, status_pending = [], []
    for proof in market.get("suspensions", []):
        code = proof["security"]
        require(code in needed, "002112_SUSPENSION_NOT_A_PORTFOLIO_SECURITY")
        if proof.get("delisting_date"):
            ended = iso(proof["delisting_date"])
            literal = f"{date.fromisoformat(ended).year}年{date.fromisoformat(ended).month}月{date.fromisoformat(ended).day}日"
            verified = []
            for claim in proof["delisting_claims"]:
                source.original(claim["source_hash"])
                source.claim(claim)
                text = plain(claim["quote"])
                verified.append("终止上市" in text and literal in text and code in text)
            require(any(verified), "002112_ACTUAL_TERMINATED_LISTING")
            # 法定注销与普通暂停交易不同。只证明先前连续停牌，不把摘牌后
            # 零行当成交0，也不临时借新公司成交或删正额头寸求出分。
            covered = set()
            for interval in proof["intervals"]:
                start = iso(interval["start_date"])
                for claim in interval["claims"]:
                    source.claim(claim)
                    text = plain(claim["quote"])
                    if all(word in text for word in ["连续停牌", "直至终止上市", "不再复牌"]):
                        d = date.fromisoformat(start)
                        require(f"{d.year}年{d.month}月{d.day}日" in text, "002112_TERMINATION_START_DATE")
                        covered.update(day for day in expected if start <= day < ended)
            status_pending.append(dict(security=code, status="CORPORATE_ACTION_METHOD_PENDING", effective_date=ended,
                dates=[day for day in expected if day >= ended],
                reason="已实际核正式摘牌/换股。固定报告期资本敞口与60日代理对法定承继尚无已核对应调整规则及必要观察；不借新证券成交、不删正额、不将摘牌后当普通停牌补零。"))
        else:
            try:
                covered = suspension_days(source, code, proof["intervals"], expected)
            except ValueError as error:
                status_pending.append(dict(security=code, status="SUSPENSION_STATE_EVIDENCE_PENDING", reason=str(error),
                                           remaining=proof.get("remaining", [])))
                covered = set()
        for day in covered:
            require(day not in amounts[code] or amounts[code][day] == 0, "002112_SUSPENSION_QUOTE_CONFLICT")
            if day not in amounts[code]:
                amounts[code][day] = D(0)
                suspension_audit.append(dict(security=code, date=day, reason="VERIFIED_FULL_DAY_SUSPENSION", amount_cny="0"))
    listing = {}
    for digest in market["listing_refs"]:
        for row in source.provider_rows(digest, ["symbol", "list_date", "ts_code"]):
            if row["symbol"] in needed and row.get("list_date"):
                listing[row["symbol"]] = iso(row["list_date"])
    gaps, average = list(status_pending), {}
    for code in sorted(needed):
        if code not in listing:
            gaps.append(dict(security=code, reason="缺真实上市日原件"))
            continue
        applicable = [d for d in expected if d >= listing[code]]
        missing = [d for d in applicable if d not in amounts[code]]
        if missing:
            gaps.append(dict(security=code, reason="缺真实成交，须核停牌或其他实际原因；没有填零", dates=missing))
        elif not applicable or sum(amounts[code][d] for d in applicable) <= 0:
            gaps.append(dict(security=code, reason="实际可用窗口无正成交，不能生成可变现承诺"))
        else:
            average[code] = sum(amounts[code][d] for d in applicable) / len(applicable)
    if gaps:
        return None, gaps
    values = {}
    for code in ["002112", *codes]:
        stocks = books[code]["stocks"]
        total = sum(D(row["market_value"]) for row in stocks)
        values[code] = sum(D(row["market_value"]) / total * D(row["market_value"]) / average[row["security"]]
                           for row in stocks if D(row["market_value"]) > 0)
    node = dict(kind="rank", target=str(values["002112"]), direction="lower",
        peers=[dict(product_id="CN_PUBLIC_FUND:" + c, value=str(values[c])) for c in codes],
        expected_products=["CN_PUBLIC_FUND:" + c for c in codes])
    node["verified_suspension_observations"] = suspension_audit
    return node, []


def compile_observations(inputs, *, root):
    # 快照入口使用80位小数环境；独立准备与冷重放必须使用同一精度。
    with localcontext() as context:
        context.prec = 80
        return _compile_observations(inputs, root=root)


def _compile_observations(inputs, *, root):
    from app.services.fund_rating_active_monthly_admission import replay_monthly_admission
    from app.services.fund_rating_002112_liquidity_admission import compile_admission

    require(inputs.get("version") == VERSION and inputs["fund_code"] == "002112", "002112_PORTFOLIO_SCOPE")
    source = RawArchive(inputs, Path(root))
    spec = inputs["portfolio_originals"]
    admission = replay_monthly_admission(source.json(inputs["monthly_admission_ref"]), root=Path(root))
    codes = admission["structure_codes"]
    require(set(spec["portfolios"]) == set(codes) | {"002112"}, "002112_PORTFOLIO_COMPLETE_GROUP")
    books = {code: _book(source, code, item, inputs["holding_date"]) for code, item in spec["portfolios"].items()}
    observations, missing = {}, {}
    classification = spec["classification"]
    require(classification["holding_date"] == inputs["holding_date"] and classification["published_on"] <= source.asof,
            "002112_CLASSIFICATION_REPORT_PERIOD")
    source.original(classification["article_ref"])
    source.claim(classification["period_claim"])
    held = date.fromisoformat(inputs["holding_date"])
    half = "上半年" if held.month == 6 else "下半年"
    require(f"{held.year}年{half}" in plain(classification["period_claim"]["quote"])
            and classification["period_claim"]["source_hash"] == classification["article_ref"]
            and source.sources[classification["pdf_ref"]]["url"] in source.body(classification["article_ref"]),
            "002112_CLASSIFICATION_PERIOD_AND_ATTACHMENT")
    source.original(classification["pdf_ref"])
    industries = {code: row["industry_code"] for code, row in parse_classifier(source.pages(classification["pdf_ref"])).items()}
    for item in classification["ipo_supplements"]:
        source.original(item["source_ref"])
        body = plain(source.body(item["source_ref"]))
        require(item["security"] in body and item["industry_code"] in body, "002112_CLASSIFICATION_IPO_ORIGINAL")
        require(item["security"] not in industries, "002112_CLASSIFICATION_NO_REPLACEMENT")
        industries[item["security"]] = item["industry_code"]
    identities = {row["code"]: row["orgId"] for row in source.json(spec["issuer_identity_ref"])["stockList"]}
    try:
        values = {code: concentration(book["stocks"], identities, industries) for code, book in books.items()}
        for n, key in enumerate(["issuer_concentration", "industry_concentration"]):
            observations[key] = dict(kind="rank", target=str(values["002112"][n]), direction="lower",
                peers=[dict(product_id="CN_PUBLIC_FUND:" + c, value=str(values[c][n])) for c in codes],
                expected_products=["CN_PUBLIC_FUND:" + c for c in codes])
        observations["asset_quality"] = _quality(source, spec, books["002112"])
    except (ValueError, KeyError) as error:
        missing["holdings"] = [dict(reason=str(error))]
        for key in ["issuer_concentration", "industry_concentration", "asset_quality"]:
            observations.pop(key, None)
    liquidity = compile_admission(source.json(inputs["liquidity_admission_ref"]), root=Path(root))
    if liquidity["pending"]:
        missing["liquidity"] = liquidity["pending"]
    else:
        capacity, gaps = _capacity(source, inputs, liquidity["admitted_codes"], books)
        if gaps:
            missing["liquidity"] = gaps
        else:
            observations.update(_target_liquidity(source, inputs))
            observations["market_capacity"] = capacity
    return dict(version=VERSION, observations=observations, missing=missing,
        ready_dimensions=[d for d in ["holdings", "liquidity"] if d not in missing],
        structure_count=len(codes), liquidity_count=len(liquidity["admitted_codes"]),
        securities=len({r["security"] for b in books.values() for r in b["stocks"]}),
        missing_peers_removed=False, imputed=False)
