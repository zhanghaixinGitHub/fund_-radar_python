"""002112 的公开截面执行重放。公司全部证券合并，不把股票检查当证券检查。"""

import re
from collections import defaultdict
from datetime import date
from decimal import Decimal as D

from app.services.fund_rating_active_execution import _assets, _stocks, _section, MONEY
from app.services.fund_rating_thematic_source import RawArchive, plain, require

VERSION = "R5_002112_COMBINED_ISSUER_EXECUTION_ORIGINALS_V1"


def combined_issuer_exposure(securities, identities, nav):
    """一个发行人多个证券必须相加；政府债券须显式分类，不以代码猜发行人。"""
    totals = defaultdict(lambda: D(0))
    missing = []
    for row in securities:
        if row.get("issuer_kind") == "CENTRAL_GOVERNMENT":
            continue
        identity = identities.get(row["security"])
        if not identity:
            missing.append(row["security"])
            continue
        totals[identity] += D(row["amount_cny"])
    return dict(amounts={k: str(v) for k, v in sorted(totals.items())},
                maximum_pct=str(max(totals.values(), default=D(0)) / D(nav) * 100),
                missing_securities=sorted(set(missing)))


def qualified_cash(principal, government_bonds, period):
    """只计本金及可证剩余期限不超过一年的政府债；不计利息/备付金/应收。"""
    value = D(principal)
    unknown = []
    for row in government_bonds:
        if not row.get("maturity"):
            unknown.append(row["security"])
        elif 0 < (date.fromisoformat(row["maturity"]) - date.fromisoformat(period)).days <= 365:
            value += D(row["amount_cny"])
    return value, unknown


def positive_compliance(text):
    text = plain(text)
    return all(word in text for word in ["基金合同", "基金运作合法合规", "无损害基金份额持有人利益的行为"]) and not any(
        word in text for word in ["但是", "除外", "尚未", "未完成", "待整改"])


def _amount(text, label):
    found = re.search(re.escape(label) + r"\s*(" + MONEY + r")", text)
    require(found is not None, "002112_EXEC_AMOUNT:" + label)
    return D(found[1].replace(",", ""))


def _bond_rows(body, full):
    start = r"[78]\.6\s*期末按公允价值" if full else r"5\.5\s*报告期末按公允价值"
    # 季度只披露前五债券。若未覆盖债券合计，保留未知，不能补零。
    matches = list(re.finditer(start, body))
    if not matches:
        return []
    tail = body[matches[-1].start():]
    stop = re.search(r"\n\s*[578]\.\d+\s*(?:报告期末|期末|投资)", tail[20:])
    if stop:
        tail = tail[:20 + stop.start()]
    return [dict(security=m[1], name=m[2].strip(), amount_cny=str(D(m[3].replace(",", ""))))
            for m in re.finditer(r"(?:^|\n)\s*\d+\s+(\d{6})\s+(.+?)\s+(" + MONEY + r")\s+[\d.]+", tail)]


def compile_observations(inputs, *, root):
    """数据未知返回具体事实缺口；不把未适配异常伪装成外部等待或正常75。"""
    spec = inputs["execution_originals"]
    require(spec.get("version") == VERSION and inputs["fund_code"] == "002112", "002112_EXEC_SCOPE")
    source = RawArchive(inputs, root)
    contract = plain(source.body(spec["contract_ref"]))
    clauses = ["股票占基金资产的比例为0-95%", "不低于基金资产的5%",
               "本基金持有一家公司发行的证券,其市值不超过基金资产净值的10%",
               "到期日在一年以内的政府债券不低于基金资产净值的5%", "现金不包括结算备付金、存出保证金、应收申购款"]
    require(all(plain(x) in contract for x in clauses), "002112_EXEC_ACTUAL_CONTRACT")
    require(source.sources[spec["contract_ref"]]["published_on"] <= spec["effective_from"]
            and spec["effective_from"] <= "2023-09-30", "002112_EXEC_EFFECTIVE_CONTRACT")
    effective = plain(source.body(spec["effective_notice_ref"]))
    require(inputs["legal_name"] in effective and "上述修改自2021年7月2日起生效" in effective,
            "002112_EXEC_EFFECTIVE_NOTICE")
    for digest in inputs["current_catalog_refs"]:
        for row in source.json(digest)["Data"]:
            if spec["effective_from"] < row["PUBLISHDATEDesc"][:10] <= inputs["as_of_date"]:
                require(not ("基金合同" in row["TITLE"] and any(w in row["TITLE"] for w in ["修改", "修订", "变更"])),
                        "002112_EXEC_NEW_CONTRACT_REVIEW_REQUIRED")
    # 目录由原真实目录推导，调用者不能挑删亏损季度或缺件全文。
    from app.services.fund_rating_active_execution import _catalog
    catalog_input = dict(inputs, catalog_refs=inputs["current_catalog_refs"], quarters=spec["quarters"], full_reports=spec["full_reports"])
    catalog = _catalog(source, catalog_input)
    identities = {}
    for digest in spec["issuer_registry_refs"]:
        ref = source.original(digest)
        require(ref.get("api") == "stock_basic", "002112_EXEC_ISSUER_REGISTRY_SOURCE")
        raw = source.json(digest)
        data = raw.get("data", raw)
        fields = data["fields"]
        require(all(k in fields for k in ["symbol", "fullname"]), "002112_EXEC_ISSUER_REGISTRY_SCHEMA")
        for values in data["items"]:
            row = dict(zip(fields, values, strict=True))
            if row.get("fullname"):
                old = identities.setdefault(row["symbol"], row["fullname"])
                require(old == row["fullname"], "002112_EXEC_ISSUER_IDENTITY_CONFLICT")
    maturity = {}
    for code, bond in spec["government_bonds"].items():
        text = plain(source.body(bond["source_hash"]))
        end = date.fromisoformat(bond["maturity"])
        require(code in text and "国债" in text and (f"{end.year}年{end.month}月{end.day}日" in text
                or f"{end.year}年{end.month:02}月{end.day:02}日" in text), "002112_EXEC_GOVERNMENT_MATURITY")
        maturity[code] = bond["maturity"]
    rows, missing = [], []
    for role, full in [("quarters", False), ("full_reports", True)]:
        for period, digest in sorted(spec[role].items()):
            source.original(digest)
            body = source.body(digest)
            text = plain(body)
            entry = catalog[role][period]
            require(inputs["legal_name"] in text and "002112" in text and "001412" in text,
                    "002112_EXEC_REPORT_IDENTITY")
            require(source.sources[digest]["published_on"] == entry["PUBLISHDATEDesc"][:10], "002112_EXEC_REPORT_PUBLICATION")
            from app.services.fund_quality_v2_disclosures import report_period
            require(str(report_period(body)[1]) == period, "002112_EXEC_REPORT_PERIOD")
            assets, total = _assets(body)
            stocks = _stocks(body, full)
            stock = D(assets["权益投资"]["amount_cny"])
            fixed = D(assets["固定收益投资"]["amount_cny"])
            if full:
                balance = _section(body, r"[67]\.1\s*资产负债表", r"[67]\.2\s*利润表")
                nav = _amount(balance, "净资产合计")
                require(sum(D(r["amount_cny"]) for r in stocks) == stock, "002112_EXEC_FULL_STOCK_SUM")
            else:
                match = re.search(r"4\.期末基金资产净值\s+(" + MONEY + r")\s+(" + MONEY + r")", body)
                require(match is not None, "002112_EXEC_QUARTER_NAV")
                nav = sum(D(v.replace(",", "")) for v in match.groups())
            require(0 < nav <= total, "002112_EXEC_FINANCIAL_DENOMINATORS")
            local = []
            for phrase in ["本基金本报告期末未投资股指期货", "本基金本报告期末未持有权证"]:
                if phrase not in text:
                    local.append("缺期末明确工具持仓结论：" + phrase)
            if any(not assets[k]["explicit_dash_zero"] for k in ["基金投资", "贵金属投资", "金融衍生品投资", "买入返售金融资产"]):
                local.append("新增非股票/债券工具须取得逐项证券、发行人和合同范围原件")
            bonds = _bond_rows(body, full) if fixed else []
            if sum(D(r["amount_cny"]) for r in bonds) != fixed:
                local.append("债券逐只金额未与固定收益资产合计闭合，不能只核股票发行人")
            for bond in bonds:
                # 明确国债名与已核官方发行公告共同确认政府发行人。
                if bond["security"] in maturity and "国债" in bond["name"]:
                    bond.update(issuer_kind="CENTRAL_GOVERNMENT", maturity=maturity[bond["security"]])
            exposure = combined_issuer_exposure(stocks + bonds, identities, nav)
            if exposure["missing_securities"]:
                local.append("缺证券对应法律发行人：" + ",".join(exposure["missing_securities"]))
            breaches = {}
            for key, value, limit in [("stock_total_assets", stock / total * 100, D(95)),
                                      ("stock_nav_alternative", stock / nav * 100, D(95)),
                                      ("gross_asset_nav", total / nav * 100, D(140)),
                                      ("combined_company_securities", D(exposure["maximum_pct"]), D(10))]:
                if value > limit:
                    breaches[key] = str(value - limit)
            principal = None
            cash_value = None
            if full:
                note = _section(body, r"[67]\.4\.7\.1\s*(?:货币资金|银行存款)", r"[67]\.4\.7\.2\s*交易性金融资产")
                principal = _amount(note, "等于：本金")
                cash_value, unknown_bonds = qualified_cash(principal, [b for b in bonds if b.get("issuer_kind") == "CENTRAL_GOVERNMENT"], period)
                if unknown_bonds:
                    local.append("缺政府债真实到期日：" + ",".join(unknown_bonds))
                if cash_value / nav * 100 < 5:
                    local.append("实际可证合格现金下界不足5%，需要其他可计政府债或例外/整改原件")
                if (principal + fixed) / total * 100 < 5:
                    local.append("实际可证非股资产下界不足5%，不能将应收/备付金当合格本金")
                trustee = _section(body, r"5\.2\s*托管人对报告期内", r"5\.3\s*托管人")
                if not re.search(r"托管人未发现损害基金(?:份额)?持有人利益的行为", plain(trustee)):
                    local.append("托管人实际履职正文尚不能确认正常执行")
            compliance = _section(body, r"4\.2\s*管理人对报告期内", r"4\.3\s*")
            if not positive_compliance(compliance):
                local.append("管理人本期运作履职正文需进一步核明")
            if breaches:
                local.append("已见公开截面约束冲突，须实际原因、合法例外及整改：" + str(breaches))
            rows.append(dict(period=period, role=role, source_hash=digest, stock_cny=str(stock), nav_cny=str(nav),
                             total_assets_cny=str(total), securities=stocks + bonds, combined_issuer=exposure,
                             cash_principal_cny=None if principal is None else str(principal),
                             qualified_cash_cny=None if cash_value is None else str(cash_value), missing=local,
                             quarter_full_security_book_observed=full, daily_compliance_certified=False))
            missing.extend(dict(period=period, source_hash=digest, detail=value) for value in local)
    full_by_period = {r["period"]: r for r in rows if r["role"] == "full_reports"}
    for quarter in [r for r in rows if r["role"] == "quarters" and r["period"] in full_by_period]:
        other = full_by_period[quarter["period"]]
        require(all(quarter[key] == other[key] for key in ["nav_cny", "stock_cny", "total_assets_cny"]), "002112_EXEC_CROSS_REPORT_CONFLICT")
    refs = sorted(source.used)
    reason = "真实合同逐期配对，全部公开完整证券按法律发行人合并10%，本金/真实短期国债及非股下界逐期复算；季度只核公开截面，不认证未公开每日账簿。"
    observations = {} if missing else {key: dict(kind="state", state="NORMAL_PUBLIC_EXECUTION", reason=reason, source_refs=refs)
                                       for key in ["allocation", "mandate"]}
    return dict(version=VERSION, status="UNKNOWN" if missing else "COMPLETE", observations=observations,
                rows=rows, missing=missing, used_source_refs=refs, normal_point_not_assumed=True,
                public_observation_boundary="季度完整资产类别和前十股票；全文完整证券与本金；未披露每日及管理人全组合约束不宣称已证")
