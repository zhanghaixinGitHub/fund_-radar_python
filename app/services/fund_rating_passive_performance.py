"""由完整原件独立编译被动基金收益、风险、稳定性三个分项。

复用R5的performance和冻结换分器。新版本遵从前日净值扣现金的披露增长口径；
旧现金再投资版本只用于重放已经封存的历史快照，不能静默改写旧分。
不联网、不写库，不把其他五项、部分权重折算或新等级塞入本次结果。
当前仅开放已核准的160119资料契约；所有来源、同行准入和分红均须重新核验。
"""

import json
import re
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal, localcontext
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_direct_index import Originals, _contract, _history, _table
from app.services.fund_rating_direct_index_math import performance
from app.services.fund_rating_inputs import canonical
from app.services.fund_rating_market_review import normalized_text
from app.services.fund_rating_metrics import total_return
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen, number
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS

LEGACY_VERSION = "R5_PASSIVE_PERFORMANCE_ORIGINALS_20261029_1"
VERSION = "R5_PASSIVE_PERFORMANCE_ORIGINALS_20261029_2"
SELECTED = {"return", "risk", "stability"}


def disclosed_growth(units, cash):
    """冻结标准净值增长：NAV_t/(NAV_前交易日-当日单位分红)。

    仅用于无拆分且已由调用方核全交易日、现金事件及单位的序列。
    与除息日现金再投定义不同；二者不能互换或在同一个同行组中混用。
    依据本轮2026-10-19封存的return-statistic-alignment-protocol。
    """
    units, cash = list(map(number, units)), list(map(number, cash))
    if not units or len(units) != len(cash) or any(x <= 0 for x in units) or any(x < 0 for x in cash):
        raise ValueError("R5_DISCLOSED_GROWTH_INVALID_OBSERVATIONS")
    values = [Decimal(1)]
    for index in range(1, len(units)):
        denominator = units[index - 1] - cash[index]
        if denominator <= 0:
            raise ValueError("R5_DISCLOSED_GROWTH_CASH_EXCEEDS_PREVIOUS_NAV")
        values.append(values[-1] * units[index] / denominator)
    return values


def _js_value(text, name):
    """只把已知var右值当JSON读取，绝不eval来源脚本。"""
    found = re.search(r"var\s+" + re.escape(name) + r"\s*=\s*(.+?);", text)
    if not found:
        raise ValueError("PEER_JS_FIELD_MISSING:" + name)
    return json.loads(found[1], parse_float=Decimal)


def contracts(source, peer, code):
    """核同份额各期法律产品、95/5指数及同一目标ETF；转型候选保留原文排除。"""
    reports = peer["reports"]
    if set(reports) != {"2023", "2024", "2025", "current"}:
        raise ValueError("PEER_FULL_CONTRACT_HISTORY_REQUIRED:" + code)
    targets, legal_names, clauses, excerpts = {}, {}, {}, {}
    for year, digest in reports.items():
        ref = source.sources[digest]
        if ref.get("archive_kind") == "DERIVED":
            raise ValueError("PEER_CONTRACT_ORIGINAL_REQUIRED")
        pages = source.pages(digest)
        text = normalized_text("\n".join(pages))
        end = "2026-06-30" if year == "current" else year + "-12-31"
        try:
            period = report_period("\n".join(pages))
        except ValueError:
            # 首份年报可以从本基金真实成立日开始；括号是日期附注，
            # 只接受原文完整起止，不凭年报标题补造1月1日。
            dated = re.search(
                r"本报告期自(20\d{2})年(\d{1,2})月(\d{1,2})日"
                r"\(基金合同生效日\)起至(20\d{2})年(\d{1,2})月(\d{1,2})日止",
                text[:6000],
            )
            if not dated:
                raise
            period = (date(*map(int, dated.groups()[:3])), date(*map(int, dated.groups()[3:])))
        needed_start = max(year + "-01-01", "2023-09-28") if year != "current" else "2026-01-01"
        if str(period[1]) != end or str(period[0]) > needed_start:
            raise ValueError("PEER_CONTRACT_PERIOD:" + code)
        if code not in text[:15000]:
            raise ValueError("PEER_SHARE_IDENTITY:" + code)
        match = re.search(r"基金名称(.+?)基金简称", text)
        if not match:
            raise ValueError("PEER_LEGAL_NAME_REQUIRED:" + code)
        legal_names[year] = match[1]
        # 只核本联接的产品说明；不能误取下一节目标ETF的100%基准。
        own_matches = list(re.finditer(r"2\.2基金产品说明(.+?)(?:2\.2\.1目标基金产品说明|2\.3基金管理人)", text))
        if not own_matches:
            raise ValueError("PEER_OWN_CONTRACT_SECTION:" + code)
        # 年报目录也含相同标题，只取最后一份真实正文，不能把页码点线当合同。
        own = own_matches[-1]
        relation_matches = list(re.finditer(r"2\.1\.1目标基金基本情况(.+?)(?:2\.2基金产品说明|基金产品说明)", text))
        match = relation_matches[-1] if relation_matches else None
        if not match:
            return {
                "admitted": False,
                "reason": "三年窗口存在非ETF联接路线",
                "period": year,
                "source_hash": digest,
                "quote": text[text.find("基金名称") : text.find("基金名称") + 350],
            }
        target = re.search(r"基金(?:主)?代码(\d{6})", match[1])
        if not target:
            raise ValueError("PEER_TARGET_CODE_REQUIRED:" + code)
        targets[year] = target[1]
        excerpts[year] = match[0]
        clauses[year] = _contract(source, digest, code, "中证500")
        # 表格窄列会把“业绩比较基/准”拆到公式两侧；正文边界已精确排除
        # 目标ETF下一节，不再额外强求该标题五字连续而丢掉真实公式。
        if not all(x in own[1] for x in ["中证500指数", "95%", "5%", "活期", "税后"]):
            raise ValueError("PEER_OWN_95_5_REQUIRED:" + code + ":" + year)
    if len(set(targets.values())) != 1:
        raise ValueError("PEER_TARGET_ROUTE_CHANGE:" + code)
    # 名称变更不冒充新产品；目标代码和历史原件分别保留，供指挥核查法律更名。
    return {
        "admitted": True,
        "target_etfs": targets,
        "legal_names": legal_names,
        "clauses": clauses,
        "target_identity_excerpts": excerpts,
    }


def peer_history(source, peer, code, days, asof, *, disclosed_statistic=False):
    """以双源逐日单位净值和原公告分红复算；原复权序列仅作差异审计。"""
    rows = _table(source.json(peer["nav"]))
    dividends = _table(source.json(peer["dividends"]))
    text = source.raw[peer["second_nav_js"]].decode("utf-8-sig")
    if _js_value(text, "fS_code") != code:
        raise ValueError("PEER_SECOND_NAV_IDENTITY")
    units = _js_value(text, "Data_netWorthTrend")
    accumulated = _js_value(text, "Data_ACWorthTrend")

    def js_day(value):
        return datetime.fromtimestamp(value / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat()

    js = {}
    for row in units:
        day = js_day(row["x"])
        if days[0] <= day <= asof:
            if day in js:
                raise ValueError("PEER_SECOND_NAV_DUPLICATE:" + code)
            js[day] = row
    ac = {js_day(r[0]): number(r[1]) for r in accumulated}
    original, known_events = {}, {}
    for row in rows:
        day = date.fromisoformat(row["nav_date"]).isoformat()
        if (
            row["ts_code"] != peer["ts_code"]
            or day in original
            or date.fromisoformat(row["ann_date"]).isoformat() > asof
        ):
            raise ValueError("PEER_NAV_SCOPE_OR_DUPLICATE:" + code)
        original[day] = row
    if not set(days) <= original.keys() or not set(days) <= js.keys():
        raise ValueError("PEER_COMMON_DATES_MISSING:" + code)
    # 原响应中完全相同的重复分红行只认一个实际事件，冲突行绝不合并。
    duplicates = []
    for event in dividends:
        if event["ts_code"] != peer["ts_code"]:
            raise ValueError("PEER_DIVIDEND_IDENTITY")
        if not event.get("ex_date"):
            raise ValueError("PEER_DIVIDEND_EX_DATE_REQUIRED")
        day = date.fromisoformat(event["ex_date"]).isoformat()
        if not days[0] <= day <= asof:
            continue
        if date.fromisoformat(event["ann_date"]).isoformat() > asof or event["div_proc"] != "实施":
            raise ValueError("PEER_DIVIDEND_NOT_IMPLEMENTED")
        if day in known_events:
            if known_events[day] != event:
                raise ValueError("PEER_CONFLICTING_DIVIDEND")
            duplicates.append(day)
        known_events[day] = event
    confirmations = peer.get("cash_confirmations", {})
    if set(confirmations) != set(known_events):
        raise ValueError("PEER_CASH_ANNOUNCEMENT_SCOPE:" + code)
    cash = {}
    for day, event in known_events.items():
        claim = confirmations[day]
        document = normalized_text("\n".join(source.pages(claim["source_hash"])))
        if (
            normalized_text(claim["quote"]) not in document
            or code not in document
            or "元/10份" not in document
            or "再投资" not in document
            or number(claim["cash_per_10_units"]) / 10 != number(event["div_cash"])
        ):
            raise ValueError("PEER_CASH_ORIGINAL_CLAIM_MISMATCH:" + code)
        cash[day] = number(event["div_cash"])
    second_events = {}
    for day, row in js.items():
        note = row.get("unitMoney") or ""
        if note:
            found = re.fullmatch(r"分红:每份派现金([\d.]+)元", normalized_text(note))
            if found:
                second_events[day] = number(found[1])
            elif normalized_text(note) == "分红:每10份派现金元" and day in cash:
                # 该数据文件仅留分红日期，金额空白；金额来自已核A类原公告，
                # 不把空白当0，也不谎称第二来源提供了金额。
                second_events[day] = cash[day]
            else:
                raise ValueError("PEER_UNCOMPILED_SPLIT_OR_EVENT:" + code + ":" + day + ":" + note)
    if cash != second_events:
        raise ValueError("PEER_CASH_SOURCE_CONFLICT:" + code)
    with localcontext() as context:
        context.prec = 70
        unit_values, cash_values, adjusted = [], [], []
        first_ac_gap = None
        cumulative_cash = Decimal(0)
        for day in days:
            row = original[day]
            unit, acc = number(row["unit_nav"]), number(row["accum_nav"])

            def published_nav(value):
                literal = number(value)
                rounded = literal.quantize(Decimal(".0001"))
                # 净值公开精度为四位；仅接纳JS序列化出现的1e-12以内
                # 二进制尾数，如1.5251999999999999，不吞掉一个公开最小单位差。
                return rounded if abs(literal - rounded) <= Decimal("1e-12") else literal

            if unit != published_nav(js[day]["y"]) or acc != published_nav(ac[day]) or unit <= 0:
                raise ValueError("PEER_DUAL_SOURCE_NAV_CONFLICT:" + code + ":" + day)
            if first_ac_gap is None:
                first_ac_gap = acc - unit
            cumulative_cash += cash.get(day, Decimal(0))
            if acc - unit != first_ac_gap + cumulative_cash:
                raise ValueError("PEER_CUMULATIVE_CASH_OR_SPLIT_UNCLOSED:" + code + ":" + day)
            unit_values.append(str(unit))
            cash_values.append(str(cash.get(day, Decimal(0))))
            adjusted.append(number(row["adj_nav"]))
        # 版本1历史快照保持原现金再投定义可重放；新版本严格沿已冻结披露口径。
        values = (
            disclosed_growth(unit_values, cash_values)
            if disclosed_statistic
            else total_return(unit_values, cash_values, ["1"] * len(days))
        )
        discrepancies = [abs(value - old / adjusted[0]) for value, old in zip(values, adjusted, strict=True)]
    return list(map(str, values)), {
        "dates": len(days),
        "source_nav_rows": len(rows),
        "second_source_nav_rows": len(js),
        "cash_events": cash,
        "duplicate_identical_event_rows_retained": duplicates,
        "provider_adjusted_nav_max_normalized_difference": str(max(discrepancies)),
        "provider_adjusted_nav_not_used_or_rewritten": True,
        "return_method": (
            "FROZEN_DISCLOSED_GROWTH_PREVIOUS_NAV_EX_CASH"
            if disclosed_statistic
            else "LEGACY_CASH_REINVESTMENT_EX_DATE_REPLAY_ONLY"
        ),
        "missing_values_filled": 0,
        "cross_share_history_used": False,
    }


def calculate(inputs, *, root):
    """先逐字节复核来源，再复用同一不可变输入的解析和计算结果。"""
    Originals(inputs, root)
    return deepcopy(_calculate_cached(canonical(inputs), str(root.resolve())))


@lru_cache(maxsize=24)
def _calculate_cached(serialized, root_name):
    """完整重核各原件及准入，再用标准R5计算明确三维；总分与等级保持空。"""
    inputs, root = json.loads(serialized), Path(root_name)
    if inputs.get("version") not in {VERSION, LEGACY_VERSION} or inputs.get("fund_code") != "160119":
        raise ValueError("R5_PASSIVE_PERFORMANCE_SCOPE")
    source = Originals(inputs, root)
    own = inputs["target_execution_inputs"]
    if any(own[k] != inputs[k] for k in ["fund_code", "as_of_date"]):
        raise ValueError("TARGET_EXECUTION_SCOPE")
    history = _history(source, own)
    frozen = source.json(inputs["cohort_ref"])
    admissions = {code: contracts(source, peer, code) for code, peer in inputs["peers"].items()}
    admitted = sorted(c for c, a in admissions.items() if a["admitted"])
    excluded = sorted(c for c, a in admissions.items() if not a["admitted"])
    # 同一法律产品的不同份额不能充当独立同行，也不能在名单中重复计权。
    # 名称取各产品当前合同原文，比较前仅归一化空白和原有全角字符。
    legal_names = [normalized_text(admissions[c]["legal_names"]["current"]) for c in admitted]
    if normalized_text(own["legal_name"]) in legal_names or len(legal_names) != len(set(legal_names)):
        raise ValueError("R5_PASSIVE_PERFORMANCE_DUPLICATE_LEGAL_PRODUCT")
    if (
        sorted(inputs["peers"]) != frozen["candidate_peers"]
        or admitted != frozen["qualified_peers"]
        or excluded != sorted(frozen["exclusions"])
        or not admitted
    ):
        raise ValueError("FROZEN_COHORT_OR_ADMISSION_CHANGED")
    values = {
        "160119": performance(
            history["dates"], history["month_ends"], history["adjusted_nav"], history["benchmark_index"]
        )
    }
    audit = {}
    for code in admitted:
        nav, audit[code] = peer_history(
            source,
            inputs["peers"][code],
            code,
            history["dates"],
            inputs["as_of_date"],
            disclosed_statistic=inputs["version"] == VERSION,
        )
        values[code] = performance(history["dates"], history["month_ends"], nav, history["benchmark_index"])
    nodes, all_refs = {}, sorted(inputs["sources"])
    for name in ["return_1y", "return_3y", "extra_drawdown", "extra_downside"]:
        nodes[name] = {
            "kind": "rank",
            "target": values["160119"][name],
            "direction": "higher" if name.startswith("return_") else "lower",
            "expected_products": ["CN_PUBLIC_FUND:" + c for c in admitted],
            "peers": [{"product_id": "CN_PUBLIC_FUND:" + c, "value": values[c][name]} for c in admitted],
            "source_refs": all_refs,
        }
    with localcontext() as context:
        context.prec = 70
        for name, count in [("monthly_success", 36), ("annual_success", 25)]:
            nodes[name] = {
                "kind": "windows",
                "expected_windows": count,
                "source_refs": all_refs,
                "rows": [
                    {**r, "reference": str(sum(number(values[c][name][i]["target"]) for c in admitted) / len(admitted))}
                    for i, r in enumerate(values["160119"][name])
                ],
            }
    packet = {
        "basis": PASSIVE_BASIS,
        "fund_code": "160119",
        "as_of_date": inputs["as_of_date"],
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_metrics": {
            "observations": nodes,
            "target_product_aliases": ["CN_PUBLIC_FUND:160119"],
            "material_event_pending": inputs["material_event_pending"],
        },
        "quality_public": {"dataDates": inputs["data_dates"]},
        "r4_evidence_registry": {h: {**r, "path": str(root / r["file"])} for h, r in inputs["sources"].items()},
        "raw_archive": [{"sha256": h, "kind": r.get("archive_kind", "ORIGINAL")} for h, r in inputs["sources"].items()],
    }
    scored = _calculate_frozen(packet, PROTOCOL, {"material_events": {}}, selected_dimensions=SELECTED)
    assert scored["score"] is None and scored["raw_score"] is None and scored["grade"] is None
    return {
        "dimensions": scored["dimensions"],
        "standard_calculation": scored,
        "packet": packet,
        "audit": {
            "admissions": admissions,
            "peer_history": audit,
            "performance": values,
            "publisher_cross_check": history["publisher_cross_check"],
            "date_count": len(history["dates"]),
            "month_end_count": len(history["month_ends"]),
            "qualified_peer_count": len(admitted),
            "excluded": excluded,
            "common_reference": "本A官网已核完整实际合同基准为全组共同参照；不宣称各同行现金用词相同",
        },
        "proposed_complete_score": None,
        "publication_scope": "PARTIAL_COMPONENTS_ONLY",
    }
