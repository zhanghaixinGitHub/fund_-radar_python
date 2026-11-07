"""万家006730增强基金稳定性原件编译；沿用冻结规则。

每次从本份额provider原快照、官网曲线、分红原件及年度合同重建，
只复用R5稳定性36月/25滚动年、30/70及经济宽容带，不计算被动跟踪项。
"""

import json
import re
from datetime import date
from decimal import Decimal as D
from decimal import localcontext
from pathlib import Path

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_quality_v2_paired_history import benchmark_records, paired_history
from app.services.fund_rating_public_market import compact, provider_history, public_document, publication
from app.services.fund_rating_public_reconciled import reconcile
from app.services.fund_rating_r5 import PROTOCOL, _calculate_frozen, window_frequency_score
from app.services.fund_rating_r5_rules import check_refs
from app.services.fund_rating_reviewed_quality import ACTIVE_BASIS
from app.services.material_content import content_digest, source_bytes, source_exists

LEGACY_VERSION = "WJ_ACTIVE_STABILITY_ORIGINALS_CANDIDATE_20261029_1"
VERSION = "WJ_ACTIVE_STABILITY_ORIGINALS_20261029_2"
RETURN_ESTIMAND = "R5_FROZEN_DISCLOSED_NAV_GROWTH_20261019"


def _disclosed_window(days, units, cash, a, b, *, issuer=None):
    """冻结净值增长及公开四位净值的保守舍入区间，不以发行人曲线代替。

    端点NAV各为公开值±0.00005；精确公告现金不加猜测误差。
    x/(x-cash)随x递减，故事件乘数上下界必须反向取前日NAV边界。
    即使同一NAV同时参与端点和事件，也独立放宽区间，不利用相关性缩窄。
    """
    quantum = D(".00005")
    if not isinstance(cash, dict):
        raise ValueError("WJ_FROZEN_CASH_EVENT_SCOPE_UNKNOWN")
    if any(not isinstance(day, str) or day not in days for day in cash):
        raise ValueError("WJ_FROZEN_CASH_EVENT_DATE_NOT_CLOSED")
    if not 0 <= a < b < len(days):
        raise ValueError("WJ_FROZEN_WINDOW_SCOPE")
    if any(v <= quantum or v.quantize(D(".0001")) != v for v in units.values()):
        raise ValueError("WJ_FROZEN_PUBLIC_NAV_PRECISION_OR_POSITIVITY")
    start, end = units[days[a]], units[days[b]]
    value = end / start
    lower = (end - quantum) / (start + quantum)
    upper = (end + quantum) / (start - quantum)
    events = []
    for day, amount in sorted(cash.items()):
        if not days[a] < day <= days[b]:
            continue
        if day not in days or days.index(day) == 0:
            raise ValueError("WJ_FROZEN_EVENT_PREVIOUS_TRADING_DAY_MISSING")
        previous_day = days[days.index(day) - 1]
        previous = units[previous_day]
        if amount < 0 or previous - quantum <= amount:
            raise ValueError("WJ_FROZEN_CASH_OR_PREVIOUS_NAV_BOUND")
        factor = previous / (previous - amount)
        factor_lower = (previous + quantum) / (previous + quantum - amount)
        factor_upper = (previous - quantum) / (previous - quantum - amount)
        value *= factor
        lower *= factor_lower
        upper *= factor_upper
        events.append(
            {
                "ex_date": day,
                "cash_per_share": str(amount),
                "previous_day": previous_day,
                "previous_nav": str(previous),
                "factor": str(factor),
                "factor_lower": str(factor_lower),
                "factor_upper": str(factor_upper),
            }
        )
    detail = {
        "formula": "NAVend/NAVstart*PRODUCT[previous_NAV/(previous_NAV-cash)]-1",
        "unit_nav_half_quantum": str(quantum),
        "cash_precision": "EXACT_ORIGINAL_DECLARED_AMOUNT",
        "start_unit_nav": str(start),
        "end_unit_nav": str(end),
        "events": events,
        "bound_is_conservative_independent_factors": True,
        "raw_unit_nav_lower_ratio": str(lower - 1),
        "raw_unit_nav_upper_ratio": str(upper - 1),
        "issuer_interval_used": False,
    }
    # 只有区间内无实际分红/拆分时，两种增长定义才同为NAVend/NAVstart。
    # 区间外历史现金因子在比值中约去；原reconcile先核全部事件和每日双源。
    # 有事件窗口绝不借该曲线缩窄冻结式区间，更不反推任何缺失净值。
    if issuer is not None and not events:
        if not all(day in issuer for day in [days[a], days[b]]):
            raise ValueError("WJ_FROZEN_NO_EVENT_ISSUER_DATES_MISSING")
        x, y = issuer[days[a]], issuer[days[b]]
        epsilon = D(".0000005")
        if min(x, y) <= epsilon:
            raise ValueError("WJ_FROZEN_NO_EVENT_ISSUER_BOUND_RANGE")
        issuer_lower = (y - epsilon) / (x + epsilon)
        issuer_upper = (y + epsilon) / (x - epsilon)
        lower, upper = max(lower, issuer_lower), min(upper, issuer_upper)
        if lower > upper:
            raise ValueError("WJ_FROZEN_NO_EVENT_SOURCES_DISAGREE")
        detail.update(
            issuer_interval_used=True,
            no_cash_window_same_estimand_algebraically=True,
            issuer_start_wealth=str(x),
            issuer_end_wealth=str(y),
            issuer_percent_factor_half_quantum=str(epsilon),
            issuer_lower_ratio=str(issuer_lower - 1),
            issuer_upper_ratio=str(issuer_upper - 1),
            no_values_imputed_or_quantum_reduced=True,
        )
    return value - 1, lower - 1, upper - 1, detail


def calculate(inputs: dict, *, root: Path) -> dict:
    """核所有源字节和身份再生成窗口，精度可能跨分档时拒绝点值。"""
    root = root.resolve()
    if (
        inputs.get("version") not in {LEGACY_VERSION, VERSION}
        or inputs.get("fund_code") != "006730"
        or inputs.get("share_class") != "C"
        or inputs.get("basis") != ACTIVE_BASIS
        or inputs.get("scope") != "SINGLE_STABILITY_DIMENSION_ONLY_TOTAL_AND_GRADE_NULL"
    ):
        raise ValueError("WJ_STABILITY_IDENTITY_OR_SCOPE")
    if any(k in inputs for k in ["score", "observations", "quality_metrics", "packet"]):
        raise ValueError("WJ_STABILITY_RAW_ORIGINALS_REQUIRED")
    frozen = inputs["version"] == VERSION
    if frozen and inputs.get("return_estimand") != RETURN_ESTIMAND:
        raise ValueError("WJ_STABILITY_EXPLICIT_FROZEN_ESTIMAND_REQUIRED")
    code, asof = inputs["fund_code"], inputs["as_of_date"]
    raw = {}
    refs = inputs["refs"]
    source_refs = {}
    for digest, ref in inputs["sources"].items():
        p = (root / ref["file"]).resolve()
        if not p.is_relative_to(root / "sources") or not source_exists(p):
            raise ValueError("WJ_STABILITY_SOURCE_SCOPE")
        content = source_bytes(p)
        if len(content) > 25_000_000 or content_digest(content) != digest or ref["sha256"] != digest:
            raise ValueError("WJ_STABILITY_SOURCE_BYTES_CHANGED")
        if ref.get("published_on") and ref["published_on"] > asof:
            raise ValueError("WJ_STABILITY_FUTURE_DOCUMENT")
        raw[digest] = content
    for label in ["provider", "benchmark", "dividend_2025", "dividend_2026", "half"]:
        h = refs[label]
        source_refs[label] = {**inputs["sources"][h], "sha256": h}
    provider = json.loads(raw[refs["provider"]])
    if (
        provider.get("fund_code") != code
        or provider.get("source", {}).get("enabled") is not True
        or not provider.get("source", {}).get("authorization_verified_at")
    ):
        raise ValueError("WJ_STABILITY_PROVIDER_IDENTITY_OR_AUTHORIZATION")
    _, history = provider_history(provider, date.fromisoformat(asof))
    if (
        len(history["dates"]) != 727
        or history["dates"][0] != "2023-09-28"
        or history["dates"][-1] != asof
        or len(history["month_ends"]) != 37
    ):
        raise ValueError("WJ_STABILITY_COMPLETE_THREE_YEAR_WINDOW")
    pages, _ = public_document(raw[refs["half"]], source_refs["half"], code, "中期报告")
    original = compact("\n".join(pages))
    if (
        compact(inputs["legal_name"]) not in original
        or code not in original
        or compact(inputs["manager_name"]) not in original
        or str(report_period("\n".join(pages))[1]) != inputs["holding_date"]
        or publication(pages) > asof
        or "本基金为指数增强型股票基金" not in original
    ):
        raise ValueError("WJ_STABILITY_PRIMARY_PRODUCT_OR_PERIOD")
    expression = r"业绩比较基准中证500指数收益率[*×]95%[+＋]一年期人民币定期存款利率[（(]税后[）)][*×]5%"
    contracts = {}
    if set(inputs["annual_reports"]) != {"2023", "2024", "2025"}:
        raise ValueError("WJ_STABILITY_ANNUAL_CONTRACT_COVERAGE")
    for year, digest in inputs["annual_reports"].items():
        doc = json.loads(raw[digest])["data"]
        text = compact(doc["notice_content"])
        if (
            compact(inputs["legal_name"]) not in text
            or code not in text
            or year + "年年度报告" not in compact(doc["notice_title"])
            or str(report_period(doc["notice_content"])[1]) != year + "-12-31"
            or str(doc["notice_date"])[:10] > asof
            or not re.search(expression, text)
        ):
            raise ValueError("WJ_STABILITY_ANNUAL_CONTRACT_IDENTITY_OR_BENCHMARK")
        contracts[year] = {
            "source_hash": digest,
            "quote": re.search(expression, text)[0],
            "published_on": str(doc["notice_date"])[:10],
        }
    if not re.search(expression, original):
        raise ValueError("WJ_STABILITY_CURRENT_BENCHMARK")
    records, adapter = benchmark_records(raw[refs["benchmark"]], source_refs["benchmark"], fund_code=code)
    # 已有严格对账适配逐笔重读2025C类利润分配表和2026C类公告金额、除息日，
    # 再将全部727个真实单位净值与官网累计再投资收益核对。供应商旧复权值保留。
    issuer = {d: {"syNavRate": row["fund"]} for d, row in records.items()}
    corrected = reconcile(history, issuer, provider, source_refs, root, code, date.fromisoformat(asof))
    paired = paired_history(corrected, raw[refs["benchmark"]], source_refs["benchmark"], fund_code=code)
    days, ends = paired["dates"], paired["month_ends"]
    month_ids = [date.fromisoformat(d).year * 12 + date.fromisoformat(d).month for d in ends]
    if any(b - a != 1 for a, b in zip(month_ids[:-1], month_ids[1:], strict=True)):
        raise ValueError("WJ_STABILITY_MONTH_GAP")
    # V2先完整执行旧双源/分红核验，再独立取真实单位净值；不改写发行人或供应商曲线。
    units = {r["nav_date"]: D(str(r["unit_nav"])) for r in history["rows"]}
    cash = {d: D(v) for d, v in corrected["return_source_reconciliation"]["cash_events"].items()}
    observations = {}
    precision = {}
    with localcontext() as ctx:
        ctx.prec = 70
        nav = list(map(D, paired["adjusted_nav"]))
        bench = list(map(D, paired["benchmark_index"]))
        # 发行人基准及V1现金再投资曲线为百分数四位小数，因子半量子5e-7。
        # V2自身增长使用单位净值四位小数半量子.00005，经端点/事件逐项传播。
        # 任一保守区间跨R5档即拒绝；不能沿用旧百分数精度或取区间中点消歧。
        epsilon = D(".0000005")
        for key, step in [("monthly_success", 1), ("annual_success", 12)]:
            unit = D(PROTOCOL["parameters"]["annual_return_gap_unit"]) / (12 if step == 1 else 1)
            rows = []
            checks = []
            for i in range(step, 37):
                a, b = map(days.index, [ends[i - step], ends[i]])
                target_proof = None
                if frozen:
                    target, low_target, high_target, target_proof = _disclosed_window(
                        days, units, cash, a, b, issuer=dict(zip(days, nav, strict=True))
                    )
                else:
                    # V1仅用于已归档快照原样重放；发行人现金再投资收益不冒称冻结统计式。
                    target = nav[b] / nav[a] - 1
                    low_target = (nav[b] - epsilon) / (nav[a] + epsilon) - 1
                    high_target = (nav[b] + epsilon) / (nav[a] - epsilon) - 1
                reference = bench[b] / bench[a] - 1
                low_ref = (bench[b] - epsilon) / (bench[a] + epsilon) - 1
                high_ref = (bench[b] + epsilon) / (bench[a] - epsilon) - 1
                lower = low_target - high_ref
                upper = high_target - low_ref
                lo, hi = window_frequency_score(lower, unit), window_frequency_score(upper, unit)
                if lo != hi:
                    raise ValueError("WJ_STABILITY_PRECISION_AMBIGUOUS:" + ends[i])
                if frozen and window_frequency_score(target - reference, unit) != lo:
                    # 不用原件区间中点或有利代表数掩盖数值代表与已证分档的冲突。
                    raise ValueError("WJ_STABILITY_REPRESENTATIVE_CONFLICT:" + ends[i])
                rows.append({"start": days[a], "end": days[b], "target": str(target), "reference": str(reference)})
                checks.append(
                    {
                        "start": days[a],
                        "end": days[b],
                        "delta_lower": str(lower),
                        "delta_upper": str(upper),
                        "invariant_score": str(lo),
                        **(
                            {
                                "target_ratio_lower": str(low_target),
                                "target_ratio_upper": str(high_target),
                                "benchmark_ratio_lower": str(low_ref),
                                "benchmark_ratio_upper": str(high_ref),
                                "target_interval_proof": target_proof,
                            }
                            if frozen
                            else {}
                        ),
                    }
                )
            observations[key] = {
                "kind": "windows",
                "expected_windows": len(rows),
                "rows": rows,
                "source_refs": list(raw),
            }
            precision[key] = checks
    registry = {
        h: {"path": str(root / r["file"]), "published_on": r.get("published_on")} for h, r in inputs["sources"].items()
    }
    for node in observations.values():
        check_refs(node, registry, asof)
    packet = {
        "fund_code": code,
        "as_of_date": asof,
        "basis": ACTIVE_BASIS,
        "unit_contract": "RATIO_RANKS_RATIO_WINDOWS",
        "quality_public": {"dataDates": {"nav": asof, "holdings": inputs["holding_date"]}},
        "quality_metrics": {
            "observations": observations,
            "target_product_aliases": ["006729", "006730", "PUBLIC:006729"],
            "material_event_pending": False,
        },
        "r4_evidence_registry": registry,
        "raw_archive": [{"kind": "ORIGINAL", "sha256": h} for h in raw],
    }
    result = _calculate_frozen(packet, PROTOCOL, {}, selected_dimensions={"stability"})
    if any(result[k] is not None for k in ["score", "raw_score", "grade"]):
        raise ValueError("WJ_STABILITY_COMPLETE_TOTAL_LEAK")
    return {
        "score": result["dimensions"]["stability"],
        "dimensions": result["dimensions"],
        "complete_score": None,
        "grade": None,
        "calculation_revision": PROTOCOL["revision"],
        "observations": observations,
        "standard_replay": result,
        "audit": {
            **(
                {
                    "return_estimand": RETURN_ESTIMAND,
                    "issuer_cash_curve_scope": "RECONCILIATION_AND_NO_EVENT_ALGEBRAICALLY_EQUIVALENT_INTERVAL_ONLY",
                    "statistic_protocol_sha256": "771606ecb96ebca237b1b58ccc09f0b1aba6d5a7763558ba0c4de7ae379a563c",
                }
                if frozen
                else {}
            ),
            "contracts": contracts,
            "cash_reconciliation": corrected["return_source_reconciliation"],
            "precision": precision,
            "precision_invariant_windows": 61,
            "common_dates": len(days),
            "month_ends": len(ends),
            "first": days[0],
            "last": days[-1],
            "adapter": adapter,
            "peer_required_for_stability": False,
            "other_dimensions_calculated": False,
        },
    }
