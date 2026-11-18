"""主动基金月评的真实合同复合基准构造，不推断缺失价格或合同权重。"""

from bisect import bisect_right
from copy import deepcopy
from datetime import date
from decimal import Decimal, localcontext
from pathlib import Path


def contract_composite(dates, prices, segments, *, carry_indices=()):
    """按当日生效合同逐日加权并连乘；仅明确许可的跨市场指数可向前取价。"""
    if not dates or dates != sorted(set(dates)) or not segments:
        raise ValueError("MONTHLY_COMPOSITE_DATES_OR_CONTRACT_MISSING")
    ordered = sorted(segments, key=lambda item: item["from"])
    if ordered[0]["from"] > dates[0] or len({s["from"] for s in ordered}) != len(ordered):
        raise ValueError("MONTHLY_COMPOSITE_CONTRACT_COVERAGE")
    for segment in ordered:
        if not segment.get("source_refs"):
            raise ValueError("MONTHLY_COMPOSITE_CONTRACT_SOURCE_MISSING")
        weights = [Decimal(str(v)) for v in segment["weights"].values()]
        if any(not v.is_finite() or v < 0 for v in weights) or sum(weights) != 1:
            raise ValueError("MONTHLY_COMPOSITE_CONTRACT_WEIGHTS")
    calendars = {code: sorted(values) for code, values in prices.items()}
    carried = []

    def price(code, day):
        if code not in prices:
            raise ValueError("MONTHLY_COMPOSITE_INDEX_MISSING:" + code)
        actual = day
        if day not in prices[code] and code in carry_indices:
            pos = bisect_right(calendars[code], day) - 1
            if pos >= 0:
                actual = calendars[code][pos]
                carried.append({"index": code, "valuation_date": day, "price_date": actual})
        if actual not in prices[code]:
            raise ValueError("MONTHLY_COMPOSITE_PRICE_GAP:" + code + ":" + day)
        value = Decimal(str(prices[code][actual]))
        if not value.is_finite() or value <= 0:
            raise ValueError("MONTHLY_COMPOSITE_PRICE_INVALID")
        return value

    with localcontext() as context:
        context.prec = 70
        levels = [Decimal(1)]
        for previous, current in zip(dates, dates[1:]):
            segment = max((s for s in ordered if s["from"] <= current), key=lambda s: s["from"])
            change = sum(
                Decimal(str(weight)) * (price(code, current) / price(code, previous) - 1)
                for code, weight in segment["weights"].items()
                if Decimal(str(weight))
            )
            levels.append(levels[-1] * (1 + change))
    return [str(value) for value in levels], carried


def rolling_active_bundle(bundle, history, *, provider_ref, benchmark_ref, benchmark_sources,
                          management_review, verified_at):
    """只接收已阅读的新期管理证据和真实历史；不继承旧期已完成标记冒充本期审核。"""
    result = deepcopy(bundle)
    inputs = result["evidence_v1_calculation"]["inputs"]
    cutoff = history["dates"][-1]
    same_partial = cutoff == inputs["as_of_date"] and result["evidence_v1_calculation"].get("partial_release_only") is True
    if (cutoff <= inputs["as_of_date"] and not same_partial) or len(history["month_ends"]) != 37:
        raise ValueError("MONTHLY_ACTIVE_NEW_PERIOD_HISTORY_REQUIRED")
    if management_review.get("as_of_date") != cutoff or not management_review.get("current_period_review"):
        raise ValueError("MONTHLY_ACTIVE_NEW_PERIOD_MANAGEMENT_REVIEW_REQUIRED")
    review = management_review["current_period_review"]
    source_hash = review.get("source_hash")
    if (
        source_hash not in management_review.get("sources", {})
        or not review.get("role_table_quote")
        or not review.get("expected_active_names")
        or not any(claim.get("source_hash") == source_hash and claim.get("quote") == review["role_table_quote"]
                   for claim in management_review.get("claims", []))
        or date.fromisoformat(review["report_period_end"]) > date.fromisoformat(cutoff)
        or date.fromisoformat(review["report_published_on"]) > date.fromisoformat(cutoff)
        or management_review["continuity"]["period_start"] != history["dates"][0]
        or management_review["continuity"]["period_end"] != cutoff
    ):
        raise ValueError("MONTHLY_ACTIVE_CURRENT_ROLE_EVIDENCE_OR_WINDOW_MISMATCH")
    result.update(as_of_date=cutoff, verified_at=verified_at)
    result["data_dates"]["nav"] = cutoff
    result["data_dates"]["manager"] = management_review["current_period_review"]["report_period_end"]
    inputs.update(history=deepcopy(history), as_of_date=cutoff, verified_at=verified_at,
                  management_inputs=deepcopy(management_review))
    inputs["target_refs"]["provider"] = provider_ref["sha256"]
    inputs["target_refs"]["benchmark"] = benchmark_ref["sha256"]
    refs = [provider_ref, benchmark_ref, *benchmark_sources, *management_review["sources"].values()]
    inputs["sources"].update({ref["sha256"]: ref for ref in refs})
    inputs["performance_source_hashes"] = [provider_ref["sha256"], benchmark_ref["sha256"],
                                            inputs["target_refs"]["half"]]
    return result


def build_monthly_inputs(code, previous_bundle, target, root, output, *, provider,
                         management_review, contract_review):
    """从本期审阅的合同和实际指数原件构造候选；净值、分数都重新计算。"""
    import json
    from datetime import datetime, timedelta, timezone

    from app.services.fund_exposure_common import now
    from app.services.fund_rating_management_inputs import _plain, _source_text
    from app.services.fund_rating_monthly_inputs import _derived, _index_daily
    from app.services.fund_rating_public_market import provider_history
    from app.services.fund_rating_reviewed_snapshot import _read

    if (previous_bundle["fund_code"] != code or contract_review.get("fund_code") != code
            or contract_review.get("as_of_date") != str(target)
            or management_review.get("fund_code") != code):
        raise ValueError("MONTHLY_ACTIVE_REVIEW_BINDING")
    _, history = provider_history(provider, date.fromisoformat(str(target)))
    current_role = management_review.get("current_period_review", {})
    role_source = management_review.get("sources", {}).get(current_role.get("source_hash"))
    if not role_source:
        raise ValueError("MONTHLY_ACTIVE_CURRENT_ROLE_ORIGINAL_REQUIRED")
    role_body = _source_text(root, role_source)
    if (not current_role.get("role_table_quote") or _plain(current_role["role_table_quote"]) not in role_body
            or code not in role_body or _plain(previous_bundle["identity"]["legal_name"]) not in role_body):
        raise ValueError("MONTHLY_ACTIVE_OWN_CURRENT_ROLE_ORIGINAL_REQUIRED")
    provider_ref = _derived(root, provider, label="monthly-provider-" + code)
    segments, refs, prices, carry = [], [], {}, []
    for segment in contract_review["segments"]:
        reference = segment["source"]
        body = _source_text(root, reference)
        quote = _plain(segment["quote"])
        if (not quote or quote not in body or not segment.get("reason")
                or _plain(previous_bundle["identity"]["legal_name"]) not in body
                or not reference.get("published_on") or reference["published_on"] > str(target)):
            raise ValueError("MONTHLY_ACTIVE_ACTUAL_CONTRACT_REVIEW_REQUIRED")
        refs.append(reference)
        segments.append({"from": segment["from"], "weights": segment["weights"], "source_refs": [reference["sha256"]]})
    for component in contract_review["components"]:
        index = component["index"]
        if component["format"] == "TUSHARE_INDEX_GLOBAL":
            from app.integrations.tushare_sprint_hk import fetch_hk

            if index not in {"HSI", "HKTECH"} or component.get("price_basis") != "CONTRACT_UNCONVERTED_CLOSE":
                raise ValueError("MONTHLY_ACTIVE_HK_CONTRACT_PRICE_BASIS_REQUIRED")
            records = []
            for year in range(date.fromisoformat(history["dates"][0]).year, date.fromisoformat(str(target)).year + 1):
                start = max(date(year, 1, 1), date.fromisoformat(history["dates"][0]))
                end = min(date(year, 12, 31), date.fromisoformat(str(target)))
                raw = json.loads(fetch_hk(index, start, end))
                ref = _derived(root, raw, label="monthly-hk-index-" + index + "-" + str(year))
                ref["url"] = "https://api.tushare.pro"
                refs.append(ref)
                records.extend(raw["data"]["items"])
            if any(row[0] != index for row in records):
                raise ValueError("MONTHLY_ACTIVE_HK_INDEX_IDENTITY")
            prices[index] = {
                str(date.fromisoformat(row[1])): str(row[2]) for row in records
            }
            carry.append(index)
            continue
        if component["format"] == "TUSHARE_INDEX_DAILY":
            reference = _index_daily(index, history["dates"][0], str(target), root=root)
            raw = _read(reference, root)
        else:
            reference = component["source"]
            raw = _read(reference, root)
        refs.append(reference)
        kind = component["format"]
        if kind == "TUSHARE_INDEX_DAILY":
            records = raw.get("rows", [])
            if not records or any(row[0] != index for row in records):
                raise ValueError("MONTHLY_ACTIVE_INDEX_IDENTITY")
            prices[index] = {str(row[1]): str(row[2]) for row in records}
        elif kind == "SW_OFFICIAL_DAILY":
            records = raw["data"]
            if raw.get("code") != "200" or any(row["swindexcode"] != index.removesuffix(".SI") for row in records):
                raise ValueError("MONTHLY_ACTIVE_SW_IDENTITY")
            prices[index] = {row["bargaindate"]: str(row["closeindex"]) for row in records}
        elif kind in {"CBA_CF_01", "CBA_TOTAL_QJ_00"} and index == kind:
            expected_id = "2c90818811afed8d0111c0c672b31578" if kind == "CBA_CF_01" else "2c9081e50e8767dc010e879387370009"
            if reference.get("params", {}).get("indexid") != expected_id or reference.get("url") not in {
                "https://indices.chinabond.com.cn/cbweb-mn/indices/singleIndexQueryResult",
                "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQueryResult",
            }:
                raise ValueError("MONTHLY_ACTIVE_CBA_INDEX_IDENTITY")
            metric = "CFZS_01" if kind == "CBA_CF_01" else "QJZS_00"
            prices[index] = {
                str(datetime.fromtimestamp(int(stamp) / 1000, timezone(timedelta(hours=8))).date()): str(value)
                for stamp, value in raw[metric].items() if value not in {None, "", "N/A"}
            }
        elif kind == "HK_OFFICIAL_HISTORY":
            records = raw["data"]
            if raw.get("code") != "0" or len(records) != 1 or records[0]["indexCode"] != component["official_index_code"]:
                raise ValueError("MONTHLY_ACTIVE_HK_IDENTITY")
            values = {row["date"]: Decimal(row["close"]) for row in records[0]["valueHistory"]}
            if component.get("currency") == "HKD" and component.get("fx_source"):
                import csv
                import zipfile

                fx_ref = component["fx_source"]
                if fx_ref.get("url") != "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip":
                    raise ValueError("MONTHLY_ACTIVE_ECB_FX_IDENTITY")
                _read(fx_ref, root)  # 先校核同版本原件摘要，不把任意本地表当汇率。
                refs.append(fx_ref)
                rates = {}
                with zipfile.ZipFile(root / fx_ref["file"]) as archive:
                    for name in archive.namelist():
                        for row in csv.DictReader(archive.read(name).decode("utf-8-sig").splitlines()):
                            if row.get("CNY") not in {None, "", "N/A"} and row.get("HKD") not in {None, "", "N/A"}:
                                rates[row["Date"]] = Decimal(row["CNY"]) / Decimal(row["HKD"])
                with localcontext() as context:
                    context.prec = 70
                    prices[index] = {
                        day: str(values[max(d for d in values if d <= day)] * rates[max(d for d in rates if d <= day)])
                        for day in history["dates"]
                    }
            elif component.get("currency") == "CNY":
                prices[index] = {day: str(value) for day, value in values.items()}
                carry.append(index)
            else:
                raise ValueError("MONTHLY_ACTIVE_FX_SOURCE_ADAPTER_REQUIRED")
        else:
            raise ValueError("MONTHLY_ACTIVE_COMPONENT_ADAPTER_REQUIRED:" + kind)
    try:
        history["benchmark_index"], carried = contract_composite(history["dates"], prices, segments, carry_indices=carry)
    except ValueError as error:
        if not str(error).startswith("MONTHLY_COMPOSITE_PRICE_GAP:"):
            raise
        # 缺一个真实基准日只阻断绩效四维；不填价、不删日，其余原件维度可独立复核。
        history["benchmark_index"], carried = [], []
        history["benchmark_missing"] = {
            "reason": str(error), "no_forward_fill": True, "no_dates_removed": True,
            "performance_dimensions_remain_null": ["return", "risk", "efficiency", "stability"],
        }
    history["publisher_cross_check"] = {
        "mode": "CONTRACT_DERIVED_COMPOSITE_REFERENCE", "effective_segments": segments,
        "benchmark_source_hashes": [ref["sha256"] for ref in refs], "carried_price_dates": carried,
    }
    benchmark_ref = _derived(root, {"history_dates": history["dates"], "benchmark_index": history["benchmark_index"],
                                    "proof": history["publisher_cross_check"]}, label="monthly-contract-composite-" + code)
    candidate = rolling_active_bundle(previous_bundle, history, provider_ref=provider_ref, benchmark_ref=benchmark_ref,
                                     benchmark_sources=refs, management_review=management_review, verified_at=now().isoformat())
    from app.services.fund_rating_evidence_preview import preview

    checked = preview(candidate, root, review_required=False)
    destination = Path(output) / "candidate-bundle.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_suffix(".pending")
    pending.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(destination)
    return {"fund_code": code, "as_of_date": str(target), "candidate_path": str(destination.resolve()), "preview": checked}
