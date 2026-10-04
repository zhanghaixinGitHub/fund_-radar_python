"""仅解析此前冻结的量价原件，核对横截面与上游选择风险，生成截至2025年的输入。"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import date, timedelta

import numpy as np

from scripts import fund_002112_development_timing_audit_v1 as timing

io = timing.io
ROOT = io.RESEARCH / "price-volume-development/20261001-v1"
NE = [
    "nav_return_1d",
    "nav_return_2d",
    "nav_return_10d",
    "nav_return_40d",
    "nav_volatility_5d",
    "nav_volatility_60d",
    "nav_downside_volatility_20d",
]
ME = [
    f"{code}_{field}"
    for code in timing.old.MARKETS
    for field in ("return_20d", "volatility_5d", "volatility_20d", "amount_vs_previous20", "volume_vs_previous20")
]


def nav_extra(values: list[float]) -> list[float]:
    """61个已知单位净值，收益/波动均为小数比例；不使用目标日净值或复权因子。"""
    v = np.asarray(values, dtype=float)
    if len(v) != 61 or not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("INVALID_NAV_FEATURE_WINDOW")
    returns = v[1:] / v[:-1] - 1
    return [float(v[-1] / v[-1 - n] - 1) for n in (1, 2, 10, 40)] + [
        float(np.std(returns[-5:], ddof=0)),
        float(np.std(returns, ddof=0)),
        float(np.sqrt(np.mean(np.minimum(returns[-20:], 0) ** 2))),
    ]


def market_extra(quotes: list[dict]) -> list[float]:
    """21个已收盘样本，量额只计算同供应商同字段比值，绝对单位未知不擅自换算。"""
    if len(quotes) != 21:
        raise ValueError("INCOMPLETE_MARKET_WINDOW")
    values = np.asarray([[float(q[k]) for k in ("pct_chg", "amount", "vol")] for q in quotes])
    if not np.isfinite(values).all() or (values[:, 1:] < 0).any():
        raise ValueError("INVALID_MARKET_VALUES")
    rates = values[:, 0] / 100
    denominators = values[:-1, 1:].mean(axis=0)
    if (denominators <= 0).any():
        raise ValueError("ZERO_MARKET_ACTIVITY_BASE")
    return [
        float(math.prod(1 + v for v in rates[-20:]) - 1),
        float(np.std(rates[-5:], ddof=0)),
        float(np.std(rates[-20:], ddof=0)),
        *(values[-1, 1:] / denominators - 1).tolist(),
    ]


def prepare() -> dict:
    """来源审查和特征生成只看原始输入，不访问开发评分或2026标签。"""
    if (ROOT / "source-audit.json").exists():
        return io.read(ROOT / "source-audit.json")
    original = timing.old.DEFAULT_ROOT
    manifest = io.read(original / "source-manifest.json")
    bound = {v["sha256"]: v for v in manifest["sources"]}
    used = {}

    def source(sha):
        record = bound[sha]
        path = original / record["snapshot_path"]
        if sha not in used:
            if io.sha(path) != sha:
                raise ValueError("FROZEN_SOURCE_BYTES_CHANGED")
            used[sha] = {"snapshot": str(path), "original_path": record["source_path"], "sha256": sha}
        return path

    provenance_sha = next(
        v["sha256"]
        for v in manifest["sources"]
        if v["source_path"].endswith("recent-input-completion\\20260930-v1\\input-provenance.json")
    )
    provenance = io.read(source(provenance_sha))
    daily = []
    previous_codes = None
    for day, meta in sorted(provenance["stock_dates"].items()):
        if day > "2025-12-31":
            continue
        payload = io.read(source(meta["raw_sha256"]))["data"]
        fields = payload["fields"]
        rows = [dict(zip(fields, v, strict=True)) for v in payload["items"]]
        codes = [v["ts_code"] for v in rows]
        count = Counter(codes)
        wrong = sum(timing.normalize_date(v["trade_date"]) != day for v in rows)
        if wrong or len(count) != len(codes):
            raise ValueError("RAW_DAILY_DATE_OR_DUPLICATE_CONFLICT")
        cache = io.read(source(meta["cache_sha256"]))
        cp = cache.get("payload", cache)
        current = set(codes)
        daily.append(
            {
                "date": day,
                "source_sha256": meta["raw_sha256"],
                "raw_rows": len(rows),
                "distinct_codes": len(current),
                "duplicate_rows": 0,
                "wrong_dates": wrong,
                "response_has_more": payload.get("has_more"),
                "response_declared_count": payload.get("count"),
                "zero_or_missing_volume": sum(v.get("vol") in (None, 0) for v in rows),
                "zero_or_missing_amount": sum(v.get("amount") in (None, 0) for v in rows),
                "missing_pct": sum(v.get("pct_chg") is None for v in rows),
                "entered_since_previous_response": None if previous_codes is None else len(current - previous_codes),
                "absent_since_previous_response": None if previous_codes is None else len(previous_codes - current),
                "missing_reason": cp.get("missing_reason"),
                "return_basis": cp.get("return_basis"),
                "cache_observed_rows": len(cp.get("rows", {})),
                "received_at": meta.get("received_at"),
                "units": "NOT_EXPLICITLY_DOCUMENTED_IN_FROZEN_RESPONSE",
                "scope": "RAW_PROVIDER_DATE_CROSS_SECTION_NO_CURRENT_SURVIVOR_OR_HOLDING_FILTER",
            }
        )
        previous_codes = current
    io.save_lines(ROOT / "daily-cross-section-audit.jsonl", daily)
    io.save(
        ROOT / "breadth-admission.json",
        {
            "at": io.now(),
            "decision": "SKIP_FULL_MARKET_BREADTH_IN_THIS_STAGE",
            "days": len(daily),
            "min_returned_rows": min(r["raw_rows"] for r in daily),
            "max_returned_rows": max(r["raw_rows"] for r in daily),
            "all_raw_dates_and_codes_checked": True,
            "today_listing_or_future_holding_filter_used": False,
            "reasons": [
                "NO_COMPLETE_CONTEMPORANEOUS_LISTED_SUSPENDED_DELISTED_UNIVERSE_PROOF",
                "MISSING_SOURCE_ROWS_HAVE_UNVERIFIED_CAUSE",
                "HAS_MORE_FALSE_WITH_COUNT_ZERO_IS_NOT_INDEPENDENT_COMPLETENESS_PROOF",
                "HISTORICAL_REVISION_AND_ABSOLUTE_UNITS_UNPROVEN",
            ],
            "universe_changes_not_called_new_listings_or_delistings_without_evidence": True,
            "no_forward_adjustment_factor_used": True,
        },
    )

    # 只使用本轮以前已经冻结的语料元数据判断选择偏差，不再次抽取或新增材料。
    frozen = io.Frozen(timing.event.OLD)
    materials = frozen.entry("materials")["payload"]
    public = frozen.entry("public")
    history = frozen.entry("historical")
    catalogs = frozen.entry("catalogs")
    io.save(
        ROOT / "corpus-selection-audit.json",
        {
            "at": io.now(),
            "frozen_inventory_sha256": io.sha(timing.event.OLD / "source-inventory.json"),
            "material_notice": materials.get("notice"),
            "material_as_of": materials.get("asOfDate"),
            "historical_companies": sorted({r.get("company") for r in history["rows"] if r.get("company")}),
            "public_source_rows": len(public["rows"]),
            "public_catalog_columns": len(catalogs["columns"]),
            "collection_basis": "MATERIALS_EXPLICITLY_COLLECTED_USING_DISCLOSED_HOLDINGS",
            "future_holding_influence_excluded": False,
            "decision": "QUARANTINE_COUNTS_AND_GLOBAL_FOR_THIS_STAGE",
            "reason": (
                "REMOVING_HOLDING_FIELDS_DOES_NOT_UNDO_UPSTREAM_SELECTION; point-in-time corpus universe unproven"
            ),
        },
    )
    old_reports = io.read(timing.event.ROOT / "reports.json")
    reviewed = io.read(timing.ROOT / "report-time-evidence.json")
    report_proof = []
    for item in reviewed:
        matches = [r for r in old_reports if r["raw"]["sha256"] == item["raw_sha256"]]
        receipts = [r["raw"].get("received_at") for r in matches if r["raw"].get("received_at")]
        report_proof.append(
            {
                "sha256": item["raw_sha256"],
                "matching_existing_reports": len(matches),
                "earliest_known_receipt": min(receipts) if receipts else None,
                "declared_publication": item["declared_publication"],
                "cms_bound": item["conservative_version_bound"],
                "same_version_historically_observed": False,
            }
        )
    io.save(
        ROOT / "report-cross-evidence.json",
        {
            "at": io.now(),
            "search_scope": "EXISTING_FROZEN_64_REPORTS_AND_PREVIOUS15_ORIGINAL_VERSION_REVIEWS",
            "reports": report_proof,
            "decision": "CONTINUE_H_F_AND_ALL_DEPENDENT_FEATURE_QUARANTINE",
            "no_claim_of_first_publication_in_2026": True,
            "new_archive_or_network_lookup": False,
        },
    )

    # 指数原件中有未被旧归一化快照保留的较早量价行，仍须从已绑定摘要读取，不能访问后来缓存。
    markets = {code: {} for code in timing.old.MARKETS}
    market_sources = {}
    for code, meta in provenance["market"].items():
        hashes = sorted({v["raw_sha256"] for v in meta["sources"]})
        for sha in hashes:
            for raw in timing.raw_quote_rows(source(sha)):
                if raw["ts_code"] != code:
                    continue
                day = timing.normalize_date(raw["trade_date"])
                values = {k: raw[k] for k in ("close", "pre_close", "pct_chg", "vol", "amount")}
                if day in markets[code] and markets[code][day] != values:
                    raise ValueError("MARKET_RAW_VERSION_CONFLICT")
                markets[code][day] = values
        market_sources[code] = {
            "source_hashes": hashes,
            "days": len(markets[code]),
            "first": min(markets[code]),
            "last": max(markets[code]),
            "absolute_units_verified": False,
            "new_features_units": "DIMENSIONLESS_RATIO_OR_RETURN",
        }
    old_input = io.lines(timing.ROOT / "inputs.jsonl")
    nav = io.read(timing.ROOT / "nav-through-2025.json")
    calendar_sha = next(
        v["sha256"] for v in manifest["sources"] if v["source_path"].endswith("cn_a_share_2021_2025_v1.json")
    )
    calendar = io.read(source(calendar_sha))
    closed = [(start, end) for year in calendar["years"] for start, end in year["closed_ranges"]]
    sessions = []
    day = date(2021, 1, 1)
    while day <= date(2025, 12, 31):
        key = day.isoformat()
        if day.weekday() < 5 and not any(start <= key <= end for start, end in closed):
            sessions.append(key)
        day += timedelta(days=1)
    output = []
    lineage = []
    for row in old_input:
        if row["target"] > "2025-12-31":
            raise ValueError("POST_2025_ROW_FORBIDDEN")
        origin = row["base"]
        idx = sessions.index(origin)
        end, lag, base = timing.old.choose_nav_window(sessions, nav, origin)
        assert np.allclose(base, row["groups"]["N"], atol=1e-13, rtol=0)
        end_idx = sessions.index(end)
        ndays = sessions[end_idx - 60 : end_idx + 1]
        ne = nav_extra([float(nav[d]["unit_nav"]) for d in ndays])
        quote_days = sessions[idx - 21 : idx]
        if len(quote_days) != 21 or max(quote_days) >= origin:
            raise ValueError("MARKET_WINDOW_FUTURE")
        me = []
        for code in timing.old.MARKETS:
            me.extend(market_extra([markets[code][d] for d in quote_days]))
        output.append(
            {
                **{k: row[k] for k in ("target", "base", "as_of", "session_index", "label_mature_at")},
                "groups": {"N": row["groups"]["N"], "M": row["groups"]["M"], "NE": ne, "ME": me},
            }
        )
        lineage.append(
            {
                "target": row["target"],
                "origin": origin,
                "as_of": row["as_of"],
                "nav_end": end,
                "nav_source_dates": ndays,
                "market_source_dates": quote_days,
                "nav_lag": lag,
                "first_publication_proven": False,
            }
        )
    io.save_lines(ROOT / "inputs.jsonl", output)
    io.save(ROOT / "nav-through-2025.json", nav)
    io.save(ROOT / "market-values-through-2025.json", markets)
    io.save_lines(ROOT / "feature-lineage.jsonl", lineage)
    io.save(
        ROOT / "source-manifest.json",
        {
            "at": io.now(),
            "prior_source_manifest_sha256": io.sha(original / "source-manifest.json"),
            "sources": used,
            "market_metadata": market_sources,
            "new_nav_columns": NE,
            "new_market_columns": ME,
        },
    )
    result = {
        "at": io.now(),
        "input_rows": len(output),
        "development_rows": sum(r["target"].startswith("2025") for r in output),
        "all_new_features_finite": bool(all(np.isfinite(r["groups"]["NE"] + r["groups"]["ME"]).all() for r in output)),
        "new_feature_dimensions": {"NE": len(NE), "ME": len(ME)},
        "source_files": len(used),
        "new_external_requests": 0,
        "breadth_admitted": False,
        "events_admitted": False,
        "holdings_admitted": False,
        "eligible_for": "RECONSTRUCTED_HISTORY_DEVELOPMENT_ONLY_WITH_DECLARED_TIMING_LIMITATIONS",
    }
    io.save(ROOT / "source-audit.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))
