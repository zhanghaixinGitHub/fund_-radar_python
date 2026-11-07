"""冻结同行池的完整目录、法律产品去重与原十二季预算复核；不读取分数。"""

import importlib
import re
from decimal import Decimal as D

from app.services.fund_quality_v2_disclosures import report_period
from app.services.fund_rating_active_execution import MONEY, _assets

_s = importlib.import_module((__package__ + "." if __package__ else "") + "fund_rating_thematic_source")
plain, need = _s.plain, _s.require
RETURN = ["002450", "005585", "001714", "001628", "003053", "164205", "001223"]
STRICT = ["002450", "001714", "001628", "003053", "001223"]


def dec(value):
    need(value is not None and not isinstance(value, bool), "ADMISSION_NUMBER")
    result = D(str(value))
    need(result.is_finite(), "ADMISSION_FINITE")
    return result


def raw_admission(source, inputs, theme):
    a = source.json(inputs["admission_ref"])
    need(
        a.get("base_frozen_admission_ref") == "78ec1131e6bd9d54fd7dfdc8271332fe8fd50319471a5476f97e61393f62c39f",
        "FROZEN_ADMISSION_AUTHORITY",
    )
    base = source.json(a["base_frozen_admission_ref"])
    need(a["boundary_decisions"] == base["boundary_decisions"], "FROZEN_BOUNDARY_DECISIONS_CHANGED")
    need(
        a["return_products"] == RETURN and a["strict_products"] == STRICT and a["not_blind_calibration"] is True,
        "FROZEN_COHORT",
    )
    actual = {}
    rawcount = 0
    pages = {}
    for h in a["catalog_refs"]:
        rows = source.provider_rows(h, ["ts_code", "name", "found_date", "market", "status"], allow_more=True)
        rawcount += len(rows)
        ref = source.sources[h]
        params = (
            ref.get("request_params_without_credentials")
            or ref.get("params_without_credentials")
            or ref.get("params")
            or {}
        )
        key = (params.get("market"), params.get("status"))
        need(key in {("O", "L"), ("O", "D"), ("E", "L"), ("E", "D")}, "CATALOG_REQUEST_SCOPE")
        need(all(r["market"] == key[0] and r["status"] == key[1] for r in rows), "CATALOG_REQUEST_IDENTITY")
        pages.setdefault(key, []).append((int(params.get("offset", 0)), len(rows), source.json(h)["data"]["has_more"]))
        for r in rows:
            need(r["ts_code"] not in actual or actual[r["ts_code"]] == r, "CATALOG_CONFLICT")
            actual[r["ts_code"]] = r
    need(set(pages) == {("O", "L"), ("O", "D"), ("E", "L"), ("E", "D")}, "CATALOG_FOUR_SCOPES")
    for group in pages.values():
        offset = 0
        ordered = sorted(group)
        for i, (start, count, more) in enumerate(ordered):
            need(start == offset and count > 0 and more is (i < len(ordered) - 1), "CATALOG_COMPLETE_PAGE_CHAIN")
            offset += count
    wrapper = source.json(a["catalog_wrapper_ref"])
    expected = {r["ts_code"]: r for item in wrapper["items"] for r in item["records"]}
    need(actual == expected and rawcount == len(actual), "FULL_CATALOGUE_RAW_REBUILD")
    candidates = {
        r["ts_code"][:6]
        for r in actual.values()
        if re.search(a["keyword"], r["name"])
        and r.get("found_date")
        and r["found_date"] <= "20230928"
        and r.get("fund_type") in {"混合型", "股票型"}
        and "指数" not in (r.get("invest_type") or "")
    }
    decisions = {r["code"]: r for r in a["boundary_decisions"]}
    need(candidates == set(decisions), "FULL_THEME_DISCOVERY_BOUNDARY")
    aliases = {}
    for row in a["aliases"]:
        body = plain(source.body(row["source_hash"]))
        q = plain(row["quote"])
        need(q in body and row["alias"] in q and row["representative"] in q, "SAME_PRODUCT_ALIAS_ORIGINAL")
        aliases[row["alias"]] = row["representative"]
    need(
        all(code in aliases for code, r in decisions.items() if r["state"] == "SAME_PRODUCT_SHARE_EXCLUDED"),
        "NO_ALIAS_PENDING",
    )
    admitted = {c for c, r in decisions.items() if r["state"] == "THEME_ADMITTED_CANDIDATE"}
    need(admitted == set(RETURN), "ALL_ADMITTED_PRODUCTS")
    need(
        decisions["003397"]["state"] == decisions["018590"]["state"] == "SELF_PRODUCT_EXCLUDED",
        "TARGET_ALL_SHARES_EXCLUDED",
    )
    excluded_audit = []
    for code, row in decisions.items():
        if row["state"].startswith("SEPARATE_"):
            if row["state"] == "SEPARATE_ACTUAL_REGION" and code == "013132":
                # 原十二季HYPZ下文数值重建，不能由允许50%港股的权限反推实际。
                continue
            claims = a.get("exclusion_original_claims", {}).get(code)
            need(claims and row["reason"], "EXCLUSION_ORIGINAL_REASON:" + code)
            for claim in claims:
                source.claim(claim)
            if code == "164818":
                body = source.body(claims[0]["source_hash"])
                need(
                    code in plain(body)[:20000] and str(report_period(body)[1]) == "2026-06-30",
                    "INDEX_PASSIVE_LEGAL_PRODUCT_PERIOD",
                )
                need(
                    all(
                        word in plain(body)
                        for word in ["采用指数化投资", "完全复制策略", "本基金为指数型基金", "中证传媒指数收益率×95%"]
                    ),
                    "INDEX_PASSIVE_ACTUAL_MANDATE",
                )
            if code == "004424":
                body = source.body(claims[0]["source_hash"])
                need(str(report_period(body)[1]) == "2026-06-30", "004424_ACTUAL_REGION_PERIOD")
                assets, _ = _assets(body)
                stock = dec(assets["权益投资"]["amount_cny"])
                hongkong = re.search(r"港股公允价值为人民币\s*(" + MONEY + r")\s*元", body)
                need(hongkong and 0 < dec(hongkong[1].replace(",", "")) < stock, "004424_ACTUAL_HK_VALUE")
                domestic_upper = (stock - dec(hongkong[1].replace(",", ""))) / stock * 100
                need(domestic_upper < 80, "004424_ACTUAL_REGION_EXCLUSION")
                excluded_audit.append(
                    {
                        "code": code,
                        "stock_cny": str(stock),
                        "hongkong_cny": hongkong[1],
                        "domestic_upper_pct": str(domestic_upper),
                    }
                )
            excluded_audit.append(
                {"code": code, "state": row["state"], "manual_reason": row["reason"], "claims": claims}
            )
        if row["state"] == "TERMINATED_BEFORE_WINDOW":
            original = a["termination_originals"].get(code)
            need(
                original and original["published_on"] < "2023-09-28" and original["liquidation_start"] < "2023-09-28",
                "TERMINATION_BEFORE_WINDOW:" + code,
            )
            digest = original["source"]["sha256"]
            source.claim({"source_hash": digest, "quote": original["quote"]})
            body = plain(source.body(digest))
            need(code in body and "清算" in plain(original["quote"]), "TERMINATED_LEGAL_PRODUCT:" + code)
            excluded_audit.append({"code": code, "state": row["state"], "source_hash": digest, "original": original})
    for code, docs in a["contracts"].items():
        periods = set()
        for d in docs:
            text = source.body(d["source_hash"])
            p = report_period(text)
            periods.add(str(p[1]))
            n = plain(text)
            need(code in n[:20000], "PEER_LEGAL_IDENTITY:" + code)
            clauses = d.get("clauses", "")
            need(clauses and plain(clauses) in n, "PEER_LEGAL_QUOTE:" + code)
            for q in d.get("legal_notes", []):
                need(plain(q) in n, "PEER_LEGAL_NOTE")
            if d.get("strategy_change"):
                need(plain(d["strategy_change"]) in n, "PEER_STRATEGY_ORIGINAL")
            need(
                any(w in n for w in ["文化", "文体", "传媒"]) and any(w in n for w in ["娱乐", "体育"]),
                "PEER_ACTUAL_OPPORTUNITY_SET",
            )
        need(
            {"2023-12-31", "2024-12-31", "2025-12-31", "2026-06-30"} <= periods, "PEER_NECESSARY_LEGAL_PERIODS:" + code
        )
    target = {
        r["period"]: dec(r["stock_amount_cny"]) / dec(r["nav_cny"]) * 100
        for r in theme["report_ledger"]
        if r["kind"] == "QUARTER"
    }
    budgets = {}
    for code, refs in a["budget_refs"].items():
        points = {}
        for h in refs:
            raw = source.json(h)
            need(raw.get("ErrCode") == 0 and raw["Data"]["FundCode"] == code, "HYPZ_IDENTITY")
            for q in raw["Data"]["QuarterInfos"]:
                day = q["JZRQ"]
                if day not in target:
                    continue
                rows = q["HYPZInfo"]
                need(rows and all(str(r["BZDM"]) == code and r["FSRQ"] == day for r in rows), "HYPZ_ROWS")
                need(len({str(r["HYDM"]) for r in rows}) == len(rows), "HYPZ_DUPLICATE")
                domestic = [r for r in rows if re.fullmatch("[A-S]", str(r["HYDM"]))]
                foreign = [r for r in rows if re.fullmatch(r"\d+", str(r["HYDM"]))]
                need(len(domestic) + len(foreign) == len(rows), "HYPZ_REGION_CLASSIFICATION")
                domestic_sum = sum(dec(r["ZJZBL"]) for r in domestic)
                all_sum = sum(dec(r["ZJZBL"]) for r in rows)
                radius = D(".005") * len(rows)
                lower = max(D(0), domestic_sum - D(".005") * len(domestic)) / (all_sum + radius) * 100
                v = {"gap_upper_pp": abs(target[day] - all_sum) + radius, "region_lower_pct": lower, "source_hash": h}
                need(day not in points or points[day] == v, "HYPZ_PERIOD_CONFLICT")
                points[day] = v
        need(set(points) == set(target), "TWELVE_QUARTER_BUDGET_CLOSURE:" + code)
        mean = sum(p["gap_upper_pp"] for p in points.values()) / len(points)
        maximum = max(p["gap_upper_pp"] for p in points.values())
        region = min(p["region_lower_pct"] for p in points.values())
        budgets[code] = {
            "mean_gap_upper_pp": str(mean),
            "max_gap_upper_pp": str(maximum),
            "region_lower_pct": str(region),
            "quarters": {d: {k: str(v) for k, v in r.items()} for d, r in points.items()},
        }
        if code in RETURN:
            need(mean <= 10 and maximum <= 20 and region >= 80, "RETURN_COHORT_BUDGET_OR_REGION:" + code)
        if code in STRICT:
            need(mean <= 5 and maximum <= 10 and region >= 90, "STRICT_COHORT_BUDGET_OR_REGION:" + code)
    need(all(c in budgets for c in RETURN), "PEER_BUDGET_MEMBERS")
    need(dec(budgets["013132"]["region_lower_pct"]) < 80, "013132_ACTUAL_REGION_EXCLUSION")
    need(
        any(
            dec(budgets["005585"][k]) > limit for k, limit in [("mean_gap_upper_pp", D(5)), ("max_gap_upper_pp", D(10))]
        ),
        "005585_STRICT_EXCLUSION",
    )
    need(
        any(
            dec(budgets["164205"][k]) > limit for k, limit in [("mean_gap_upper_pp", D(5)), ("max_gap_upper_pp", D(10))]
        ),
        "164205_STRICT_EXCLUSION",
    )
    return {
        "catalogue_shares_rebuilt": len(actual),
        "theme_boundary_decisions": len(decisions),
        "share_aliases": aliases,
        "budgets": budgets,
        "not_blind_calibration": True,
        "exclusion_original_reviews": excluded_audit,
    }
