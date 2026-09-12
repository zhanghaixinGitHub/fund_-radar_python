"""全名单数据检查与历史重建；基金类型由公开档案及基准核验，绝不按旧三只限范围。"""

import re
from datetime import date, datetime

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services.direction_1d_protocol import calendar, digest, features, input_days, window
from app.services.direction_1d_selection import available_at_prediction


def classify(p: dict) -> dict:
    """正向核验境内基准；跨境成分、商品、FOF等保留独立原因，不用名称作准入正证据。"""
    description = " ".join(str(p.get(k) or "") for k in ("fund_name", "source_fund_type", "invest_type", "benchmark"))
    reason, group = None, None
    if re.search(
        r"QDII|FOF|REIT|货币|黄金|原油|商品|美元|港元|港股|恒生|海外|越南|日本|纳斯达克|全球|标普石油",
        description,
        re.I,
    ):
        reason = "SPECIAL_POLICY_REQUIRED"
    elif not p.get("profile_hash") or p.get("source_code") != "TUSHARE_PRO_FUND" or p.get("status") != "ACTIVE":
        reason = "GROUP_UNVERIFIED"
    elif not re.search(
        r"沪深|中证|国证|中国A股|上证|上海证券|中国债|申银万国|中债|iBoxx亚债基金中国指数", p.get("benchmark") or ""
    ):
        reason = "GROUP_UNVERIFIED"
    elif p["fund_type"] in {"STOCK", "MIXED", "BOND"}:
        group = {"STOCK": "CN_EQUITY", "MIXED": "CN_MIXED", "BOND": "CN_BOND"}[p["fund_type"]]
    else:
        reason = "SPECIAL_POLICY_REQUIRED"
    # 已有产品主表按管理人和来源主产品建立，不凭净值相关性推定份额家族。
    family = str(p["fund_master_id"]) if p.get("fund_master_id") and p.get("master_name") else None
    if not family:
        reason, group = "GROUP_UNVERIFIED", None
    return {
        "group_id": group,
        "product_family_id": family,
        "classification_reason": reason,
        "group_evidence": {
            "source": p.get("source_code"),
            "profile_hash": p.get("profile_hash"),
            "benchmark": p.get("benchmark"),
            "invest_type": p.get("invest_type"),
            "rule": "DOMESTIC_BENCHMARK_ASSET_TYPE_V1",
            "historical_mapping_evidence": "CURRENT_PROFILE_ASSUMED",
        },
        "currency": "CNY" if group else "UNVERIFIED",
        "cross_border": reason == "SPECIAL_POLICY_REQUIRED",
    }


def inventory(codes: list[str], now: datetime | None = None) -> dict:
    if len(codes) > 100 or any(not re.fullmatch(r"\d{6}", c) for c in codes):
        raise ValueError("INVALID_SCOPE")
    now = now or repo.clock()
    w = window(now)
    wanted = input_days(date.fromisoformat(w["base_nav_date"]))
    with get_engine().connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        try:
            source = repo.source(c)
        except ValueError:
            # 来源撤回仍返回全部范围的拒绝原因，不能让整张覆盖表消失或继续算分。
            return {
                "items": [
                    {
                        "fund_code": code,
                        "fund_name": code,
                        "group_id": None,
                        "status": "SOURCE_UNAVAILABLE",
                        "reason_codes": ["SOURCE_UNAVAILABLE"],
                        "history_count": 0,
                        "missing_count": len(wanted),
                        "missing_dates": [str(d) for d in wanted],
                        "latest_nav_date": None,
                        "model_ids": [],
                        "next_window_model_available": False,
                        "current_input_complete": False,
                        "observed_at": now.isoformat(),
                    }
                    for code in sorted(set(codes))
                ],
                "checked_count": len(set(codes)),
                "window": w,
                "scope_hash": digest(sorted(set(codes))),
                "checked_at": now.isoformat(),
                "model_released": False,
                "up_probability": None,
            }
        registry = repo.models(c)
        ps = {p["fund_code"]: p for p in repo.profiles(c, codes)}
        result = []
        for code in sorted(set(codes)):
            p = ps.get(code)
            if not p:
                result.append(
                    {
                        "fund_code": code,
                        "fund_name": code,
                        "reason_codes": ["SOURCE_UNAVAILABLE"],
                        "status": "SOURCE_UNAVAILABLE",
                    }
                )
                continue
            mapping = classify(p)
            rows = repo.navs(c, code, source["source_id"], wanted[0], wanted[-1])
            points = {r["nav_date"]: r for r in rows}
            missing = [str(d) for d in wanted if d not in points]
            counts = (
                c.execute(
                    text("""SELECT min(nav_date) first,max(nav_date) last,count(*) count
              FROM nav_daily WHERE fund_code=:code AND source_id=:source"""),
                    {"code": code, "source": source["source_id"]},
                )
                .mappings()
                .one()
            )
            reasons = [mapping["classification_reason"]] if mapping["classification_reason"] else []
            if counts["count"] < 61:
                reasons.append("HISTORY_TOO_SHORT")
            elif missing:
                reasons.append("DATA_PENDING")
            elif not reasons:
                try:
                    features([points[d]["unit_nav"] for d in wanted])
                except ValueError as error:
                    reasons.append(str(error))
            eligible = [m for m in registry if m["group_id"] == mapping["group_id"] and m["expires_at"] > now]
            active = [m for m in eligible if available_at_prediction(m, now)]
            if mapping["group_id"] and not eligible:
                reasons.append("MODEL_PENDING")
            elif eligible:
                from app.services.direction_1d_inference import load_model
                from app.services.direction_1d_protocol import score

                valid_ids = set()
                for model_row in eligible:
                    try:
                        score(load_model(model_row), [0.0] * 7)
                        valid_ids.add(model_row["model_id"])
                    except (ValueError, OSError, KeyError):
                        continue
                if not valid_ids:
                    reasons.append("MODEL_UNAVAILABLE")
                active = [m for m in active if m["model_id"] in valid_ids]
                eligible = [m for m in eligible if m["model_id"] in valid_ids]
                if not active:
                    reasons.append("MODEL_NOT_ACTIVE_FOR_WINDOW")
            result.append(
                {
                    "fund_code": code,
                    "fund_name": p["fund_name"],
                    "fund_type": p["fund_type"],
                    "share_class": p["share_class"],
                    **mapping,
                    "history_start": str(counts["first"]) if counts["first"] else None,
                    "history_end": str(counts["last"]) if counts["last"] else None,
                    "history_count": counts["count"],
                    "missing_dates": missing,
                    "missing_count": len(missing),
                    "latest_nav_date": str(counts["last"]) if counts["last"] else None,
                    "observed_at": now.isoformat(),
                    "source_code": source["source_code"],
                    "calendar_version": w["calendar_version"],
                    "event_status": "UNKNOWN",
                    "status": ("MODEL_PENDING" if reasons[0] == "MODEL_NOT_ACTIVE_FOR_WINDOW" else reasons[0])
                    if reasons
                    else "READY_EXPERIMENTAL",
                    "reason_codes": reasons,
                    "model_ids": [str(m["model_id"]) for m in eligible],
                    "current_input_complete": not missing,
                    "next_window_model_available": bool(eligible),
                    "prediction_status": "WAITING_DATA"
                    if missing
                    else "MODEL_UNAVAILABLE"
                    if not active
                    else w["status"],
                }
            )
    return {
        "items": result,
        "checked_count": len(result),
        "scope_hash": digest(sorted(set(codes))),
        "window": w,
        "checked_at": now.isoformat(),
        "model_released": False,
        "up_probability": None,
    }


def history(codes: list[str]) -> dict:
    """仅读2021—2024已使用开发历史；2025受保护、2026不隐式解封。"""
    with get_engine().connect().execution_options(isolation_level="REPEATABLE READ") as c, c.begin():
        source = repo.source(c)
        now = c.execute(text("SELECT clock_timestamp()")).scalar_one()
        ps = repo.profiles(c, codes)
        data = []
        for p in ps:
            mapping = classify(p)
            if not mapping["group_id"]:
                continue
            rows = repo.navs(c, p["fund_code"], source["source_id"], date(2021, 1, 1), date(2024, 12, 31))
            data.append(
                {
                    "fund_code": p["fund_code"],
                    **mapping,
                    "rows": [
                        {
                            "date": str(r["nav_date"]),
                            "nav": str(r["unit_nav"]),
                            "ann_date": str(r["ann_date"]) if r["ann_date"] else None,
                            "source_hash": r["content_hash"],
                        }
                        for r in rows
                    ],
                }
            )
        return {
            "kind": "HISTORICAL_RECONSTRUCTION",
            "availability_evidence": "ASSUMED",
            "captured_at": now.isoformat(),
            "source_code": source["source_code"],
            "retention_days": source["retention_days"],
            "funds": data,
            "calendar_hash": calendar()[1],
            "history_usage_manifest": [
                {"start": "2021-01-01", "end": "2024-12-31", "use": "DEVELOPMENT_ALREADY_USED_20D"},
                {"start": "2025-01-01", "end": "2025-12-31", "use": "PROTECTED_EXCLUDED"},
                {"start": "2026-01-01", "end": "2026-12-31", "use": "NO_RETROSPECTIVE_TRAINING_FORWARD_ASSESSED_ONLY"},
            ],
        }
