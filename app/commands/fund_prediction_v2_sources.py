"""冻结第二版本期真实资料和逐周期日期，不批准资料、不生成方向。"""

import argparse
import json
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

from app.services.comprehensive_multi_prediction import review_calendar
from app.services.direction_1d_protocol import digest
from app.services.prediction_window_v2 import fund_window, is_midnight, window
from app.services.fund_exposure_common import now, read
from app.services.fund_information_snapshot import prepare
from app.services.fund_prediction_snapshot_compat import materials_input_identity
from app.services.fund_workflow_materials import registered_materials
from app.services.prediction_contract import PredictionFailure, add_months, prediction_policy, target_dates
from app.services.prediction_daily_guidance import review_guidance
from app.services.prediction_review_v2 import policy_v2
from app.services.prediction_trial_v2 import current_holdings, local_path, save_once, storage_check
from app.services.prediction_v2_baseline import baseline_ready

HORIZONS = ("T5_V1", "T20_V1", "M6_V1")
QDII_CALENDAR_REVIEW_ONLY = {
    "COMPREHENSIVE_CALENDAR_POLICY_REQUIRED",
    "COMPREHENSIVE_VN_CONTRACT_CALENDAR_REQUIRED",
}


def failure(error):
    """只保存有限原因和缺项，数据库或网络异常正文不写入资料包。"""
    if isinstance(error, PredictionFailure):
        return {
            "type": type(error).__name__,
            "code": error.payload.get("code"),
            "summary": error.payload.get("summary"),
            "details": error.payload.get("details", {}),
        }
    message = str(error)
    code = (
        "SOURCE_FILE_MISSING"
        if isinstance(error, FileNotFoundError)
        else (message if message.isupper() and len(message) < 120 else "SOURCE_FREEZE_FAILED")
    )
    return {
        "type": type(error).__name__,
        "code": code,
        "missing_field": message if isinstance(error, KeyError) else None,
    }


def _missing_dates(horizon, current, reason):
    """保留待核状态；半年名义日期只表示六个月，不表示真实估值日。"""
    nominal = str(add_months(date.fromisoformat(current["base_nav_date"]), 6)) if horizon == "M6_V1" else None
    return {
        "horizonId": horizon,
        "baseNavDate": current["base_nav_date"],
        "startDate": current["target_nav_date"],
        "endDate": None,
        "nominalEndDate": nominal,
        "endDateStatus": "UNVERIFIED",
        "calendarId": None,
        "calendarHash": None,
        "missingEvidence": reason,
    }


def _partial_calendar_materials(at, code, material_path, material, current, calendar_failure):
    """已知QDII窗口缺件仅供实际审阅；原件仍真校验，净值/归档版本只读。"""
    from sqlalchemy import text

    from app.db.session import get_nav_preview_engine
    from app.services.fund_comprehensive_materials import checked_documents

    documents = {d["id"]: d for d in checked_documents(at, code, material)}
    for key, kind in (("prospectus_ref", "PROSPECTUS"), ("report_ref", "REPORT")):
        document = documents.get((material.get("calendar_evidence") or {}).get(key), {})
        if document.get("kind") != kind or code not in document.get("codes", []):
            raise ValueError("V2_PARTIAL_OWN_ORIGINAL_REQUIRED")
        if document.get("raw_path"):
            from app.services.material_content import content_digest, source_bytes, source_exists

            raw = Path(document["raw_path"])
            if not source_exists(raw):
                raise FileNotFoundError("V2_PARTIAL_ORIGINAL_FILE_MISSING")
            if content_digest(source_bytes(raw)) != document.get("raw_sha256"):
                raise ValueError("V2_PARTIAL_ORIGINAL_HASH_MISMATCH")
    base = date.fromisoformat(current["base_nav_date"])
    with get_nav_preview_engine().connect() as connection:
        rows = connection.execute(text("""
            SELECT n.nav_date,n.ann_date,n.source_id,n.unit_nav,n.content_hash,n.updated_at,
                   v.version_id,v.received_at,v.expires_at
            FROM nav_daily n JOIN source_registry s ON s.source_id=n.source_id
            LEFT JOIN LATERAL (
                SELECT version_id,received_at,expires_at FROM direction_1d_source_version
                WHERE fund_code=n.fund_code AND source_id=n.source_id AND kind='NAV'
                  AND business_date=n.nav_date AND payload->>'upstream_hash'=n.content_hash
                ORDER BY received_at DESC LIMIT 1
            ) v ON TRUE
            WHERE n.fund_code=:code AND n.nav_date BETWEEN :start AND :base AND s.enabled=true
            ORDER BY n.nav_date,n.source_id
        """), {"code": code, "start": base - timedelta(days=7), "base": base}).mappings()
        nav = []
        for row in rows:
            if row["updated_at"] > at or (row["ann_date"] and row["ann_date"] > at.date()):
                continue
            value = {
                "nav_date": str(row["nav_date"]), "ann_date": str(row["ann_date"]) if row["ann_date"] else None,
                "source_id": str(row["source_id"]), "unit_nav": str(row["unit_nav"]),
                "upstream_hash": row["content_hash"], "updated_at": row["updated_at"].isoformat(),
            }
            if row["version_id"]:
                value.update(version_id=str(row["version_id"]), observed_at=row["received_at"].isoformat(),
                             expires_at=row["expires_at"].isoformat())
            nav.append(value)
    data = deepcopy(material)
    data.update(
        as_of=at.isoformat(), window=deepcopy(current), materials_path=str(Path(material_path).resolve()),
        materials_hash=digest(material), nav=nav,
        time_context={"baseline_complete": any(v["nav_date"] == str(base) for v in nav)},
        valuation_status={"ready": False, "reason": calendar_failure},
        partial_material_assembly={
            "assembled_at": at.isoformat(), "source_ready": False,
            "reason": calendar_failure, "no_original_dates_changed": True,
            "requires_actual_reading": True, "nav_versions_read_only": True,
        },
    )
    return data


def freeze_sources(directory, *, codes=None, analysis_requests=None, holdings_provider=current_holdings, clock=None):
    """逐只冻结当前持仓。失败保留缺口；成功清单可以直接交给第二版 prepare。"""
    directory = local_path(directory)
    if directory.exists():
        raise ValueError("V2_SOURCE_DIRECTORY_MUST_BE_NEW")
    if codes and analysis_requests:
        raise ValueError("V2_SOURCE_CHOOSE_CODES_OR_REQUESTS")
    inventory, at = holdings_provider(), clock or now()
    current = window(at)
    early = is_midnight(current)
    requests = {}
    for path in analysis_requests or []:
        request = read(local_path(path))
        code = request.get("fund_code")
        if (
            request.get("format") != "FUND_PREDICTION_V2_DAILY_ANALYSIS_REQUEST"
            or request.get("owner_hash") != inventory["owner_hash"]
            or request.get("policy_hash") != digest(policy_v2())
            or (request.get("window") != current if early else any(
                request.get("window", {}).get(k) != current[k] for k in ("base_nav_date", "target_nav_date")
            ))
        ):
            raise ValueError("V2_SOURCE_REQUEST_OWNER_POLICY_OR_WINDOW_MISMATCH")
        if code in requests and any(
            request.get(k) != requests[code].get(k) for k in ("materials_hash", "materials_input_hash")
        ):
            raise ValueError("V2_SOURCE_REQUEST_MATERIALS_CONFLICT")
        requests[code] = request
    selected = sorted(set(requests if analysis_requests else codes or inventory["scope"]))
    if not selected or not set(selected) <= set(inventory["scope"]):
        raise ValueError("V2_SOURCE_OUTSIDE_CURRENT_HOLDINGS")
    rows, selection = [], []
    for code in selected:
        row = {
            "fund_code": code,
            "captured_at": at.isoformat(),
            "source_ready": False,
            "status": "WAITING_MATERIALS",
            "periods": {},
        }
        preparing_materials = False
        try:
            handoff = registered_materials(code, at=at)
            if not handoff:
                raise ValueError("V2_REGISTERED_MATERIALS_REQUIRED")
            material_path = Path(handoff["materials_path"])
            material = read(material_path)
            row.update(
                materials_path=str(material_path),
                materials_hash=digest(material),
                legacy_material_gaps=handoff.get("gaps", []),
            )
            if material.get("fund_code") != code or digest(material) != handoff.get("materials_hash"):
                raise ValueError("V2_SOURCE_MATERIAL_IDENTITY_MISMATCH")
            request = requests.get(code)
            if request and (
                request.get("materials_hash") != digest(material)
                or request.get("materials_input_hash") != materials_input_identity(material)
            ):
                raise ValueError("V2_SOURCE_REQUEST_MATERIALS_CHANGED")
            # 共用准备器保留全部来源状态；第二版区分基金基准与历史个股缺口。
            preparing_materials = True
            data = prepare(at, code, material_path, planning=current) if early else prepare(at, code, material_path)
            preparing_materials = False
            if data.get("fund_code") != code or any(
                data.get("window", {}).get(k) != current[k] for k in ("base_nav_date", "target_nav_date")
            ):
                raise ValueError("V2_SOURCE_PREPARED_WINDOW_OR_FUND_MISMATCH")
            if data.get("materials_hash") != digest(material):
                raise ValueError("V2_SOURCE_PREPARED_MATERIAL_HASH_MISMATCH")
            from app.services.fund_information_contracts import time_context
            from app.services.fund_prediction_nav_identity import previous_qdii_nav_date
            from app.services.vietnam_reference_fx_review import apply_reference_review

            if not early:
                apply_reference_review(data, material)
            reference = None if early else previous_qdii_nav_date(data.get("nav", []), current["base_nav_date"], data)
            if reference:
                data["reference_nav_date"] = reference
            if reference or (data.get("valuation_status") or {}).get("prediction_reference_fx"):
                data["time_context"] = time_context(data)
            valuation_window = current
            try:
                calendar, calendar_failure = review_calendar(data, code, at), None
                if early:
                    valuation_window = fund_window(current, calendar)
            except (ValueError, KeyError) as error:
                calendar, calendar_failure = None, failure(error)
            if early and calendar is not None:
                if valuation_window != current:
                    data = prepare(at, code, material_path, planning=valuation_window, valuations=calendar)
                    if data.get("materials_hash") != digest(material):
                        raise ValueError("V2_SOURCE_PREPARED_MATERIAL_HASH_MISMATCH")
                data["planning_window"] = current
                data["forecast_calendar_verified"] = True
            packet = {"format": "CODEX_FUND_INPUT_V1", "input_hash": digest(data), "input": data}
            source = directory / "funds" / code / "T1" / "input.json"
            save_once(source, packet)
            row.update(
                source_path=str(source),
                source_hash=digest(packet),
                source_ready=baseline_ready(data),
                time_context=data.get("time_context", {}),
            )
            if reference:
                row["reference_nav_date"] = reference
            for horizon in HORIZONS:
                reason = calendar_failure
                if horizon == "M6_V1" and calendar is not None:
                    from app.services.prediction_natural_month import natural_month_target

                    dates = natural_month_target(reference or valuation_window["base_nav_date"],
                                                 valuation_window["target_nav_date"])
                    dates["baseNavDate"] = valuation_window["base_nav_date"]
                    reason = None
                elif calendar is None:
                    dates = _missing_dates(horizon, valuation_window, reason)
                else:
                    try:
                        dates = (target_dates(calendar, at, horizon, valuation_window=valuation_window)
                                 if early else target_dates(calendar, at, horizon))
                    except (ValueError, KeyError) as error:
                        reason = failure(error)
                        dates = _missing_dates(horizon, valuation_window, reason)
                contents = {
                    "data": data,
                    "dates": dates,
                    "horizon_id": horizon,
                    "prediction_policy": prediction_policy(),
                    "analysis_policy": policy_v2(),
                }
                sibling = {"format": "CODEX_COMPREHENSIVE_MULTI_V1", "input_hash": digest(contents), "input": contents}
                save_once(directory / "funds" / code / horizon / "input.json", sibling)
                row["periods"][horizon] = {
                    "status": dates["endDateStatus"],
                    "end_date": dates["endDate"],
                    "nominal_end_date": dates.get("nominalEndDate"),
                    "failure": reason,
                }
            row["status"] = "FROZEN" if row["source_ready"] else "FROZEN_WITH_BASELINE_GAPS"
            selection.append(
                {
                    "fund_code": code,
                    "source_kind": "FROZEN",
                    "source_path": str(source),
                    "sample_label": "本人当前持仓的第二版本期资料",
                    "reason": "冻结真实现有材料；缺口和日历按实际保留，方向另行逐周期审阅。",
                }
            )
        except (ValueError, KeyError, FileNotFoundError) as error:
            row["failure"] = failure(error)
            profile = next((p for p in inventory["profiles"] if p["fund_code"] == code), {})
            calendar_failure = str(error)
            if preparing_materials and calendar_failure in QDII_CALENDAR_REVIEW_ONLY and "QDII" in str(
                profile.get("fund_name", "")
            ).upper():
                try:
                    partial = _partial_calendar_materials(at, code, material_path, material, current, calendar_failure)
                    from app.services.fund_prediction_nav_identity import (
                        current_prediction_nav_identity,
                        nav_identity,
                        previous_qdii_nav_date,
                    )

                    base = current["base_nav_date"]
                    reference = None if early else previous_qdii_nav_date(partial["nav"], base, profile)
                    if reference and nav_identity(partial["nav"], reference) == current_prediction_nav_identity(
                        code, base, at=at
                    ):
                        # 保留真实参考净值身份；当前行情缺件仍禁止输出方向。
                        partial["reference_nav_date"] = reference
                    path = directory / "funds" / code / "partial-input.json"
                    save_once(path, partial)
                except (ValueError, KeyError, FileNotFoundError) as partial_error:
                    row["partial_failure"] = failure(partial_error)
                else:
                    row.update(status="FROZEN_WITH_CALENDAR_GAPS", source_path=str(path),
                               source_hash=digest(partial), source_ready=False)
                    selection.append({
                        "fund_code": code, "source_kind": "PARTIAL_MATERIALS", "source_path": str(path),
                        "sample_label": "本人当前持仓的真实缺件审阅资料",
                        "reason": "原件已核但QDII估值窗口未齐，仅供实际缺件审阅，不能放行方向。",
                    })
        rows.append(row)
        save_once(directory / "funds" / code / "freeze-status.json", row)
    final = holdings_provider()
    if final["owner_hash"] != inventory["owner_hash"] or not set(selected) <= set(final["scope"]):
        save_once(directory / "scope-changed.json", {"status": "HOLDINGS_CHANGED_DURING_FREEZE"})
        raise ValueError("V2_SOURCE_HOLDINGS_CHANGED_DURING_FREEZE")
    # prepare 的 selection 参数是普通 JSON 数组，不是业务封套；创建后不能覆盖。
    selection_path = directory / "selection.json"
    with selection_path.open("x", encoding="utf-8") as handle:
        json.dump(selection, handle, ensure_ascii=False, indent=2)
    if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
        raise OSError("V2_SOURCE_SELECTION_READBACK_MISMATCH")
    result = {
        "status": "SUCCEEDED"
        if len(selection) == len(selected)
        else "PARTIAL_SUCCESS"
        if selection
        else "WAITING_MATERIALS",
        "captured_at": at.isoformat(),
        "owner_hash": inventory["owner_hash"],
        "window": current,
        "policy_hash": digest(policy_v2()),
        "selection_path": str(selection_path),
        "requested_count": len(selected),
        "frozen_count": len(selection),
        "source_ready_count": sum(r["source_ready"] for r in rows),
        "funds": rows,
        "production_publication": False,
        "directions_generated": 0,
    }
    guidance = review_guidance()
    save_once(directory / "review-guidance.json", guidance)
    result.update(review_guidance_path=str(directory / "review-guidance.json"), review_guidance_hash=digest(guidance))
    save_once(directory / "freeze-summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help=".local-runs 下全新的资料冻结目录")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--fund-code", action="append", help="可重复；省略时取本人当前全部持仓")
    group.add_argument("--analysis-request", action="append", type=Path, help="每日第二版分析请求，可重复")
    args = parser.parse_args()
    instant = now()
    guard = storage_check(instant.replace(hour=0, minute=0, second=0, microsecond=0).isoformat())
    result = freeze_sources(args.run_dir, codes=args.fund_code, analysis_requests=args.analysis_request)
    save_once(args.run_dir / "storage-check.json", guard)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "selection_path",
                    "requested_count",
                    "frozen_count",
                    "source_ready_count",
                    "directions_generated",
                )
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if result["frozen_count"] == result["requested_count"] else 2)


if __name__ == "__main__":
    main()
