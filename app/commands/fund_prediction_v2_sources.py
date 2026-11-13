"""冻结第二版本期真实资料和逐周期日期，不批准资料、不生成方向。"""

import argparse
import json
from datetime import date
from pathlib import Path

from app.services.comprehensive_multi_prediction import review_calendar
from app.services.direction_1d_protocol import digest, window
from app.services.fund_exposure_common import now, read
from app.services.fund_information_snapshot import prepare
from app.services.fund_prediction_snapshot_compat import materials_input_identity
from app.services.fund_workflow_materials import registered_materials
from app.services.prediction_contract import PredictionFailure, add_months, prediction_policy, target_dates
from app.services.prediction_review_v2 import policy_v2
from app.services.prediction_trial_v2 import current_holdings, local_path, save_once, storage_check

HORIZONS = ("T5_V1", "T20_V1", "M6_V1")


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


def freeze_sources(directory, *, codes=None, analysis_requests=None, holdings_provider=current_holdings, clock=None):
    """逐只冻结当前持仓。失败保留缺口；成功清单可以直接交给第二版 prepare。"""
    directory = local_path(directory)
    if directory.exists():
        raise ValueError("V2_SOURCE_DIRECTORY_MUST_BE_NEW")
    if codes and analysis_requests:
        raise ValueError("V2_SOURCE_CHOOSE_CODES_OR_REQUESTS")
    inventory, at = holdings_provider(), clock or now()
    current = window(at)
    requests = {}
    for path in analysis_requests or []:
        request = read(local_path(path))
        code = request.get("fund_code")
        if (
            request.get("format") != "FUND_PREDICTION_V2_DAILY_ANALYSIS_REQUEST"
            or request.get("owner_hash") != inventory["owner_hash"]
            or request.get("policy_hash") != digest(policy_v2())
            or any(request.get("window", {}).get(k) != current[k] for k in ("base_nav_date", "target_nav_date"))
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
            # 共用资料准备器核来源、净值与基准。这里只读其结果，不重写 source_ready。
            data = prepare(at, code, material_path)
            if data.get("fund_code") != code or any(
                data.get("window", {}).get(k) != current[k] for k in ("base_nav_date", "target_nav_date")
            ):
                raise ValueError("V2_SOURCE_PREPARED_WINDOW_OR_FUND_MISMATCH")
            if data.get("materials_hash") != digest(material):
                raise ValueError("V2_SOURCE_PREPARED_MATERIAL_HASH_MISMATCH")
            packet = {"format": "CODEX_FUND_INPUT_V1", "input_hash": digest(data), "input": data}
            source = directory / "funds" / code / "T1" / "input.json"
            save_once(source, packet)
            row.update(
                source_path=str(source),
                source_hash=digest(packet),
                source_ready=data.get("time_context", {}).get("baseline_complete") is True,
                time_context=data.get("time_context", {}),
            )
            try:
                calendar, calendar_failure = review_calendar(data, code, at), None
            except (ValueError, KeyError) as error:
                calendar, calendar_failure = None, failure(error)
            for horizon in HORIZONS:
                reason = calendar_failure
                if calendar is None:
                    dates = _missing_dates(horizon, current, reason)
                else:
                    try:
                        dates = target_dates(calendar, at, horizon)
                    except (ValueError, KeyError) as error:
                        reason = failure(error)
                        dates = _missing_dates(horizon, current, reason)
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
