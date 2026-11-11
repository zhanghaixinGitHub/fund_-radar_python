"""核对本人正式预测并保留待处理清单；不生成方向、不修改历史预测。

正式结果只读；资料日期校验沿用原入口，可能幂等追加真实净值观察，不续期或改写旧观察。
"""

import argparse
import json
import re
from pathlib import Path

import httpx
from dotenv import dotenv_values
from sqlalchemy import URL, bindparam, create_engine, text

from app.core.config import get_settings
from app.db.session import get_engine
from app.services.direction_1d_protocol import digest, window
from app.services.fund_exposure_common import now, read
from app.services.fund_materials import snapshot
from app.services.fund_materials_store import versioned_save
from app.services.fund_prediction_followthrough import HORIZONS, recovery_actions, verify_outputs
from app.services.fund_workflow_materials import receipt_path, registered_materials, workflow_root
from app.services.material_content import material_read_cache, source_exists


def owner_scope(settings, owner):
    """沿用Java本人有效性和关注校验，分页获取，不能扩大为全市场或其他账号。"""
    codes, after = [], ""
    with httpx.Client(
        timeout=httpx.Timeout(20, connect=5),
        trust_env=False,
        headers={"X-Service-Token": settings.ai_service_token.get_secret_value()},
    ) as client:
        for _ in range(100):
            response = client.get(
                settings.core_service_base_url + f"/internal/v1/direction-1d/sync/users/{owner}/held-fund-codes",
                params={"after": after},
            )
            response.raise_for_status()
            try:
                page = response.json()
            except ValueError:
                raise ValueError("OWNER_SCOPE_RESPONSE_INVALID") from None
            if not isinstance(page, list) or any(
                not isinstance(code, str) or not re.fullmatch(r"\d{6}", code) for code in page
            ):
                raise ValueError("OWNER_SCOPE_RESPONSE_INVALID")
            if not page:
                return codes
            if page != sorted(set(page)) or page[0] <= after:
                raise ValueError("OWNER_SCOPE_PAGINATION_INVALID")
            codes.extend(page)
            after = page[-1]
    raise ValueError("OWNER_SCOPE_PAGINATION_LIMIT")


@material_read_cache
def saved_records(owner, codes, target, core_env):
    """交叉核对Java本人留档和Python多周期原记录；不接受仅生成未归档的结果。"""
    cfg = dotenv_values(core_env)
    url = URL.create(
        "postgresql+psycopg",
        username=cfg["FUND_CORE_DB_USERNAME"],
        password=cfg["FUND_CORE_DB_PASSWORD"],
        host=cfg.get("FUND_CORE_DB_HOST", "localhost"),
        port=int(cfg.get("FUND_CORE_DB_PORT", "54329")),
        database=cfg.get("FUND_CORE_DB_NAME", "fund_core"),
    )
    engine = create_engine(url)
    rows = []
    try:
        with engine.connect() as c:
            c.execute(text("SET TRANSACTION READ ONLY"))
            t1_query = text("""SELECT f.forecast_id::text AS id,f.fund_code,f.content_hash,f.payload_json,r.status,
                    r.content_hash AS receipt_hash FROM direction_1d_forecast f
                    JOIN direction_1d_user_forecast u USING(forecast_id)
                    JOIN direction_1d_forecast_receipt r USING(forecast_id)
                    WHERE u.user_id=:owner AND f.target_nav_date=:target
                    AND f.fund_code IN :codes""").bindparams(bindparam("codes", expanding=True))
            t1 = c.execute(t1_query, {"owner": owner, "target": target, "codes": codes}).mappings()
            for item in t1:
                payload = (
                    json.loads(item["payload_json"]) if isinstance(item["payload_json"], str) else item["payload_json"]
                )
                if item["fund_code"] not in codes or item["status"] != "VERIFIED":
                    continue
                if digest(payload) != item["content_hash"] or item["receipt_hash"] != item["content_hash"]:
                    raise ValueError("OWNER_T1_READBACK_HASH_INVALID")
                rows.append(
                    {
                        "id": item["id"],
                        "fund_code": item["fund_code"],
                        "horizon": "T1",
                        "base_date": payload["base_nav_date"],
                        "target_date": target,
                        "end_date": target,
                        "owner_verified": True,
                        "_snapshot_id": payload.get("input_snapshot_id"),
                        "_snapshot_hash": payload.get("input_hash"),
                    }
                )
            multi_query = text("""SELECT prediction_id::text AS id,payload FROM prediction_user_link
                WHERE user_id=:owner AND payload->>'startDate'=:target
                AND payload->>'fundCode' IN :codes""").bindparams(bindparam("codes", expanding=True))
            multi = c.execute(multi_query, {"owner": owner, "target": target, "codes": codes}).mappings()
            linked = {
                item["id"]: item["payload"]
                for item in multi
                if item["payload"].get("fundCode") in codes and item["payload"].get("startDate") == target
            }
        # 新目标日尚无本人多周期归档是正常待办状态。空 expanding 参数会被
        # SQLAlchemy 编成 INTEGER 空集，与 PostgreSQL UUID 主键比较时报错；
        # 此时没有原记录需要核对，直接保留空结果，仍由后续守卫要求继续生成。
        originals = {}
        if linked:
            with get_engine().connect() as c:
                c.execute(text("SET TRANSACTION READ ONLY"))
                query = text(
                    "SELECT prediction_id::text AS id,content_hash,payload FROM fund_prediction_record "
                    "WHERE prediction_id IN :ids"
                ).bindparams(bindparam("ids", expanding=True))
                originals = {r["id"]: r for r in c.execute(query, {"ids": list(linked)}).mappings()}
            for identifier, payload in linked.items():
                original = originals.get(identifier)
                if original is None:
                    raise ValueError("OWNER_MULTI_ORIGINAL_MISSING")
                if payload != original["payload"] or digest(payload) != original["content_hash"]:
                    raise ValueError("OWNER_MULTI_READBACK_HASH_INVALID")
                if payload.get("generationStatus") == "SUCCEEDED":
                    rows.append(
                        {
                            "id": identifier,
                            "fund_code": payload["fundCode"],
                            "horizon": payload["horizonId"],
                            "base_date": payload["baseNavDate"],
                            "target_date": target,
                            "end_date": payload["endDate"],
                            "owner_verified": True,
                            "_snapshot_id": (payload.get("featureSnapshot") or {}).get("inputSnapshotId"),
                            "_snapshot_hash": payload.get("featureHash"),
                        }
                    )
    finally:
        engine.dispose()
    if not rows:
        return []
    material_identities = {}
    # 同期资料变化后必须继续新输入审阅，不能仅以旧正文和本人档案相同就结束。
    with get_engine().connect() as c:
        c.execute(text("SET TRANSACTION READ ONLY"))
        for kind, table in (("T1", "direction_1d_snapshot"), ("MULTI", "prediction_feature_snapshot")):
            selected = [r for r in rows if (r["horizon"] == "T1") == (kind == "T1")]
            ids = sorted({r["_snapshot_id"] for r in selected if r["_snapshot_id"]})
            snapshots = {}
            if ids:
                # 表名仅来自固定二选一；基金、账号和UUID始终绑定参数，空集合不发UUID查询。
                query = text(
                    f"SELECT snapshot_id::text AS id,content_hash,payload FROM {table} WHERE snapshot_id IN :ids "
                    "ORDER BY COALESCE(payload->>'fund_code',payload->'data'->>'fund_code'),snapshot_id"
                ).bindparams(bindparam("ids", expanding=True))
                # 原件可能较大，流式验证后只保留身份/资料摘要，不把全部全文副本堆在内存。
                # 每份快照含完整原文。默认动态缓冲会把多只基金的大正文同时
                # 拉入内存；一行一验保留全部哈希检查，降低分页和磁盘争用。
                for source_row in (
                    c.execution_options(stream_results=True, yield_per=1, max_row_buffer=1)
                    .execute(query, {"ids": ids})
                    .mappings()
                ):
                    payload = source_row["payload"]
                    from app.services.fund_prediction_snapshot_compat import checked_snapshot, materials_input_identity

                    payload = checked_snapshot(payload, source_row["content_hash"])
                    data = payload if kind == "T1" else payload["data"]
                    material_hash = data.get("materials_hash")
                    if material_hash not in material_identities:
                        original_material = read(Path(data["materials_path"]))
                        if digest(original_material) != material_hash or original_material.get("fund_code") != data.get(
                            "fund_code"
                        ):
                            raise ValueError("OWNER_INPUT_ORIGINAL_MATERIALS_INVALID")
                        material_identities[material_hash] = materials_input_identity(original_material)
                    snapshots[source_row["id"]] = {
                        "content_hash": source_row["content_hash"],
                        "fund_code": data.get("fund_code"),
                        "target_date": data.get("window", {}).get("target_nav_date"),
                        "materials_hash": data.get("materials_hash"),
                        "materials_input_hash": material_identities[material_hash],
                    }
            for row in selected:
                original = snapshots.get(row.pop("_snapshot_id"))
                expected = row.pop("_snapshot_hash")
                if not original or original["content_hash"] != expected:
                    raise ValueError(
                        "OWNER_PREDICTION_INPUT_SNAPSHOT_REFERENCE_INVALID:" + row["fund_code"] + ":" + kind
                    )
                if original["fund_code"] != row["fund_code"] or original["target_date"] != target:
                    raise ValueError("OWNER_PREDICTION_INPUT_IDENTITY_INVALID")
                row["materials_hash"] = original["materials_hash"]
                row["materials_input_hash"] = original["materials_input_hash"]
                row["snapshot_verified"] = True
    return rows


def horizon_dates(packet, code, at):
    """逐周期保留原日历结果，长周期不适配不能使已正式读回的一日结果失效。

    这里只解释原适配器的真实失败，不增加估值日、不改适用基金范围。原适配器
    明确报告覆盖边界或产品不在支持范围时单独留存；其他缺项仍然是可处理行动。
    """
    from app.services.comprehensive_multi_prediction import review_calendar
    from app.services.prediction_contract import PredictionFailure, target_dates

    dates, actions, failures = {}, [], []
    try:
        calendar = review_calendar(packet, code, at)
    except (ValueError, KeyError) as error:
        calendar, initial_error = None, error
    else:
        initial_error = None
    for horizon in HORIZONS[1:]:
        try:
            if initial_error is not None:
                raise initial_error
            dates[horizon] = target_dates(calendar, at, horizon)
        except (ValueError, KeyError) as error:
            payload = error.payload if isinstance(error, PredictionFailure) else {}
            failure = {"horizon": horizon, "failure": type(error).__name__ + ":" + str(error), "backend": payload}
            failures.append(failure)
            details = payload.get("details", {})
            outside = (
                payload.get("code") == "CALENDAR_RANGE_MISSING"
                and calendar is not None
                and details.get("calendarId") == calendar.calendar_id
                and details.get("coverageEnd") == calendar.coverage_end.isoformat()
                and details.get("requestedDate", "") > details.get("coverageEnd", "")
            )
            coverage_proof = None
            if not outside and calendar is not None and payload.get("code") == "CALENDAR_RANGE_MISSING":
                # 原 target_dates 在剩余交易日不足时以 start 调用 missing，错误中的
                # requestedDate 因而仍在覆盖内。用原周期长度和已验证的真实会话数
                # 证明边界不足，不推算或新增任何未来交易日。
                from bisect import bisect_left

                from app.services.prediction_contract import prediction_policy

                horizon_policy = next(h for h in prediction_policy()["horizons"] if h["horizon_id"] == horizon)
                base = calendar.required_base(at)
                available = len(calendar.sessions) - bisect_left(calendar.sessions, base) - 1
                if horizon_policy["unit"] == "TRADING_SESSION" and 0 <= available < horizon_policy["length"]:
                    outside = True
                    coverage_proof = {
                        "baseDate": base.isoformat(),
                        "availableSessionsAfterBase": available,
                        "requiredSessions": horizon_policy["length"],
                        "verifiedCoverageEnd": calendar.coverage_end.isoformat(),
                    }
            unsupported = (
                payload.get("code") == "CALENDAR_POLICY_MISSING"
                and details.get("fundCode") == code
                and payload.get("summary") == "该基金不在本次境内股票、混合和债券基金的已支持范围"
            )
            if outside or unsupported:
                dates[horizon] = {
                    "endDateStatus": "OUTSIDE_VERIFIED_CALENDAR" if outside else "UNSUPPORTED_PRODUCT_CALENDAR",
                    "fundCode": code,
                    "backendFailure": payload,
                    "calendarSourceHash": calendar.source_hash if calendar is not None else digest(packet),
                    "coverageProof": coverage_proof,
                }
            else:
                actions.append({"action": "RECHECK_CURRENT_INPUT_AND_CALENDAR", **failure})
    return dates, actions, failures


@material_read_cache
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fund-code", action="append", help="可选，仅检查本人关注内指定基金；定时任务默认全部本人关注")
    parser.add_argument("--blocked-file", type=Path, help="真实外部阻断的逐周期证据JSON数组，不能使用待核验作为原因")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--core-env", type=Path, default=Path(r"D:\Project-FullStack\workSpace01\02-java\.env"))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    binding = json.loads((project / ".local-runs/automation-owner/account.json").read_text(encoding="utf-8"))
    owner = binding["owner_user_id"]
    settings, at = get_settings(), now()
    current_window = window(at)
    # 同一次定时运行的目标窗口不可跨收市时间静默漂移；未做完必须保留原期失败。
    if args.output.exists():
        previous = read(args.output)
        if previous["owner_user_id"] != owner:
            raise ValueError("RECOVERY_LEDGER_OWNER_CHANGED")
        if previous["window"]["target_nav_date"] != current_window["target_nav_date"]:
            if previous["status"] == "CLOSED":
                print(json.dumps({"status": "CLOSED", "reused_previous_closed_window": True}, ensure_ascii=True))
                return
            previous.update(status="MISSED_DEADLINE", checked_at=at.isoformat(), all_predictions_generated=False)
            versioned_save(args.output, previous)
            print(json.dumps({"status": "MISSED_DEADLINE", "original_target": previous["window"]["target_nav_date"]}))
            raise SystemExit(2)
    scope = owner_scope(settings, owner)
    codes = sorted(set(args.fund_code or scope))
    if not set(codes) <= set(scope):
        raise ValueError("FUND_OUTSIDE_BOUND_OWNER_SCOPE")
    records = saved_records(owner, codes, current_window["target_nav_date"], args.core_env)
    exceptions = json.loads(args.blocked_file.read_text(encoding="utf-8")) if args.blocked_file else []
    results = []
    for code in codes:
        state = workflow_root() / "news/funds" / (code + ".json")
        news = read(state) if state.exists() else {}
        handoff, material, dates, failure = None, None, {}, None
        try:
            handoff = registered_materials(code, at=at)
        except (ValueError, OSError, KeyError) as error:
            failure = type(error).__name__ + ":" + str(error)
        # 旧交接可供识别新增证券，不能被当作本期已批准。
        old_path = receipt_path(code)
        old = read(old_path) if old_path.exists() else {}
        if old.get("materials_path") and source_exists(Path(old["materials_path"])):
            material = read(Path(old["materials_path"]))
            if material.get("fund_code") != code:
                raise ValueError("RECOVERY_MATERIAL_FUND_MISMATCH")
        reports = [r for r in (snapshot(code) or {}).get("reports", []) if r["publishedDate"] <= str(at.date())]
        report = max(reports, key=lambda r: (r["endDate"], r["publishedDate"])) if reports else None
        actions = recovery_actions(
            code,
            report,
            material,
            news,
            handoff,
            current_window["base_nav_date"],
            str(at.date()),
            target_reports={
                target: max(
                    (snapshot(target) or {}).get("reports", []),
                    key=lambda r: (r["endDate"], r["publishedDate"]),
                    default=None,
                )
                for target in (material or {}).get("exposure_chain", [code])[1:]
            },
        )
        if handoff and handoff.get("reviewed") and not handoff.get("gaps"):
            try:
                from app.services.fund_information_snapshot import prepare

                packet = prepare(at, code, handoff["materials_path"])
                dates, calendar_actions, calendar_failures = horizon_dates(packet, code, at)
                actions.extend(calendar_actions)
            except (ValueError, KeyError) as error:
                failure = type(error).__name__ + ":" + str(error)
                actions.append({"action": "RECHECK_CURRENT_INPUT_AND_CALENDAR", "failure": failure})
        from app.services.fund_prediction_snapshot_compat import materials_input_identity

        current_material_identity = materials_input_identity(read(Path(handoff["materials_path"]))) if handoff else None
        result = verify_outputs(
            code,
            current_window,
            dates,
            records,
            actions,
            exceptions,
            materials_hash=(handoff or {}).get("materials_hash"),
            materials_input_hash=current_material_identity,
        )
        result["prepare_failure"] = failure
        result["period_dates"] = dates
        results.append(result)
    pending = [r["fund_code"] for r in results if r["status"] == "ACTION_REQUIRED"]
    report = {
        "checked_at": at.isoformat(),
        "owner_user_id": owner,
        "window": current_window,
        "fund_count": len(codes),
        "status": "ACTION_REQUIRED" if pending else "CLOSED",
        "all_predictions_generated": all(h["status"] == "SAVED_AND_READ_BACK" for r in results for h in r["horizons"]),
        "pending_funds": pending,
        "funds": results,
    }
    versioned_save(args.output, report)
    print(
        json.dumps(
            {"status": report["status"], "pending_funds": pending, "output": str(args.output.resolve())},
            ensure_ascii=True,
        )
    )
    raise SystemExit(2 if pending else 0)


if __name__ == "__main__":
    main()
