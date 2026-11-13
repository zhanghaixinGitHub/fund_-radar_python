"""正式采用第二版分析：本机显式发布、数据库追加留档、页面只读公共字段。"""

import copy
import hashlib
import json
import re
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from app.db.session import get_engine
from app.services.direction_1d_protocol import ZONE, canonical, digest
from app.services.prediction_review_v2 import policy_v2
from app.services.prediction_trial_v2 import current_holdings, storage_check

PROJECT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT / ".local-runs"
HORIZONS = ("T1", "T5_V1", "T20_V1", "M6_V1")
VERSION = "COMPREHENSIVE_REVIEW_V2"
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def _stamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("V2_PUBLICATION_TIMEZONE_REQUIRED")
    return result.astimezone(ZONE)


def _load(path, cache):
    path = Path(path).resolve()
    if not path.is_relative_to(SOURCE_ROOT.resolve()):
        raise ValueError("V2_PUBLICATION_SOURCE_OUTSIDE_WORKSPACE")
    key = str(path)
    if key not in cache:
        if path.stat().st_size > MAX_SOURCE_BYTES:
            raise ValueError("V2_PUBLICATION_SOURCE_TOO_LARGE")
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8-sig"))
        if isinstance(value, dict) and "hash" in value and "payload" in value:
            if digest(value["payload"]) != value["hash"]:
                raise ValueError("V2_PUBLICATION_ENVELOPE_HASH_MISMATCH")
            value = value["payload"]
        cache[key] = (value, hashlib.sha256(raw).hexdigest())
    return cache[key][0]


def _strings(value):
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        return list(dict.fromkeys(item for item in value if isinstance(item, str) and item.strip()))
    return []


def _case_source(row, source):
    if row["source"]["kind"] == "ORIGINAL_RESULT":
        return source
    matches = [
        a
        for a in source.get("assessments", [])
        if a.get("fund_code") == row["fund_code"] and a.get("horizon") == row["horizon"]
    ]
    if len(matches) != 1:
        raise ValueError("V2_PUBLICATION_ASSESSMENT_NOT_UNIQUE")
    return matches[0]


def _public_period(row, assessment, original, source, reviewed_at):
    review = assessment.get("review", {})
    period_review = assessment.get("period_assessment", {})
    period = row["period"]
    decision = "ABSTAIN" if row["decision"] == "BLOCKED" else row["decision"]
    primary = assessment.get("primary_driver") or period_review.get("primary_driver")
    counter = assessment.get("counter_tradeoff") or period_review.get("counter_tradeoff") or review.get("tradeoff")
    conditions = _strings(
        assessment.get("reversal_conditions")
        or assessment.get("change_conditions")
        or review.get("change_conditions")
        or review.get("recovery_conditions")
    )
    limitations = _strings(assessment.get("limitations"))
    limitations += _strings(
        assessment.get("remaining")
        or assessment.get("remaining_specific")
        or assessment.get("missing_effect")
        or period_review.get("missing_effect")
    )
    if row["decision"] == "BLOCKED":
        limitations += _strings(period.get("reason")) or ["本周期日期或资料尚未核齐，暂不判断。"]
    if not primary or not counter or not conditions:
        raise ValueError("V2_PUBLICATION_REASON_OR_CONDITIONS_MISSING")
    source_cutoff = (
        assessment.get("source_cutoff")
        or assessment.get("information_cutoff")
        or source.get("source_cutoff")
        or source.get("as_of")
        or source.get("created_at")
        or source.get("reviewed_at")
        or reviewed_at
    )
    source_cutoff = _stamp(source_cutoff).isoformat()
    conditional = row["horizon"] == "M6_V1" and (
        period.get("end_date") is None or assessment.get("output_kind") == "CONDITIONAL_OUTLOOK"
    )
    if conditional and period.get("end_date") is not None:
        raise ValueError("V2_PUBLICATION_CONDITIONAL_END_DATE_INVALID")
    nominal = period.get("nominal_end_date")
    if conditional and nominal is None:
        from app.services.prediction_contract import add_months

        nominal = str(add_months(date.fromisoformat(period["base_date"]), 6))
    result = {
        "horizonId": row["horizon"],
        "decision": decision,
        "kind": "CONDITIONAL_OUTLOOK" if conditional else "DATED_ANALYSIS",
        "confidence": "LOW",
        "baseNavDate": period["base_date"],
        "startDate": period["start_date"],
        "endDate": period.get("end_date"),
        "nominalEndDate": nominal,
        "analysisAt": reviewed_at,
        "sourceCutoff": source_cutoff,
        "summary": assessment.get("summary") or primary,
        "primaryReason": primary,
        "counterReason": counter,
        "conditions": conditions,
        "limitations": list(dict.fromkeys(limitations)),
        "awaitingReason": ("；".join(limitations) or "正反因素尚未形成明确优势，暂不判断。")
        if decision == "ABSTAIN"
        else "官方未来日历尚未核齐，保留半年条件展望。"
        if conditional
        else None,
        "originallyForecast": (
            row["horizon"] == "T1"
            and row["source"]["kind"] == "ORIGINAL_RESULT"
            and original["decision"] in {"UP", "DOWN"}
        ),
        "excludedFromAccuracy": True,
    }
    return result


def prepare_release(ledger_path: Path, *, holdings=None, clock=None):
    """复用已验收的研究，不生成新方向；把本机依据封入可审计正式档案。"""
    clock = clock or datetime.now(ZONE)
    cache = {}
    ledger_path = Path(ledger_path).resolve()
    ledger = _load(ledger_path, cache)
    manifest = _load(ledger_path.parent / "case-manifest.json", cache)
    if digest(manifest) != ledger["parent_manifest_hash"]:
        raise ValueError("V2_PUBLICATION_MANIFEST_MISMATCH")
    parent = _load(manifest["parent_ledger_path"], cache)
    original = _load(manifest["original_summary_path"], cache)
    if digest(parent) != manifest["parent_ledger_hash"] or digest(original) != manifest["original_summary_hash"]:
        raise ValueError("V2_PUBLICATION_PARENT_HASH_MISMATCH")
    verification = ledger["verification"]
    if (
        verification.get("old_36_results_unchanged") is not True
        or verification.get("previous_23_ledger_unchanged") is not True
        or verification.get("exact_reading_spans_checked", 0) <= 0
        or verification.get("passed_related_tests", 0) <= 0
        or verification.get("storage_check", {}).get("count") != 0
    ):
        raise ValueError("V2_PUBLICATION_PREVIOUS_VERIFICATION_INCOMPLETE")
    rows = ledger["latest_36"]
    codes = {r["fund_code"] for r in rows}
    if (
        len(rows) != 36
        or len(codes) != 9
        or len({(r["fund_code"], r["horizon"]) for r in rows}) != 36
        or any({r["horizon"] for r in rows if r["fund_code"] == c} != set(HORIZONS) for c in codes)
    ):
        raise ValueError("V2_PUBLICATION_36_UNIQUE_CASES_REQUIRED")
    if any(not re.fullmatch(r"[0-9]{6}", c) for c in codes):
        raise ValueError("V2_PUBLICATION_FUND_CODE_INVALID")
    inventory = holdings if holdings is not None else current_holdings()
    if inventory["owner_hash"] != manifest["owner_hash"] or not codes <= set(inventory["scope"]):
        raise ValueError("V2_PUBLICATION_CURRENT_OWNER_OR_HOLDINGS_MISMATCH")
    originals = {}
    policy = policy_v2()
    policy_hash = digest(policy)
    for old in original["results"]:
        record = _load(old["result_path"], cache)
        if digest(record) != old["result_hash"] or record["policy_hash"] != policy_hash:
            raise ValueError("V2_PUBLICATION_ORIGINAL_RESULT_OR_POLICY_CHANGED")
        originals[record["fund_code"], record["horizon"]] = record
    if set(originals) != {(r["fund_code"], r["horizon"]) for r in rows}:
        raise ValueError("V2_PUBLICATION_ORIGINAL_CASE_SET_MISMATCH")
    # 原件和引文的深验收已经留档；这里逐一重核其内容摘要，防止验收后被替换。
    for receipt in verification.get("raw_receipt_checks", []):
        path = Path(receipt["path"]).resolve()
        if not path.is_relative_to(SOURCE_ROOT.resolve()):
            raise ValueError("V2_PUBLICATION_RECEIPT_PATH_INVALID")
        actual = (
            digest(_load(path, cache))
            if receipt["scheme"] == "PAYLOAD"
            else hashlib.sha256(path.read_bytes()).hexdigest()
        )
        if actual != receipt["hash"]:
            raise ValueError("V2_PUBLICATION_RAW_RECEIPT_CHANGED")
    views, cases, packets = {}, [], {}
    for row in rows:
        if row["decision"] not in {"UP", "DOWN", "ABSTAIN"} or row["confidence"] != "LOW":
            raise ValueError("V2_PUBLICATION_DECISION_INVALID")
        old = originals[row["fund_code"], row["horizon"]]
        if digest(old) != row["original_result_hash"]:
            raise ValueError("V2_PUBLICATION_ORIGINAL_RESULT_MISMATCH")
        source = _load(row["source"]["path"], cache)
        if digest(source) != row["source"]["hash"]:
            raise ValueError("V2_PUBLICATION_SOURCE_HASH_MISMATCH")
        assessment = _case_source(row, source)
        if assessment["decision"] != row["decision"] or assessment["confidence"] != "LOW":
            raise ValueError("V2_PUBLICATION_SOURCE_DECISION_MISMATCH")
        period, old_period = row["period"], old["period"]
        if any(period.get(k) != old_period.get(k) for k in ("base_date", "start_date")):
            raise ValueError("V2_PUBLICATION_PERIOD_IDENTITY_CHANGED")
        if period.get("end_date") != old_period.get("end_date"):
            # 唯一日期修复必须指向已验收、与总回执绑定的官方日历；其他周期复用原验收。
            evidence = _load(period["source_path"], cache)
            if (
                digest(evidence) != period["source_hash"]
                or digest(evidence) != verification["calendar_verification_hash"]
                or evidence["periods"][row["horizon"]]["endDate"] != period["end_date"]
            ):
                raise ValueError("V2_PUBLICATION_PERIOD_REPAIR_MISMATCH")
        reviewed = (
            assessment.get("reviewed_at")
            or assessment.get("research_at")
            or assessment.get("evaluated_at")
            or source.get("reviewed_at")
            or source.get("created_at")
        )
        reviewed = _stamp(reviewed)
        if reviewed > clock or _stamp(ledger["created_at"]) > clock:
            raise ValueError("V2_PUBLICATION_FUTURE_REVIEW_TIME")
        code = row["fund_code"]
        view = views.setdefault(
            code,
            {
                "fundCode": code,
                "reviewedAt": reviewed.isoformat(),
                "marketAsOf": row["period"]["base_date"],
                "periods": [],
            },
        )
        if view["marketAsOf"] != row["period"]["base_date"]:
            raise ValueError("V2_PUBLICATION_BASE_DATE_MISMATCH")
        view["reviewedAt"] = max(_stamp(view["reviewedAt"]), reviewed).isoformat()
        view["periods"].append(_public_period(row, assessment, old, source, reviewed.isoformat()))
        cases.append(
            {
                "case": row,
                "analysis": assessment,
                "original_record": old,
                "original_source_cutoff": old.get("source_cutoff"),
                "source_hash": digest(source),
            }
        )
        # 原始数值放在JSON字符串中，避免JSONB把-0.0归一为0.0后破坏原件摘要。
        packets[digest(source)] = {"canonical_json": canonical(source)}
    for view in views.values():
        view["periods"].sort(key=lambda r: HORIZONS.index(r["horizonId"]))
    archive = {
        "version": VERSION,
        "policy": policy,
        "policy_hash": policy_hash,
        "ledger": ledger,
        "manifest": manifest,
        "cases": cases,
        "source_packets": packets,
        "source_receipts": [
            {"path": k, "file_sha256": v[1], "payload_hash": digest(v[0])} for k, v in sorted(cache.items())
        ],
        "owner_audit": {"owner_hash": inventory["owner_hash"], "current_fund_codes": sorted(inventory["scope"])},
        "excluded_from_accuracy": True,
    }
    content_hash = digest(archive)
    return {
        "version": VERSION,
        "policyHash": policy_hash,
        "contentHash": content_hash,
        "archive": archive,
        "views": views,
        "summary": {
            "records": len(cases),
            "funds": len(views),
            "decisions": dict(Counter(r["decision"] for r in rows)),
            "t1RetrospectivesAtAdoption": sum(r["horizon"] == "T1" for r in rows),
            "excludedFromAccuracy": True,
        },
    }


def adopt_release(ledger_path: Path, *, adopt=False):
    """默认预览；显式采用才追加数据库记录，不触碰旧预测和准确率。"""
    prepared = prepare_release(ledger_path)
    return _publish(prepared, adopt=adopt)


def _publish(prepared, *, adopt=False):
    if not adopt:
        return {"status": "PREVIEW", **{k: prepared[k] for k in ("version", "policyHash", "contentHash", "summary")}}
    storage_check(
        prepared["archive"].get("ledger", {}).get("created_at") or prepared["archive"]["manifest"]["captured_at"]
    )
    current = current_holdings()
    if current["owner_hash"] != prepared["archive"]["owner_audit"]["owner_hash"] or not set(prepared["views"]) <= set(
        current["scope"]
    ):
        raise ValueError("V2_PUBLICATION_HOLDINGS_CHANGED_BEFORE_ADOPTION")
    with get_engine().begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(hashtext('prediction_review_v2_release'))"))
        release_id = uuid4()
        connection.execute(
            text("""INSERT INTO prediction_review_v2_release
          (release_id,version,policy_hash,content_hash,archive,views)
          VALUES(:id,:version,:policy,:hash,CAST(:archive AS jsonb),CAST(:views AS jsonb))
          ON CONFLICT(content_hash) DO NOTHING"""),
            {
                "id": release_id,
                "version": VERSION,
                "policy": prepared["policyHash"],
                "hash": prepared["contentHash"],
                "archive": canonical(prepared["archive"]),
                "views": canonical(prepared["views"]),
            },
        )
        stored = (
            connection.execute(
                text("SELECT * FROM prediction_review_v2_release WHERE content_hash=:hash"),
                {"hash": prepared["contentHash"]},
            )
            .mappings()
            .one()
        )
        if digest(stored["archive"]) != prepared["contentHash"] or stored["views"] != prepared["views"]:
            raise ValueError("V2_PUBLICATION_DATABASE_READBACK_MISMATCH")
    # 先提交并读回正式结果，再完成同一期的新请求；重复采用也可补完先前中断的进度。
    from app.services.prediction_review_v2_requests import complete_from_release

    complete_from_release(prepared, stored["release_id"], stored["adopted_at"])
    return {
        "status": "ADOPTED",
        "version": VERSION,
        "releaseId": str(stored["release_id"]),
        "adoptedAt": stored["adopted_at"].isoformat(),
        "policyHash": prepared["policyHash"],
        "contentHash": prepared["contentHash"],
        "reused": stored["release_id"] != release_id,
        "summary": prepared["summary"],
    }


def _active(connection, fund_code=None):
    # 统一启动迁移前的只读检查允许尚无新表；其他数据库错误原样上报。
    if connection.scalar(text("SELECT to_regclass('prediction_review_v2_release')")) is None:
        return None
    where = "WHERE views ? :code" if fund_code is not None else ""
    return (
        connection.execute(
            text(f"""SELECT release_id,version,policy_hash,adopted_at,views
      FROM prediction_review_v2_release {where} ORDER BY adopted_at DESC,release_id DESC LIMIT 1"""),
            {"code": fund_code} if fund_code is not None else {},
        )
        .mappings()
        .first()
    )


def adoption_status():
    """正式入口按数据库采用记录判断版本；不靠版本字符串或本机试验文件。"""
    with get_engine().connect() as connection:
        row = _active(connection)
    if row is None:
        return None
    return {
        "version": row["version"],
        "policyHash": row["policy_hash"],
        "adoptedAt": row["adopted_at"].isoformat(),
        "releaseId": str(row["release_id"]),
    }


def public_view(view, adopted_at, *, clock=None):
    """读取只改变展示种类；不改原分析时刻、方向和旧记录。"""
    result = copy.deepcopy(view)
    result["adoptedAt"] = adopted_at.isoformat() if isinstance(adopted_at, datetime) else adopted_at
    clock = clock or datetime.now(ZONE)
    for period in result["periods"]:
        # 页面使用日常中文；原审阅及其校验值仍留在只追加的审计原文中。
        def display_text(value):
            if not isinstance(value, str):
                return value
            value = re.sub(r"(?<![A-Za-z0-9_])LOW(?![A-Za-z0-9_])", "低把握", value)
            return value.replace("CONDITIONAL_OUTLOOK", "半年条件展望").replace(
                "本项不是ASSET_INCOME收益模型", "不能按利息收入推算回报"
            )

        for field in ("summary", "primaryReason", "counterReason", "awaitingReason"):
            if field in period:
                period[field] = display_text(period[field])
        for field in ("conditions", "limitations"):
            period[field] = [display_text(value) for value in period.get(field, [])]
        if period["horizonId"] == "T1":
            end = period.get("endDate") or period.get("startDate")
            ended = end and clock >= datetime.combine(datetime.fromisoformat(end).date(), time(15), ZONE)
            if ended or period.get("kind") == "RETROSPECTIVE":
                period["kind"] = "RETROSPECTIVE"
                note = "这个一日分析的旧区间已经结束，仅供回看，不是下一交易日预测。"
                if note not in period["limitations"]:
                    period["limitations"].append(note)
                period["note"] = note
    return result


def current_review(fund_code):
    """页面只从正式数据库读取，不接触.local-runs、个人持仓或外部资料。"""
    if not re.fullmatch(r"[0-9]{6}", fund_code):
        raise ValueError("V2_PUBLICATION_FUND_CODE_INVALID")
    with get_engine().connect() as connection:
        row = _active(connection, fund_code)
    if row is None or fund_code not in row["views"]:
        return None
    return public_view(row["views"][fund_code], row["adopted_at"])


def read_review(fund_code):
    """供页面与新请求入口共用的正式读函数。"""
    return current_review(fund_code)


def prepare_trial_release(directory: Path, *, holdings=None, clock=None):
    """任意当前持仓的严格第二版审阅结果，重新验收后封包；不重新生成观点。"""
    from app.services.fund_prediction_snapshot_compat import materials_input_identity
    from app.services.prediction_review_v2 import validate_cross_horizons, validate_review_v2
    from app.services.prediction_trial_v2 import checked_input

    clock = clock or datetime.now(ZONE)
    directory = Path(directory).resolve()
    cache = {}
    manifest = _load(directory / "manifest.json", cache)
    policy = policy_v2()
    if manifest["policy_hash"] != digest(policy):
        raise ValueError("V2_PUBLICATION_POLICY_CHANGED")
    inventory = holdings if holdings is not None else current_holdings()
    entries = manifest["selection"]
    selected = {e["fund_code"] for e in entries}
    if (
        not entries
        or len(selected) != len(entries)
        or len(entries) > 500
        or manifest["inventory"]["owner_hash"] != inventory["owner_hash"]
        or not selected <= set(inventory["scope"])
    ):
        raise ValueError("V2_PUBLICATION_CURRENT_OWNER_OR_HOLDINGS_MISMATCH")
    candidates = list((directory / "summaries").glob("*.json"))
    if not candidates or len(candidates) > 100:
        raise ValueError("V2_PUBLICATION_EVALUATED_SUMMARY_REQUIRED")
    summaries = [_load(p, cache) for p in candidates]
    summary = max(summaries, key=lambda s: _stamp(s["checked_at"]))
    expected = {(c, h) for c in selected for h in HORIZONS}
    if (
        summary.get("all_result_hashes_read_back") is not True
        or summary.get("owner_scope_read_back") is not True
        or len(summary["results"]) != len(expected)
        or {(r["fund_code"], r["horizon"]) for r in summary["results"]} != expected
    ):
        raise ValueError("V2_PUBLICATION_ALL_EVALUATED_PERIODS_REQUIRED")
    inputs, validated, cases, views = {}, [], [], {}
    for entry in entries:
        packet, data = checked_input(directory, entry)
        if digest(packet) != entry["packet_hash"]:
            raise ValueError("V2_PUBLICATION_TRIAL_INPUT_CHANGED")
        inputs[entry["fund_code"]] = (packet, data)
    for row in summary["results"]:
        record = _load(row["result_path"], cache)
        if digest(record) != row["result_hash"] or record["decision"] != row["decision"]:
            raise ValueError("V2_PUBLICATION_RESULT_CHANGED")
        code, h = row["fund_code"], row["horizon"]
        packet, data = inputs[code]
        review = record["review"]
        evaluated = _stamp(record["evaluated_at"])
        identity = digest({"packet": digest(packet), "review": review, "policy": manifest["policy_hash"]})
        if (
            evaluated > clock
            or evaluated > _stamp(summary["checked_at"])
            or record.get("identity") != identity
            or record.get("review_hash") != digest(review)
            or record.get("packet_hash") != digest(packet)
            or record["policy_hash"] != digest(policy)
            or record["fund_code"] != code
            or record["horizon"] != h
            or record["period"] != packet["periods"][h]
        ):
            raise ValueError("V2_PUBLICATION_STRICT_RESULT_IDENTITY_INVALID")
        checked = validate_review_v2(packet, data, review, at=evaluated)
        if checked.decision not in {"UP", "DOWN", "ABSTAIN", "BLOCKED"} or checked.decision != record["decision"]:
            raise ValueError("V2_PUBLICATION_DECISION_INVALID")
        validated.append(checked)
        review_path = directory / "reviews" / (record["review_hash"] + ".json")
        if _load(review_path, cache) != review:
            raise ValueError("V2_PUBLICATION_REVIEW_ORIGINAL_CHANGED")
        source = _load(packet["source_path"], cache)
        if digest(source) != packet["source_hash"]:
            raise ValueError("V2_PUBLICATION_SOURCE_CHANGED")
        row_view = {
            "fund_code": code,
            "horizon": h,
            "decision": record["decision"],
            "period": record["period"],
            "source": {"kind": "ORIGINAL_RESULT"},
        }
        view = views.setdefault(
            code,
            {
                "fundCode": code,
                "reviewedAt": evaluated.isoformat(),
                "marketAsOf": record["period"]["base_date"],
                "periods": [],
            },
        )
        view["reviewedAt"] = max(_stamp(view["reviewedAt"]), evaluated).isoformat()
        if view["marketAsOf"] != record["period"]["base_date"]:
            raise ValueError("V2_PUBLICATION_BASE_DATE_MISMATCH")
        item = _public_period(row_view, record, record, record, evaluated.isoformat())
        # 材料归属留在私有审计。没有原材料绑定时照实展示分析，但不能完成每日事前任务。
        material_path, material_hash = data.get("materials_path"), data.get("materials_hash")
        materials_identity = None
        if material_path and material_hash:
            material = _load(material_path, cache)
            if digest(material) != material_hash or material.get("fund_code") != code:
                raise ValueError("V2_PUBLICATION_MATERIAL_IDENTITY_MISMATCH")
            materials_identity = materials_input_identity(material)
        end = record["period"].get("end_date")
        deadline = datetime.combine(datetime.fromisoformat(record["period"]["start_date"]).date(), time(15), ZONE)
        ex_ante = bool(
            packet.get("source_ready")
            and materials_identity
            and end
            and evaluated < deadline
            and _stamp(record["source_cutoff"]) <= evaluated
            and record["period"].get("status") == "RESOLVED"
        )
        item["originallyForecast"] = bool(h == "T1" and ex_ante and record["decision"] in {"UP", "DOWN"})
        view["periods"].append(item)
        cases.append(
            {
                "case": row_view,
                "analysis": record,
                "original_record": record,
                "workflow_identity": {
                    "owner_hash": inventory["owner_hash"],
                    "materials_hash": material_hash,
                    "materials_input_hash": materials_identity,
                    "snapshot_verified": True,
                    "ex_ante_eligible": ex_ante,
                    "analysis_version": VERSION,
                    "base_date": record["period"]["base_date"],
                    "target_date": record["period"]["start_date"],
                    "end_date": end,
                },
            }
        )
    validate_cross_horizons(validated)
    for view in views.values():
        view["periods"].sort(key=lambda v: HORIZONS.index(v["horizonId"]))
    archive = {
        "version": VERSION,
        "publication_type": "STRICT_V2_TRIAL",
        "policy": policy,
        "policy_hash": digest(policy),
        "manifest": manifest,
        "summary": summary,
        "cases": cases,
        "source_packets": {
            c: {"canonical_json": canonical({"packet": packet, "data": data})} for c, (packet, data) in inputs.items()
        },
        "source_receipts": [
            {"path": k, "file_sha256": v[1], "payload_hash": digest(v[0])} for k, v in sorted(cache.items())
        ],
        "owner_audit": {"owner_hash": inventory["owner_hash"], "current_fund_codes": sorted(inventory["scope"])},
        "excluded_from_accuracy": True,
    }
    return {
        "version": VERSION,
        "policyHash": digest(policy),
        "contentHash": digest(archive),
        "archive": archive,
        "views": views,
        "summary": {
            "records": len(cases),
            "funds": len(views),
            "excludedFromAccuracy": True,
            "decisions": dict(Counter(r["decision"] for r in summary["results"])),
        },
    }


def publish_trial(directory: Path, *, adopt=False):
    """只有完成严格审阅的目录可发布；显式采用仍保留当时的真实窗口。"""
    return _publish(prepare_trial_release(directory), adopt=adopt)


def daily_records(owner, codes, target):
    """每日任务的私有核对接口；不读取本机文件，不把历史研究当成本期事前方向。"""
    codes = list(dict.fromkeys(codes))
    if not codes:
        return []
    owner_hash = digest(owner)
    result = []
    with get_engine().connect() as connection:
        if connection.scalar(text("SELECT to_regclass('prediction_review_v2_release')")) is None:
            return []
        db_clock = connection.scalar(text("SELECT clock_timestamp()"))
        # 每只基金取自身最近采用的版本；单只新发布不会使其他基金的结果消失。
        for code in codes:
            stored = (
                connection.execute(
                    text("""SELECT release_id,adopted_at,archive,policy_hash
              FROM prediction_review_v2_release WHERE views ? :code
              ORDER BY adopted_at DESC,release_id DESC LIMIT 1"""),
                    {"code": code},
                )
                .mappings()
                .first()
            )
            if stored is None or stored["archive"].get("publication_type") != "STRICT_V2_TRIAL":
                continue
            for item in stored["archive"]["cases"]:
                row, identity = item["case"], item.get("workflow_identity", {})
                if (
                    row["fund_code"] != code
                    or identity.get("owner_hash") != owner_hash
                    or identity.get("target_date") != str(target)
                ):
                    continue
                cutoff = datetime.combine(datetime.fromisoformat(identity["target_date"]).date(), time(15), ZONE)
                eligible = identity.get("ex_ante_eligible") is True and stored["adopted_at"] < cutoff
                # 区间结束后的旧结果可审计，但不再作为待生成的未来方向。
                expired = db_clock >= cutoff
                kind = (
                    "CONDITIONAL_OUTLOOK"
                    if row["horizon"] == "M6_V1"
                    else "DATED_FORECAST"
                    if eligible and not expired
                    else "RETROSPECTIVE"
                )
                result.append(
                    {
                        "id": str(stored["release_id"]) + ":" + code + ":" + row["horizon"],
                        "fund_code": code,
                        "horizon": row["horizon"],
                        "decision": "ABSTAIN" if row["decision"] == "BLOCKED" else row["decision"],
                        "base_date": identity["base_date"],
                        "target_date": identity["target_date"],
                        "end_date": identity.get("end_date"),
                        "owner_verified": True,
                        "snapshot_verified": identity.get("snapshot_verified") is True,
                        "materials_input_hash": identity.get("materials_input_hash"),
                        "materials_hash": identity.get("materials_hash"),
                        "analysis_version": VERSION,
                        "record_kind": kind,
                        "ex_ante_eligible": bool(eligible and not expired),
                        "policy_hash": stored["policy_hash"],
                    }
                )
    return result
