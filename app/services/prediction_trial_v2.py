"""当前本人模拟持仓的V2独立试验；业务库只读，结果与正式预测分开。"""

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import text

from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read, save
from app.services.prediction_review_v2 import policy_v2, validate_cross_horizons, validate_review_v2

PROJECT = Path(__file__).resolve().parents[2]
HORIZONS = ("T1", "T5_V1", "T20_V1", "M6_V1")


def local_path(path):
    """来源与输出限制在D盘当前工程，不能回落旧C盘或使用任意外部路径。"""
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT / ".local-runs"):
        raise ValueError("V2_PATH_OUTSIDE_LOCAL_RUNS")
    return path


def current_holdings():
    from app.commands.fund_prediction_followthrough import owner_scope
    from app.core.config import get_settings
    from app.db.session import get_engine
    from app.repositories.direction_1d import profiles

    identity = json.loads((PROJECT / ".local-runs/automation-owner/account.json").read_text(encoding="utf-8"))
    owner = identity["owner_user_id"]
    scope = owner_scope(get_settings(), owner)
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        rows = profiles(connection, scope) if scope else []
    if {r["fund_code"] for r in rows} != set(scope):
        raise ValueError("V2_HOLDING_PROFILE_MISSING")
    # 只保存业务档案和账号摘要，不输出凭据或账号配置文件正文。
    fields = ("fund_code", "fund_name", "fund_type", "invest_type", "source_fund_type", "benchmark", "profile_hash")
    return {"owner_hash": digest(owner), "scope": scope, "profiles": [{k: r.get(k) for k in fields} for r in rows]}


def storage_check(since):
    """只查本次范围的存储事件；事件日志读取失败也不能继续业务写入。"""
    if sys.platform != "win32":
        raise ValueError("V2_WINDOWS_STORAGE_CHECK_REQUIRED")
    instant = datetime.fromisoformat(since)
    if instant.tzinfo is None:
        raise ValueError("V2_STORAGE_TIMEZONE_REQUIRED")
    stamp = instant.isoformat()
    command = (
        "$ErrorActionPreference='Stop';"
        f"$trialSince=[datetimeoffset]::Parse('{stamp}').LocalDateTime;"
        "try {$trialEvents=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$trialSince;Level=2,3} "
        "-ErrorAction Stop | Where-Object {$_.ProviderName -match 'disk|ntfs|stor|volmgr'})} "
        "catch {if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') {$trialEvents=@()} else {throw}};"
        "@{checked_at=(Get-Date -Format o);count=$trialEvents.Count;read_succeeded=$true} | ConvertTo-Json -Compress"
    )
    # 跟随统一终端使用PowerShell 7，避免旧Windows PowerShell继承新版模块路径后无法读取事件。
    shell = shutil.which("pwsh.exe") or "powershell.exe"
    process = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", command], capture_output=True, text=True, timeout=30
    )
    if process.returncode:
        raise ValueError("V2_STORAGE_EVENT_READ_FAILED")
    result = json.loads(process.stdout)
    if result["count"]:
        raise OSError("V2_NEW_STORAGE_ERROR_STOP_WRITES")
    return result


def save_once(path, payload):
    if path.exists():
        if read(path) != payload:
            raise ValueError("V2_IMMUTABLE_ARTIFACT_CONFLICT")
        return
    save(path, payload)
    if read(path) != payload:
        raise OSError("V2_WRITE_READBACK_MISMATCH")


def load_source(path, kind):
    source = read(local_path(path))
    if kind == "FROZEN":
        if source.get("format") != "CODEX_FUND_INPUT_V1" or digest(source["input"]) != source["input_hash"]:
            raise ValueError("V2_ORIGINAL_PACKET_INVALID")
        return source, source["input"]
    if kind == "PARTIAL_MATERIALS":
        return source, source
    raise ValueError("V2_SOURCE_KIND_INVALID")


def prepare_trial(directory, selection, *, holdings=None, clock=None):
    """记下本次持仓范围及使用的资料，复用原件，不重新采集或生成方向。"""
    directory = local_path(directory)
    if (directory / "manifest.json").exists():
        raise ValueError("V2_TRIAL_ALREADY_PREPARED")
    at = clock or now()
    inventory = holdings or current_holdings()
    entries = json.loads(Path(selection).read_text(encoding="utf-8"))
    codes = [row["fund_code"] for row in entries]
    if not codes or len(codes) != len(set(codes)) or not set(codes) <= set(inventory["scope"]):
        raise ValueError("V2_SELECTION_OUTSIDE_CURRENT_HOLDINGS")
    profiles = {row["fund_code"]: row for row in inventory["profiles"]}
    packets = []
    for row in entries:
        code, kind = row["fund_code"], row["source_kind"]
        path = local_path(row["source_path"])
        source, data = load_source(path, kind)
        if data["fund_code"] != code:
            raise ValueError("V2_SOURCE_FUND_MISMATCH")
        cutoff = data.get("as_of") or data.get("workflow_reviewed_at")
        if not cutoff or datetime.fromisoformat(cutoff) > at:
            raise ValueError("V2_SOURCE_CUTOFF_INVALID")
        window = data.get("window", {})
        base = window.get("base_nav_date") or data.get("calendar_evidence", {}).get("base_date")
        target = window.get("target_nav_date") or data.get("calendar_evidence", {}).get("target_date")
        if not base or not target:
            raise ValueError("V2_SOURCE_WINDOW_MISSING")
        ready = kind == "FROZEN" and data.get("time_context", {}).get("baseline_complete") is True
        periods = {}
        for horizon in HORIZONS:
            periods[horizon] = {
                "status": "UNVERIFIED",
                "base_date": base,
                "start_date": target,
                "end_date": None,
                "reason": "产品适用估值规则或官方日历尚未核齐",
            }
            if horizon == "T1" and ready:
                periods[horizon].update(status="RESOLVED", end_date=target, reason=None)
            elif horizon != "T1" and kind == "FROZEN":
                sibling = path.parent.parent / horizon / "input.json"
                if sibling.exists():
                    original = read(sibling)
                    contents = original.get("input", {})
                    dates, other = contents.get("dates", {}), contents.get("data", {})
                    if (
                        original.get("format") != "CODEX_COMPREHENSIVE_MULTI_V1"
                        or digest(contents) != original.get("input_hash")
                        or other.get("fund_code") != code
                        or dates.get("horizonId") != horizon
                        or dates.get("baseNavDate") != base
                        or dates.get("startDate") != target
                    ):
                        raise ValueError("V2_PERIOD_ORIGINAL_MISMATCH")
                    periods[horizon].update(
                        status=dates["endDateStatus"],
                        end_date=dates["endDate"],
                        # 名义六个月日期可确定；保留它不代表已核实最终估值日。
                        nominal_end_date=dates.get("nominalEndDate"),
                        calendar_id=dates.get("calendarId"),
                        calendar_hash=dates.get("calendarHash"),
                        source_path=str(sibling),
                        source_hash=digest(original),
                        reason=None,
                    )
        packets.append(
            {
                "format": "FUND_PREDICTION_V2_TRIAL_INPUT",
                "fund_code": code,
                "captured_at": at.isoformat(),
                "source_kind": kind,
                "source_path": str(path),
                "source_hash": digest(source),
                "source_cutoff": cutoff,
                "source_ready": ready,
                "input_mode": "EXISTING_FROZEN_INPUT_REVIEW" if ready else "INCOMPLETE_MATERIAL_REVIEW",
                "profile": profiles[code],
                "sample_label": row["sample_label"],
                "selection_reason": row["reason"],
                "periods": periods,
            }
        )
    # 所有只读检查完成后再发布输入；中断时原选择清单和首次输入仍保留。
    manifest = {
        "version": policy_v2()["version"],
        "policy_hash": digest(policy_v2()),
        "policy_snapshot": policy_v2(),
        "captured_at": at.isoformat(),
        "inventory": inventory,
        "selection": [
            {"fund_code": p["fund_code"], "packet_hash": digest(p), "sample_label": p["sample_label"]} for p in packets
        ],
        "production_publication": False,
    }
    for packet in packets:
        save_once(directory / "inputs" / (packet["fund_code"] + ".json"), packet)
    save_once(directory / "manifest.json", manifest)
    return manifest


def checked_input(directory, entry):
    packet = read(directory / "inputs" / (entry["fund_code"] + ".json"))
    if digest(packet) != entry["packet_hash"]:
        raise ValueError("V2_TRIAL_INPUT_CHANGED")
    source, data = load_source(packet["source_path"], packet["source_kind"])
    if digest(source) != packet["source_hash"]:
        raise ValueError("V2_ORIGINAL_SOURCE_CHANGED")
    for period in packet["periods"].values():
        if period.get("source_path") and digest(read(local_path(period["source_path"]))) != period["source_hash"]:
            raise ValueError("V2_PERIOD_SOURCE_CHANGED")
    return packet, data


def evaluate_trial(directory, reviews_file, *, holdings_provider=current_holdings, clock=None):
    """实际导入逐份审阅，先全部校验再落结果；不接受已清仓或其他账号的样本。"""
    directory = local_path(directory)
    manifest = read(directory / "manifest.json")
    if manifest["policy_hash"] != digest(policy_v2()):
        raise ValueError("V2_POLICY_CHANGED_NEW_TRIAL_REQUIRED")
    inventory = holdings_provider()
    selected = {item["fund_code"] for item in manifest["selection"]}
    if inventory["owner_hash"] != manifest["inventory"]["owner_hash"] or not selected <= set(inventory["scope"]):
        raise ValueError("V2_CURRENT_HOLDING_OR_OWNER_CHANGED")
    values = json.loads(Path(reviews_file).read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("V2_REVIEWS_LIST_REQUIRED")
    keys = [(v.get("fund_code"), v.get("horizon")) for v in values]
    expected = {(code, h) for code in selected for h in HORIZONS}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("V2_ALL_SELECTED_PERIODS_REQUIRED")
    at = clock or now()
    indexed = {(v["fund_code"], v["horizon"]): v for v in values}
    validated = []
    for entry in manifest["selection"]:
        packet, data = checked_input(directory, entry)
        for h in HORIZONS:
            r = validate_review_v2(packet, data, indexed[entry["fund_code"], h], at=at)
            validated.append((packet, r))
    validate_cross_horizons([r for _, r in validated])
    # 验证期间用户可能清仓；保存前再次核对，不能沿用准备阶段的旧持仓。
    final_scope = holdings_provider()
    if final_scope["owner_hash"] != inventory["owner_hash"] or not selected <= set(final_scope["scope"]):
        raise ValueError("V2_HOLDINGS_CHANGED_BEFORE_SAVE")
    results = []
    for packet, r in validated:
        review = r.model_dump()
        identity = digest({"packet": digest(packet), "review": review, "policy": manifest["policy_hash"]})
        result_path = directory / "results" / r.fund_code / r.horizon / (identity + ".json")
        if result_path.exists():
            result = read(result_path)
            if result.get("identity") != identity or result.get("review_hash") != digest(review):
                raise ValueError("V2_RESULT_IDENTITY_MISMATCH")
        else:
            result = {
                "identity": identity,
                "fund_code": r.fund_code,
                "horizon": r.horizon,
                "sample_label": packet["sample_label"],
                "classification": r.classification.model_dump(),
                "decision": r.decision,
                "output_kind": r.output_kind,
                "period_assessment": r.period_assessment.model_dump(),
                "summary": r.summary,
                "evaluated_at": at.isoformat(),
                "source_cutoff": packet["source_cutoff"],
                "input_mode": packet["input_mode"],
                "period": packet["periods"][r.horizon],
                "packet_hash": digest(packet),
                "policy_hash": manifest["policy_hash"],
                "review_hash": digest(review),
                "review": review,
                "production_publication": False,
                "validation_status": "UNVALIDATED",
                "confidence": "LOW",
                "up_probability": None,
                "follow_ups": [
                    {"check": key, "effect": c.gap_effect, "reason": c.impact_reason, "next": c.follow_up}
                    for key, c in r.checks.items()
                    if c.gap_effect != "NONE"
                ],
                "verification_scope": "结构、引用、日期及身份；经济推理仍由实际审阅负责",
            }
            save_once(result_path, result)
        save_once(directory / "reviews" / (digest(review) + ".json"), review)
        results.append(
            {
                "fund_code": r.fund_code,
                "horizon": r.horizon,
                "decision": r.decision,
                "result_path": str(result_path),
                "result_hash": digest(result),
                "summary": result["summary"],
            }
        )
    summary = {
        "version": manifest["version"],
        "checked_at": at.isoformat(),
        "fund_count": len(selected),
        "review_count": len(results),
        "results": results,
        "counts": {
            s: sum(row["decision"] == s for row in results) for s in ("UP", "DOWN", "FLAT", "ABSTAIN", "BLOCKED")
        },
        "production_publication": False,
        "forecast_accuracy_measured": False,
        "conditional_outlook_count": sum(r.output_kind == "CONDITIONAL_OUTLOOK" for _, r in validated),
        "conditional_direction_count": sum(
            r.output_kind == "CONDITIONAL_OUTLOOK" and r.decision in {"UP", "DOWN", "FLAT"} for _, r in validated
        ),
        "dated_direction_count": sum(
            r.output_kind == "DATED_ANALYSIS" and r.decision in {"UP", "DOWN", "FLAT"} for _, r in validated
        ),
        "owner_scope_read_back": True,
        "all_result_hashes_read_back": True,
    }
    for row in results:
        if digest(read(Path(row["result_path"]))) != row["result_hash"]:
            raise ValueError("V2_FINAL_RESULT_READBACK_FAILED")
    save_once(directory / "summaries" / (digest(summary) + ".json"), summary)
    return summary
