"""SPX按需同步：复用已验权客户端，单独保存手动证据，不修改冻结实验及旧定时回执。"""

import hashlib
import json
import os
import traceback
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from time import monotonic
from uuid import uuid4

import httpx

from app.core.logging import get_logger
from app.services import direction_1d_overnight_collect as original
from app.services.direction_1d_protocol import ZONE, canonical, digest

PROJECT = Path(__file__).resolve().parents[2]
CONTRACT_ROOT = PROJECT / ".local-runs/direction-1d-overnight-live-20260913"
ROOT = PROJECT / ".local-runs/direction-1d-spx-manual"
COOLDOWN_SECONDS = 60
# 这是按60秒最短间隔推导的记录完整性上界，不是每日请求额度。
MAX_RECORDS_PER_DAY = 24 * 60 * 60 // COOLDOWN_SECONDS + 1
STAGES = {"SOURCE": "检查数据源", "FETCH": "获取标普500行情", "SAVE": "校验并保存行情"}
ERROR_MESSAGES = {
    "OVERNIGHT_SOURCE_UNAVAILABLE": "数据源未启用或授权、留存、限速配置不完整，请检查数据源登记。",
    "OVERNIGHT_SOURCE_ID_CHANGED": "当前数据源与采集配置不一致，请检查数据源登记。",
    "OVERNIGHT_TOKEN_MISSING": "未配置Tushare凭据，请检查采集服务的环境配置。",
    "OVERNIGHT_OFFICIAL_ENDPOINT_REQUIRED": "数据源地址不是已核准的官方接口，请检查采集服务配置。",
    "OVERNIGHT_PROVIDER_HTTP_FAILED": "Tushare接口返回HTTP错误，请稍后重试；本次未自动重试。",
    "OVERNIGHT_PROVIDER_BUSINESS_FAILED": (
        "Tushare拒绝了本次请求，请检查账号的接口权限或调用额度。当前回执无法区分这两种原因。"
    ),
    "OVERNIGHT_RESPONSE_LIMIT": "数据源响应超出大小或耗时上限，本次已停止。",
    "OVERNIGHT_RESPONSE_CONTAINS_CREDENTIAL": "数据源返回了不应留存的敏感内容，本次已停止。",
    "OVERNIGHT_CLOCK_DISCONTINUITY": "同步期间系统时间异常或跨日，无法确认获取时间，请检查电脑时间后重新同步。",
    "OVERNIGHT_UNFINISHED_MARKET_SESSION": "返回行情包含尚未收盘的交易时段，未计为有效记录。",
    "OVERNIGHT_CALENDAR_UNAVAILABLE": "当前日期不在已核验的交易日历范围内，请补齐日历配置。",
    "OVERNIGHT_CALENDAR_INVALID": "交易日历校验失败，请检查本地日历配置。",
    "OVERNIGHT_CONTRACT_CHANGED": "采集配置或绑定文件已变化，请检查本地采集配置。",
    "SPX_EVIDENCE_CHANGED": "本地同步记录完整性校验失败，请检查留档文件。",
    "SPX_TIMEOUT": "连接或等待Tushare响应超时，请检查网络后重试。",
    "SPX_NETWORK_ERROR": "无法连接Tushare，请检查网络、代理或域名解析后重试。",
    "SPX_SOURCE_CHECK_FAILED": "读取数据源登记失败，请检查数据库连接和采集服务日志。",
    "SPX_STORAGE_ERROR": "本地同步记录无法读写，请检查磁盘空间及目录权限。",
    "SPX_INVALID_RESPONSE": "行情日期、价格或响应格式校验未通过，本次数据不能使用。",
    "SPX_UNKNOWN_ERROR": "同步未完成，当前回执没有更详细原因，请按本次记录编号检查采集服务日志。",
}
logger = get_logger(__name__)


def error_code(error: Exception, stage: str) -> str:
    """只公开已知错误码及固定说明；供应商、数据库异常原文可能含凭据，不透传。"""
    if isinstance(error, ValueError) and str(error) in ERROR_MESSAGES:
        return str(error)
    if isinstance(error, httpx.TimeoutException):
        return "SPX_TIMEOUT"
    if isinstance(error, httpx.RequestError):
        return "SPX_NETWORK_ERROR"
    if isinstance(error, OSError):
        return "SPX_STORAGE_ERROR"
    if stage == "SOURCE":
        return "SPX_SOURCE_CHECK_FAILED"
    if isinstance(error, (ValueError, KeyError, TypeError)):
        return "SPX_INVALID_RESPONSE"
    return "SPX_UNKNOWN_ERROR"


def progress(folder: Path, reservation: dict, stage: str) -> None:
    """开始每个真实阶段前独占留档；状态轮询仅读取这些标记，不虚构耗时百分比。"""
    save_new(folder / f"stage-{stage}.json", {"reservation_hash": digest(reservation), "stage": stage})


def save_new(path: Path, payload: dict) -> None:
    """同目录刷盘后独占发布带指纹的完整文件，进程中断不留下半份可用回执。"""
    temporary = path.with_name(f".{path.name}-{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical({"payload": payload, "hash": digest(payload)}).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_saved(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if digest(value["payload"]) != value["hash"]:
        raise ValueError("SPX_EVIDENCE_CHANGED")
    return value["payload"]


@contextmanager
def operation_lock(root: Path):
    """跨线程/进程互斥；系统在进程退出时释放锁，服务重启后无需删除旧锁文件。"""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "operation.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        acquired = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
        except OSError:
            pass
        try:
            yield acquired
        finally:
            if acquired:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def current_plan(at: datetime) -> dict:
    """交易日按当天08:00研究窗口取数；休息日取最近两次已收盘行情，仅作资料参考。"""
    plan = original.day_plan(at.date())
    if plan["status"] == "TRADING_DAY":
        return plan
    completed = [(day, close) for day, close in original.market_sessions().items() if close <= at]
    if len(completed) < 2:
        raise ValueError("OVERNIGHT_CALENDAR_UNAVAILABLE")
    selected = completed[-2:]
    return {
        "status": "REFERENCE_ONLY",
        "target_date": None,
        "required_us_dates": [day for day, _ in selected],
        "close_events": {day: close.isoformat() for day, close in selected},
    }


def summary(root: Path, reservation: dict, at: datetime) -> dict:
    """恢复回执时核验原始响应、结果和保存时间；只向页面返回固定的公共行情字段。"""
    attempt_id = reservation["attempt_id"]
    folder = root / "days" / reservation["day"] / str(reservation["number"])
    result_file = folder / "result.json"
    base = {
        "attemptId": attempt_id,
        "requestedAt": reservation["started_at"],
        "receivedAt": None,
        "persistedAt": None,
        "targetDate": reservation["plan"]["target_date"],
        "latestUsDate": None,
        "close": None,
        "previousClose": None,
        "dailyChangePct": None,
        "overnightChangePct": None,
        "rowCount": 0,
        "missingUsDates": [],
        "usableBeforeU08": False,
        "stage": "SOURCE",
        "stageLabel": STAGES["SOURCE"],
        "completedSteps": 0,
        "totalSteps": len(STAGES),
        "errorCode": None,
    }
    for number, stage in enumerate(STAGES):
        stage_file = folder / f"stage-{stage}.json"
        if stage_file.exists():
            marker = read_saved(stage_file)
            if marker != {"reservation_hash": digest(reservation), "stage": stage}:
                raise ValueError("SPX_EVIDENCE_CHANGED")
            base.update(stage=stage, stageLabel=STAGES[stage], completedSteps=number)
    if not result_file.exists():
        interrupted = at - datetime.fromisoformat(reservation["started_at"]) > timedelta(minutes=2)
        return base | {
            "state": "INTERRUPTED" if interrupted else "RUNNING",
            "message": "上次同步中断，未取得完整结果，可重新手动同步。"
            if interrupted
            else f"正在{base['stageLabel']}，请等待本次结果。",
        }
    result = read_saved(result_file)
    if result["reservation_hash"] != digest(reservation):
        raise ValueError("SPX_EVIDENCE_CHANGED")
    if result["state"] == "FAILED":
        code = result.get("error_code", "SPX_UNKNOWN_ERROR")
        if code not in ERROR_MESSAGES:
            code = "SPX_UNKNOWN_ERROR"
        return base | {"state": "FAILED", "errorCode": code, "message": ERROR_MESSAGES[code]}
    body = (folder / "response.json").read_bytes()
    validation = original.validate_response(body, reservation["plan"])
    if hashlib.sha256(body).hexdigest() != result["response_hash"] or validation != result["validation"]:
        raise ValueError("SPX_EVIDENCE_CHANGED")
    received, persisted = (datetime.fromisoformat(result[key]) for key in ("received_at", "persisted_at"))
    started = datetime.fromisoformat(reservation["started_at"])
    if not started <= received <= persisted <= at:
        raise ValueError("OVERNIGHT_CLOCK_DISCONTINUITY")
    if at >= datetime.fromisoformat(result["retention_until"]):
        return base | {"state": "EXPIRED", "message": "上次行情留存有效期已过，请重新同步。"}
    rows = validation["rows"]
    latest = max(rows, default=None)
    row = rows.get(latest, {})
    plan = reservation["plan"]
    complete = validation["status"] == "COMPLETE"
    timely = (
        complete
        and plan["status"] == "TRADING_DAY"
        and persisted < datetime.fromisoformat(plan["deadline"])
        and started.date().isoformat() == plan["target_date"]
        and all(datetime.fromisoformat(close) <= started for close in plan["close_events"].values())
    )
    state = "ON_TIME" if timely else "LATE" if plan["status"] == "TRADING_DAY" else "REFERENCE_ONLY"
    if not complete:
        state = "INCOMPLETE"
    messages = {
        "ON_TIME": "同步成功，所需行情已在当天早上8点前收到并保存。",
        "LATE": "同步成功，已超过当天早上8点；保留为晚到资料，不算提前取得。",
        "REFERENCE_ONLY": "同步成功，今天不是中国交易日，行情仅作资料参考。",
        "INCOMPLETE": "已保存本次响应，但所需美股日期尚未齐全，不能作为完整隔夜输入。",
    }
    return base | {
        "state": state,
        "message": messages[state],
        "receivedAt": received.isoformat(),
        "persistedAt": persisted.isoformat(),
        "latestUsDate": latest,
        "close": row.get("close"),
        "previousClose": row.get("pre_close"),
        "dailyChangePct": row.get("pct_chg"),
        "overnightChangePct": validation["diagnostic_return"] * 100 if complete and plan.get("deadline") else None,
        "rowCount": len(rows),
        "missingUsDates": validation["missing_dates"],
        "usableBeforeU08": timely,
        "stage": "DONE",
        "stageLabel": "同步完成" if complete else "已保存，数据不全",
        "completedSteps": len(STAGES),
    }


def status(*, root: Path = ROOT, contract_root: Path = CONTRACT_ROOT, clock=original.now) -> dict:
    """只读状态；浏览器打开、刷新和服务重启均不会请求Tushare或重新执行采集。"""
    at = clock().astimezone(ZONE)
    value = {
        "mode": "MANUAL",
        "serverTime": at.isoformat(),
        "canSync": False,
        "availability": "BLOCKED",
        "message": "",
        "attemptsToday": 0,
        "maxAttemptsPerDay": None,
        "nextAllowedAt": None,
        "lastAttempt": None,
        "performedNow": False,
        "errorCode": None,
    }
    try:
        contract = original.load_contract(contract_root)
        if at.year != 2026 or at < datetime.fromisoformat(contract["installed_at"]):
            raise ValueError("OVERNIGHT_CALENDAR_UNAVAILABLE")
        reservations = []
        day_folder = root / "days" / str(at.date())
        # 只扫描当天数字目录，既兼容原1—4号记录，也允许按需执行第5次及以后。
        folders = list(day_folder.iterdir()) if day_folder.exists() else []
        if len(folders) > MAX_RECORDS_PER_DAY:
            raise ValueError("SPX_EVIDENCE_CHANGED")
        for folder in folders:
            if not folder.name.isdecimal() or not folder.is_dir():
                continue
            number = int(folder.name)
            if not 1 <= number <= MAX_RECORDS_PER_DAY:
                raise ValueError("SPX_EVIDENCE_CHANGED")
            path = folder / "reserved.json"
            if path.exists():
                item = read_saved(path)
                if item["day"] != str(at.date()) or item["number"] != number:
                    raise ValueError("SPX_EVIDENCE_CHANGED")
                reservations.append(item)
        reservations.sort(key=lambda item: item["number"])
        if [item["number"] for item in reservations] != list(range(1, len(reservations) + 1)):
            raise ValueError("SPX_EVIDENCE_CHANGED")
        value["attemptsToday"] = len(reservations)
        latest = reservations[-1] if reservations else None
        if latest is None and (root / "latest.json").exists():
            pointer = read_saved(root / "latest.json")
            day = date.fromisoformat(pointer["day"])
            number = pointer["number"]
            if not 1 <= number <= MAX_RECORDS_PER_DAY or day > at.date():
                raise ValueError("SPX_EVIDENCE_CHANGED")
            latest = read_saved(root / "days" / str(day) / str(number) / "reserved.json")
            if latest["day"] != str(day) or latest["number"] != number:
                raise ValueError("SPX_EVIDENCE_CHANGED")
        if latest:
            value["lastAttempt"] = summary(root, latest, at)
            next_allowed = datetime.fromisoformat(latest["started_at"]) + timedelta(seconds=COOLDOWN_SECONDS)
            value["nextAllowedAt"] = next_allowed.isoformat()
            if value["lastAttempt"]["state"] == "RUNNING":
                value.update(availability="RUNNING", message="本次同步正在执行，页面会自动更新进度。")
                return value
            if at < next_allowed:
                value.update(availability="COOLDOWN", message="本次已结束；为避免重复取数，两次同步至少间隔60秒。")
                return value
        value.update(canSync=True, availability="READY", message="可按需同步，无每日4次限制。", nextAllowedAt=None)
    except (ValueError, OSError, KeyError, TypeError) as error:
        code = error_code(error, "STATUS")
        value.update(errorCode=code, message=ERROR_MESSAGES[code])
    return value


def synchronize(
    *,
    root: Path = ROOT,
    contract_root: Path = CONTRACT_ROOT,
    clock=original.now,
    query=original.fetch_spx,
    source_reader=original.source_status,
) -> dict:
    """用户单击只尝试一次有界请求；无每日次数额度，保留60秒间隔和跨入口互斥，不自动重试。"""
    with operation_lock(root) as acquired:
        value = status(root=root, contract_root=contract_root, clock=clock)
        if not acquired:
            return value | {
                "canSync": False,
                "availability": "RUNNING",
                "message": "已有手动同步正在执行，页面会自动更新进度。",
            }
        if not value["canSync"]:
            return value
        contract, at = original.load_contract(contract_root), clock().astimezone(ZONE)
        plan = current_plan(at)
        number = value["attemptsToday"] + 1
        folder = root / "days" / str(at.date()) / str(number)
        folder.mkdir(parents=True, exist_ok=True)
        reservation = {
            "attempt_id": uuid4().hex,
            "day": str(at.date()),
            "number": number,
            "started_at": at.isoformat(),
            "plan": plan,
            "contract_hash": digest(contract),
            "protocol": "SPX_MANUAL_OBSERVATION_V1",
            "max_requests": 1,
        }
        save_new(folder / "reserved.json", reservation)
        pointer = root / f"latest-{uuid4().hex}.json"
        save_new(pointer, {"day": str(at.date()), "number": number})
        os.replace(pointer, root / "latest.json")
        result = {"reservation_hash": digest(reservation), "state": "FAILED", "api_calls": 0}
        logger.info("spx_manual.synchronize   >>> 开始手动同步 attempt=%s", reservation["attempt_id"])
        stage = "SOURCE"
        try:
            progress(folder, reservation, stage)
            source = source_reader()
            original.validate_source(source)
            if source["source_id"] != contract["source_id"]:
                raise ValueError("OVERNIGHT_SOURCE_ID_CHANGED")
            stage = "FETCH"
            progress(folder, reservation, stage)
            began, wall_start = monotonic(), clock()
            result["api_calls"] = 1
            body = query(
                date.fromisoformat(plan["required_us_dates"][0]), date.fromisoformat(plan["required_us_dates"][-1])
            )
            received = clock()
            if received < wall_start or abs((received - wall_start).total_seconds() - (monotonic() - began)) > 5:
                raise ValueError("OVERNIGHT_CLOCK_DISCONTINUITY")
            stage = "SAVE"
            progress(folder, reservation, stage)
            original.write_body(folder / "response.json", body)
            validation = original.validate_response(body, plan)
            persisted = clock()
            if not at <= wall_start <= received <= persisted or persisted.date() != at.date():
                raise ValueError("OVERNIGHT_CLOCK_DISCONTINUITY")
            if any(datetime.fromisoformat(plan["close_events"][day]) > wall_start for day in validation["rows"]):
                raise ValueError("OVERNIGHT_UNFINISHED_MARKET_SESSION")
            result.update(
                state="SAVED",
                received_at=received.isoformat(),
                persisted_at=persisted.isoformat(),
                validation=validation,
                response_hash=hashlib.sha256(body).hexdigest(),
                retention_until=(
                    persisted + timedelta(days=min(contract["retention_days"], source["retention_days"]))
                ).isoformat(),
            )
        except Exception as error:
            # 原始异常可能携带凭据；保存稳定错误类型与完整定位栈，不记录外部错误文本。
            result.update(
                error_code=error_code(error, stage),
                error_type=type(error).__name__,
                error_stack=[
                    {"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name}
                    for frame in traceback.extract_tb(error.__traceback__)
                ],
            )
            logger.warning(
                "spx_manual.synchronize   >>> 同步失败 attempt=%s type=%s stack=%s",
                reservation["attempt_id"],
                type(error).__name__,
                result["error_stack"],
            )
        save_new(folder / "result.json", result)
        value = status(root=root, contract_root=contract_root, clock=clock)
        logger.info(
            "spx_manual.synchronize   >>> 手动同步结束 attempt=%s state=%s api_calls=%s",
            reservation["attempt_id"],
            result["state"],
            result["api_calls"],
        )
        return value | {"performedNow": True}
