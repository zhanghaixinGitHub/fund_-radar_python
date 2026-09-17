"""目标17日仅一次公开IF排名实收，具体合约来自R95实际收到的主力映射。"""

import hashlib
from datetime import date, datetime, time, timedelta
from time import monotonic

import httpx

from app.core.config import get_settings
from app.integrations import tushare_sprint_futures_holdings as parser
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_current_basis_data as quote_live
from app.services import direction_1d_sprint_futures_member_data as data

URL = "https://api.tushare.pro"
MAX_BYTES = 1048576
require = data.require


def root():
    return base.ROOT / "round-120/live-if-member"


def in_window(t, u, at):
    require(at.tzinfo is not None, "TIMEZONE_REQUIRED")
    local = at.astimezone(base.ZONE)
    return (
        t == "2026-09-16"
        and u == "2026-09-17"
        and local.date() == date(2026, 9, 17)
        and time(8) <= local.time() < time(8, 30)
    )


def ready(u):
    return (root() / u / "snapshot.json").is_file()


def query(t, contract):
    require(
        contract.startswith("IF") and contract.endswith(".CFX") and len(contract) == 10 and contract[2:6].isdigit(),
        "ACTUAL_CONTRACT_INVALID",
    )
    return {
        "api_name": "fut_holding",
        "params": {"trade_date": t.replace("-", ""), "symbol": contract[:6], "exchange": "CFFEX"},
        "fields": ",".join(parser.FIELDS),
    }


def quote_context(t, u):
    """完整验证原R95实际快照与原文，只复用其已预约预算，不读取历史映射替代。"""
    ph = base.digest(base.read(base.ROOT / "round-95/plan.json"))
    source = quote_live.load_live(t, u, ph)
    value = {
        "plan_hash": ph,
        "source_hash": base.digest(source),
        "source_at": source["at"],
        "available": source["available"],
    }
    if source["available"]:
        point = source["snapshot"]["points"][t]
        raw = (quote_live.live_root() / u / "raw/fut_daily.json").read_bytes()
        q = quote_live.parser.parse(raw, "fut_daily", t.replace("-", ""), t.replace("-", ""))[t]
        require(q["close"] == point["futures_close"], "ACTUAL_QUOTE_CHANGED")
        value.update(
            {
                "main_contract": point["main_contract"],
                "total_open_interest": q["oi"],
                "quote_raw_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return value


def raw_identity(path):
    if not path.exists():
        return None
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def fetch(request_query):
    cfg = get_settings()
    require(cfg.tushare_api_url == URL, "LIVE_URL_CHANGED")
    token = cfg.tushare_token.get_secret_value()
    require(bool(token), "LIVE_TOKEN_MISSING")
    began = monotonic()
    with httpx.Client(timeout=httpx.Timeout(20, connect=5), follow_redirects=False) as client:
        with client.stream("POST", URL, json=request_query | {"token": token}) as response:
            raw = bytearray()
            for chunk in response.iter_bytes():
                raw.extend(chunk)
                require(len(raw) <= MAX_BYTES and monotonic() - began < 30, "LIVE_RAW_LIMIT")
            status = response.status_code
    require(token.encode() not in raw, "SECRET_IN_RESPONSE")
    return bytes(raw), status


def reconstruct(t, u, ph):
    require(ph == base.digest(base.read(base.ROOT / "round-120/plan.json")), "MODEL_PLAN_CHANGED")
    folder = root() / u
    source = quote_context(t, u)
    points, error, request_hash, receipt_hash, completed, received = {}, None, None, None, source["source_at"], None
    request_path = folder / "request.json"
    reserved, requests = 0, 0
    if not source["available"]:
        require(not request_path.exists(), "REQUEST_WITHOUT_ACTUAL_MAIN_CONTRACT")
        error = "ACTUAL_MAIN_SOURCE_UNAVAILABLE"
    else:
        request = base.read(request_path)
        receipt = base.read(folder / "response.json")
        started = datetime.fromisoformat(request["at"])
        ended = datetime.fromisoformat(receipt["at"])
        require(
            request["t"] == t
            and request["u"] == u
            and request["plan_hash"] == ph
            and request["attempt"] == 1
            and request["url"] == URL,
            "LIVE_REQUEST_CHANGED",
        )
        require(
            request["query"] == query(t, source["main_contract"])
            and request["quote_context_hash"] == base.digest(source),
            "ACTUAL_MAIN_BINDING_CHANGED",
        )
        require(receipt["request_hash"] == base.digest(request), "LIVE_RECEIPT_CHANGED")
        require(
            datetime.fromisoformat(source["source_at"]) <= started <= ended
            and in_window(t, u, started)
            and in_window(t, u, ended),
            "LIVE_TIME_INVALID",
        )
        require(receipt["raw"] == raw_identity(folder / "raw.bin"), "LIVE_RAW_CHANGED")
        reserved, requests = 1, None
        request_hash, receipt_hash, completed = base.digest(request), base.digest(receipt), receipt["at"]
        if receipt["status"] == "RECEIVED":
            raw = (folder / "raw.bin").read_bytes()
            require(0 < len(raw) <= MAX_BYTES and type(receipt["http_status"]) is int, "LIVE_RESPONSE_INVALID")
            requests, received = 1, completed
            if receipt["http_status"] == 200:
                try:
                    point = parser.parse(raw, source["main_contract"][:6], [t])[t]
                    oi = source["total_open_interest"]
                    require(
                        0 < point["ranked_long_contracts"] <= oi and 0 < point["ranked_short_contracts"] <= oi,
                        "LIVE_RANKED_TOTAL_EXCEEDS_OI",
                    )
                    points[t] = {k: v for k, v in point.items() if k != "rows"} | {"total_open_interest": oi}
                except (ValueError, TypeError, KeyError) as exc:
                    error = type(exc).__name__ + ":" + base.error_code(exc)
            else:
                error = "HTTP_" + str(receipt["http_status"])
        else:
            require(receipt["status"] in ("FAILED_NO_RETRY", "INTERRUPTED_NO_RETRY"), "LIVE_FAILURE_CHANGED")
            if receipt["status"] == "FAILED_NO_RETRY":
                require(receipt["raw"] is None, "FAILED_REQUEST_HAS_RAW")
            error = receipt["error_type"]
    feature = data.extend({}, t, u, points)["futures_member"]
    return {
        "t": t,
        "u": u,
        "plan_hash": ph,
        "request_hash": request_hash,
        "receipt_hash": receipt_hash,
        "completed_at": completed,
        "received_at": received,
        "points": points,
        "feature": feature,
        "available": feature["available"],
        "error": error,
        "quote_context": source,
        "reserved_request_slots": reserved,
        "new_source_requests": requests,
    }


def load_live(t, u, ph):
    snapshot = base.read(root() / u / "snapshot.json")
    actual = reconstruct(t, u, ph)
    require({k: v for k, v in snapshot.items() if k != "at"} == actual, "LIVE_SNAPSHOT_CHANGED")
    at = datetime.fromisoformat(snapshot["at"])
    require(datetime.fromisoformat(actual["completed_at"]) <= at and in_window(t, u, at), "LIVE_SNAPSHOT_LATE")
    return snapshot


def capture(t, u, ph):
    """缺少真实映射不请求排名；请求已预约后无论成功、失败或中断都不再请求。"""
    folder = root() / u
    if ready(u):
        return load_live(t, u, ph)
    if not in_window(t, u, base.now()) or not in_window(t, u, base.now() + timedelta(seconds=35)):
        return None
    source = quote_context(t, u)
    folder.mkdir(parents=True, exist_ok=True)
    if source["available"] and not (folder / "request.json").exists():
        request = {
            "at": base.now().isoformat(),
            "t": t,
            "u": u,
            "plan_hash": ph,
            "query": query(t, source["main_contract"]),
            "url": URL,
            "quote_context_hash": base.digest(source),
            "attempt": 1,
        }
        base.save(folder / "request.json", request)
        try:
            raw, status = fetch(request["query"])
        except Exception as exc:
            receipt = {
                "at": base.now().isoformat(),
                "request_hash": base.digest(request),
                "status": "FAILED_NO_RETRY",
                "raw": None,
                "error_type": type(exc).__name__,
            }
        else:
            with (folder / "raw.bin").open("xb") as stream:
                stream.write(raw)
            receipt = {
                "at": base.now().isoformat(),
                "request_hash": base.digest(request),
                "status": "RECEIVED",
                "http_status": status,
                "raw": raw_identity(folder / "raw.bin"),
            }
        base.save(folder / "response.json", receipt)
    elif source["available"] and not (folder / "response.json").exists():
        request = base.read(folder / "request.json")
        require(request["t"] == t and request["u"] == u and request["plan_hash"] == ph, "INTERRUPTED_REQUEST_CHANGED")
        base.save(
            folder / "response.json",
            {
                "at": base.now().isoformat(),
                "request_hash": base.digest(request),
                "status": "INTERRUPTED_NO_RETRY",
                "raw": raw_identity(folder / "raw.bin"),
                "error_type": "REQUEST_RESERVED_WITHOUT_RECEIPT",
            },
        )
    snapshot = reconstruct(t, u, ph) | {"at": base.now().isoformat()}
    require(in_window(t, u, datetime.fromisoformat(snapshot["at"])), "LIVE_SNAPSHOT_LATE")
    base.save(folder / "snapshot.json", snapshot)
    return load_live(t, u, ph)
