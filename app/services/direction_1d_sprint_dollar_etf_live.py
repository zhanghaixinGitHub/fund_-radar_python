"""美元相关ETF一次实收：保留当时请求、原文及失败，禁止历史回填和请求重试。"""

import hashlib
from datetime import date, datetime, time

from app.integrations import sina_sprint_dollar_etf as parser
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_dollar_etf_data as data
from app.services import direction_1d_sprint_overnight as overnight

URL = parser.url_for("UUP")


def root():
    return b.ROOT / "round-111/live-dollar-etf"


def require(condition, reason):
    if not condition:
        raise ValueError("DOLLAR_ETF_LIVE_" + reason)


def in_window(t, u, at):
    """研究期内只有这一对T/U允许新请求；历史日期不能冒充实际采集日期。"""
    require(at.tzinfo is not None, "TIMEZONE_REQUIRED")
    local = at.astimezone(b.ZONE)
    return (
        t == "2026-09-16"
        and u == "2026-09-17"
        and local.date() == date(2026, 9, 17)
        and time(8) <= local.time() < time(8, 30)
    )


def query(t):
    return {"symbol": "UUP", "url": URL}


def fetch(t):
    """只读一次免费公开行情；原冻结适配器限制为UUP、200KB和20秒，不重试。"""
    raw, _ = parser.fetch_etf("UUP")
    return raw, 200


def raw_identity(path):
    if not path.exists():
        return None
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def reconstruct(t, u, ph):
    """从实际请求和原文重新解析；供应商失败、缺日或中断只形成不可用状态。"""
    r = root() / u
    request, receipt = b.read(r / "request.json"), b.read(r / "response.json")
    started, completed = datetime.fromisoformat(request["at"]), datetime.fromisoformat(receipt["at"])
    require(
        request["t"] == t
        and request["u"] == u
        and request["plan_hash"] == ph
        and request["query"] == query(t)
        and request["url"] == URL
        and request["attempt"] == 1,
        "REQUEST_CHANGED",
    )
    require(
        receipt["request_hash"] == b.digest(request)
        and started <= completed
        and in_window(t, u, started)
        and in_window(t, u, completed),
        "RECEIPT_OR_TIME_CHANGED",
    )
    aligned = overnight.alignment(t, u)
    require(all(datetime.fromisoformat(v) <= started for v in aligned["close_events"].values()), "UNCLOSED_SESSION")
    points, error = {}, None
    if receipt["status"] == "RECEIVED":
        raw = (r / "raw.bin").read_bytes()
        require(
            0 < len(raw) <= parser.MAX_BYTES
            and receipt["raw"] == raw_identity(r / "raw.bin")
            and type(receipt["http_status"]) is int,
            "RAW_CHANGED",
        )
        if receipt["http_status"] == 200:
            try:
                full = parser.parse(raw, "UUP", aligned["required_us_dates"][-1])
                points = {d: full[d] for d in aligned["new_us_dates"] if d in full}
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                # 解析器自身错误码不含Token或供应商完整响应；不能将缺记录当成平盘。
                error = type(exc).__name__ + ":" + b.error_code(exc)
        else:
            error = "HTTP_" + str(receipt["http_status"])
    else:
        require(receipt["status"] in ("FAILED_NO_RETRY", "INTERRUPTED_NO_RETRY"), "FAILURE_CHANGED")
        require(receipt["raw"] == raw_identity(r / "raw.bin"), "ORPHAN_RAW_CHANGED")
        if receipt["status"] == "FAILED_NO_RETRY":
            require(receipt["raw"] is None, "FAILED_REQUEST_HAS_RAW")
        # 中断后观察到的时间不是行情接收时间；孤立原文保持封存，不作为模型输入。
        error = receipt["error_type"]
    feature = data.features(t, u, points)
    return {
        "t": t,
        "u": u,
        "plan_hash": ph,
        "request_hash": b.digest(request),
        "receipt_hash": b.digest(receipt),
        "completed_at": receipt["at"],
        "received_at": receipt["at"] if receipt["status"] == "RECEIVED" else None,
        "points": points,
        "feature": feature,
        "available": feature["available"],
        "error": error,
        "reserved_request_slots": 1,
        "new_source_requests": 1 if receipt["status"] == "RECEIVED" else None,
        "scope": "UUP_CLOSED_US_SESSIONS_INTRADAY_PRICE_RATIO",
        "historical_first_publication_verified": False,
    }


def load_live(t, u, ph):
    snapshot = b.read(root() / u / "snapshot.json")
    actual = reconstruct(t, u, ph)
    require({k: v for k, v in snapshot.items() if k != "at"} == actual, "SNAPSHOT_CHANGED")
    created = datetime.fromisoformat(snapshot["at"])
    require(datetime.fromisoformat(actual["completed_at"]) <= created and in_window(t, u, created), "SNAPSHOT_LATE")
    return snapshot


def capture(t, u, ph):
    """最多一个请求；共用研究run锁，另以独占请求文件保留预算，断电后也不补发。"""
    r = root() / u
    if (r / "snapshot.json").exists():
        return load_live(t, u, ph)
    if not in_window(t, u, b.now()):
        return None
    r.mkdir(parents=True, exist_ok=True)
    if not (r / "request.json").exists():
        request = {
            "at": b.now().isoformat(),
            "t": t,
            "u": u,
            "plan_hash": ph,
            "url": URL,
            "query": query(t),
            "attempt": 1,
        }
        b.save(r / "request.json", request)
        try:
            raw, status = fetch(t)
        except Exception as exc:
            receipt = {
                "at": b.now().isoformat(),
                "request_hash": b.digest(request),
                "raw": None,
                "status": "FAILED_NO_RETRY",
                "error_type": type(exc).__name__,
            }
        else:
            # 写盘失败必须显式抛出；后续仅能记录中断，不能伪造网络失败或再次查询。
            with (r / "raw.bin").open("xb") as f:
                f.write(raw)
            receipt = {
                "at": b.now().isoformat(),
                "request_hash": b.digest(request),
                "status": "RECEIVED",
                "http_status": status,
                "raw": raw_identity(r / "raw.bin"),
            }
        b.save(r / "response.json", receipt)
    elif not (r / "response.json").exists():
        request = b.read(r / "request.json")
        require(request["t"] == t and request["u"] == u and request["plan_hash"] == ph, "REQUEST_CHANGED")
        b.save(
            r / "response.json",
            {
                "at": b.now().isoformat(),
                "request_hash": b.digest(request),
                "raw": raw_identity(r / "raw.bin"),
                "status": "INTERRUPTED_NO_RETRY",
                "error_type": "REQUEST_RESERVED_WITHOUT_COMPLETION_RECEIPT",
            },
        )
    snapshot = reconstruct(t, u, ph) | {"at": b.now().isoformat()}
    require(in_window(t, u, datetime.fromisoformat(snapshot["at"])), "SNAPSHOT_LATE")
    b.save(r / "snapshot.json", snapshot)
    return load_live(t, u, ph)
