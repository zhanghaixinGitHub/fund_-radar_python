"""KURE未来输入一次实收；只在08:00至08:30采集，不回填、不重试、不冒充历史为未来。"""

import hashlib
from datetime import date, datetime, time

from app.integrations import sina_sprint_china_health_etf as api
from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_china_health_data as data
from app.services import direction_1d_sprint_overnight as overnight


def root():
    return b.ROOT / "round-104/live-health"


def require(condition, reason):
    if not condition:
        raise ValueError("KURE_LIVE_" + reason)


def in_window(t, u, at):
    require(at.tzinfo is not None, "TIMEZONE_REQUIRED")
    overnight.alignment(t, u)
    local = at.astimezone(b.ZONE)
    return local.date() == date(2026, 9, 17) and str(local.date()) == u and time(8) <= local.time() < time(8, 30)


def reconstruct(t, u, ph):
    """读回真实请求和响应；失败只表示该次供应商读取不可用，不能自行补造一份成功行情。"""
    r = root() / u
    request, receipt = b.read(r / "request.json"), b.read(r / "response.json")
    started, received = datetime.fromisoformat(request["at"]), datetime.fromisoformat(receipt["at"])
    require(
        request["t"] == t
        and request["u"] == u
        and request["plan_hash"] == ph
        and request["symbol"] == "KURE"
        and request["url"] == api.url_for("KURE")
        and request["attempt"] == 1,
        "REQUEST_CHANGED",
    )
    require(
        receipt["request_hash"] == b.digest(request)
        and started <= received
        and in_window(t, u, started)
        and in_window(t, u, received),
        "RECEIPT_OR_TIME_CHANGED",
    )
    aligned = overnight.alignment(t, u)
    require(all(datetime.fromisoformat(s) <= started for s in aligned["close_events"].values()), "UNCLOSED_SESSION")
    points, error = {}, None
    if receipt["status"] == "RECEIVED":
        raw = (r / "raw.bin").read_bytes()
        require(
            receipt["http_status"] == 200
            and 0 < len(raw) <= api.MAX_BYTES
            and receipt["bytes"] == len(raw)
            and hashlib.sha256(raw).hexdigest() == receipt["sha256"],
            "RAW_CHANGED",
        )
        try:
            full = api.parse(raw, "KURE", aligned["required_us_dates"][-1])
            points = {d: full[d] for d in aligned["new_us_dates"] if d in full}
        except (ValueError, TypeError, KeyError) as exc:
            error = type(exc).__name__ + ":" + str(exc)[:100]
    else:
        require(receipt["status"] == "FAILED_NO_RETRY" and not (r / "raw.bin").exists(), "FAILURE_CHANGED")
        error = receipt["error_type"]
    feature = data.features(t, u, points)
    return {
        "t": t,
        "u": u,
        "plan_hash": ph,
        "request_hash": b.digest(request),
        "receipt_hash": b.digest(receipt),
        "received_at": receipt["at"],
        "points": points,
        "available": feature["available"],
        "feature": feature,
        "error": error,
        "new_source_requests": 1,
    }


def load_live(t, u, ph):
    snapshot = b.read(root() / u / "snapshot.json")
    actual = reconstruct(t, u, ph)
    require({k: v for k, v in snapshot.items() if k != "at"} == actual, "SNAPSHOT_CHANGED")
    created = datetime.fromisoformat(snapshot["at"])
    require(datetime.fromisoformat(actual["received_at"]) <= created and in_window(t, u, created), "SNAPSHOT_LATE")
    return snapshot


def capture(t, u, ph):
    """一个目标日最多一次新GET；断电留下请求时不会再次请求，保持真实缺口。"""
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
            "symbol": "KURE",
            "url": api.url_for("KURE"),
            "attempt": 1,
        }
        b.save(r / "request.json", request)
        try:
            raw, headers = api.fetch_etf("KURE")
            with (r / "raw.bin").open("xb") as f:
                f.write(raw)
            receipt = {
                "at": b.now().isoformat(),
                "request_hash": b.digest(request),
                "status": "RECEIVED",
                "http_status": 200,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "headers": headers,
            }
        except Exception as exc:
            # 原文写盘异常不归为正常网络失败，保留可见错误而非生成虚假失败收据。
            if (r / "raw.bin").exists():
                raise
            receipt = {
                "at": b.now().isoformat(),
                "request_hash": b.digest(request),
                "status": "FAILED_NO_RETRY",
                "error_type": type(exc).__name__,
            }
        b.save(r / "response.json", receipt)
    if not (r / "response.json").exists():
        return None
    actual = reconstruct(t, u, ph)
    snapshot = actual | {"at": b.now().isoformat()}
    require(in_window(t, u, datetime.fromisoformat(snapshot["at"])), "SNAPSHOT_LATE")
    b.save(r / "snapshot.json", snapshot)
    return load_live(t, u, ph)
