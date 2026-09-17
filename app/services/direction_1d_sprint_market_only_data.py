"""纯市场一日模型的数据链：预测输入不读取基金净值，标签另按公告成熟。

复用已验收的SPX、FXI/CNYA成交价和CNYA估值差来源，未公告的基金净值不被伪装成输入。
"""

import hashlib
import json
from datetime import date, datetime, time

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_cnya_data as cnya
from app.services import direction_1d_sprint_etf_joint as reference
from app.services import direction_1d_sprint_overnight as overnight
from app.services import direction_1d_sprint_us_etf_data as etfs

SLOTS = ("0700", "0730", "0800")


def root():
    return base.ROOT / "round-70/market-inputs"


def scope():
    """固定本人原30只研究基金的身份；不由净值是否及时决定当天是否预测。"""
    history = base.read(base.ROOT / "history.json")
    return sorted(
        [{"code": f["fund_code"], "family": f["family"], "group": f["group"]} for f in history["funds"]],
        key=lambda f: f["code"],
    )


def feature_values(t, u, spx_rows, etf_rows, cnya_rows):
    """原三输入分别为SPX区间变化、FXI同日开到收、CNYA估值差，单位百分数。"""
    s = overnight.extend([0.0] * 30, overnight.alignment(t, u), spx_rows)
    e, c = etfs.features(t, u, etf_rows), cnya.features(t, u, cnya_rows)
    return {
        "features": [s[30] * 100, e[0], c[0]],
        "available": bool(e[-1] and c[-1]),
        "us_sessions": s[31],
        "etf_available": bool(e[-1]),
        "cnya_available": bool(c[-1]),
    }


def dataset():
    """重新核对市场特征和两日标签；不能只改外层JSON摘要就替换扩展数据。"""
    folder = base.ROOT / "nav-independent-market-feasibility-v1"
    spec, result, bundle = (base.read(folder / n) for n in ("plan.json", "result.json", "rows.json"))
    history, spx = base.read(base.ROOT / "history.json"), base.read(overnight.root() / "spx.json")
    if (
        result["plan_hash"] != base.digest(spec)
        or bundle["plan_hash"] != base.digest(spec)
        or result["rows_hash"] != base.digest(bundle)
        or spec["source_hashes"]["history"] != base.digest(history)
        or spec["source_hashes"]["spx"] != base.digest(spx)
        or spec["source_hashes"]["r50plan"] != base.digest(reference.plan())
        or spec["calendar_hash"] != base.calendar()[1]
        or spec["script_sha256"] != hashlib.sha256((folder / "check.py").read_bytes()).hexdigest()
    ):
        raise ValueError("MARKET_ONLY_HISTORY_MANIFEST_CHANGED")
    if base.now() >= datetime.fromisoformat(history["expires_at"]) or base.now() >= datetime.fromisoformat(
        spx["expires_at"]
    ):
        raise ValueError("MARKET_ONLY_HISTORY_EXPIRED")
    e, c = etfs.history(), cnya.history()
    if bundle["source_market_hashes"] != {"etfs": base.digest(e), "cnya": base.digest(c)}:
        raise ValueError("MARKET_ONLY_ORIGINAL_MARKET_CHANGED")
    funds = {f["fund_code"]: f for f in history["funds"]}
    nav = {code: {r["date"]: r for r in f["rows"]} for code, f in funds.items()}
    days = list(map(str, base.calendar()[0]))
    successor = dict(zip(days[:-1], days[1:], strict=True))
    old_keys = {(r["code"], r["u"]) for r in reference.dataset()[0]}
    cache, seen = {}, set()
    for row in bundle["rows"]:
        t, u, code = row["t"], row["u"], row["code"]
        if code not in funds or successor.get(t) != u or (code, u) in seen:
            raise ValueError("MARKET_ONLY_SCOPE_OR_ADJACENCY_CHANGED")
        seen.add((code, u))
        if (t, u) not in cache:
            cache[t, u] = feature_values(t, u, spx["rows"], e, c)
        a, z = nav[code][t], nav[code][u]
        truth = base.label(a["nav"], z["nav"])
        if (
            row["market"] != cache[t, u]
            or row["y"] != truth["y"]
            or row["actual_direction"] != truth["actual_direction"]
            or row["mature"] != max(successor[u], a["ann_date"], z["ann_date"])
            or row["label_t_hash"] != a["source_hash"]
            or row["label_u_hash"] != z["source_hash"]
            or row["family"] != funds[code]["family"]
            or row["group"] != funds[code]["group"]
            or row["old_question"] != ((code, u) in old_keys)
            or row["fund_nav_used_as_predictor"] is not False
        ):
            raise ValueError("MARKET_ONLY_ROW_OR_MATURITY_CHANGED")
    if len(bundle["rows"]) != 38949 or not old_keys <= seen:
        raise ValueError("MARKET_ONLY_OLD_QUESTIONS_REMOVED")
    return bundle["rows"], {"rows_hash": base.digest(bundle), "proof_hash": base.digest(result)}


def deadline(target):
    return datetime.combine(date.fromisoformat(target), time(8, 30), base.ZONE)


def ended(at):
    return at >= datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])


def slot_for(at):
    if at.tzinfo is None:
        raise ValueError("MARKET_ONLY_TIMEZONE_REQUIRED")
    local = at.astimezone(base.ZONE)
    if not time(7) <= local.time() < time(8, 30):
        return None
    return f"{local.hour:02d}{'00' if local.minute < 30 else '30'}"


def spx_observation(target, t, origin, slot):
    """旧SPX响应必须有原预约并重算解析结果；新补采还核对真实HTTP字节摘要。"""
    if slot not in SLOTS or origin not in ("ROUND3", "MARKET_ONLY"):
        raise ValueError("MARKET_ONLY_SPX_ROUTE_INVALID")
    aligned = overnight.alignment(t, target)
    if origin == "ROUND3":
        folder = overnight.root() / "live" / target
        reservation = base.read(folder / f"{slot}-reserved.json")
        observation = base.read(folder / f"{slot}-response.json")
        body = json.dumps(observation["response"], ensure_ascii=False)
    else:
        folder = root() / target / "spx"
        reservation = base.read(folder / f"{slot}-reserved.json")
        observation = base.read(folder / f"{slot}-response.json")
        raw = (folder / f"{slot}-response.bin").read_bytes()
        if hashlib.sha256(raw).hexdigest() != observation["body_sha256"] or len(raw) != observation["bytes"]:
            raise ValueError("MARKET_ONLY_SPX_RAW_CHANGED")
        body = raw.decode("utf-8")
    received, requested = datetime.fromisoformat(observation["received_at"]), datetime.fromisoformat(reservation["at"])
    if (
        observation["plan"] != aligned
        or reservation["plan"] != aligned
        or observation["parsed"] != overnight.validate_response(body, aligned["required_us_dates"])
        or observation["parsed"]["status"] != "COMPLETE"
        or not requested <= received < deadline(target)
        or any(datetime.fromisoformat(closed) >= received for closed in aligned["close_events"].values())
        or slot_for(requested) != slot
        or str(requested.astimezone(base.ZONE).date()) != target
    ):
        raise ValueError("MARKET_ONLY_SPX_TIME_OR_RESPONSE_CHANGED")
    return observation, {
        "origin": origin,
        "slot": slot,
        "hash": base.digest(observation),
        "request_hash": base.digest(reservation),
    }


def capture_spx(at, t, target):
    """先复用旧时槽响应；仅当本时槽没有旧预约时补采，最多三个时槽，不自动重试。"""
    slot = slot_for(at)
    if ended(at) or slot is None or str(at.astimezone(base.ZONE).date()) != target:
        return None
    for earlier in SLOTS:
        for origin, directory in (
            ("ROUND3", overnight.root() / "live" / target),
            ("MARKET_ONLY", root() / target / "spx"),
        ):
            if not (directory / f"{earlier}-response.json").exists():
                continue
            try:
                return spx_observation(target, t, origin, earlier)
            except ValueError:
                continue
    folder = root() / target / "spx"
    reservation = folder / f"{slot}-reserved.json"
    if reservation.exists() or (overnight.root() / "live" / target / f"{slot}-reserved.json").exists():
        return None
    aligned = overnight.alignment(t, target)
    if any(datetime.fromisoformat(closed) >= at for closed in aligned["close_events"].values()):
        return None
    overnight.source()
    if ended(base.now()) or base.now() >= deadline(target):
        return None
    base.save(reservation, {"at": base.now().isoformat(), "max_requests": 1, "plan": aligned})
    try:
        required = aligned["required_us_dates"]
        body = overnight.fetch_spx(date.fromisoformat(required[0]), date.fromisoformat(required[-1]))
        received = base.now()
        raw = body.encode("utf-8") if isinstance(body, str) else body
        with (folder / f"{slot}-response.bin").open("xb") as output:
            output.write(raw)
        value = {
            "received_at": received.isoformat(),
            "plan": aligned,
            "parsed": overnight.validate_response(raw, required),
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        base.save(folder / f"{slot}-response.json", value)
        return spx_observation(target, t, "MARKET_ONLY", slot)
    except Exception as exc:
        base.save(folder / f"{slot}-failure.json", {"at": base.now().isoformat(), "error": base.error_code(exc)})
        return None


def load(target):
    value = base.read(root() / target / "input.json")
    t = value["base"]
    spx, ref = spx_observation(target, t, value["spx_ref"]["origin"], value["spx_ref"]["slot"])
    e, c = etfs.load(target), cnya.load(target)
    created = datetime.fromisoformat(value["at"])
    if (
        value["target"] != target
        or created >= deadline(target)
        or value["spx_ref"] != ref
        or value["etf_hash"] != base.digest(e)
        or value["cnya_hash"] != base.digest(c)
        or e["base"] != t
        or c["base"] != t
        or any(datetime.fromisoformat(stamp) > created for stamp in (spx["received_at"], e["at"], c["at"]))
        or value["market"] != feature_values(t, target, spx["parsed"]["rows"], e["rows"], c["rows"])
        or value["fund_nav_used_as_predictor"] is not False
    ):
        raise ValueError("MARKET_ONLY_LIVE_INPUT_CHANGED")
    return value


def capture(at):
    slot = slot_for(at)
    target = str(at.astimezone(base.ZONE).date())
    days = list(map(str, base.calendar()[0]))
    if ended(at) or slot is None or target not in days or days.index(target) == 0:
        return None
    path = root() / target / "input.json"
    if path.exists():
        return load(target)
    t = days[days.index(target) - 1]
    s = capture_spx(at, t, target)
    if s is None:
        return None
    e, c = etfs.capture(base.now()), cnya.capture(base.now())
    if e is None or c is None or base.now() >= deadline(target):
        return None
    spx, ref = s
    value = {
        "at": base.now().isoformat(),
        "base": t,
        "target": target,
        "spx_ref": ref,
        "etf_hash": base.digest(e),
        "cnya_hash": base.digest(c),
        "market": feature_values(t, target, spx["parsed"]["rows"], e["rows"], c["rows"]),
        "fund_nav_used_as_predictor": False,
    }
    base.save(path, value)
    checked = load(target)
    if base.now() >= deadline(target):
        raise ValueError("MARKET_ONLY_INPUT_READBACK_LATE")
    return checked
