"""只把R109实际T日股票响应与事前冻结的20日基线组合；不请求行情、不复用历史T输入。"""

from datetime import datetime

from app.services import direction_1d_sprint as b
from app.services import direction_1d_sprint_market_stock_activity as model
from app.services import direction_1d_sprint_stock_breadth_live as parent


def require(condition, reason):
    if not condition:
        raise ValueError("STOCK_ACTIVITY_LIVE_" + reason)


def in_window(t, u, at):
    return parent.in_window(t, u, at)


def ready(u):
    return (parent.root() / u / "snapshot.json").is_file()


def freeze_baseline():
    """只冻结首目标日前T之前20个实际源日；逐原文复算，未来推理只需核验这些原文摘要。"""
    path = model.root() / "future-baseline.json"
    require(not path.exists(), "BASELINE_ALREADY_FROZEN")
    p = b.read(model.data.root() / "qualification-plan.json")
    q = b.read(model.data.root() / "qualification-result.json")
    require(
        q["plan_hash"] == b.digest(p) and q["status"] == "QUALIFIED_STOCK_ACTIVITY_WITH_LIMITS", "QUALIFICATION_CHANGED"
    )
    for name, expected in p["code"].items():
        require(model.core.sha(b.PROJECT / name) == expected, "SOURCE_CODE_CHANGED")
    days = list(map(str, b.calendar()[0]))
    u = model.FIRST_TARGET
    t = days[days.index(u) - 1]
    dates = days[days.index(t) - 20 : days.index(t)]
    require(len(dates) == 20 and all(d < t for d in dates), "BASELINE_DATES_INVALID")
    stock = model.data.old.old
    seed = b.read(stock.root() / "reused-history.json")
    rows, raw_sources = {}, {}
    for day in dates:
        raw_path = (
            b.ROOT / seed["sources"][day]["raw_path"]
            if day in seed["sources"]
            else stock.root() / "raw" / (day + ".json")
        )
        digest = model.core.sha(raw_path)
        require(digest == q["source_proof"]["raw_hashes"][day], "BASELINE_RAW_CHANGED")
        row = model.data.features.parse(raw_path.read_bytes(), day)
        require(row == q["features"][day] and row["available"], "BASELINE_DERIVED_CHANGED")
        rows[day] = row
        raw_sources[day] = {"path": raw_path.relative_to(b.ROOT).as_posix(), "sha256": digest}
    value = {
        "at": b.now().isoformat(),
        "t": t,
        "u": u,
        "dates": dates,
        "rows": rows,
        "raw_sources": raw_sources,
        "source_qualification_hash": b.digest(q),
        "calendar_hash": b.calendar()[1],
        "new_source_requests": 0,
    }
    require(datetime.fromisoformat(value["at"]) < model.core.data.deadline(u), "BASELINE_FROZEN_LATE")
    b.save(path, value)
    return value


def load_baseline(t, u):
    """绑定模型资格和运行意图，并重验20份原文字节；禁止T、U或不连续的日期混入基线。"""
    value = b.read(model.root() / "future-baseline.json")
    intent = b.read(model.root() / "runtime-intent.json")
    plan = b.read(model.root() / "plan.json")
    require(b.digest(value) == intent["baseline_hash"], "BASELINE_CHANGED")
    require(value["source_qualification_hash"] == plan["source_qualification_hash"], "BASELINE_SOURCE_CHANGED")
    days = list(map(str, b.calendar()[0]))
    require(t in days and u in days and days.index(u) == days.index(t) + 1, "BASELINE_TARGET_INVALID")
    expected = days[days.index(t) - 20 : days.index(t)]
    require(
        value["t"] == t
        and value["u"] == u == model.FIRST_TARGET
        and value["dates"] == expected
        and len(expected) == 20,
        "BASELINE_TARGET_CHANGED",
    )
    require(set(value["rows"]) == set(value["raw_sources"]) == set(expected), "BASELINE_POINTS_CHANGED")
    require(value["calendar_hash"] == b.calendar()[1] and all(d < t for d in expected), "BASELINE_CALENDAR_CHANGED")
    for day, meta in value["raw_sources"].items():
        require(model.core.sha(b.ROOT / meta["path"]) == meta["sha256"], "BASELINE_RAW_CHANGED")
        model.data.features.validate_day(value["rows"][day], day)
        require(value["rows"][day]["available"], "BASELINE_UNAVAILABLE")
    return value


def load_live(t, u, ph):
    require(ph == b.digest(b.read(model.root() / "plan.json")), "MODEL_PLAN_CHANGED")
    baseline = load_baseline(t, u)
    parent_plan = b.digest(b.read(b.ROOT / "round-109/plan.json"))
    source = parent.load_live(t, u, parent_plan)
    require(
        datetime.fromisoformat(baseline["at"]) <= datetime.fromisoformat(source["at"]), "BASELINE_AFTER_ACTUAL_SOURCE"
    )
    points = dict(baseline["rows"])
    require(t not in points, "HISTORICAL_T_NOT_ALLOWED")
    if source["available"]:
        # 父加载器已经重验请求、实收时间及原文摘要，本层从同一字节独立派生成交集中度。
        raw = (parent.root() / u / "raw.bin").read_bytes()
        value = model.data.features.parse(raw, t)
        require(
            value["included_set_sha256"] == source["points"][t]["included_set_sha256"], "ACTUAL_SOURCE_UNIVERSE_CHANGED"
        )
        points[t] = value
    feature = model.data.extend({}, t, u, points)["stock_activity"]
    return {
        "at": source["at"],
        "t": t,
        "u": u,
        "plan_hash": ph,
        "parent_plan_hash": parent_plan,
        "parent_source_hash": b.digest(source),
        "baseline_hash": b.digest(baseline),
        "baseline_at": baseline["at"],
        "receipt_hash": source["receipt_hash"],
        "points": points,
        "feature": feature,
        "available": feature["available"],
        "received_at": source["received_at"],
        "reserved_request_slots": 0,
        "new_source_requests": 0,
    }
