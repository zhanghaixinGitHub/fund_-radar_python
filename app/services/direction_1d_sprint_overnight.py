"""第三轮：同题加入中国T收盘至U08:30间SPX变化，历史与实际到达证据分开。

只延续已验证的个人本地SPX能力，不修改旧隔夜任务、来源登记或前两轮文件。
未来答案必须引用第二轮已验收输入；07:00/07:30/08:00三个半小时槽最多各查询一次。
"""

import hashlib
import json
import time
from bisect import bisect_right
from calendar import monthrange
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from datetime import time as day_time
from decimal import Decimal
from functools import lru_cache
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.integrations.tushare_overnight import fetch_spx
from app.services import direction_1d_overnight_audit as old_audit
from app.services import direction_1d_overnight_collect as old_live
from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_market as market

CANDIDATES = ("LR32_252", "TREE32_252")
OLD = base.PROJECT / ".local-runs/direction-1d-overnight-study-20260913"
OLD_LIVE = base.PROJECT / ".local-runs/direction-1d-overnight-live-20260913"
CALENDAR_2025 = base.PROJECT / "app/data/calendars/nyse_cash_2025_sprint_v1.json"


def root():
    return base.ROOT / "round-03"


def plain(path):
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint():
    result = market.fingerprint()
    for name in (
        "app/services/direction_1d_sprint_overnight.py",
        "app/integrations/tushare_overnight.py",
        "app/services/direction_1d_overnight_audit.py",
        "app/services/direction_1d_overnight_collect.py",
        "app/data/calendars/nyse_cash_2025_sprint_v1.json",
        "app/data/calendars/nyse_cash_2021_2024_v1.json",
        "app/data/calendars/nyse_cash_2026_v1.json",
    ):
        result[name] = hashlib.sha256((base.PROJECT / name).read_bytes()).hexdigest()
    return result


def source():
    value = market.active()
    contract = old_live.load_contract(OLD_LIVE)
    if str(value["source_id"]) != contract["source_id"]:
        raise ValueError("OVERNIGHT_SOURCE_ID_CHANGED")
    return value


@lru_cache(maxsize=1)
def sessions():
    """各年份官方日历分别读取，包含临时休市及夏令时，不改旧四年日历。"""
    out = old_audit.sessions() | old_live.market_sessions()
    spec = plain(CALENDAR_2025)
    holidays, early = set(spec["holidays"]), set(spec["early_closes"])
    if spec["calendar_id"] != "NYSE_CASH_2025_SPRINT_V1" or holidays & early:
        raise ValueError("NYSE_2025_CALENDAR_INVALID")
    d = date(2025, 1, 1)
    while d.year == 2025:
        if d.weekday() < 5 and str(d) not in holidays:
            out[str(d)] = datetime.combine(
                d, day_time(13 if str(d) in early else 16), ZoneInfo("America/New_York")
            ).astimezone(base.ZONE)
        d += timedelta(days=1)
    return dict(sorted(out.items()))


@lru_cache(maxsize=4096)
def alignment(t, u):
    days = list(map(str, base.calendar()[0]))
    if t not in days or days.index(u) != days.index(t) + 1:
        raise ValueError("OVERNIGHT_CN_PAIR_INVALID")
    events = sessions()
    keys = list(events)
    closes = list(events.values())
    start = datetime.combine(date.fromisoformat(t), day_time(15), base.ZONE)
    cutoff = datetime.combine(date.fromisoformat(u), day_time(8, 30), base.ZONE)
    before, last = bisect_right(closes, start) - 1, bisect_right(closes, cutoff) - 1
    if before < 0 or last < before:
        raise ValueError("OVERNIGHT_BASE_UNCOVERED")
    required = keys[before : last + 1]
    return {
        "base": t,
        "target": u,
        "deadline": cutoff.isoformat(),
        "required_us_dates": required,
        "new_us_dates": required[1:],
        "close_events": {d: events[d].isoformat() for d in required},
    }


def same_previous_close(prior, reported):
    """相邻收盘/前收盘任一仅两位时，允许半分以内显示舍入差，不修正原价。"""
    prior, reported = Decimal(str(prior)), Decimal(str(reported))
    error = abs(prior - reported)
    return error <= Decimal("0.0001") or (
        any(v == v.quantize(Decimal("0.01")) for v in (prior, reported)) and error <= Decimal("0.005000001")
    )


def extend(x, aligned, points):
    """无美股开市时才允许0变化；少一条应有行情即拒绝，绝不沿用昨天当成今天。"""
    required = aligned["required_us_dates"]
    if len(x) != 30 or any(d not in points for d in required):
        raise ValueError("OVERNIGHT_INPUT_INCOMPLETE")
    close = [float(points[d]["close"]) for d in required]
    if not np.isfinite(close).all() or min(close) <= 0:
        raise ValueError("OVERNIGHT_VALUE_INVALID")
    if any(
        not same_previous_close(points[a]["close"], points[b]["pre_close"])
        for a, b in zip(required, required[1:], strict=False)
    ):
        raise ValueError("OVERNIGHT_PREVIOUS_CLOSE_CHANGED")
    return list(x) + [close[-1] / close[0] - 1, float(len(required) - 1)]


def plan():
    path = next(
        (
            root() / name
            for name in ("plan-v4.json", "plan-v3.json", "plan-v2.json", "plan.json")
            if (root() / name).exists()
        ),
        root() / "plan.json",
    )
    if path.exists():
        p = base.read(path)
        if p.get("revision") in (2, 3, 4):
            previous = "plan.json" if p["revision"] == 2 else f"plan-v{p['revision'] - 1}.json"
            if p["previous_plan_hash"] != base.digest(base.read(root() / previous)):
                raise ValueError("ROUND_03_PREVIOUS_PLAN_CHANGED")
        if p["code"] != fingerprint():
            raise ValueError("ROUND_03_CODE_CHANGED")
        return p
    p = {
        "at": base.now().isoformat(),
        "candidates": list(CANDIDATES),
        "code": fingerprint(),
        "parent_market_hash": base.digest(base.read(market.root() / "market.json")),
        "parent_history_hash": base.digest(base.read(base.ROOT / "history.json")),
        "development_year": 2025,
        "new_features": ["spx_return_between_T15_and_U0830", "new_us_session_count"],
        "threshold": 0.5,
        "primary": "highest 2025 development score fixed before future results",
        "max_new_historical_requests": 22,
        "max_development_fits": 24,
        "max_forward_fits": 6,
        "max_live_requests_per_target": 3,
        "live_slots": ["0700", "0730", "0800"],
        "purpose": "USER_AUTHORIZED_PERSONAL_LOCAL_SPX_RESEARCH",
        "new_cost_cny": 0,
        "historical_availability": "ASSUMED_NOT_TRUE_FORWARD",
        "held_out_2026_scores_read": False,
        "spx_document": "https://tushare.pro/document/2?doc_id=211",
    }
    base.save(path, p)
    return p


def validate_response(body, required):
    """兼容实测两位/四位数值舍入，价格链只接受显示精度内差异，特征只从价格计算。

    2026年2月实测pct_chg只保留两位，最大误差应为0.005个百分点；数值本身带
    更多小数的记录仍用原0.0001界限。不是按0.1或整数的显示位数扩大容差。
    相邻收盘和前收盘任一只保留两位时允许半分以内差异，其他日期、缺日、正值要求不变。
    旧校验器不变。两种兼容都有原始响应证据，不因模型评分调整校验。
    """
    payload = json.loads(body)
    if type(payload.get("code")) is not int or payload["code"] != 0:
        raise ValueError("OVERNIGHT_PROVIDER_BUSINESS_FAILED")
    data = payload.get("data") or {}
    if data.get("fields") != old_live.FIELDS or not isinstance(data.get("items"), list) or len(data["items"]) > 32:
        raise ValueError("OVERNIGHT_FIELDS_OR_ROWS_INVALID")
    rows, rounded = {}, []
    for values in data["items"]:
        if not isinstance(values, list) or len(values) != 5:
            raise ValueError("OVERNIGHT_ROW_INVALID")
        code, day, close, previous, pct = values
        if not isinstance(day, str) or len(day) != 8 or not day.isdigit():
            raise ValueError("OVERNIGHT_DATE_INVALID")
        day = datetime.strptime(day, "%Y%m%d").date().isoformat()
        if code != "SPX" or day not in required or day in rows:
            raise ValueError("OVERNIGHT_UNEXPECTED_OR_DUPLICATE_DATE")
        if any(type(v) not in (int, float) for v in (close, previous, pct)):
            raise ValueError("OVERNIGHT_PRICE_INVALID")
        if not np.isfinite([close, previous, pct]).all() or min(close, previous) <= 0:
            raise ValueError("OVERNIGHT_PRICE_INVALID")
        two_decimals = Decimal(str(pct)) == Decimal(str(pct)).quantize(Decimal("0.01"))
        error = abs((close / previous - 1) * 100 - pct)
        if error > (0.005000001 if two_decimals else 0.0001):
            raise ValueError("OVERNIGHT_RETURN_INCONSISTENT")
        if error > 0.0001:
            rounded.append(day)
        rows[day] = {"close": close, "pre_close": previous, "pct_chg": pct}
    missing = sorted(set(required) - rows.keys())
    if missing:
        return {"status": "INCOMPLETE", "rows": rows, "missing_dates": missing, "two_decimal_rounding_dates": rounded}
    if any(
        not same_previous_close(rows[a]["close"], rows[b]["pre_close"])
        for a, b in zip(required, required[1:], strict=False)
    ):
        raise ValueError("OVERNIGHT_PREVIOUS_CLOSE_CHANGED")
    return {"status": "COMPLETE", "rows": rows, "missing_dates": [], "two_decimal_rounding_dates": rounded}


def parsed_query(start, end):
    """复用有界原文客户端和数值校验；请求范围内逐个核对官方应有交易日。"""
    body = fetch_spx(start, end)
    required = [d for d in sessions() if start <= date.fromisoformat(d) <= end]
    parsed = validate_response(body, required)
    if parsed["status"] != "COMPLETE":
        raise ValueError("OVERNIGHT_INPUT_INCOMPLETE")
    return {
        "at": base.now().isoformat(),
        "start": str(start),
        "end": str(end),
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "response": json.loads(body),
        "rows": parsed["rows"],
    }


def acquire():
    p = plan()
    current = source()
    if (root() / "spx.json").exists():
        return {"status": "ALREADY_CAPTURED"}
    study = plain(OLD / "study.json")
    if base.digest(study) != plain(OLD / "study-receipt.json")["hash"]:
        raise ValueError("OLD_SPX_STUDY_CHANGED")
    if base.now() >= datetime.fromisoformat(study["source_expires_at"]):
        raise ValueError("OLD_SPX_EVIDENCE_EXPIRED")
    payloads = []
    for year in range(2021, 2025):
        name = f"data/history-{year}.json"
        raw = (OLD / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != study["input_files"][name]:
            raise ValueError("OLD_SPX_FILE_CHANGED")
        payloads.append(json.loads(raw))
    points = old_audit.validate_history(payloads)
    if old_audit.coverage(points, old_audit.sessions())["status"] != "COMPLETE":
        raise ValueError("OLD_SPX_COVERAGE_INVALID")
    proof = root() / "data/probe.json"
    if not proof.exists():
        base.save(proof, parsed_query(date(2025, 1, 2), date(2025, 1, 2)))
        time.sleep(max(7.0, 60 / current["rate_limit_per_minute"]))
    end = date.fromisoformat(base.window(base.now())["base_nav_date"])
    for year in range(2025, end.year + 1):
        for month in range(1, 13):
            start = date(year, month, 1)
            if start > end:
                break
            path = root() / f"data/{year}-{month:02d}.json"
            if not path.exists():
                base.save(path, parsed_query(start, min(date(year, month, monthrange(year, month)[1]), end)))
                time.sleep(max(7.0, 60 / current["rate_limit_per_minute"]))
            points.update(base.read(path)["rows"])
    events = {d: close for d, close in sessions().items() if d <= str(end)}
    if set(events) != set(points):
        raise ValueError("SPX_CALENDAR_COVERAGE_MISMATCH")
    dates = sorted(points)
    if any(
        not same_previous_close(points[a]["close"], points[b]["pre_close"])
        for a, b in zip(dates, dates[1:], strict=False)
    ):
        raise ValueError("SPX_HISTORY_DISCONTINUITY")
    value = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "rows": points,
        "old_study_hash": base.digest(study),
        "expires_at": study["source_expires_at"],
        "historical_first_versions_verified": False,
    }
    base.save(root() / "spx.json", value)
    return {
        "rows": len(points),
        "reused_rows": 1005,
        "new_data_requests": len(list((root() / "data").glob("*.json"))) + p.get("extra_provider_requests", 0),
    }


def dataset():
    data = base.read(root() / "spx.json")
    if base.now() >= datetime.fromisoformat(data["expires_at"]):
        raise ValueError("SPX_EVIDENCE_EXPIRED")
    rows, excluded = market.dataset()
    if excluded:
        raise ValueError("PARENT_MARKET_INPUT_INCOMPLETE")
    cache, result = {}, []
    for r in rows:
        if r["u"] not in cache:
            aligned = alignment(r["t"], r["u"])
            cache[r["u"]] = extend([0.0] * 30, aligned, data["rows"])[-2:]
        result.append(r | {"x": r["x"] + cache[r["u"]]})
    return result


def fit(rows, name, cutoff):
    selected = [r for r in rows if r["mature"] < cutoff]
    dates = set(sorted({r["u"] for r in selected})[-252:])
    selected = [r for r in selected if r["u"] in dates]
    classes = Counter(r["y"] for r in selected)
    if len(dates) < 120 or len(classes) < 2 or min(classes.values()) < 20:
        raise ValueError("FIT_SAMPLE_INSUFFICIENT")
    counts = Counter((r["u"], r["family"]) for r in selected)
    weights = [1 / counts[r["u"], r["family"]] for r in selected]
    x, y = np.asarray([r["x"] for r in selected]), np.asarray([r["y"] for r in selected])
    if x.shape[1] != 32 or not np.isfinite(x).all():
        raise ValueError("FIT_FEATURE_INVALID")
    if name == "LR32_252":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=0))
        model.fit(x, y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    elif name == "TREE32_252":
        model = HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=7,
            learning_rate=0.05,
            l2_regularization=10,
            min_samples_leaf=40,
            early_stopping=False,
            random_state=0,
        )
        model.fit(x, y, sample_weight=weights)
    else:
        raise ValueError("CANDIDATE_NOT_FROZEN")
    return {
        "model": model,
        "n": 32,
        "fit_rows": len(selected),
        "fit_dates": len(dates),
        "fit_end": max(r["u"] for r in selected),
        "fit_hash": base.digest(selected),
    }


def train():
    p = plan()
    source()
    if (root() / "result.json").exists():
        return base.read(root() / "result.json")
    if p["parent_market_hash"] != base.digest(base.read(market.root() / "market.json")) or p[
        "parent_history_hash"
    ] != base.digest(base.read(base.ROOT / "history.json")):
        raise ValueError("PARENT_INPUT_CHANGED")
    rows = dataset()
    groups = sorted({r["group"] for r in rows})
    predictions = defaultdict(list)
    with threadpool_limits(limits=2):
        for q in range(1, 5):
            start = f"2025-{q * 3 - 2:02d}-01"
            end = "2026-01-01" if q == 4 else f"2025-{q * 3 + 1:02d}-01"
            for group in groups:
                grouped = [r for r in rows if r["group"] == group]
                exam = [r for r in grouped if start <= r["u"] < end]
                control = base.read(market.root() / f"folds/{q}-{group}-TREE30_252.json")
                if {(r["code"], r["u"]) for r in exam} != {(r["code"], r["u"]) for r in control}:
                    raise ValueError("COMMON_EXAM_MISMATCH")
                predictions["TREE30_252"].extend(control)
                predictions["ALWAYS_UP"].extend(r | {"prediction": 1} for r in control)
                for name in CANDIDATES:
                    path = root() / f"folds/{q}-{group}-{name}.json"
                    if path.exists():
                        predicted = base.read(path)
                    else:
                        trained = fit(grouped, name, start)
                        scores = trained["model"].predict_proba([r["x"] for r in exam])[:, 1]
                        predicted = [
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                            | {"prediction": int(score > 0.5)}
                            for r, score in zip(exam, scores, strict=True)
                        ]
                        base.save(path, predicted)
                    predictions[name].extend(predicted)
            base.save(root() / "progress.json", {"quarter": q, "at": base.now().isoformat()}, replace=True)
        metrics = {n: base.metrics(v) for n, v in predictions.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        bundle = {
            n: {g: fit([r for r in rows if r["group"] == g], n, str(base.now().date())) for g in groups}
            for n in CANDIDATES
        }
    path = root() / "models.joblib"
    joblib.dump(bundle, path)
    result = {
        "at": base.now().isoformat(),
        "metrics": metrics,
        "winner": winner,
        "code": fingerprint(),
        "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "spx_hash": base.digest(base.read(root() / "spx.json")),
        "held_out_2026_scores_read": False,
        "kind": "DEVELOPMENT_ASSUMED_AVAILABILITY_NOT_FORWARD",
        "new_cost_cny": 0,
    }
    base.save(root() / "result.json", result)
    return result


def models():
    result = base.read(root() / "result.json")
    path = root() / "models.joblib"
    if result["code"] != fingerprint() or hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]:
        raise ValueError("ROUND_03_MODEL_OR_CODE_CHANGED")
    return result, joblib.load(path)


def preflight():
    """实际已封存历史题目推理验收，不生成或计数真实未来答案。"""
    _, bundle = models()
    rows = dataset()
    chosen = {}
    for row in reversed(rows):
        chosen.setdefault(row["code"], row)
    count = 0
    with threadpool_limits(limits=2):
        for r in chosen.values():
            for groups in bundle.values():
                score = float(groups[r["group"]]["model"].predict_proba([r["x"]])[0, 1])
                if not 0 <= score <= 1:
                    raise ValueError("SCORE_INVALID")
                count += 1
    value = {"at": base.now().isoformat(), "kind": "DRY_RUN_NOT_FORWARD", "model_checks": count}
    base.save(root() / "preflight.json", value)
    return value


def parent_check(path):
    parent = base.read(path)
    receipt = base.read(market.root() / "receipts" / parent["u"] / path.name)
    deadline = datetime.combine(date.fromisoformat(parent["u"]), day_time(8, 30), base.ZONE)
    if (
        receipt["status"] != "VERIFIED"
        or receipt["forecast_hash"] != base.digest(parent)
        or datetime.fromisoformat(receipt["readback_at"]) >= deadline
    ):
        raise ValueError("PARENT_FORECAST_NOT_VERIFIED")
    return parent


def tick():
    at = base.now()
    end = datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])
    w = base.window(at)
    if at >= end or w["status"] != "OPEN" or not day_time(7) <= at.time() < day_time(8, 30):
        return report()
    target = w["target_nav_date"]
    if target != str(at.date()):
        return report()
    parents = [
        p
        for p in (market.root() / "forward" / target).glob("*.json")
        if not (root() / "forward" / target / p.name).exists()
    ]
    if not parents:
        return report()
    source()
    aligned = alignment(w["base_nav_date"], target)
    if any(datetime.fromisoformat(t) >= at for t in aligned["close_events"].values()):
        raise ValueError("OVERNIGHT_UNFINISHED_MARKET_SESSION")
    slot = f"{at.hour:02d}{'00' if at.minute < 30 else '30'}"
    reservation = root() / f"live/{target}/{slot}-reserved.json"
    if reservation.exists():
        return report()
    base.save(reservation, {"at": at.isoformat(), "max_requests": 1, "plan": aligned})
    required = aligned["required_us_dates"]
    body = fetch_spx(date.fromisoformat(required[0]), date.fromisoformat(required[-1]))
    received = base.now()
    parsed = validate_response(body, aligned["required_us_dates"])
    observation = {
        "received_at": received.isoformat(),
        "response": json.loads(body),
        "parsed": parsed,
        "plan": aligned,
        "first_provider_publication_claimed": False,
    }
    base.save(root() / f"live/{target}/{slot}-response.json", observation)
    deadline = datetime.fromisoformat(aligned["deadline"])
    if parsed["status"] != "COMPLETE" or received >= deadline:
        return report()
    manifest, bundle = models()
    for path in parents:
        if base.now() >= deadline:
            break
        parent = parent_check(path)
        x = extend(parent["x"], aligned, parsed["rows"])
        original = base.read(base.ROOT / "forward" / target / path.name)
        if parent["parent_hash"] != base.digest(original):
            raise ValueError("ORIGINAL_PARENT_CHANGED")
        answers = {}
        with threadpool_limits(limits=2):
            for name, groups in bundle.items():
                value = float(groups[original["group"]]["model"].predict_proba([x])[0, 1])
                answers[name] = {"prediction": int(value > 0.5), "research_score": value}
        payload = {
            "at": base.now().isoformat(),
            "u": target,
            "code": original["code"],
            "x": x,
            "parent_market_hash": base.digest(parent),
            "original_hash": base.digest(original),
            "observation_hash": base.digest(observation),
            "observation_file": f"live/{target}/{slot}-response.json",
            "model_hash": manifest["model_sha256"],
            "answers": answers,
            "status": "MODEL_NOT_RELEASED",
        }
        saved = root() / "forward" / target / path.name
        base.save(saved, payload)
        readback = base.now()
        verified = base.read(saved) == payload and readback < deadline
        base.save(
            root() / "receipts" / target / path.name,
            {
                "readback_at": readback.isoformat(),
                "forecast_hash": base.digest(payload),
                "status": "VERIFIED" if verified else "LATE_OR_INVALID",
            },
        )
    return report()


def report():
    if not (root() / "result.json").exists():
        return {"phase": "NOT_TRAINED"}
    paired, good, late, pending, closed_good = defaultdict(list), 0, 0, 0, 0
    original_report = base.report()
    for path in (root() / "forward").glob("*/*.json"):
        value = base.read(path)
        receipt_path = root() / "receipts" / value["u"] / path.name
        receipt = base.read(receipt_path) if receipt_path.exists() else {}
        if receipt.get("status") != "VERIFIED" or receipt.get("forecast_hash") != base.digest(value):
            late += 1
            continue
        good += 1
        closed_good += value["u"] in original_report["closed_targets"]
        parent = parent_check(market.root() / "forward" / value["u"] / path.name)
        original = base.read(base.ROOT / "forward" / value["u"] / path.name)
        if value["parent_market_hash"] != base.digest(parent) or value["original_hash"] != base.digest(original):
            raise ValueError("PARENT_HASH_CHANGED")
        path = base.ROOT / "outcomes" / value["u"] / path.name
        if not path.exists():
            pending += 1
            continue
        outcome = base.read(path)
        if outcome["forecast_hash"] != base.digest(original):
            raise ValueError("OUTCOME_HASH_CHANGED")
        for name, answer in (value["answers"] | parent["answers"] | original["answers"]).items():
            paired[name].append(original | outcome | {"prediction": answer["prediction"]})
    due = original_report["due_eligible_predictions"]
    scope_due = len(original_report["closed_targets"]) * original_report["watchlist_funds"]
    value = {
        "at": base.now().isoformat(),
        "primary_candidate": base.read(root() / "result.json")["winner"],
        "verified_forecasts": good,
        "invalid_or_late": late,
        "pending": pending,
        "eligible_coverage": closed_good / due if due else None,
        "whole_watchlist_coverage": closed_good / scope_due if scope_due else None,
        "missing_due_predictions": due - closed_good,
        "matched_forward_metrics": {k: base.metrics(v) for k, v in paired.items()},
        "new_cost_cny": 0,
        "model_released": False,
    }
    base.save(root() / "report.json", value, replace=True)
    return value
