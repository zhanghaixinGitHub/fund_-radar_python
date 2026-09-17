"""纯市场一日模型的独立提前预测和实际结果核对，不依赖旧NAV模型的父答案。"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

from app.services import direction_1d_sprint as base
from app.services import direction_1d_sprint_etf_runtime_v2 as runtime
from app.services import direction_1d_sprint_market_only as model
from app.services import direction_1d_sprint_market_only_data as data


def sprint_end():
    return datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"])


def validate_forecast(value, receipt, source, manifest, bundle, scope):
    """回读时重新算答案并核对真实时刻、固定范围和全部绑定，不接受后补或被修改的答案。"""
    identity = next((f for f in scope if f["code"] == value["code"]), None)
    created = datetime.fromisoformat(value["at"])
    readback = datetime.fromisoformat(receipt["readback_at"])
    if (
        identity is None
        or any(value[k] != identity[k] for k in ("code", "group", "family"))
        or value["u"] < model.FIRST_TARGET
        or value["u"] != source["target"]
        or value["t"] != source["base"]
        or not datetime.fromisoformat(source["at"]) <= created <= readback < data.deadline(value["u"])
        or datetime.fromisoformat(manifest["at"]) > created
        or created >= sprint_end()
        or receipt["status"] != "VERIFIED"
        or receipt["forecast_hash"] != base.digest(value)
        or value["input_hash"] != base.digest(source)
        or value["scope_hash"] != base.digest(scope)
        or value["plan_hash"] != manifest["plan_hash"]
        or value["model_hash"] != manifest["model_sha256"]
        or value["nav_inputs_used"] is not False
        or value["base_nav_at_forecast"] is not None
        or value["base_nav_observation_status"] != "NOT_READ_BY_MARKET_MODEL"
        or value["status"] != "MODEL_NOT_RELEASED"
        or not runtime.answers_match(value["answers"], model.answers(source["market"], value["group"], bundle))
    ):
        raise ValueError("MARKET_ONLY_FORECAST_INVALID")
    return value


def write_forecasts(source, manifest, bundle):
    scope = data.scope()
    target = source["target"]
    for fund in scope:
        if base.now() >= min(data.deadline(target), sprint_end()):
            break
        path = model.root() / "forward" / target / f"{fund['code']}.json"
        if path.exists():
            continue
        value = fund | {
            "at": base.now().isoformat(),
            "t": source["base"],
            "u": target,
            "input_hash": base.digest(source),
            "scope_hash": base.digest(scope),
            "plan_hash": manifest["plan_hash"],
            "model_hash": manifest["model_sha256"],
            "nav_inputs_used": False,
            "base_nav_at_forecast": None,
            "base_nav_observation_status": "NOT_READ_BY_MARKET_MODEL",
            "answers": model.answers(source["market"], fund["group"], bundle),
            "status": "MODEL_NOT_RELEASED",
        }
        base.save(path, value)
        readback = base.read(path)
        received = base.now()
        receipt = {
            "readback_at": received.isoformat(),
            "forecast_hash": base.digest(value),
            "status": "VERIFIED" if received < data.deadline(target) and readback == value else "LATE_OR_INVALID",
        }
        base.save(model.root() / "receipts" / target / path.name, receipt)
        validate_forecast(value, receipt, source, manifest, bundle, scope)


def verified_forecasts(manifest, bundle):
    scope, inputs, valid, invalid = data.scope(), {}, [], []
    for path in sorted((model.root() / "forward").glob("*/*.json")):
        try:
            value = base.read(path)
            if path.parent.name != value["u"] or path.stem != value["code"]:
                raise ValueError("MARKET_ONLY_FORECAST_PATH_CHANGED")
            target = value["u"]
            if target not in inputs:
                inputs[target] = data.load(target)
            receipt = base.read(model.root() / "receipts" / target / path.name)
            valid.append(validate_forecast(value, receipt, inputs[target], manifest, bundle, scope))
        except Exception as exc:
            invalid.append({"path": path.relative_to(model.root()).as_posix(), "error": base.error_code(exc)})
    return valid, invalid


def observed_pair(forecast, snapshot, at):
    """只核对两日均实际收到且已公告的RAW净值；历史快照内未被重新观察的旧行不能充数。"""
    closed = datetime.combine(date.fromisoformat(forecast["u"]), time(18), base.ZONE)
    if (
        at < closed
        or datetime.fromisoformat(snapshot["expires_at"]) <= at
        or datetime.fromisoformat(snapshot["at"]) > at
    ):
        return None
    fund = next((f for f in snapshot["funds"] if f["fund_code"] == forecast["code"]), None)
    if fund is None:
        return None
    points = {r["date"]: r for r in fund["rows"]}
    a, z = points.get(forecast["t"]), points.get(forecast["u"])
    for row in (a, z):
        if row is None or not row.get("ann_date") or not row.get("received_at"):
            return None
        received = datetime.fromisoformat(row["received_at"])
        if received.tzinfo is None or received > at or date.fromisoformat(row["ann_date"]) > at.date():
            return None
    if datetime.fromisoformat(z["received_at"]) < closed:
        return None
    # label本身拒绝非正数、NaN和无穷值；不调整为复权净值或另一个日期。
    return {"t_source": a, "u_source": z, **base.label(a["nav"], z["nav"])}


def pair_version(pair):
    return {k: {field: pair[k][field] for field in ("date", "nav", "ann_date")} for k in ("t_source", "u_source")}


def observation_paths(forecasts):
    """只扫描本次待核对目标之后的三天观察文件，按首次实际观察顺序保留结果。"""
    if not forecasts:
        return []
    lower = min(f["u"] for f in forecasts).replace("-", "") + "T180000"
    return sorted(p for p in (base.ROOT / "observations").glob("*.json") if p.stem >= lower)


def observe_outcomes(forecasts, at):
    due = [f for f in forecasts if at >= datetime.combine(date.fromisoformat(f["u"]), time(18), base.ZONE)]
    if not due:
        return
    first = {}
    for f in due:
        path = model.root() / "outcomes" / f["u"] / f"{f['code']}.json"
        if path.exists():
            first[f["code"], f["u"]] = base.read(path)
    for path in observation_paths(due):
        snapshot = base.read(path)
        if datetime.fromisoformat(snapshot["at"]) > at:
            continue
        reference = {
            "observation_path": path.relative_to(base.ROOT).as_posix(),
            "observation_hash": base.digest(snapshot),
        }
        for f in due:
            pair = observed_pair(f, snapshot, at)
            if pair is None:
                continue
            key = f["code"], f["u"]
            if key not in first:
                value = pair | reference | {"observed_at": at.isoformat(), "forecast_hash": base.digest(f)}
                base.save(model.root() / "outcomes" / f["u"] / f"{f['code']}.json", value)
                first[key] = value
            elif pair_version(pair) != pair_version(first[key]):
                # 每一种实际数值/公告版本独立保存，收到时间变化本身不是净值修订。
                version = base.digest(pair_version(pair))[:24]
                dest = model.root() / "revisions" / f["u"] / f["code"] / f"{version}.json"
                if not dest.exists():
                    base.save(
                        dest,
                        pair
                        | reference
                        | {
                            "observed_at": at.isoformat(),
                            "forecast_hash": base.digest(f),
                            "first_outcome_hash": base.digest(first[key]),
                        },
                    )


def refresh_if_needed(forecasts, at):
    """旧分支未刷新而新分支有成熟未核对答案时，每半小时最多共享一次30基金刷新。"""
    pending = [
        f
        for f in forecasts
        if at >= datetime.combine(date.fromisoformat(f["u"]), time(18), base.ZONE)
        and not (model.root() / "outcomes" / f["u"] / f"{f['code']}.json").exists()
    ]
    if not pending or at >= sprint_end():
        return
    slot = at.replace(minute=at.minute // 30 * 30, second=0, microsecond=0)
    latest = base.ROOT / "latest-nav.json"
    if latest.exists() and datetime.fromisoformat(base.read(latest)["at"]) >= slot:
        return
    reservation = model.root() / "nav-requests" / f"{slot:%Y%m%dT%H%M}.json"
    if reservation.exists():
        return
    source = base.source()
    history = base.read(base.ROOT / "history.json")
    if base.now() >= datetime.fromisoformat(history["expires_at"]):
        raise ValueError("MARKET_ONLY_NAV_HISTORY_EXPIRED")
    cfg = base.get_settings()
    if cfg.tushare_api_url != "https://api.tushare.pro":
        raise ValueError("UPSTREAM_URL_UNEXPECTED")
    base.save(reservation, {"at": base.now().isoformat(), "max_fund_requests": len(history["funds"]), "max_retries": 0})
    out = {
        "at": base.now().isoformat(),
        "funds": [],
        "errors": [],
        "expires_at": history["expires_at"],
        "source_branch": "MARKET_ONLY",
    }
    pause = max(0.35, 60 / min(source["rate_limit_per_minute"], 120))
    with base.TushareFundClient(
        token=cfg.tushare_token.get_secret_value(),
        api_url=cfg.tushare_api_url,
        connect_timeout_seconds=5,
        read_timeout_seconds=20,
        max_retries=0,
        catalog_max_rows_per_query=15000,
    ) as client:
        for fund in history["funds"]:
            # 每次请求前留出连接/读取超时余量，截止后不启动下一请求，也不重试。
            if base.now() + timedelta(seconds=26) >= sprint_end():
                out["errors"].append({"code": fund["fund_code"], "error": "SPRINT_DEADLINE_REACHED"})
                break
            try:
                records = client.list_nav_history(
                    fund["source_fund_code"],
                    start_date=base.now().date() - timedelta(days=130),
                    end_date=base.now().date(),
                )
                received = base.now()
                rows = []
                for r in records:
                    if r.nav_date > received.date() or (r.ann_date and r.ann_date > received.date()):
                        raise ValueError("FUTURE_NAV_RESPONSE")
                    rows.append(
                        {
                            "date": str(r.nav_date),
                            "nav": str(r.unit_nav),
                            "ann_date": str(r.ann_date) if r.ann_date else None,
                            "received_at": received.isoformat(),
                        }
                    )
                out["funds"].append({k: v for k, v in fund.items() if k != "rows"} | {"rows": rows})
            except Exception as exc:
                out["errors"].append({"code": fund["fund_code"], "error": base.error_code(exc)})
            if base.now() + timedelta(seconds=pause) < sprint_end():
                base.time.sleep(pause)
    stamp = base.now().strftime("%Y%m%dT%H%M%S%f")
    base.save(base.ROOT / "observations" / f"{stamp}.json", out)
    base.save(latest, out, replace=True)


def validated_outcome(f, value, cache, at):
    ref = value["observation_path"]
    path = base.ROOT / ref
    if Path(ref).is_absolute() or path.resolve().parent != (base.ROOT / "observations").resolve():
        raise ValueError("MARKET_ONLY_OUTCOME_PATH_INVALID")
    if ref not in cache:
        cache[ref] = base.read(path)
    snapshot = cache[ref]
    pair = observed_pair(f, snapshot, datetime.fromisoformat(value["observed_at"]))
    if (
        pair is None
        or value["observation_hash"] != base.digest(snapshot)
        or value["forecast_hash"] != base.digest(f)
        or any(value[k] != v for k, v in pair.items())
        or datetime.fromisoformat(value["observed_at"]) > at
    ):
        raise ValueError("MARKET_ONLY_OUTCOME_CHANGED")
    return pair


def report(manifest=None, bundle=None):
    if not (model.root() / "result.json").exists():
        return {"phase": "NOT_TRAINED"}
    if manifest is None:
        manifest, bundle = model.models()
    forecasts, invalid = verified_forecasts(manifest, bundle)
    paired, cache, revised = defaultdict(list), {}, set()
    at = base.now()
    for f in forecasts:
        path = model.root() / "outcomes" / f["u"] / f"{f['code']}.json"
        if not path.exists():
            continue
        outcome = base.read(path)
        pair = validated_outcome(f, outcome, cache, at)
        for revision in (model.root() / "revisions" / f["u"] / f["code"]).glob("*.json"):
            v = base.read(revision)
            validated_outcome(f, v, cache, at)
            if v["first_outcome_hash"] != base.digest(outcome):
                raise ValueError("MARKET_ONLY_REVISION_PARENT_CHANGED")
            revised.add((f["code"], f["u"]))
        for name, answer in f["answers"].items():
            paired[name].append(f | pair | {"prediction": answer["prediction"]})
    closed = [
        str(day)
        for day in base.calendar()[0]
        if str(day) >= model.FIRST_TARGET and data.deadline(str(day)) <= min(at, sprint_end())
    ]
    closed_predictions = sum(f["u"] in closed for f in forecasts)
    matched = len(next(iter(paired.values()), []))
    value = {
        "at": at.isoformat(),
        "verified_forecasts": len(forecasts),
        "invalid_forecasts": invalid,
        "pending": len(forecasts) - matched,
        "matched": matched,
        "closed_targets": closed,
        "matched_forward_metrics": {n: base.metrics(v) for n, v in paired.items()},
        "outcome_policy": (
            "FIRST_OBSERVED_RAW_NAV_PAIR; retain revisions separately, never overwrite first answer or outcome"
        ),
        "revised_questions": len(revised),
        "eligible_funds": len(data.scope()),
        "watchlist_funds": 43,
        "missing_due_predictions": len(closed) * len(data.scope()) - closed_predictions,
        "eligible_coverage": closed_predictions / (len(closed) * len(data.scope())) if closed else None,
        "whole_watchlist_coverage": closed_predictions / (len(closed) * 43) if closed else None,
        "new_cost_cny": 0,
        "status": "MODEL_NOT_RELEASED",
    }
    base.save(model.root() / "report.json", value, replace=True)
    return value


def tick():
    manifest, bundle = model.models()
    at = base.now()
    if at >= sprint_end():
        return report(manifest, bundle)
    if str(at.date()) >= model.FIRST_TARGET and data.slot_for(at):
        try:
            source = data.capture(at)
            if source is not None:
                write_forecasts(source, manifest, bundle)
        except Exception as exc:
            # 输入尚未完整或来源校验失败也不阻止已存在预测的结果核对。
            base.save(
                model.root() / "capture-status.json",
                {"at": base.now().isoformat(), "error": base.error_code(exc)},
                replace=True,
            )
            forecasts, _ = verified_forecasts(manifest, bundle)
            observe_outcomes(forecasts, base.now())
            if base.error_code(exc) == "CNYA_LIVE_SLOT_OR_BUDGET_EXHAUSTED":
                return report(manifest, bundle)
            raise
    forecasts, _ = verified_forecasts(manifest, bundle)
    observe_outcomes(forecasts, base.now())
    refresh_if_needed(forecasts, base.now())
    observe_outcomes(forecasts, base.now())
    return report(manifest, bundle)
