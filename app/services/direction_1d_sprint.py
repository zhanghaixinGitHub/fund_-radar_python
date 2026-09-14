"""三天自主研究入口：新协议、只读业务库、独立保存开发成绩和真实未来答案。

研究可以继续扩展，但每轮必须先落盘方案。旧120日试验和正式模型登记不受影响。
所有日期按北京时间；收益标签沿用单位净值严格上涨，持平记为不涨并单列。
"""

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from datetime import time as day_time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from threadpoolctl import threadpool_limits

from app.core.config import get_settings
from app.db.session import get_nav_preview_engine
from app.integrations.tushare import TushareFundClient
from app.repositories import direction_1d as repo
from app.services.direction_1d_data import classify
from app.services.direction_1d_independent_audit import live_context
from app.services.direction_1d_inference import load_model
from app.services.direction_1d_protocol import ZONE, calendar, canonical, digest, features, input_days, label, window
from app.services.direction_1d_protocol import score as registered_score

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / ".local-runs/direction-1d-sprint-20260914"
# 不使用无界搜索。每轮8个固定候选，最多4个开发季度×3组，加一次未来模型拟合。
CANDIDATES = (
    "LR7_504",
    "LR18_504",
    "TREE7_504",
    "TREE18_504",
    "EXTRA18_504",
    "LR18_252",
    "TREE18_252",
    "EXTRA18_252",
)


def now():
    return datetime.now(ZONE)


def save(path: Path, payload, *, replace=False):
    """原始证据只创建一次；报告等派生状态原子替换，哈希供后续读取校验。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical({"hash": digest(payload), "payload": payload})
    if not replace:
        with path.open("x", encoding="utf-8") as f:
            f.write(raw)
    else:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(raw, encoding="utf-8")
        os.replace(tmp, path)


def read(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["hash"] != digest(value["payload"]):
        raise ValueError("EVIDENCE_HASH_MISMATCH")
    return value["payload"]


def error_code(exc):
    # 连接器异常可能带SQL或地址，只记录异常类型和我们定义的安全状态码。
    msg = str(exc)
    return msg if isinstance(exc, ValueError) and msg.replace("_", "").isupper() else type(exc).__name__


def initialize():
    if (ROOT / "protocol.json").exists():
        return read(ROOT / "protocol.json")
    start = now()
    spec = {
        "study": "DIRECTION_1D_SPRINT_20260914",
        "started_at": start.isoformat(),
        "deadline_at": (start + timedelta(days=3)).isoformat(),
        "new_cost_cny": 0,
        "challenge_accuracy": 0.8,
        "coverage_target": 0.95,
        "target": "NEXT_TRADING_DAY_RAW_UNIT_NAV_UP_VS_NON_UP",
        "selection_period": "2025 quarterly walk-forward; 2024 only if 2025 unavailable",
        "reserved_audit_period": "2026-01-01 through 2026-09-11; no selection by this period's score",
        "historical_availability": "RECONSTRUCTED_ANN_DATE_ONLY_NOT_TRUE_FORWARD",
        "primary_metric": "mean over target dates of mean over product families",
        "baselines": ["ALWAYS_UP", "ALWAYS_NON_UP", "MOMENTUM", "TRAIN_MAJORITY", "LR7_504"],
        "candidates": list(CANDIDATES),
        "confidence_threshold": 0.5,
        "research_rounds_may_expand_with_new_predeclared_plan": True,
        "forward_policy": "first successful answer per fund and target, written before 08:30; immutable",
        "forward_model_policy": "freeze active bundle separately for each target date before reading its outcome",
        "long_term_success": "not established by three days; report distinct dates, coverage and class balance",
        "release": "MODEL_NOT_RELEASED",
        "business_database_writes": 0,
    }
    save(ROOT / "protocol.json", spec)
    return spec


def source():
    with get_nav_preview_engine().connect() as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        row = dict(repo.source(c))
        clock = c.execute(text("SELECT clock_timestamp()")).scalar_one()
        if abs((clock - now()).total_seconds()) > 5:
            raise ValueError("CLOCK_SKEW")
        if not row.get("authorization_verified_at") or row["rate_limit_per_minute"] <= 0:
            raise ValueError("SOURCE_NOT_VERIFIED")
        return row


def snapshot():
    """冻结当前本人关注范围及已有净值；不读其他人的关注，也不登记业务模型。"""
    initialize()
    if (ROOT / "history.json").exists():
        return {"status": "ALREADY_CAPTURED"}
    context = live_context(
        PROJECT / ".local-runs/direction-1d-calibration-preflight-20260913/runtime-after.json",
        Path("C:/ideaProject/workSpace12/.env"),
    )
    if not (ROOT / "scope.json").exists():
        save(ROOT / "scope.json", context)
    funds = []
    with get_nav_preview_engine().connect() as c, c.begin():
        c.execute(text("SET TRANSACTION READ ONLY"))
        s = repo.source(c)
        for p in context["profiles"]:
            group = classify(p, prediction=True)
            if not group["group_id"]:
                continue
            rows = repo.navs(c, p["fund_code"], s["source_id"], date(2021, 1, 1), now().date())
            funds.append(
                {
                    "fund_code": p["fund_code"],
                    "source_fund_code": p["source_fund_code"],
                    "group": group["group_id"],
                    "family": group["product_family_id"],
                    "rows": [
                        {
                            "date": str(r["nav_date"]),
                            "nav": str(r["unit_nav"]),
                            "ann_date": str(r["ann_date"]) if r["ann_date"] else None,
                            "source_hash": r["content_hash"],
                        }
                        for r in rows
                    ],
                }
            )
    capture = {
        "at": now().isoformat(),
        "funds": funds,
        "expires_at": (now() + timedelta(days=context["source"]["retention_days"])).isoformat(),
        "availability": "HISTORICAL_RECONSTRUCTION",
    }
    save(ROOT / "history.json", capture)
    return {"funds": len(funds), "nav_rows": sum(len(f["rows"]) for f in funds)}


def vector(values, extended):
    """扩展仅用T及以前净值：1/2/3/10日收益、短期波动/上涨占比/反转等11项。"""
    x = features(values)
    if not extended:
        return x
    v = np.asarray(values, dtype=float)
    r = v[1:] / v[:-1] - 1
    extra = [
        *(v[-1] / v[-1 - k] - 1 for k in (1, 2, 3, 10)),
        float(np.std(r[-5:])),
        float(np.mean(r[-5:] > 0)),
        float(np.mean(r[-20:] > 0)),
        float(r[-1] - np.mean(r[-5:])),
        float(np.mean(r[-5:]) - np.mean(r[-20:])),
        float(np.min(r[-5:])),
        float(np.max(r[-5:])),
    ]
    return x + extra


def samples(history):
    days, _ = calendar()
    result = []
    for fund in history["funds"]:
        points = {r["date"]: r for r in fund["rows"]}
        for i in range(60, len(days) - 2):
            t, u, maturity = map(str, days[i : i + 3])
            wanted = [str(d) for d in days[i - 60 : i + 2]]
            if not all(d in points for d in wanted):
                continue
            # 公告日期晚于预测截止日期的输入不算可用；历史无精确时刻依然标注假设。
            if any((points[d].get("ann_date") or d) > u for d in wanted[:-1]):
                continue
            vals = [points[d]["nav"] for d in wanted[:-1]]
            try:
                x = vector(vals, True)
                answer = label(points[t]["nav"], points[u]["nav"])
            except ValueError:
                continue
            result.append(
                {
                    "code": fund["fund_code"],
                    "family": fund["family"],
                    "group": fund["group"],
                    "t": t,
                    "u": u,
                    "mature": max(maturity, points[u].get("ann_date") or u),
                    "x": x,
                    "y": answer["y"],
                    "actual_direction": answer["actual_direction"],
                    "momentum": int(float(vals[-1]) > float(vals[-2])),
                }
            )
    return sorted(result, key=lambda r: (r["u"], r["family"], r["code"]))


def fit(rows, name, cutoff):
    """整天切分并留成熟间隔；同一产品的不同份额按同日家族倒数加权。"""
    eligible = [r for r in rows if r["mature"] < cutoff]
    dates = sorted({r["u"] for r in eligible})[-int(name.split("_")[1]) :]
    selected_dates = set(dates)
    eligible = [r for r in eligible if r["u"] in selected_dates]
    if len(dates) < 120 or min(Counter(r["y"] for r in eligible).values(), default=0) < 20:
        raise ValueError("FIT_SAMPLE_INSUFFICIENT")
    if len({r["y"] for r in eligible}) < 2:
        raise ValueError("FIT_SINGLE_CLASS")
    counts = Counter((r["u"], r["family"]) for r in eligible)
    w = np.asarray([1 / counts[(r["u"], r["family"])] for r in eligible])
    n = 7 if "7_" in name else 18
    x, y = np.asarray([r["x"][:n] for r in eligible]), np.asarray([r["y"] for r in eligible])
    if name.startswith("LR"):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=0))
        model.fit(x, y, standardscaler__sample_weight=w, logisticregression__sample_weight=w)
    else:
        model = (
            HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=7,
                learning_rate=0.05,
                l2_regularization=10,
                min_samples_leaf=40,
                early_stopping=False,
                random_state=0,
            )
            if name.startswith("TREE")
            else ExtraTreesClassifier(n_estimators=150, max_depth=6, min_samples_leaf=30, n_jobs=2, random_state=0)
        )
        model.fit(x, y, sample_weight=w)
    return {
        "model": model,
        "n": n,
        "majority": int(np.average(y, weights=w) > 0.5),
        "fit_end": max(r["u"] for r in eligible),
        "fit_rows": len(eligible),
        "fit_dates": len(dates),
    }


def metrics(rows):
    if not rows:
        return {"count": 0, "accuracy": None}
    family_days = defaultdict(list)
    date_rows = defaultdict(list)
    for r in rows:
        family_days[(r["u"], r["family"])].append(int(r["prediction"] == r["y"]))
    for (day, _), values in family_days.items():
        date_rows[day].append(float(np.mean(values)))
    recalls = [
        np.mean([r["prediction"] == c for r in rows if r["y"] == c]) for c in (0, 1) if any(r["y"] == c for r in rows)
    ]
    return {
        "count": len(rows),
        "distinct_dates": len(date_rows),
        "family_dates": len(family_days),
        "accuracy": float(np.mean([np.mean(v) for v in date_rows.values()])),
        "raw_accuracy": float(np.mean([r["prediction"] == r["y"] for r in rows])),
        "balanced_accuracy": float(np.mean(recalls)) if len(recalls) == 2 else None,
        "actual_up_rate": float(np.mean([r["y"] for r in rows])),
        "flat_count": sum(r.get("actual_direction") == "FLAT" for r in rows),
    }


def train():
    initialize()
    if (ROOT / "round-01/result.json").exists():
        return read(ROOT / "round-01/result.json")
    history = read(ROOT / "history.json")
    if now() >= datetime.fromisoformat(history["expires_at"]):
        raise ValueError("EVIDENCE_EXPIRED")
    rows = samples(history)
    year = 2025 if sum(r["u"].startswith("2025") for r in rows) > 1000 else 2024
    plan = {
        "candidates": list(CANDIDATES),
        "development_year": year,
        "history_hash": digest(history),
        "created_at": now().isoformat(),
        "source_code_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "held_out_scores_read": False,
        "threshold": 0.5,
        "groups": sorted({r["group"] for r in rows}),
    }
    plan_path = ROOT / "round-01/plan.json"
    if plan_path.exists():
        old = read(plan_path)
        if any(old[k] != plan[k] for k in ("history_hash", "candidates", "development_year", "source_code_hash")):
            raise ValueError("ROUND_INPUT_CHANGED")
    else:
        save(plan_path, plan)
    predictions = defaultdict(list)
    with threadpool_limits(limits=2):
        for quarter in range(1, 5):
            start = f"{year}-{3 * quarter - 2:02d}-01"
            end = f"{year + 1}-01-01" if quarter == 4 else f"{year}-{3 * quarter + 1:02d}-01"
            for group in plan["groups"]:
                grouped = [r for r in rows if r["group"] == group]
                exam = [r for r in grouped if start <= r["u"] < end]
                if not exam:
                    continue
                for name in CANDIDATES:
                    path = ROOT / f"round-01/folds/{quarter}-{group}-{name}.json"
                    if path.exists():
                        tested = read(path)
                    else:
                        fitted = fit(grouped, name, start)
                        scores = fitted["model"].predict_proba([r["x"][: fitted["n"]] for r in exam])[:, 1]
                        tested = [
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                            | {"prediction": int(s > 0.5), "momentum": r["momentum"], "majority": fitted["majority"]}
                            for r, s in zip(exam, scores, strict=True)
                        ]
                        save(path, tested)
                    predictions[name].extend(tested)
            save(
                ROOT / "progress.json",
                {"phase": "TRAIN", "completed_quarter": quarter, "at": now().isoformat()},
                replace=True,
            )
        results = {name: metrics(values) for name, values in predictions.items()}
        reference = predictions["LR7_504"]
        for name, key in (
            ("ALWAYS_UP", 1),
            ("ALWAYS_NON_UP", 0),
            ("MOMENTUM", "momentum"),
            ("TRAIN_MAJORITY", "majority"),
        ):
            results[name] = metrics([r | {"prediction": r[key] if isinstance(key, str) else key} for r in reference])
        winner = max(CANDIDATES, key=lambda n: (results[n]["accuracy"], -CANDIDATES.index(n)))
        bundle = {
            name: {
                group: fit([r for r in rows if r["group"] == group], name, str(now().date()))
                for group in plan["groups"]
            }
            for name in dict.fromkeys(("LR7_504", winner))
        }
    # 原已登记模型独立保留；LR7_504是重新训练的对照，不能冒充页面上的旧模型。
    originals = {}
    for row in read(ROOT / "scope.json")["registry"]:
        if row["group_id"] in originals:
            continue
        if datetime.fromisoformat(row["expires_at"]) <= now():
            continue
        originals[row["group_id"]] = {
            "kind": "REGISTERED_7",
            "payload": load_model(row),
            "model_id": row["model_id"],
            "expires_at": row["expires_at"],
        }
    if set(originals) != set(plan["groups"]):
        raise ValueError("ORIGINAL_MODEL_GROUP_MISSING")
    bundle["ORIGINAL7"] = originals
    model_path = ROOT / "round-01/models.joblib"
    joblib.dump(bundle, model_path)
    result = {
        "at": now().isoformat(),
        "winner": winner,
        "development_year": year,
        "metrics": results,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "kind": "DEVELOPMENT_ONLY",
        "forward_model_names": list(bundle),
        "source_code_hash": plan["source_code_hash"],
        "held_out_scores_read": False,
        "conclusion": "真实未来成绩仍待形成；历史最高值不是成功证明。",
    }
    save(ROOT / "round-01/result.json", result)
    return result


def refresh():
    """只查询已授权fund_nav与固定本人范围，响应独立保存，不触发全市场同步。"""
    s = source()
    history = read(ROOT / "history.json")
    if now() >= datetime.fromisoformat(history["expires_at"]):
        raise ValueError("EVIDENCE_EXPIRED")
    cfg = get_settings()
    if cfg.tushare_api_url != "https://api.tushare.pro":
        raise ValueError("UPSTREAM_URL_UNEXPECTED")
    out = {"at": now().isoformat(), "funds": [], "errors": [], "expires_at": history["expires_at"]}
    with TushareFundClient(
        token=cfg.tushare_token.get_secret_value(),
        api_url=cfg.tushare_api_url,
        connect_timeout_seconds=5,
        read_timeout_seconds=20,
        max_retries=0,
        catalog_max_rows_per_query=15000,
    ) as client:
        for fund in history["funds"]:
            try:
                records = client.list_nav_history(
                    fund["source_fund_code"], start_date=now().date() - timedelta(days=130), end_date=now().date()
                )
                # 每次观测只保存运行所需的近130日，完整训练历史只封存一份。
                lower = str(now().date() - timedelta(days=130))
                points = {r["date"]: r for r in fund["rows"] if r["date"] >= lower}
                for r in records:
                    if r.nav_date > now().date() or (r.ann_date and r.ann_date > now().date()):
                        raise ValueError("FUTURE_NAV_RESPONSE")
                    points[str(r.nav_date)] = {
                        "date": str(r.nav_date),
                        "nav": str(r.unit_nav),
                        "ann_date": str(r.ann_date) if r.ann_date else None,
                        "received_at": now().isoformat(),
                    }
                out["funds"].append(fund | {"rows": list(points.values())})
            except Exception as exc:
                out["errors"].append({"code": fund["fund_code"], "error": error_code(exc)})
            time.sleep(max(0.35, 60 / min(s["rate_limit_per_minute"], 120)))
    stamp = now().strftime("%Y%m%dT%H%M%S%f")
    save(ROOT / f"observations/{stamp}.json", out)
    save(ROOT / "latest-nav.json", out, replace=True)
    return out


def model_answers(models, group, x):
    """分清已登记的原7项模型和本轮重训模型；分数只作研究量，不展示成可靠概率。"""
    answers = {}
    for name, groups in models.items():
        fitted = groups[group]
        if fitted.get("kind") == "REGISTERED_7":
            if datetime.fromisoformat(fitted["expires_at"]) <= now():
                raise ValueError("ORIGINAL_MODEL_EXPIRED")
            value = registered_score(fitted["payload"], x[:7])
        else:
            with threadpool_limits(limits=2):
                value = float(fitted["model"].predict_proba([x[: fitted["n"]]])[0, 1])
        answers[name] = {"prediction": int(value > 0.5), "research_score": value}
    return answers


def tick():
    """到期只报告；错过窗口不补写；各基金首次成功答案不可被下一次运行覆盖。"""
    spec = read(ROOT / "protocol.json")
    if now() >= datetime.fromisoformat(spec["deadline_at"]):
        return report()
    w = window(now())
    pending = [
        p for p in (ROOT / "forward").glob("*/*.json") if not (ROOT / "outcomes" / p.parent.name / p.name).exists()
    ]
    if w["status"] != "OPEN" and not pending:
        save(
            ROOT / "last-tick.json",
            {"at": now().isoformat(), "window": w, "errors": [], "action": "WAITING_FOR_NEXT_WINDOW"},
            replace=True,
        )
        return report()
    result = read(ROOT / "round-01/result.json")
    path = ROOT / "round-01/models.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]:
        raise ValueError("MODEL_HASH_MISMATCH")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != result["source_code_hash"]:
        raise ValueError("FORWARD_CODE_CHANGED_REVIEW_REQUIRED")
    # joblib只读取本研究自行训练、哈希核对成功的本地包，不接受外来模型文件。
    models = joblib.load(path)
    history = refresh()
    if now() >= datetime.fromisoformat(history["expires_at"]):
        raise ValueError("EVIDENCE_EXPIRED")
    w = window(now())
    for fund in history["funds"]:
        points = {r["date"]: r for r in fund["rows"]}
        if w["status"] == "OPEN":
            target_path = ROOT / f"forward/{w['target_nav_date']}/{fund['fund_code']}.json"
            wanted = list(map(str, input_days(date.fromisoformat(w["base_nav_date"]))))
            if not target_path.exists() and all(d in points for d in wanted):
                try:
                    x = vector([points[d]["nav"] for d in wanted], True)
                    answers = model_answers(models, fund["group"], x)
                    answers.update(
                        {
                            "ALWAYS_UP": {"prediction": 1},
                            "ALWAYS_NON_UP": {"prediction": 0},
                            "MOMENTUM": {
                                "prediction": int(float(points[wanted[-1]]["nav"]) > float(points[wanted[-2]]["nav"]))
                            },
                        }
                    )
                    captured = now()
                    if captured >= datetime.fromisoformat(w["deadline_at"]):
                        raise ValueError("MISSED_DEADLINE")
                    save(
                        target_path,
                        {
                            "code": fund["fund_code"],
                            "family": fund["family"],
                            "group": fund["group"],
                            "base": wanted[-1],
                            "u": w["target_nav_date"],
                            "at": captured.isoformat(),
                            "base_nav": points[wanted[-1]]["nav"],
                            "inputs": [points[d] for d in wanted],
                            "deadline": w["deadline_at"],
                            "models_hash": result["model_sha256"],
                            "answers": answers,
                            "status": "EXPERIMENTAL_NOT_RELEASED",
                        },
                    )
                except ValueError as exc:
                    history["errors"].append({"code": fund["fund_code"], "error": error_code(exc)})
        for forecast_path in (ROOT / "forward").glob(f"*/{fund['fund_code']}.json"):
            f = read(forecast_path)
            outcome_path = ROOT / f"outcomes/{f['u']}/{fund['fund_code']}.json"
            target = points.get(f["u"])
            # 当前已拿到且目标交易日已收盘，才允许形成结果；保留原始基准净值及修订提示。
            if (
                target
                and not outcome_path.exists()
                and now() >= datetime.combine(date.fromisoformat(f["u"]), day_time(18), ZONE)
            ):
                save(
                    outcome_path,
                    {
                        **label(f["base_nav"], target["nav"]),
                        "observed_at": now().isoformat(),
                        "target_source": target,
                        "forecast_hash": digest(f),
                        "base_revision_detected": points.get(f["base"], {}).get("nav") != f["base_nav"],
                    },
                )
    save(ROOT / "last-tick.json", {"at": now().isoformat(), "window": w, "errors": history["errors"]}, replace=True)
    return report()


def report():
    spec = read(ROOT / "protocol.json")
    forecasts = [read(p) for p in (ROOT / "forward").glob("*/*.json")]
    measured = defaultdict(list)
    for f in forecasts:
        path = ROOT / f"outcomes/{f['u']}/{f['code']}.json"
        if path.exists():
            outcome = read(path)
            if outcome["forecast_hash"] != digest(f):
                raise ValueError("FORECAST_OUTCOME_HASH_MISMATCH")
            for name, answer in f["answers"].items():
                measured[name].append(f | outcome | {"prediction": answer["prediction"]})
    count = next(iter(measured.values()), [])
    days, _ = calendar()
    start, end = datetime.fromisoformat(spec["started_at"]), datetime.fromisoformat(spec["deadline_at"])
    targets = [str(d) for d in days if start < datetime.combine(d, day_time(8, 30), ZONE) <= end]
    closed = [d for d in targets if datetime.combine(date.fromisoformat(d), day_time(8, 30), ZONE) <= now()]
    eligible_count = len(read(ROOT / "history.json")["funds"]) if (ROOT / "history.json").exists() else 0
    scope_count = len(read(ROOT / "scope.json")["codes"]) if (ROOT / "scope.json").exists() else 0
    closed_count = sum(f["u"] in closed for f in forecasts)
    eligible_due, full_due = len(closed) * eligible_count, len(closed) * scope_count
    data = {
        "at": now().isoformat(),
        "deadline_at": spec["deadline_at"],
        "phase": "CHECKPOINT_REACHED" if now() >= datetime.fromisoformat(spec["deadline_at"]) else "RUNNING",
        "future_forecasts": len(forecasts),
        "mature_outcomes": len(count),
        "pending": len(forecasts) - len(count),
        "distinct_mature_dates": len({r["u"] for r in count}),
        "forward_metrics": {n: metrics(v) for n, v in measured.items()},
        "planned_targets": targets,
        "closed_targets": closed,
        "watchlist_funds": scope_count,
        "eligible_funds": eligible_count,
        "unsupported_funds": scope_count - eligible_count,
        "due_eligible_predictions": eligible_due,
        "missing_eligible_predictions": eligible_due - closed_count,
        "eligible_coverage": closed_count / eligible_due if eligible_due else None,
        "whole_watchlist_coverage": closed_count / full_due if full_due else None,
        "new_cost_cny": 0,
        "model_released": False,
        "conclusion": "三天检查点不足以证明长期80%或所有方法均不可行。",
    }
    save(ROOT / "report.json", data, replace=True)
    return data
