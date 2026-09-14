"""三天研究第二轮：在同一份净值题目上加入沪深300、中证500量价特征。

本模块独立版本化，不修改首轮冻结代码。新答案引用原答案哈希并单独保存；
已有Windows任务通过新入口顺序执行首轮及本轮，两者共用进程互斥锁。
"""

import hashlib
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from app.integrations.tushare_market_reference import TushareMarketReferenceClient
from app.services import direction_1d_sprint as base

INDICES = ("000300.SH", "000905.SH")
CANDIDATES = ("LR30_252", "TREE30_252", "EXTRA30_252")


def root():
    return base.ROOT / "round-02"


def fingerprint():
    names = (
        "app/services/direction_1d_sprint_market.py",
        "app/services/direction_1d_sprint.py",
        "app/services/direction_1d_protocol.py",
        "app/integrations/tushare_market_reference.py",
    )
    return {n: hashlib.sha256((base.PROJECT / n).read_bytes()).hexdigest() for n in names}


def active():
    """截止后只可读本地报告；来源授权和有效期由已有只读入口检查。"""
    spec = base.read(base.ROOT / "protocol.json")
    if base.now() >= datetime.fromisoformat(spec["deadline_at"]):
        raise ValueError("SPRINT_DEADLINE_REACHED")
    source = base.source()
    if "index_daily" not in source["authorized_api_names"]:
        raise ValueError("INDEX_DAILY_NOT_AUTHORIZED")
    history = base.read(base.ROOT / "history.json")
    if base.now() >= datetime.fromisoformat(history["expires_at"]):
        raise ValueError("EVIDENCE_EXPIRED")
    return source


def plan():
    """在下载和评分之前固定三个假设、同题比较口径及最大查询/拟合数量。"""
    path = root() / "plan.json"
    if path.exists():
        value = base.read(path)
        if value["code"] != fingerprint():
            raise ValueError("ROUND_02_CODE_CHANGED")
        return value
    original = base.read(base.ROOT / "round-01/result.json")
    spec = {
        "at": base.now().isoformat(),
        "candidates": list(CANDIDATES),
        "indices": list(INDICES),
        "history_hash": base.digest(base.read(base.ROOT / "history.json")),
        "code": fingerprint(),
        "development_year": original["development_year"],
        "threshold": 0.5,
        "control": "TREE18_252",
        "primary_metric": "equal target-date then product-family weights",
        "features": "18 NAV + each index 1/5/20d return,20d volatility,amount/previous20d mean-1 + 2 size spreads",
        "index_amount_unit": "thousand CNY; model uses dimensionless ratio",
        "max_historical_requests": 14,
        "max_development_fits": 36,
        "max_forward_fits": 9,
        "historical_input_period": "2021-01-01 through latest completed NAV base date",
        "held_out_2026_scores_read": False,
        "new_cost_cny": 0,
        "future_primary": "highest development score, fixed before first future answer",
        "availability": "historical close reconstructed now; future actual receipts required",
        "fund_nav_document": "https://tushare.pro/document/2?doc_id=119",
        "index_document": "https://tushare.pro/document/1?doc_id=95",
        "ann_date_finding": "Announcement date is not documented as first-publication time; do not backdate",
    }
    base.save(path, spec)
    return spec


def client():
    cfg = base.get_settings()
    if cfg.tushare_api_url != "https://api.tushare.pro":
        raise ValueError("UPSTREAM_URL_UNEXPECTED")
    return TushareMarketReferenceClient(
        token=cfg.tushare_token.get_secret_value(),
        api_url=cfg.tushare_api_url,
        connect_timeout_seconds=5,
        read_timeout_seconds=20,
        max_retries=0,
        catalog_max_rows_per_query=2000,
        max_rows_per_query=1000,
    )


def query(c, index, start, end):
    """每次最多一年；保存日期、收盘、千元成交额，不把缺失成交额填0。"""
    if index not in INDICES or (end - start).days > 366 or start > end or end > base.now().date():
        raise ValueError("INDEX_QUERY_OUT_OF_SCOPE")
    records = c.list_index_activity(index, start_date=start, end_date=end)
    dates = [r.trade_date for r in records]
    if not records or len(set(dates)) != len(dates) or any(not start <= d <= end for d in dates):
        raise ValueError("INDEX_RESPONSE_DATES_INVALID")
    result = {}
    for r in records:
        if r.amount is None or not np.isfinite([float(r.close_price), float(r.amount)]).all():
            raise ValueError("INDEX_AMOUNT_MISSING")
        if r.close_price <= 0 or r.amount <= 0:
            raise ValueError("INDEX_VALUE_INVALID")
        result[str(r.trade_date)] = {"close": str(r.close_price), "amount": str(r.amount)}
    return {
        "index": index,
        "start": str(start),
        "end": str(end),
        "received_at": base.now().isoformat(),
        "rows": result,
        "kind": "UPSTREAM_READ_ONLY_RESPONSE",
    }


def acquire():
    p = plan()
    source = active()
    path = root() / "market.json"
    if path.exists():
        return {"status": "ALREADY_CAPTURED"}
    end = date.fromisoformat(base.window(base.now())["base_nav_date"])
    points = {code: {} for code in INDICES}
    with client() as c:
        # 先验证近期小样本的真实权限和覆盖，再扩展历史；失败不自动购买权限。
        for index in INDICES:
            proof = root() / f"market/probe-{index}.json"
            if not proof.exists():
                base.save(proof, query(c, index, end - timedelta(days=6), end))
                time.sleep(max(0.5, 60 / source["rate_limit_per_minute"]))
        for year in range(2021, end.year + 1):
            for index in INDICES:
                saved = root() / f"market/{year}-{index}.json"
                if not saved.exists():
                    base.save(saved, query(c, index, date(year, 1, 1), min(date(year, 12, 31), end)))
                    time.sleep(max(0.5, 60 / source["rate_limit_per_minute"]))
                points[index].update(base.read(saved)["rows"])
    payload = {
        "at": base.now().isoformat(),
        "plan_hash": base.digest(p),
        "indices": points,
        "expires_at": base.read(base.ROOT / "history.json")["expires_at"],
        "historical_only": True,
    }
    base.save(path, payload)
    return {"index_rows": {code: len(rows) for code, rows in points.items()}, "last_day": str(end)}


def market_vector(nav, anchor, indices):
    """只取T及以前连续21个交易日，目标日行情即使已存在也不读取。"""
    sessions = list(map(str, base.calendar()[0]))
    i = sessions.index(anchor)
    if i < 20 or len(nav) != 18:
        raise ValueError("MARKET_FEATURE_WINDOW_INVALID")
    wanted = sessions[i - 20 : i + 1]
    extra = []
    for index in INDICES:
        rows = indices[index]
        if any(d not in rows for d in wanted):
            raise ValueError("MARKET_INPUT_INCOMPLETE")
        close = np.asarray([float(rows[d]["close"]) for d in wanted])
        amount = np.asarray([float(rows[d]["amount"]) for d in wanted])
        if not np.isfinite([*close, *amount]).all() or np.any(close <= 0) or np.any(amount <= 0):
            raise ValueError("MARKET_VALUE_INVALID")
        extra += [
            *(float(close[-1] / close[-1 - k] - 1) for k in (1, 5, 20)),
            float(np.std(close[1:] / close[:-1] - 1)),
            float(amount[-1] / amount[:-1].mean() - 1),
        ]
    return list(nav) + extra + [extra[5] - extra[0], extra[6] - extra[1]]


def dataset():
    market = base.read(root() / "market.json")
    if base.now() >= datetime.fromisoformat(market["expires_at"]):
        raise ValueError("EVIDENCE_EXPIRED")
    result, rejected, cache = [], Counter(), {}
    for r in base.samples(base.read(base.ROOT / "history.json")):
        if r["t"] not in cache:
            try:
                cache[r["t"]] = market_vector([0.0] * 18, r["t"], market["indices"])[18:]
            except ValueError as exc:
                cache[r["t"]] = base.error_code(exc)
        x = cache[r["t"]]
        if isinstance(x, str):
            rejected[x] += 1
        else:
            result.append(r | {"x": r["x"] + x})
    return result, dict(rejected)


def fit(rows, name, cutoff):
    """复制首轮明确配方，仅把输入长度变为30；训练标签仍严格早于截止且已成熟。"""
    selected = [r for r in rows if r["mature"] < cutoff]
    dates = set(sorted({r["u"] for r in selected})[-252:])
    selected = [r for r in selected if r["u"] in dates]
    classes = Counter(r["y"] for r in selected)
    if len(dates) < 120 or len(classes) < 2 or min(classes.values()) < 20:
        raise ValueError("FIT_SAMPLE_INSUFFICIENT")
    counts = Counter((r["u"], r["family"]) for r in selected)
    weights = [1 / counts[r["u"], r["family"]] for r in selected]
    x, y = np.asarray([r["x"] for r in selected]), np.asarray([r["y"] for r in selected])
    if x.shape[1] != 30 or not np.isfinite(x).all():
        raise ValueError("FIT_FEATURE_INVALID")
    if name == "LR30_252":
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1500, random_state=0))
        model.fit(x, y, standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
    else:
        if name == "TREE30_252":
            model = HistGradientBoostingClassifier(
                max_iter=100,
                max_leaf_nodes=7,
                learning_rate=0.05,
                l2_regularization=10,
                min_samples_leaf=40,
                early_stopping=False,
                random_state=0,
            )
        elif name == "EXTRA30_252":
            model = ExtraTreesClassifier(n_estimators=150, max_depth=6, min_samples_leaf=30, n_jobs=2, random_state=0)
        else:
            raise ValueError("CANDIDATE_NOT_FROZEN")
        model.fit(x, y, sample_weight=weights)
    return {
        "model": model,
        "n": 30,
        "fit_end": max(r["u"] for r in selected),
        "fit_dates": len(dates),
        "fit_rows": len(selected),
        "fit_hash": base.digest(selected),
    }


def train():
    p = plan()
    active()
    path = root() / "result.json"
    if path.exists():
        return base.read(path)
    rows, excluded = dataset()
    if base.digest(base.read(base.ROOT / "history.json")) != p["history_hash"]:
        raise ValueError("HISTORY_CHANGED")
    by_name = defaultdict(list)
    groups = sorted({r["group"] for r in rows})
    year = p["development_year"]
    with threadpool_limits(limits=2):
        for q in range(1, 5):
            start = f"{year}-{q * 3 - 2:02d}-01"
            end = f"{year + 1}-01-01" if q == 4 else f"{year}-{q * 3 + 1:02d}-01"
            for group in groups:
                grouped = [r for r in rows if r["group"] == group]
                exam = [r for r in grouped if start <= r["u"] < end]
                keys = {(r["code"], r["u"]) for r in exam}
                old = base.read(base.ROOT / f"round-01/folds/{q}-{group}-TREE18_252.json")
                control = [r for r in old if (r["code"], r["u"]) in keys]
                if len(control) != len(exam):
                    raise ValueError("COMMON_EXAM_MISMATCH")
                by_name["TREE18_252"].extend(control)
                by_name["ALWAYS_UP"].extend(r | {"prediction": 1} for r in control)
                for name in CANDIDATES:
                    saved = root() / f"folds/{q}-{group}-{name}.json"
                    if saved.exists():
                        values = base.read(saved)
                    else:
                        fitted = fit(grouped, name, start)
                        scores = fitted["model"].predict_proba([r["x"] for r in exam])[:, 1]
                        values = [
                            {k: r[k] for k in ("code", "family", "group", "u", "y", "actual_direction")}
                            | {"prediction": int(score > 0.5)}
                            for r, score in zip(exam, scores, strict=True)
                        ]
                        base.save(saved, values)
                    by_name[name].extend(values)
            base.save(root() / "progress.json", {"quarter": q, "at": base.now().isoformat()}, replace=True)
        metrics = {n: base.metrics(v) for n, v in by_name.items()}
        winner = max(CANDIDATES, key=lambda n: (metrics[n]["accuracy"], -CANDIDATES.index(n)))
        models = {
            n: {g: fit([r for r in rows if r["group"] == g], n, str(base.now().date())) for g in groups}
            for n in CANDIDATES
        }
    file = root() / "models.joblib"
    joblib.dump(models, file)
    result = {
        "at": base.now().isoformat(),
        "metrics": metrics,
        "winner": winner,
        "excluded": excluded,
        "model_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        "code": fingerprint(),
        "market_hash": base.digest(base.read(root() / "market.json")),
        "held_out_2026_scores_read": False,
        "kind": "DEVELOPMENT_ONLY",
        "new_cost_cny": 0,
        "conclusion": "历史开发比较不能代替真实未来结果；三条分支如实保留，不按未来成绩切换主候选。",
    }
    base.save(path, result)
    return result


def models():
    result = base.read(root() / "result.json")
    if result["code"] != fingerprint():
        raise ValueError("ROUND_02_CODE_CHANGED")
    path = root() / "models.joblib"
    if hashlib.sha256(path.read_bytes()).hexdigest() != result["model_sha256"]:
        raise ValueError("ROUND_02_MODEL_HASH_MISMATCH")
    return result, joblib.load(path)


def preflight():
    """真实数据推理验收，明确不写forward目录、不把历史分数计作提前预测。"""
    _, bundle = models()
    points = base.read(root() / "market.json")["indices"]
    anchor = date.fromisoformat(base.window(base.now())["base_nav_date"])
    wanted = list(map(str, base.input_days(anchor)))
    count = 0
    for f in base.read(base.ROOT / "latest-nav.json")["funds"]:
        nav = {r["date"]: r["nav"] for r in f["rows"]}
        x = market_vector(base.vector([nav[d] for d in wanted], True), wanted[-1], points)
        for groups in bundle.values():
            with threadpool_limits(limits=2):
                score = float(groups[f["group"]]["model"].predict_proba([x])[0, 1])
            if not 0 <= score <= 1:
                raise ValueError("SCORE_INVALID")
            count += 1
    value = {"at": base.now().isoformat(), "kind": "DRY_RUN_NOT_FORWARD", "model_checks": count}
    base.save(root() / "preflight.json", value)
    return value


def tick():
    """在首轮已封存的同题输入上添加新答案；必须当前窗口仍打开，且实际持久化未超时。"""
    w = base.window(base.now())
    if base.now() >= datetime.fromisoformat(base.read(base.ROOT / "protocol.json")["deadline_at"]):
        return report()
    if w["status"] != "OPEN":
        return report()
    targets = [
        p
        for p in (base.ROOT / "forward" / w["target_nav_date"]).glob("*.json")
        if not (root() / "forward" / w["target_nav_date"] / p.name).exists()
    ]
    if not targets:
        return report()
    active()
    manifest, bundle = models()
    anchor = date.fromisoformat(w["base_nav_date"])
    received = {}
    with client() as c:
        for index in INDICES:
            received[index] = query(c, index, anchor - timedelta(days=50), anchor)
            time.sleep(0.5)
    stamp = base.now().strftime("%Y%m%dT%H%M%S%f")
    market = {
        "at": base.now().isoformat(),
        "responses": received,
        "indices": {i: r["rows"] for i, r in received.items()},
    }
    base.save(root() / f"observations/{stamp}.json", market)
    deadline = datetime.fromisoformat(w["deadline_at"])
    for path in targets:
        if base.now() >= deadline:
            break
        parent = base.read(path)
        if (
            datetime.fromisoformat(parent["at"]) >= deadline
            or parent["base"] != str(anchor)
            or datetime.fromtimestamp(path.stat().st_mtime, base.ZONE) >= deadline
        ):
            raise ValueError("PARENT_FORECAST_TIME_INVALID")
        nav = base.vector([r["nav"] for r in parent["inputs"]], True)
        x = market_vector(nav, parent["base"], market["indices"])
        answers = {}
        with threadpool_limits(limits=2):
            for name, groups in bundle.items():
                value = float(groups[parent["group"]]["model"].predict_proba([x])[0, 1])
                answers[name] = {"prediction": int(value > 0.5), "research_score": value}
        saved = root() / "forward" / parent["u"] / path.name
        payload = {
            "at": base.now().isoformat(),
            "u": parent["u"],
            "code": parent["code"],
            "parent_hash": base.digest(parent),
            "market_hash": base.digest(market),
            "market_file": f"observations/{stamp}.json",
            "model_hash": manifest["model_sha256"],
            "x": x,
            "answers": answers,
            "status": "MODEL_NOT_RELEASED",
        }
        base.save(saved, payload)
        # 保存后回读并记录真实核对时刻；越过截止的文件保留证据但不会计入成功预测。
        persisted = base.now()
        valid = base.read(saved) == payload and persisted < deadline
        base.save(
            root() / "receipts" / parent["u"] / path.name,
            {
                "forecast_hash": base.digest(payload),
                "readback_at": persisted.isoformat(),
                "status": "VERIFIED" if valid else "LATE_OR_INVALID",
            },
        )
    return report()


def report():
    result_path = root() / "result.json"
    if not result_path.exists():
        return {"phase": "NOT_TRAINED"}
    groups = defaultdict(list)
    verified, late, pending = 0, 0, 0
    for path in (root() / "forward").glob("*/*.json"):
        value = base.read(path)
        receipt_path = root() / "receipts" / value["u"] / path.name
        receipt = base.read(receipt_path) if receipt_path.exists() else {}
        if receipt.get("status") != "VERIFIED" or receipt.get("forecast_hash") != base.digest(value):
            late += 1
            continue
        verified += 1
        parent = base.read(base.ROOT / "forward" / value["u"] / path.name)
        if value["parent_hash"] != base.digest(parent):
            raise ValueError("PARENT_HASH_CHANGED")
        outcome_path = base.ROOT / "outcomes" / value["u"] / path.name
        if not outcome_path.exists():
            pending += 1
            continue
        outcome = base.read(outcome_path)
        if outcome["forecast_hash"] != base.digest(parent):
            raise ValueError("PARENT_OUTCOME_HASH_CHANGED")
        for name, answer in (value["answers"] | parent["answers"]).items():
            groups[name].append(parent | outcome | {"prediction": answer["prediction"]})
    parent_report = base.report()
    due = parent_report["due_eligible_predictions"]
    closed = sum(
        p.parent.name in parent_report["closed_targets"] and base.read(p).get("status") == "VERIFIED"
        for p in (root() / "receipts").glob("*/*.json")
    )
    value = {
        "at": base.now().isoformat(),
        "primary_candidate": base.read(result_path)["winner"],
        "verified_forecasts": verified,
        "invalid_or_late": late,
        "pending": pending,
        "eligible_coverage": closed / due if due else None,
        "whole_watchlist_coverage": (
            closed / (parent_report["watchlist_funds"] * len(parent_report["closed_targets"])) if due else None
        ),
        "missing_due_predictions": due - closed,
        "matched_forward_metrics": {k: base.metrics(v) for k, v in groups.items()},
        "model_released": False,
        "new_cost_cny": 0,
    }
    base.save(root() / "report.json", value, replace=True)
    return value
