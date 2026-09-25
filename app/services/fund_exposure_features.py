"""按当时可得资料构造 002112 的持仓输入；历史重建与真实提前记录分开。"""

from collections import Counter
from datetime import date, datetime, time
from math import prod

from sqlalchemy import text

from app.db.session import get_engine
from app.repositories import direction_1d as repo
from app.services.direction_1d_protocol import FEATURES, ZONE, calendar, digest, features, label
from app.services.fund_exposure_common import FUND, ROOT, initialize, now, read, save
from app.services.fund_exposure_quotes import INDICES, permission, reports

EXPOSURE_FEATURES = (
    "disclosed_nav_return_1d",
    "disclosed_nav_return_5d",
    "up_stock_nav_weight",
    "nav_weighted_amount_vs_previous20",
    "nav_weight_concentration",
    "disclosed_nav_weight",
    "reported_stock_nav_weight",
    "report_age_days",
    "full_disclosure",
)
MARKET_FEATURES = ("csi300_return_1d", "csi300_return_5d", "csi500_return_1d", "csi500_return_5d")


def specification():
    """在读取标签前固定输入定义；成交活跃指标只采用成交额，不叠加高度相似的换手率。"""
    value = {
        "version": "EXPOSURE_FEATURES_V1",
        "nav": list(FEATURES),
        "holdings": list(EXPOSURE_FEATURES),
        "market": list(MARKET_FEATURES),
        "lookback_sessions": 21,
        "minimum_known_weight_quote_coverage": 1.0,
        "market_references": INDICES,
        "industry_policy": "REPORT_BROAD_INDUSTRY_PLUS_DISCLOSED_STOCK_BASKET_PROXY_NO_CURRENT_TAG_BACKFILL",
        "amount_unit": "THOUSAND_CNY",
        "volume_unit": "HUNDRED_SHARES",
        "price_unit": "CNY",
        "return_unit": "FRACTION",
        "turnover_source_unit": "PERCENT",
        "weight_denominator": "FUND_NAV_NOT_STOCK_ASSETS",
        "missing_quote": "EXCLUDE_NOT_ZERO",
        "historical_quote_availability": "NEXT_TRADING_DAY_0800_RECONSTRUCTED",
        "recipe": "same frozen multinomial logistic regression as current three state protocol",
        "comparison": "2021-2023 fit, 2024 development check; common dates plus baseline full coverage",
        "minimum_class_dates": 30,
        "minimum_fit_dates": 252,
        "maximum_fits": 6,
        "selection_metric": "same-date correct days and per-class recall; evidence insufficient prevents adoption",
        "simple_control": "training-set majority with fixed tie order FLAT DOWN UP",
    }
    path = ROOT / "feature-spec.json"
    if path.exists():
        if read(path) != value:
            raise ValueError("EXPOSURE_FEATURE_SPEC_CHANGED")
    else:
        save(path, value)
    return value


def select_report(rs, as_of, *, live=False):
    """先选最新报告期，再选该期已公开的版本；较晚发布的旧年报不能挤掉新季报。"""
    candidates = []
    for r in rs:
        if datetime.fromisoformat(r["available_at"]) > as_of:
            continue
        if (
            live
            and max(datetime.fromisoformat(r["raw"]["received_at"]), datetime.fromisoformat(r["parsed_at"])) > as_of
        ):
            continue
        candidates.append(r)
    if not candidates:
        raise ValueError("EXPOSURE_NO_AVAILABLE_REPORT")
    return max(candidates, key=lambda r: (r["report_end"], r["available_at"], r["raw"]["sha256"]))


def load_bundle():
    """加载有限研究目录并验证证据有效期；仅当前许可允许的来源可进入输入。"""
    source = permission()
    if not {"daily", "index_daily"}.issubset(source["authorized_api_names"]):
        raise ValueError("EXPOSURE_API_NOT_AUTHORIZED")
    days = {p.stem: read(p) for p in (ROOT / "stock-days").glob("*.json")}
    indices = {p.stem: read(p) for p in (ROOT / "indices").glob("*.json") if p.stem in INDICES}
    current = now()
    for receipt in [d["receipt"] for d in days.values()] + [r for v in indices.values() for r in v["receipts"]]:
        if datetime.fromisoformat(receipt["expires_at"]) <= current:
            raise ValueError("EXPOSURE_EVIDENCE_EXPIRED")
    return {"reports": reports(), "days": days, "indices": indices}


def calculate(bundle, base: date, as_of: datetime, *, live=False, research_sessions=None):
    """只估算披露股票的净资产贡献，不冒充实时持仓或基金收益。

    缺失任何正权重股票的必需窗口时，整组持仓输入不可用；同时保存覆盖率和股票名单。
    披露为 0.00% 的小仓位保留原表，不能反推其精确权重，因此不参与乘积。
    """
    # 独立研究可以传入已另行核验的更早日历；正式和原冻结研究默认行为不变。
    # 不允许真实预测借此绕开正式日历，调用方仍须保存研究日历的版本与哈希。
    if live and research_sessions is not None:
        raise ValueError("EXPOSURE_RESEARCH_CALENDAR_NOT_FOR_LIVE")
    sessions = research_sessions if research_sessions is not None else calendar()[0]
    i = sessions.index(base)
    if i < 20 or as_of <= datetime.combine(base, time(15), ZONE):
        raise ValueError("EXPOSURE_INPUT_TIME_INVALID")
    needed = [str(d) for d in sessions[i - 20 : i + 1]]
    if not live and as_of < datetime.combine(sessions[i + 1], time(8), ZONE):
        raise ValueError("EXPOSURE_HISTORICAL_QUOTE_NOT_YET_AVAILABLE")
    report = select_report(bundle["reports"], as_of, live=live)
    age = (base - date.fromisoformat(report["report_end"])).days
    missing = []
    weighted, weighted5, up_weight, activity, hhi, available_weight = [0.0] * 6
    total_weight = float(report["disclosed_nav_pct"]) / 100
    quote_hashes = {}
    for holding in report["holdings"]:
        weight = float(holding["nav_weight_pct"]) / 100
        if weight == 0:
            continue
        code = holding["stock_code"]
        rows = []
        absent = []
        for d in needed:
            day = bundle["days"].get(d)
            if day is None or code not in day["rows"]:
                absent.append(d)
                continue
            receipt = day["receipt"]
            if live and datetime.fromisoformat(receipt["received_at"]) > as_of:
                absent.append(d)
                continue
            rows.append(day["rows"][code])
            quote_hashes[d] = receipt["sha256"]
        if absent:
            missing.append(
                {
                    "stock_code": code,
                    "dates": absent,
                    "nav_weight": weight,
                    "reason": "SOURCE_NOT_RETURNED_OR_NOT_AVAILABLE_CAUSE_UNVERIFIED",
                }
            )
            continue
        mean_amount = sum(r["amount"] for r in rows[:-1]) / 20
        if mean_amount <= 0:
            missing.append(
                {"stock_code": code, "dates": needed, "nav_weight": weight, "reason": "AMOUNT_BASE_NOT_POSITIVE"}
            )
            continue
        available_weight += weight
        weighted += weight * rows[-1]["pct_chg"] / 100
        weighted5 += weight * (prod(1 + r["pct_chg"] / 100 for r in rows[-5:]) - 1)
        up_weight += weight if rows[-1]["pct_chg"] > 0 else 0
        activity += weight * (rows[-1]["amount"] / mean_amount - 1)
        hhi += weight**2
    x = [
        weighted,
        weighted5,
        up_weight,
        activity,
        hhi,
        total_weight,
        float(report["stock_nav_pct"]) / 100,
        float(age),
        float(report["full_stock_disclosure"]),
    ]
    valid = not missing and 0 <= age <= initialize()["max_report_age_days"] and total_weight > 0
    market = []
    market_errors = []
    market_hashes = {}
    for code in INDICES:
        index = bundle["indices"].get(code)
        if index is None or any(d not in index["rows"] for d in needed[-5:]):
            market_errors.append(code + ":MISSING")
            continue
        # 聚合文件只能在其所含版本均实际接收后用于真实记录，保守地使用最晚回执。
        if live and any(datetime.fromisoformat(r["received_at"]) > as_of for r in index["receipts"]):
            market_errors.append(code + ":NOT_RECEIVED")
            continue
        vals = [index["rows"][d]["pct_chg"] / 100 for d in needed[-5:]]
        market.extend([vals[-1], prod(1 + v for v in vals) - 1])
        market_hashes[code] = digest(index)
    return {
        "base_nav_date": str(base),
        "as_of": as_of.isoformat(),
        "mode": "LIVE_INPUT_ONLY" if live else "HISTORICAL_RECONSTRUCTION",
        "report_raw_sha256": report["raw"]["sha256"],
        "report_payload_hash": digest(report),
        "report_end": report["report_end"],
        "report_available_at": report["available_at"],
        "report_received_at": report["raw"]["received_at"],
        "full_disclosure": report["full_stock_disclosure"],
        "reported_industries": report["reported_industries"],
        "report_age_days": age,
        "disclosed_nav_weight": total_weight,
        "unexplained_nav_weight": max(0, 1 - total_weight),
        "quote_coverage_of_disclosed_weight": available_weight / total_weight if total_weight else None,
        "missing": missing,
        "quote_hashes": quote_hashes,
        "market_hashes": market_hashes,
        "holdings_features": x if valid else None,
        "market_features": market if not market_errors else None,
        "market_errors": market_errors,
        "status": "AVAILABLE" if valid else "INCOMPLETE_OR_STALE",
    }


def development_nav():
    """只读取截至 2024 的开发净值；2025 留存标签及 2026 历史成绩不进入这条查询。"""
    path = ROOT / "nav-development.json"
    if path.exists():
        return read(path)
    with get_engine().connect() as c:
        source = repo.source(c)
        rows = repo.navs(c, FUND, source["source_id"], date(2021, 1, 1), date(2024, 12, 31))
    value = {
        "fund_code": FUND,
        "received_at": now().isoformat(),
        "source_id": str(source["source_id"]),
        "rows": [
            {
                **r,
                "nav_date": str(r["nav_date"]),
                "unit_nav": str(r["unit_nav"]),
                "ann_date": str(r["ann_date"]) if r["ann_date"] else None,
                "updated_at": str(r["updated_at"]),
            }
            for r in rows
        ],
    }
    save(path, value)
    return value


def build_dataset():
    """按同一交易日构建比较题目；不随机切分、不用 2025 或回算的 2026 答案。"""
    spec = specification()
    bundle = load_bundle()
    nav = development_nav()
    mapping = {r["nav_date"]: r for r in nav["rows"]}
    sessions, _ = calendar()
    rows, unavailable = [], []
    for i in range(60, len(sessions) - 1):
        base, target = sessions[i : i + 2]
        if target > date(2024, 12, 31):
            break
        need = [str(d) for d in sessions[i - 60 : i + 2]]
        if any(d not in mapping for d in need):
            unavailable.append({"target": str(target), "reason": "NAV_GAP"})
            continue
        as_of = datetime.combine(target, time(8), ZONE)
        if any(mapping[d]["ann_date"] and mapping[d]["ann_date"] > str(target) for d in need[:-1]):
            unavailable.append({"target": str(target), "reason": "NAV_NOT_PUBLIC_BY_TARGET"})
            continue
        try:
            x = features([mapping[d]["unit_nav"] for d in need[:-1]])
            exposure = calculate(bundle, base, as_of)
        except ValueError as exc:
            unavailable.append({"target": str(target), "reason": str(exc)})
            continue
        answer = label(mapping[str(base)]["unit_nav"], mapping[str(target)]["unit_nav"])
        rows.append(
            {
                "base": str(base),
                "target": str(target),
                "nav_features": x,
                "exposure": exposure,
                "actual_direction": answer["actual_direction"],
            }
        )
    result = {
        "at": now().isoformat(),
        "kind": "HISTORICAL_DEVELOPMENT_ONLY",
        "spec_hash": digest(spec),
        "plan_hash": digest(initialize()),
        "nav_hash": digest(nav),
        "rows": rows,
        "unavailable": unavailable,
        "counts": {
            "nav": len(rows),
            "holdings": sum(r["exposure"]["holdings_features"] is not None for r in rows),
            "holdings_market": sum(
                r["exposure"]["holdings_features"] is not None and r["exposure"]["market_features"] is not None
                for r in rows
            ),
        },
        "missing_reasons": dict(Counter(r["reason"] for r in unavailable)),
    }
    key = digest(result)
    save(ROOT / "datasets" / (key + ".json"), result)
    save(
        ROOT / "dataset-current.json",
        {"file": f"datasets/{key}.json", "counts": result["counts"], "at": result["at"]},
        replace=True,
    )
    return {"file": f"datasets/{key}.json", "counts": result["counts"], "unavailable": len(unavailable)}


def training_gate(rows):
    """沿用现有三分类门槛；不能把上涨/下跌改叫持平来凑数。"""
    counts = Counter(r["actual_direction"] for r in rows)
    return {
        "eligible": len({r["target"] for r in rows}) >= 252 and all(counts[c] >= 30 for c in ("UP", "FLAT", "DOWN")),
        "dates": len({r["target"] for r in rows}),
        "class_counts": {c: counts[c] for c in ("UP", "FLAT", "DOWN")},
        "minimum_per_class": 30,
        "minimum_dates": 252,
    }


def study():
    """先检查训练能否成立；失败也形成完整比较可行性结论，绝不伪造候选成绩。"""
    specification()
    dataset = read(ROOT / read(ROOT / "dataset-current.json")["file"])
    rows = dataset["rows"]
    train = [r for r in rows if r["target"] <= "2023-12-31"]
    check = [r for r in rows if "2024-01-01" <= r["target"] <= "2024-12-31"]

    def common(r):
        return r["exposure"]["holdings_features"] is not None and r["exposure"]["market_features"] is not None

    gate = training_gate(train)
    if gate["eligible"]:
        # 数据范围或标签规则发生了实质变化：此入口不擅自新增训练/登记流程。
        raise ValueError("EXPOSURE_REVIEW_REQUIRED_UNEXPECTED_ELIGIBLE_COHORT")
    majority = max(("FLAT", "DOWN", "UP"), key=lambda c: gate["class_counts"][c])
    groups = []
    for name, values in (("NAV_FULL", check), ("COMMON_DATES", [r for r in check if common(r)])):
        groups.append(
            {
                "group": name,
                "dates": len(values),
                "actual_counts": dict(Counter(r["actual_direction"] for r in values)),
                "simple_majority": majority,
                "simple_majority_correct": sum(r["actual_direction"] == majority for r in values),
                "candidate_correct": None,
                "candidate_status": "NOT_TRAINED_INSUFFICIENT_THREE_STATE_SAMPLES",
            }
        )
    with get_engine().connect() as c:
        models = [
            dict(r)
            for r in c.execute(
                text(
                    "SELECT model_id,metadata->>'protocol' AS protocol,group_id "
                    "FROM direction_1d_model ORDER BY model_id"
                )
            ).mappings()
        ]
    result = {
        "at": now().isoformat(),
        "dataset_hash": digest(dataset),
        "status": "INSUFFICIENT_EVIDENCE_KEEP_EXISTING",
        "gate": gate,
        "common_training_gate": training_gate([r for r in train if common(r)]),
        "comparison": groups,
        "fits": 0,
        "candidate_predictions": 0,
        "mature_candidate_predictions": 0,
        "model_registry_before": models,
        "model_registry_hash": digest(models),
        "model_registry_changed": False,
        "reason_zh": (
            "002112 开发训练期持平样本少于 30 个；不降低门槛、不改标签、不擅自扩展其他基金。"
            "新增数据已备好，尚不能证明预测改善。"
        ),
    }
    # 每份结论单独留存，当前指针不改写过去版本。
    key = digest(result)
    save(ROOT / "studies" / (key + ".json"), result)
    save(
        ROOT / "study-current.json",
        {"file": f"studies/{key}.json", "status": result["status"], "at": result["at"]},
        replace=True,
    )
    return {k: v for k, v in result.items() if k != "model_registry_before"}
