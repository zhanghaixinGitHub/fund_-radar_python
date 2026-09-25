"""002112 自身完整历史的独立离线训练资料；只准备数据，不拟合或登记模型。

原 2021—2023 单基金方案及十基金补充资料保持原样。本方案按 C 类自身成立后的
完整可用历史扩展，保留所有排除原因；不能挑选持平日期、改变标签或使用 A 类净值。
"""

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

from sqlalchemy import text

from app.db.session import get_engine
from app.integrations.tushare_sprint_stock_breadth_v2 import FIELDS, validate_quote_values
from app.services.direction_1d_protocol import FEATURES, RECIPE, ZONE, calendar, digest, features, label
from app.services.fund_exposure_common import FUND, ROOT, now, read, save
from app.services.fund_exposure_features import EXPOSURE_FEATURES, MARKET_FEATURES, calculate, select_report
from app.services.fund_exposure_quotes import INDICES, Provider, permission, safe_error
from app.services.fund_materials_store import versioned_save
from app.services.fund_peer_materials import QuoteDays

STORE = ROOT / "readiness-extended"
CALENDAR_FILE = Path(__file__).resolve().parents[1] / "data/calendars/cn_a_share_2015_2020_research_v1.json"
CALENDAR_HASH = "798a999a94f791cc7f856324ba4968738eee051a4b379434e4a6f8cfedce68ae"
INDEX_FIELDS = ["ts_code", "trade_date", "close", "pre_close", "pct_chg"]
PLAN = {
    "version": "FUND_002112_FULL_OWN_HISTORY_READINESS_V1",
    "fund_code": FUND,
    "purpose": "OFFLINE_EXPLORATORY_TRAINING_ONLY_NOT_PRODUCTION_ADOPTION",
    "scope_reason": "USE_COMPLETE_AVAILABLE_C_SHARE_HISTORY_NOT_LABEL_SELECTED_DATES_OR_PEERS",
    "prior_inspection_disclosed": "NAV_PRECHECK_FOUND_30_FLAT_DATES_BEFORE_HOLDINGS_CHECK_NOT_A_BLIND_PREREGISTRATION",
    "nav_start": "2015-11-16",
    "fit_end": "2023-12-31",
    "fit_as_of": "2024-01-01T08:00:00+08:00",
    "development_start": "2024-01-01",
    "development_end": "2024-12-31",
    "development_is_independent_test": False,
    "protected_label_years": [2025],
    "reconstructed_2026_labels_allowed": False,
    "target": "UNIT_NAV_DIRECTION_THREE_STATE_V2",
    "minimum_fit_dates": 252,
    "minimum_class_dates": 30,
    "nav_lookback_sessions": 61,
    "quote_lookback_sessions": 21,
    "max_report_age_days": 210,
    "minimum_positive_weight_quote_coverage": 1.0,
    "missing_quote_policy": "EXCLUDE_NEVER_ZERO",
    "report_weight_policy": "REPORTED_WEIGHT_OR_EXPLICIT_DERIVATION_FROM_SAME_REPORT_VALUE_AND_TOTAL_NAV",
    "historical_availability": "DECLARED_PUBLICATION_DATE_NEXT_DAY_0800_AND_NEXT_SESSION_QUOTES_RECONSTRUCTION",
    "features": list(FEATURES + EXPOSURE_FEATURES + MARKET_FEATURES),
    "candidates": ["NAV7", "NAV7_HOLDINGS", "NAV7_HOLDINGS_MARKET"],
    "recipe": RECIPE,
    "maximum_future_fits": 6,
    "automatic_training": False,
    "automatic_adoption": False,
    "new_purchase_cny": 0,
}


def initialize():
    """独立固定本方案；原研究文件只读，后续重试不能悄悄改变时间或样本门槛。"""
    path = STORE / "plan.json"
    if path.exists():
        if read(path) != PLAN:
            raise ValueError("READINESS_PLAN_CHANGED")
    else:
        save(path, PLAN)
    return PLAN


def sessions():
    """旧年份依据交易所公告构造，包含 2019 五一及 2020 春节临时调整。"""
    value = json.loads(CALENDAR_FILE.read_text(encoding="utf-8"))
    if digest(value) != CALENDAR_HASH:
        raise ValueError("READINESS_CALENDAR_HASH_CHANGED")
    first, last = date.fromisoformat(value["coverage_start"]), date.fromisoformat(value["coverage_end"])
    closed = set()
    for year in value["years"]:
        for left, right in year["closed_ranges"]:
            left, right = date.fromisoformat(left), date.fromisoformat(right)
            closed.update(left + timedelta(days=i) for i in range((right - left).days + 1))
    older = tuple(first + timedelta(days=i) for i in range((last - first).days + 1))
    older = tuple(d for d in older if d.weekday() < 5 and d not in closed)
    newer, current_hash = calendar()
    newer = tuple(d for d in newer if d <= date(2024, 12, 31))
    return older + newer, digest([CALENDAR_HASH, current_hash])


def checked_raw(receipt):
    """哈希和许可留存期限都有效才复用；不延长旧证据期限。"""
    path = Path(receipt["raw_path"]) if receipt.get("raw_path") else ROOT / receipt["file"]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise ValueError("READINESS_SOURCE_HASH_MISMATCH")
    if receipt.get("expires_at") and datetime.fromisoformat(receipt["expires_at"]) <= now():
        raise ValueError("READINESS_SOURCE_EXPIRED")
    return raw


def verify_derived_market(older, index):
    """从留存接口原文重建聚合行情，防止只核摘要却信任被改过的中间数字。"""
    reconstructed = defaultdict(dict)
    for code, receipt in older["receipts"].items():
        data = json.loads(checked_raw(receipt))["data"]
        if data["fields"] != FIELDS:
            raise ValueError("READINESS_QUOTE_FIELDS_CHANGED")
        for item in data["items"]:
            if item[0] != code:
                raise ValueError("READINESS_QUOTE_IDENTITY_CHANGED")
            day = datetime.strptime(item[1], "%Y%m%d").date().isoformat()
            if code in reconstructed[day]:
                raise ValueError("READINESS_QUOTE_DUPLICATE")
            reconstructed[day][code] = dict(zip(FIELDS[2:], item[2:], strict=True))
    if {d: v["rows"] for d, v in older["days"].items()} != dict(reconstructed):
        raise ValueError("READINESS_QUOTE_AGGREGATE_CHANGED")
    for value in older["days"].values():
        sources = {c: older["receipts"][c]["sha256"] for c in value["rows"]}
        if value["receipt"]["sha256"] != digest(sources) or value["receipt"]["source_hashes"] != sources:
            raise ValueError("READINESS_QUOTE_LINEAGE_CHANGED")
    for code, value in index.items():
        merged = {}
        for receipt in value["receipts"]:
            data = json.loads(checked_raw(receipt))["data"]
            fields = data["fields"]
            for item in data["items"]:
                row = dict(zip(fields, item, strict=True))
                if row["ts_code"] != code:
                    raise ValueError("READINESS_INDEX_IDENTITY_CHANGED")
                day = datetime.strptime(row["trade_date"], "%Y%m%d").date().isoformat()
                if day <= PLAN["development_end"]:
                    # 后续回执是来源的更新版本，顺序与聚合构建相同。
                    merged[day] = {k: row[k] for k in fields if k not in ("ts_code", "trade_date")}
        if merged != value["rows"]:
            raise ValueError("READINESS_INDEX_AGGREGATE_CHANGED")


def load_nav():
    """只读已保存 C 类净值，严格截在 2024 年；缺公布日期、修订痕迹均保留。"""
    source = permission()
    if "fund_nav" not in source["authorized_api_names"]:
        raise ValueError("EXPOSURE_API_NOT_AUTHORIZED")
    with get_engine().connect() as connection:
        connection.execute(text("SET TRANSACTION READ ONLY"))
        connection.execute(text("SET LOCAL statement_timeout='15000'"))
        rows = connection.execute(
            text("""SELECT nav_date,unit_nav,ann_date,content_hash FROM nav_daily
            WHERE fund_code=:code AND source_id=:source AND nav_date BETWEEN :first AND :last ORDER BY nav_date"""),
            {"code": FUND, "source": source["source_id"], "first": date(2015, 11, 16), "last": date(2024, 12, 31)},
        ).mappings()
        rows = [
            {
                "date": str(r["nav_date"]),
                "nav": str(r["unit_nav"]),
                "ann_date": str(r["ann_date"]) if r["ann_date"] else None,
                "source_hash": r["content_hash"],
            }
            for r in rows
        ]
    value = {"fund_code": FUND, "source_id": str(source["source_id"]), "at": now().isoformat(), "rows": rows}
    versioned_save(STORE / "nav.json", value)
    return value


def nav_rows(nav, trading_days, *, fund_code=FUND, family="DBFUND_001412"):
    """每条输入必须含 61 个连续交易日，答案仅为紧接着的一个交易日。"""
    mapping = {r["date"]: r for r in nav["rows"]}
    if len(mapping) != len(nav["rows"]):
        raise ValueError("READINESS_DUPLICATE_NAV")
    rows, excluded = [], []
    for i in range(60, len(trading_days) - 1):
        days = [str(d) for d in trading_days[i - 60 : i + 2]]
        base, target = days[-2:]
        reason = None
        if any(d not in mapping for d in days):
            reason = "NAV_GAP"
        elif any(not mapping[d]["ann_date"] for d in days):
            reason = "NAV_PUBLICATION_UNKNOWN"
        elif any(mapping[d]["ann_date"] > target for d in days[:-1]):
            reason = "NAV_NOT_PUBLIC_BY_TARGET"
        if reason:
            excluded.append({"base": base, "target": target, "reason": reason})
            continue
        try:
            x = features([mapping[d]["nav"] for d in days[:-1]])
            answer = label(mapping[base]["nav"], mapping[target]["nav"])
        except ValueError as exc:
            excluded.append({"base": base, "target": target, "reason": str(exc)})
            continue
        mature = datetime.combine(
            max(date.fromisoformat(mapping[target]["ann_date"]), date.fromisoformat(target)) + timedelta(days=1),
            time(8),
            ZONE,
        )
        rows.append(
            {
                "fund_code": fund_code,
                "family": family,
                "base": base,
                "target": target,
                "t": base,
                "u": target,
                "as_of": target + "T08:00:00+08:00",
                "mature_at": mature.isoformat(),
                "nav_features": x,
                "nav_input_dates": days[:-1],
                "nav_input_hashes": [mapping[d]["source_hash"] for d in days[:-1]],
                "nav_last_publication_date": max(mapping[d]["ann_date"] for d in days[:-1]),
                **answer,
            }
        )
    return rows, excluded


def indices(provider, trading_days):
    """补旧年份两个现有指数，并与公告日历逐日交叉核对；不凭净值有无猜交易日。"""
    result = {}
    expected = {str(d) for d in trading_days if d.year < 2021}
    for code in INDICES:
        value, receipt = provider.query(
            "index_daily", {"ts_code": code, "start_date": "20150101", "end_date": "20201231"}, INDEX_FIELDS
        )
        checked_raw(receipt)
        rows = {}
        for item in value["data"]["items"]:
            if item[0] != code or any(type(v) not in (int, float) or not math.isfinite(v) for v in item[2:]):
                raise ValueError("READINESS_INDEX_IDENTITY_OR_VALUE")
            day = datetime.strptime(item[1], "%Y%m%d").date().isoformat()
            if day in rows or item[2] <= 0 or item[3] <= 0 or abs(100 * (item[2] / item[3] - 1) - item[4]) > 0.002:
                raise ValueError("READINESS_INDEX_DUPLICATE_OR_RETURN")
            rows[day] = dict(zip(INDEX_FIELDS[2:], item[2:], strict=True))
        if {d for d in rows if d >= PLAN["nav_start"]} != expected:
            raise ValueError("READINESS_OFFICIAL_CALENDAR_QUOTE_MISMATCH")
        current = read(ROOT / "indices" / (code + ".json"))
        for r in current["receipts"]:
            checked_raw(r)
        result[code] = {
            "rows": {**rows, **{d: v for d, v in current["rows"].items() if d <= PLAN["development_end"]}},
            "receipts": [receipt, *current["receipts"]],
        }
    versioned_save(STORE / "indices.json", result)
    return result


def load_reports():
    result = read(STORE / "report-result.json")
    reports = [read(STORE / "reports" / (name + ".json")) for name in result["report_files"]]
    for report in reports:
        checked_raw(report["raw"])
        if report["fund_code"] != FUND or report["report_end"] > "2024-09-30":
            raise ValueError("READINESS_REPORT_SCOPE")
    return reports


def acquire_older_quotes(rows, reports, trading_days, provider, progress):
    """只拉本基金历史正权重持仓股票的所需区间；已有 2021 起原文直接复用。"""
    needed = defaultdict(set)
    positions = {str(d): i for i, d in enumerate(trading_days)}
    for row in rows:
        if row["base"] >= "2021-02-01":
            continue
        at = datetime.fromisoformat(row["as_of"])
        report = select_report(reports, at)
        i = positions[row["base"]]
        dates = [str(d) for d in trading_days[i - 20 : i + 1] if d.year < 2021]
        for holding in report["holdings"]:
            if float(holding["nav_weight_pct"]) > 0:
                needed[holding["stock_code"]].update(dates)
    by_day, receipts, errors = defaultdict(dict), {}, []
    for i, (code, days) in enumerate(sorted(needed.items())):
        progress(i, len(needed), code, "补齐本基金早期持仓行情")
        try:
            days = sorted(days)
            if not days or not code.endswith((".SH", ".SZ")):
                raise ValueError("READINESS_STOCK_SCOPE")
            value, receipt = provider.query(
                "daily",
                {"ts_code": code, "start_date": days[0].replace("-", ""), "end_date": days[-1].replace("-", "")},
                FIELDS,
            )
            checked_raw(receipt)
            seen = set()
            for item in value["data"]["items"]:
                day = datetime.strptime(item[1], "%Y%m%d").date().isoformat()
                if (
                    item[0] != code
                    or day in seen
                    or not days[0] <= day <= days[-1]
                    or date.fromisoformat(day) not in trading_days
                ):
                    raise ValueError("READINESS_QUOTE_SCOPE_OR_DUPLICATE")
                seen.add(day)
                by_day[day][code] = dict(zip(FIELDS[2:], item[2:], strict=True))
            receipts[code] = receipt
            if not seen:
                errors.append({"stock_code": code, "reason": "SOURCE_RETURNED_EMPTY"})
        except Exception as exc:
            errors.append({"stock_code": code, "reason": safe_error(exc)})
    output = {}
    for day, quotes in sorted(by_day.items()):
        # 实际早期原文涨幅常为两位小数，2018 后逐步变为四位；沿用同一除权昨收
        # 公式验证，并按原文量化精度解释舍入差。训练仍用原值，不重写原文或标签。
        precision = Counter()
        maximum_error = 0.0
        for row in quotes.values():
            digits = 2 if round(row["pct_chg"], 2) == row["pct_chg"] else 4
            tolerance = 0.0051 if digits == 2 else 0.00011
            maximum_error = max(maximum_error, validate_quote_values(row, rounding_tolerance=tolerance))
            precision[digits] += 1
        sources = {c: receipts[c]["sha256"] for c in quotes}
        output[day] = {
            "rows": quotes,
            "receipt": {
                "sha256": digest(sources),
                "source_hashes": sources,
                "received_at": max(receipts[c]["received_at"] for c in quotes),
            },
            "pct_precision_counts": dict(precision),
            "maximum_pct_formula_error": maximum_error,
        }
    value = {
        "days": output,
        "receipts": receipts,
        "errors": errors,
        "requested": {c: sorted(v) for c, v in needed.items()},
    }
    versioned_save(STORE / "older-quotes.json", value)
    return value


def build(progress=lambda *args: None):
    """补必要行情并构造完整记录；是否合格由后续统一检查判定，不把采集成功等同可训练。"""
    initialize()
    trading_days, calendar_hash = sessions()
    nav = load_nav()
    rows, excluded = nav_rows(nav, trading_days)
    source_nav_rows = rows
    provider = Provider()
    index = indices(provider, trading_days)
    reports = load_reports()
    old_quotes = acquire_older_quotes(rows, reports, trading_days, provider, progress)
    recent = QuoteDays()

    class Quotes:
        def get(self, day, default=None):
            return old_quotes["days"].get(day, default) if day < "2021-01-01" else recent.get(day, default)

    bundle = {"reports": reports, "days": Quotes(), "indices": index}
    complete, gaps = [], defaultdict(set)
    for i, row in enumerate(rows):
        if i % 100 == 0:
            progress(i, len(rows), FUND, "逐日核对本基金训练输入完整性")
        try:
            exposure = calculate(
                bundle,
                date.fromisoformat(row["base"]),
                datetime.fromisoformat(row["as_of"]),
                research_sessions=trading_days,
            )
            if exposure["holdings_features"] is None or exposure["market_features"] is None:
                excluded.append(
                    {
                        "base": row["base"],
                        "target": row["target"],
                        "reason": "HOLDINGS_OR_MARKET_INCOMPLETE",
                        "missing": exposure["missing"],
                        "report_end": exposure["report_end"],
                    }
                )
                for item in exposure["missing"]:
                    gaps[item["stock_code"]].update(item["dates"])
                continue
            complete.append(
                {
                    **row,
                    "exposure": exposure,
                    "x": row["nav_features"] + exposure["holdings_features"] + exposure["market_features"],
                }
            )
        except ValueError as exc:
            if str(exc) != "EXPOSURE_NO_AVAILABLE_REPORT":
                raise
            excluded.append({"base": row["base"], "target": row["target"], "reason": safe_error(exc)})
    train = [r for r in complete if r["target"] <= PLAN["fit_end"] and r["mature_at"] <= PLAN["fit_as_of"]]
    development = [r for r in complete if PLAN["development_start"] <= r["target"] <= PLAN["development_end"]]

    def counts(values):
        return {
            "dates": len({r["target"] for r in values}),
            "classes": dict(Counter(r["actual_direction"] for r in values)),
        }

    result = {
        "at": now().isoformat(),
        "fund_code": FUND,
        "plan_hash": digest(PLAN),
        "calendar_hash": calendar_hash,
        "nav_hash": digest(nav),
        "train": train,
        "development": development,
        "excluded": excluded,
        "nav_only_train_counts": counts(
            [r for r in source_nav_rows if r["target"] <= PLAN["fit_end"] and r["mature_at"] <= PLAN["fit_as_of"]]
        ),
        "counts": {"train": counts(train), "development": counts(development)},
        "quote_gaps": {c: sorted(v) for c, v in gaps.items()},
        "new_requests": provider.count,
        "acquisition_errors": old_quotes["errors"],
        "training_ready": False,
        "training_runs": 0,
    }
    # 本文件是检查中间成果；只有独立验收函数签发 ready 清单后才能交给训练程序。
    key = digest(result)
    save(STORE / "datasets" / (key + ".json"), result)
    versioned_save(
        STORE / "dataset-current.json", {"file": f"datasets/{key}.json", "counts": result["counts"], "at": result["at"]}
    )
    return {
        "file": f"datasets/{key}.json",
        "counts": result["counts"],
        "errors": old_quotes["errors"],
        "quote_gaps": len(gaps),
        "new_requests": provider.count,
    }


def validate_rows(rows, *, split, nav, trading_days):
    """独立重算净值、标签与时间边界；重复日期不增加样本数，缺失输入不允许进入训练。"""
    expected, _ = nav_rows(nav, trading_days)
    expected = {r["target"]: r for r in expected}
    seen = set()
    for row in rows:
        target = row["target"]
        if target in seen or target not in expected or row["fund_code"] != FUND:
            raise ValueError("READINESS_SAMPLE_IDENTITY_OR_DUPLICATE")
        seen.add(target)
        original = expected[target]
        for key in original:
            if row.get(key) != original[key]:
                raise ValueError("READINESS_NAV_OR_LABEL_CHANGED")
        if split == "train":
            if target > PLAN["fit_end"] or row["mature_at"] > PLAN["fit_as_of"]:
                raise ValueError("READINESS_TRAIN_TIME_LEAK")
        elif split == "development":
            if not PLAN["development_start"] <= target <= PLAN["development_end"]:
                raise ValueError("READINESS_DEVELOPMENT_SCOPE")
        else:
            raise ValueError("READINESS_UNKNOWN_SPLIT")
        exposure = row["exposure"]
        x = row["x"]
        if len(x) != len(PLAN["features"]) or any(type(v) not in (int, float) or not math.isfinite(v) for v in x):
            raise ValueError("READINESS_FEATURE_VALUE_OR_SHAPE")
        if x != row["nav_features"] + exposure["holdings_features"] + exposure["market_features"]:
            raise ValueError("READINESS_FEATURES_CHANGED")
        if exposure["missing"] or exposure["market_errors"] or exposure["status"] != "AVAILABLE":
            raise ValueError("READINESS_INPUTS_INCOMPLETE")
        if (
            exposure["report_available_at"] > row["as_of"]
            or not 0 <= exposure["report_age_days"] <= PLAN["max_report_age_days"]
        ):
            raise ValueError("READINESS_REPORT_TIME_LEAK")
        if abs(exposure["quote_coverage_of_disclosed_weight"] - 1) > 1e-9:
            raise ValueError("READINESS_QUOTE_COVERAGE")
    classes = {c: sum(r["actual_direction"] == c for r in rows) for c in ("UP", "FLAT", "DOWN")}
    return {
        "dates": len(seen),
        "classes": classes,
        "eligible": len(seen) >= PLAN["minimum_fit_dates"]
        and all(v >= PLAN["minimum_class_dates"] for v in classes.values()),
    }


def verify():
    """发布前从磁盘重新核验；通过仅代表允许离线训练，不代表已经训练或适合启用。"""
    initialize()
    current = read(STORE / "dataset-current.json")
    path = (STORE / current["file"]).resolve()
    if path.parent != (STORE / "datasets").resolve():
        raise ValueError("READINESS_DATASET_PATH")
    dataset = read(path)
    if path.stem != digest(dataset) or dataset["plan_hash"] != digest(PLAN):
        raise ValueError("READINESS_DATASET_HASH_CHANGED")
    nav = read(STORE / "nav.json")
    trading_days, calendar_hash = sessions()
    if dataset["nav_hash"] != digest(nav) or dataset["calendar_hash"] != calendar_hash:
        raise ValueError("READINESS_INPUT_VERSION_CHANGED")
    report_map = {r["raw"]["sha256"]: r for r in load_reports()}
    older = read(STORE / "older-quotes.json")
    for receipt in older["receipts"].values():
        checked_raw(receipt)
    index = read(STORE / "indices.json")
    for value in index.values():
        for receipt in value["receipts"]:
            checked_raw(receipt)
    verify_derived_market(older, index)
    recent = QuoteDays()

    class Quotes:
        def get(self, day, default=None):
            return older["days"].get(day, default) if day < "2021-01-01" else recent.get(day, default)

    bundle = {"reports": list(report_map.values()), "days": Quotes(), "indices": index}
    for row in dataset["train"] + dataset["development"]:
        # 重算全部持仓与大盘输入，不相信上次写入的 ready 布尔值。
        actual = calculate(
            bundle,
            date.fromisoformat(row["base"]),
            datetime.fromisoformat(row["as_of"]),
            research_sessions=trading_days,
        )
        if actual != row["exposure"]:
            raise ValueError("READINESS_EXPOSURE_REPRODUCTION_FAILED")
    train = validate_rows(dataset["train"], split="train", nav=nav, trading_days=trading_days)
    development = validate_rows(dataset["development"], split="development", nav=nav, trading_days=trading_days)
    ready = train["eligible"] and bool(dataset["development"]) and not dataset["acquisition_errors"]
    result = {
        "at": now().isoformat(),
        "fund_code": FUND,
        "training_ready": ready,
        "status": "READY_FOR_OFFLINE_TRAINING" if ready else "MORE_QUALIFIED_SAMPLES_REQUIRED",
        "dataset_file": current["file"],
        "dataset_hash": digest(dataset),
        "plan_hash": digest(PLAN),
        "train": train,
        "development": development,
        "training_runs": 0,
        "models_registered": 0,
        "production_adoption_allowed": False,
        "limitations": [
            "EXPLORATORY_HISTORY_EXTENSION",
            "HISTORICAL_AVAILABILITY_RECONSTRUCTED",
            "2024_DEVELOPMENT_ALREADY_OBSERVED_NOT_FINAL_TEST",
            "RECENT_REGIME_AND_FLAT_RECALL_REQUIRE_FUTURE_EVALUATION",
            "FINANCIAL_NEWS_AND_PDF_SEMANTIC_EVENTS_NOT_MODEL_FEATURES",
        ],
    }
    versioned_save(STORE / "readiness.json", result)
    return result
