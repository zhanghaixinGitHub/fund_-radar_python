"""002112 一日离线候选的独立训练包。此模块只准备、重验和交付输入，不拟合模型。

扩展自身历史及固定参照基金属于新的探索方案，原单基金方案、封存标签和正式预测
均不改动。文件齐全不等于行合格：以每个历史预测时刻逐日剔除缺失和未来信息。
"""

import math
import re
from collections import Counter
from datetime import datetime, time, timedelta

from app.integrations.public_fund_reports import PEERS, report_period
from app.integrations.public_fund_reports import STORE as PEER_STORE
from app.services import fund_training_readiness as own
from app.services.direction_1d_protocol import ZONE, digest
from app.services.direction_1d_training import weights
from app.services.fund_exposure_common import FUND, ROOT, now, read, save
from app.services.fund_exposure_features import calculate, select_report
from app.services.fund_materials_store import versioned_save
from app.services.fund_peer_materials import QuoteDays, nav_history

STORE = ROOT / "training-ready"
CLASSES = ("DOWN", "FLAT", "UP")
PLAN = {
    "version": "FUND_002112_POOLED_EXPLORATORY_READINESS_V1",
    "target_fund": FUND,
    "horizon_trading_days": 1,
    "target": "UNIT_NAV_DIRECTION_THREE_STATE_V2",
    "flat_rule": "EXACT_EQUAL_UNIT_NAV",
    "own_nav_start": "2015-11-16",
    "peer_nav_start": "2021-01-01",
    "fit_end": "2023-12-31",
    "fit_as_of": "2024-01-01T08:00:00+08:00",
    "development": ["2024-01-01", "2024-12-31"],
    "evaluation_fund": FUND,
    "candidate_peers_fixed_before_report_repair": PEERS,
    "cohort_rule": "REPORTED_DOMESTIC_MIXED_FUND_WITH_POSITIVE_STOCK_EXPOSURE_AND_SAME_INPUT_DEFINITION",
    "selection_uses_labels_or_performance": False,
    "style_transfer": "EXPLORATORY_NOT_SAME_STRATEGY_NOT_EVIDENCE_OF_TARGET_ACCURACY",
    "family_weighting": "EQUAL_FAMILY_THEN_EQUAL_FAMILY_DATE_NEVER_DUPLICATE_A_C",
    "minimum_fit_dates": 252,
    "minimum_class_dates": 30,
    "minimum_target_fit_dates": 252,
    "minimum_quote_coverage": 1.0,
    "nav_lookback": 61,
    "quote_lookback": 21,
    "missing_inputs": "EXCLUDE_NOT_ZERO_NOT_CARRY_FORWARD",
    "missing_latest_report": "EXCLUDE_NOT_FALL_BACK_TO_OLDER_DISCLOSURE",
    "features": own.PLAN["features"],
    "variants": {"NAV7": 7, "NAV7_HOLDINGS": 16, "NAV7_HOLDINGS_MARKET": 20},
    "recipe": own.PLAN["recipe"],
    "maximum_future_fits": 6,
    "2024_is_independent_test": False,
    "protected_label_years": [2025],
    "reconstructed_2026_labels_allowed": False,
    "automatic_training": False,
    "automatic_adoption": False,
    "new_purchase_cny": 0,
    "prior_inspection": "ORIGINAL_AND_EXTENDED_CLASS_COUNTS_ALREADY_INSPECTED_EXPLORATORY_NOT_BLIND_TEST",
}


def initialize():
    """单独固定探索方案；不得用重试修改标签、日期、候选范围或样本门槛。"""
    path = STORE / "plan.json"
    if path.exists() and read(path) != PLAN:
        raise ValueError("TRAINING_PACKAGE_PLAN_CHANGED")
    if not path.exists():
        save(path, PLAN)


def public_time(day):
    return datetime.combine(datetime.fromisoformat(day).date() + timedelta(days=1), time(8), ZONE).isoformat()


def latest_listing(catalog, at):
    available = [r for r in catalog if r["available_at"] <= at]
    return max(available, key=lambda r: (r["report_end"], r["available_at"]), default=None)


def cohort_evidence(code, reports):
    """共同点是预测目标和输入口径，而非宣称投资风格相同；策略差异留待目标基金验证。"""
    if code not in PEERS or not reports:
        raise ValueError("TRAINING_COHORT_REPORT_MISSING")
    masters = {r["fund_master_code"] for r in reports}
    if len(masters) != 1:
        raise ValueError("TRAINING_COHORT_FAMILY_CHANGED")
    contexts = []
    for report in reports:
        text = re.sub(r"\s+", "", report["product_description_excerpt"])
        title = re.sub(r"\s+", "", report["title"])
        if (
            code not in text
            or "混合" not in title
            or any(k in title for k in ("ETF", "联接", "QDII", "FOF", "指数型", "指数增强"))
        ):
            raise ValueError("TRAINING_COHORT_PRODUCT_INCOMPATIBLE")
        if not all(k in text for k in ("投资目标", "投资策略", "业绩比较基准")):
            raise ValueError("TRAINING_COHORT_STRATEGY_EVIDENCE_MISSING")
        start, end = text.index("投资目标"), text.index("风险收益特征") if "风险收益特征" in text else len(text)
        contexts.append(
            {
                "report_sha256": report["raw"]["sha256"],
                "report_end": report["report_end"],
                "description": text[start : min(end, start + 3500)],
                "url": report["raw"]["url"],
            }
        )
    return {
        "fund_code": code,
        "family": "FUND_" + next(iter(masters)),
        "evidence": contexts,
        "compatibility": "COMMON_TARGET_AND_INPUT_UNITS_NOT_IDENTICAL_STYLE",
    }


def sources():
    """读取各基金自身净值和报告；历史缺口明列，原研究的任何文件都不重写。"""
    own_nav = read(own.STORE / "nav.json")
    own_reports = own.load_reports()
    source_catalog = read(own.STORE / "catalog.json")["reports"]
    own_catalog = []
    for item in source_catalog:
        period = report_period(item["title"])
        if period and "2015-12-31" <= period[0] <= "2024-09-30":
            day = datetime.fromtimestamp(int(item["catalog_activation_ms"]) / 1000, ZONE).date().isoformat()
            own_catalog.append({"report_end": period[0], "report_type": period[1], "available_at": public_time(day)})
    funds = {
        FUND: {
            "nav": own_nav,
            "reports": own_reports,
            "catalog": own_catalog,
            "family": "FUND_001412",
            "cohort": {"role": "TARGET_OWN_C_SHARE"},
        }
    }
    excluded = []
    for nav in nav_history()["funds"]:
        code = nav["fund_code"]
        path = PEER_STORE / code / "manifest.json"
        if not path.exists():
            excluded.append({"fund_code": code, "reason": "TRAINING_COHORT_REPORT_MISSING"})
            continue
        manifest = read(path)
        reports = [read(PEER_STORE / doc["file"]) for doc in manifest["documents"].values()]
        for report in reports:
            own.checked_raw(report["raw"])
            if report["fund_code"] != code:
                raise ValueError("TRAINING_PACKAGE_REPORT_IDENTITY")
        try:
            evidence = cohort_evidence(code, reports)
        except ValueError as exc:
            excluded.append({"fund_code": code, "reason": str(exc)})
            continue
        catalog = [
            {
                "report_end": e["report_end"],
                "report_type": e["report_type"],
                "available_at": public_time(e["published_date"]),
            }
            for e in manifest["catalog"]
        ]
        funds[code] = {
            "nav": nav,
            "reports": reports,
            "catalog": catalog,
            "family": evidence["family"],
            "cohort": evidence,
        }
    families = [v["family"] for v in funds.values()]
    if len(families) != len(set(families)):
        raise ValueError("TRAINING_PACKAGE_DUPLICATE_SHARE_FAMILY")
    return funds, excluded


def summarize(rows):
    """同时数记录、家族日期及真实日期；同一天多个基金不伪装成多个市场日期。"""
    return {
        "rows": len(rows),
        "dates": len({r["target"] for r in rows}),
        "classes": {k: sum(r["actual_direction"] == k for r in rows) for k in CLASSES},
        "class_distinct_dates": {k: len({r["target"] for r in rows if r["actual_direction"] == k}) for k in CLASSES},
    }


def eligibility(train, development):
    keys = [(r["family"], r["target"]) for r in train]
    if len(keys) != len(set(keys)):
        raise ValueError("TRAINING_PACKAGE_DUPLICATE_FAMILY_DATE")
    count = summarize(train)
    target = summarize([r for r in train if r["fund_code"] == FUND])
    ready = (
        count["dates"] >= PLAN["minimum_fit_dates"]
        and min(count["class_distinct_dates"].values()) >= PLAN["minimum_class_dates"]
        and target["dates"] >= PLAN["minimum_target_fit_dates"]
        and min(target["classes"].values()) > 0
        and bool(development)
    )
    return {
        "training_ready": ready,
        "train": count,
        "target_train": target,
        "target_development": summarize(development),
    }


def construct(funds, older, index, progress=lambda *args: None, *, read_only=False):
    """生成与独立复验复用同一确定性计算；复验重新读取原文并重算所有输入与答案。"""
    days, _ = own.sessions()
    recent = QuoteDays(allow_fetch=not read_only)

    class Quotes:
        def get(self, day, default=None):
            return older["days"].get(day, default) if day < "2021-01-01" else recent.get(day, default)

    quotes = Quotes()
    candidates, excluded = [], []
    for code, value in funds.items():
        rows, rejected = own.nav_rows(value["nav"], days, fund_code=code, family=value["family"])
        candidates.extend(r for r in rows if r["target"] <= PLAN["fit_end"] or code == FUND)
        excluded.extend(
            {"fund_code": code, **r}
            for r in rejected
            if r["target"] >= (PLAN["own_nav_start"] if code == FUND else PLAN["peer_nav_start"])
        )
    candidates.sort(key=lambda r: (r["target"], r["fund_code"]))
    train, development = [], []
    for i, row in enumerate(candidates):
        if i % 150 == 0:
            progress(i, len(candidates), FUND, "逐日复核合格训练资料")
        value = funds[row["fund_code"]]
        at = datetime.fromisoformat(row["as_of"])
        try:
            selected = select_report(value["reports"], at)
            newest = latest_listing(value["catalog"], row["as_of"])
            if newest and (
                (newest["report_end"], newest["available_at"]) > (selected["report_end"], selected["available_at"])
            ):
                raise ValueError("TRAINING_LATEST_DISCLOSURE_MISSING")
            # 同一报告期年报/中报比季报更全，缺失时不能默默用前十持仓替代。
            if (
                newest
                and newest["report_end"] == selected["report_end"]
                and newest["report_type"] != selected["report_type"]
            ):
                raise ValueError("TRAINING_LATEST_DISCLOSURE_MISSING")
            exposure = calculate(
                {"reports": value["reports"], "days": quotes, "indices": index},
                datetime.fromisoformat(row["base"]).date(),
                at,
                research_sessions=days,
            )
            if (
                exposure["holdings_features"] is None
                or exposure["market_features"] is None
                or exposure["missing"]
                or exposure["market_errors"]
            ):
                raise ValueError("TRAINING_INPUTS_INCOMPLETE")
            vector = row["nav_features"] + exposure["holdings_features"] + exposure["market_features"]
            if len(vector) != 20 or any(not math.isfinite(v) for v in vector):
                raise ValueError("TRAINING_FEATURE_VALUE_OR_SHAPE")
            if abs(exposure["quote_coverage_of_disclosed_weight"] - 1) > 1e-9:
                raise ValueError("TRAINING_QUOTE_COVERAGE")
            complete = {**{k: v for k, v in row.items() if k != "y"}, "x": vector, "exposure": exposure}
            if row["target"] <= PLAN["fit_end"]:
                if row["mature_at"] > PLAN["fit_as_of"]:
                    raise ValueError("TRAINING_IMMATURE_LABEL")
                train.append(complete)
            elif row["fund_code"] == FUND and PLAN["development"][0] <= row["target"] <= PLAN["development"][1]:
                development.append(complete)
            else:
                raise ValueError("TRAINING_TIME_SCOPE")
        except ValueError as exc:
            if str(exc) not in {
                "EXPOSURE_NO_AVAILABLE_REPORT",
                "TRAINING_LATEST_DISCLOSURE_MISSING",
                "TRAINING_INPUTS_INCOMPLETE",
                "TRAINING_IMMATURE_LABEL",
            }:
                raise
            excluded.append(
                {"fund_code": row["fund_code"], "base": row["base"], "target": row["target"], "reason": str(exc)}
            )
    return {"train": train, "development": development, "excluded": excluded}


def build(progress=lambda *args: None):
    initialize()
    funds, excluded_funds = sources()
    older, index = read(own.STORE / "older-quotes.json"), read(own.STORE / "indices.json")
    own.verify_derived_market(older, index)
    data = construct(funds, older, index, progress)
    snapshot = {"funds": funds, "older_quotes": older, "indices": index, "excluded_funds": excluded_funds}
    source_hash = digest(snapshot)
    source_path = STORE / "sources" / (source_hash + ".json")
    if not source_path.exists():
        save(source_path, snapshot)
    result = {
        **data,
        "plan_hash": digest(PLAN),
        "calendar_hash": own.sessions()[1],
        "source_file": f"sources/{source_hash}.json",
        "source_hash": source_hash,
        "summary": eligibility(data["train"], data["development"]),
        "cohort": {c: v["cohort"] for c, v in funds.items()},
        "excluded_funds": excluded_funds,
        "excluded_reasons": dict(Counter(r["reason"] for r in data["excluded"])),
        "weights": weights(data["train"]).tolist(),
        "training_runs": 0,
    }
    key = digest(result)
    path = STORE / "datasets" / (key + ".json")
    if not path.exists():
        save(path, result)
    # 只是待验收指针；ready 清单必须在下方独立重算完成后发布。
    versioned_save(STORE / "candidate.json", {"file": f"datasets/{key}.json", "sha256": key})
    return verify(progress)


def verify(progress=lambda *args: None, *, pointer="candidate.json", publish=True):
    """训练前可再次调用；只读重算不能发网络请求，失效或篡改立即拒绝旧的 ready 标记。"""
    if read(STORE / "plan.json") != PLAN:
        raise ValueError("TRAINING_PACKAGE_PLAN_CHANGED")
    current = read(STORE / pointer)
    path = (STORE / current["file"]).resolve()
    if path.parent != (STORE / "datasets").resolve():
        raise ValueError("TRAINING_DATASET_PATH")
    data = read(path)
    if digest(data) != current["sha256"] or path.stem != current["sha256"]:
        raise ValueError("TRAINING_DATASET_HASH_CHANGED")
    if data["plan_hash"] != digest(PLAN) or data["calendar_hash"] != own.sessions()[1]:
        raise ValueError("TRAINING_PROTOCOL_CHANGED")
    source_path = (STORE / data["source_file"]).resolve()
    if source_path.parent != (STORE / "sources").resolve():
        raise ValueError("TRAINING_SOURCE_PATH")
    snapshot = read(source_path)
    if digest(snapshot) != data["source_hash"] or source_path.stem != data["source_hash"]:
        raise ValueError("TRAINING_SOURCE_CHANGED")
    for code, value in snapshot["funds"].items():
        for report in value["reports"]:
            own.checked_raw(report["raw"])
        if code != FUND and cohort_evidence(code, value["reports"]) != data["cohort"][code]:
            raise ValueError("TRAINING_COHORT_CHANGED")
    own.verify_derived_market(snapshot["older_quotes"], snapshot["indices"])
    replay = construct(snapshot["funds"], snapshot["older_quotes"], snapshot["indices"], progress, read_only=True)
    if any(replay[key] != data[key] for key in ("train", "development", "excluded")):
        raise ValueError("TRAINING_INPUT_REPLAY_FAILED")
    if weights(data["train"]).tolist() != data["weights"]:
        raise ValueError("TRAINING_WEIGHTS_CHANGED")
    status = eligibility(data["train"], data["development"])
    if status != data["summary"]:
        raise ValueError("TRAINING_SUMMARY_CHANGED")
    result = {
        **status,
        "file": current["file"],
        "sha256": current["sha256"],
        "verified_at": now().isoformat(),
        "training_runs": 0,
        "models_registered": 0,
        "production_adoption_allowed": False,
        "status": "READY_FOR_OFFLINE_TRAINING" if status["training_ready"] else "MORE_QUALIFIED_DATA_REQUIRED",
        "limitations": [
            "EXPLORATORY_POOLED_TRANSFER",
            "HISTORICAL_AVAILABILITY_RECONSTRUCTED",
            "2024_ALREADY_OBSERVED_DEVELOPMENT_NOT_FINAL_TEST",
            "FINANCIAL_NEWS_EVENTS_NOT_FEATURES",
            "NO_PRODUCTION_ADOPTION",
        ],
    }
    if publish:
        versioned_save(STORE / "readiness.json", result)
        if status["training_ready"]:
            versioned_save(STORE / "ready.json", result)
    return result


def training_inputs(variant="NAV7_HOLDINGS_MARKET", progress=lambda *args: None):
    """交给后续三分类训练器的实际数组；重新验收、同日比较，不在这里执行 fit 或登记。"""
    if variant not in PLAN["variants"]:
        raise ValueError("TRAINING_VARIANT_INVALID")
    result = verify(progress, pointer="ready.json", publish=False)
    if not result["training_ready"]:
        raise ValueError("TRAINING_NOT_READY")
    data = read(STORE / result["file"])
    size = PLAN["variants"][variant]
    return {
        "fund_code": FUND,
        "horizon": 1,
        "variant": variant,
        "features": PLAN["features"][:size],
        "X_train": [r["x"][:size] for r in data["train"]],
        "y_train": [r["actual_direction"] for r in data["train"]],
        "sample_weight": data["weights"],
        "X_development": [r["x"][:size] for r in data["development"]],
        "y_development": [r["actual_direction"] for r in data["development"]],
        "dataset_hash": result["sha256"],
        "recipe": PLAN["recipe"],
        "training_runs": 0,
    }


def supplement(*, progress=lambda *args: None):
    """手动、一键和后台任务共用的补齐环节；上层持有同一个维护锁并跟踪至复验结束。"""
    from app.integrations.dbfund_reports import acquire

    progress(0, 1, FUND, "检查本基金早期报告，复用已保存的原文")
    reports = acquire(extended_history=True, progress=progress)
    nav = own.load_nav()
    signature = digest({"nav_rows": nav["rows"], "reports": own.load_reports(), "calendar": own.sessions()[1]})
    checkpoint = STORE / "own-input-checkpoint.json"
    old = read(checkpoint) if checkpoint.exists() else {}
    cache = own.STORE / "older-quotes.json"
    index = own.STORE / "indices.json"
    rebuild = (
        old.get("signature") != signature
        or not cache.exists()
        or not index.exists()
        or bool(old.get("result", {}).get("errors"))
    )
    if rebuild:
        progress(0, 1, FUND, "补齐早期股票行情和指数，已有成功文件继续复用")
        own_result = own.build(progress)
        versioned_save(checkpoint, {"signature": signature, "result": own_result})
    else:
        own_result = old["result"]
        own.verify_derived_market(read(cache), read(index))
    result = build(progress)
    errors = [{"scope": "OWN_HISTORY_REPORT", **e} for e in reports["errors"]]
    errors.extend({"scope": "OWN_HISTORY_QUOTES", **e} for e in own_result.get("errors", []))
    if not result["training_ready"]:
        errors.append({"reason": "MORE_QUALIFIED_TRAINING_DATA_REQUIRED"})
    return {
        "created": 0,
        "updated": 1 if result["training_ready"] else 0,
        "skipped": reports["reports"],
        "errors": errors,
        "readiness": result,
        "training_runs": 0,
        "new_requests": own_result.get("new_requests", 0) if rebuild else 0,
    }
