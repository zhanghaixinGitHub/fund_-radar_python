"""特殊类型月评新期输入装配；只消费真实新观察和实际审阅，不批准用途。"""

import json
from copy import deepcopy
from datetime import date
from pathlib import Path

from sqlalchemy import text

from app.services.fund_exposure_common import now
from app.services.fund_materials_store import versioned_save
from app.services.fund_rating_pipeline import archive_json
from app.services.fund_rating_typed_disclosure import calculate


def observed_provider_rows(code, start_date, end_date, *, session):
    """仅取授权来源实际存在的净值；外国估值日期随后须由发行人完整列表核准。"""
    start, end = date.fromisoformat(str(start_date)), date.fromisoformat(str(end_date))
    if start > end or (end - start).days > 4 * 366:
        raise ValueError("MONTHLY_TYPED_PROVIDER_WINDOW")
    source = (
        session.execute(
            text(
                "SELECT source_id,enabled,authorization_verified_at FROM source_registry "
                "WHERE source_code='TUSHARE_PRO_FUND'"
            )
        )
        .mappings()
        .one()
    )
    if not source["enabled"] or not source["authorization_verified_at"]:
        raise ValueError("MONTHLY_TYPED_PROVIDER_AUTHORIZATION")
    rows = []
    for year in range(start.year, end.year + 1):
        rows.extend(
            session.execute(
                text(
                    "SELECT nav_date,ann_date,unit_nav,accumulated_nav,adjusted_nav,content_hash "
                    "FROM nav_daily WHERE fund_code=:code AND source_id=:source "
                    "AND nav_date BETWEEN :start AND :end ORDER BY nav_date"
                ),
                {
                    "code": code,
                    "source": source["source_id"],
                    "start": max(start, date(year, 1, 1)),
                    "end": min(end, date(year, 12, 31)),
                },
            )
            .mappings()
            .all()
        )
    if not rows or len({r["nav_date"] for r in rows}) != len(rows):
        raise ValueError("MONTHLY_TYPED_PROVIDER_EMPTY_OR_DUPLICATE")
    return {
        "fund_code": code,
        "source": {
            "enabled": True,
            "source_code": "TUSHARE_PRO_FUND",
            "authorization_verified_at": str(source["authorization_verified_at"]),
        },
        "history": {"rows": [{k: str(v) if v is not None else None for k, v in row.items()} for row in rows]},
        "observed_at": now().isoformat(),
    }


def prepare_typed_monthly(
    code,
    previous_bundle,
    target_date,
    root,
    output,
    *,
    replacements,
    new_sources,
    period_review,
    event_audit=None,
    management_inputs=None,
):
    """路径替换均显式提供；原件、日历、基准及逐项分数仍由生产类型计算器重放。

    replacements 为路径元组到值的映射，例如 ("refs", "provider")。调用方必须
    实际审阅 period_review；本函数不从旧日期或无公告结果生成批准。
    """
    target = str(date.fromisoformat(str(target_date)))
    if (
        previous_bundle.get("fund_code") != code
        or not period_review
        or period_review.get("fund_code") != code
        or period_review.get("as_of_date") != target
        or len(period_review.get("reason", "").strip()) < 60
        or not period_review.get("evidence")
    ):
        raise ValueError("MONTHLY_TYPED_ACTUAL_PERIOD_REVIEW_REQUIRED")
    bundle = deepcopy(previous_bundle)
    raw = bundle["typed_disclosure_calculation"]
    inputs = raw["inputs"]
    required = (
        ("refs", "provider")
        if raw["profile"] in {"MIXED", "CONVERTIBLE", "BOND_INDEX"}
        else (
            ("nav_ref",)
            if raw["profile"] == "GOLD"
            else (("nav_chart_ref",) if inputs.get("adapter") == "TH_VIETNAM" else ("provider_ref",))
        )
    )
    if required not in replacements:
        raise ValueError("MONTHLY_TYPED_NEW_NAV_SOURCE_REQUIRED")
    for reference in new_sources:
        inputs["sources"][reference["sha256"]] = reference
    for path, value in replacements.items():
        if not path or any(part in {"score", "dimensions", "precomputed_scores"} for part in path):
            raise ValueError("MONTHLY_TYPED_INPUT_PATH_REQUIRED")
        node = inputs
        for part in path[:-1]:
            node = node[part]
        node[path[-1]] = deepcopy(value)

    def nested(value, path):
        for part in path:
            value = value[part]
        return value

    if nested(inputs, required) == nested(previous_bundle["typed_disclosure_calculation"]["inputs"], required):
        raise ValueError("MONTHLY_TYPED_UNCHANGED_NAV_SOURCE")
    if period_review.get("holding_date") != inputs["holding_date"]:
        raise ValueError("MONTHLY_TYPED_FULL_REPORT_DATE_REVIEW_REQUIRED")
    inputs["monthly_period_review"] = deepcopy(period_review)
    inputs["as_of_date"] = bundle["as_of_date"] = target
    inputs["verified_at"] = bundle["verified_at"] = now().isoformat()
    bundle["data_dates"]["nav"] = target
    if "window_start" in inputs:
        inputs["window_start"] = replacements.get(("window_start",), inputs["window_start"])
    if management_inputs is not None:
        inputs["management_inputs"] = deepcopy(management_inputs)
    if event_audit is not None:
        if (
            event_audit.get("fund_code") != code
            or event_audit.get("as_of_date") != target
            or not event_audit.get("complete")
            or not event_audit.get("reviews")
        ):
            raise ValueError("MONTHLY_TYPED_EVENT_AUDIT_REQUIRED")
        reference = {
            **archive_json(root, event_audit),
            "kind": "json",
            "archive_kind": "DERIVED",
            "label": "monthly-actual-event-audit-" + code,
        }
        inputs["sources"][reference["sha256"]] = reference
        inputs["event_review"] = {"complete": True, "source_hash": reference["sha256"]}
    raw["input_archive"] = archive_json(root, inputs)
    # 不把编译异常伪装为零分；候选和失败仍归档供后续接力。
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    candidate_path = folder / "candidate-bundle.json"
    candidate_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        result = calculate(bundle, root=Path(root))
        receipt = {
            "fund_code": code,
            "as_of_date": target,
            "checked_at": now().isoformat(),
            "reviewed": False,
            "candidate_path": str(candidate_path.resolve()),
            "calculation": result,
        }
    except (KeyError, ValueError, FileNotFoundError) as error:
        receipt = {
            "fund_code": code,
            "as_of_date": target,
            "checked_at": now().isoformat(),
            "reviewed": False,
            "candidate_path": str(candidate_path.resolve()),
            "failure": type(error).__name__ + ":" + str(error),
        }
    versioned_save(folder / "preparation.json", receipt)
    return receipt
