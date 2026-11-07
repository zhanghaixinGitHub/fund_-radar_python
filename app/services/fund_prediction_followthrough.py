"""每日预测补齐清单与结束条件，不形成方向、不放宽原资料准入。

资料同步的终态与预测完成是两个条件。该模块将可继续处理的公告、新持仓及
正式保存步骤列成清单，并要求本人正式结果读回后才能收口，防止待核验被当成完成。
"""

import hashlib
from datetime import datetime
from pathlib import Path

from app.services.direction_1d_protocol import ZONE, digest

HORIZONS = ("T1", "T5_V1", "T20_V1", "M6_V1")
EXTERNAL_REASONS = {
    "SOURCE_UNAVAILABLE",
    "AUTH_REQUIRED",
    "IDENTITY_CONFLICT",
    "NO_DIRECTION_ADVANTAGE",
    "VERIFIED_NON_TRADING_TARGET",
}


def recovery_actions(code, report, material, news, handoff, base_date, coverage_end=None, *, target_reports=None):
    """比对本基金披露与已整理资料，返回具体下一步，不自动批准原文或调用预测。"""
    actions = []
    if not report:
        actions.append({"action": "COLLECT_OWN_REPORT"})
    companies = {c["code"]: c for c in (material or {}).get("companies", [])}

    def quote_ready(company):
        """休市参考必须再次核对完整原文证据，不能由日期差自行视为已补齐。"""
        from app.services.fund_valuation_sessions import verified_jp_holiday_reference

        quote = company.get("quote") or {}
        if quote.get("date") == base_date:
            return True
        data = material or {}
        evidence = data.get("calendar_evidence", {})
        return bool(
            verified_jp_holiday_reference(
                data,
                {d["id"]: d for d in data.get("documents", [])},
                company["code"],
                quote,
                {"base_nav_date": base_date, "target_nav_date": evidence.get("target_date")},
            )
        )

    new_codes, incomplete = [], []
    for holding in (report or {}).get("holdings", []):
        stock = holding["stockCode"]
        company = companies.get(stock)
        if not company:
            new_codes.append(stock)
        elif not company.get("financials") or not quote_ready(company):
            incomplete.append(stock)
    if new_codes or incomplete:
        actions.append(
            {
                "action": "SUPPLEMENT_HOLDING_MATERIALS",
                "new_securities": sorted(set(new_codes)),
                "incomplete_securities": sorted(set(incomplete)),
                "required": [
                    "VERIFY_SECURITY_IDENTITY",
                    "BASE_DATE_QUOTE",
                    "APPLICABLE_FINANCIAL_AND_BUSINESS_ORIGINALS",
                ],
            }
        )
    for error in news.get("errors", []):
        if error.get("article_id"):
            actions.append({"action": "VERIFY_NOTICE_IDENTITY_OR_SOURCE", "article_id": error["article_id"]})
    if (
        news.get("status") != "SUCCEEDED"
        or not news.get("catalog_complete")
        or coverage_end
        and news.get("coverage", {}).get("end", "") < coverage_end
    ):
        actions.append({"action": "RECOVER_NOTICE_COVERAGE"})
    if not handoff or not handoff.get("reviewed") or handoff.get("gaps"):
        actions.append({"action": "REVIEW_COMPREHENSIVE_MATERIALS", "gaps": (handoff or {}).get("gaps", [])})
    if report and material:
        reviewed = material.get("source_manifest", {}).get("latest_partial_report", {}).get("report", {})
        if report["endDate"] > material.get("report", {}).get("endDate", "") and (
            reviewed.get("endDate") != report["endDate"] or reviewed.get("id") != report.get("id")
        ):
            actions.append(
                {"action": "REVIEW_NEW_REPORT", "report_id": report.get("id"), "report_end": report["endDate"]}
            )
    if material and target_reports is not None:
        reviewed_targets = material.get("source_manifest", {}).get("target_partial_reports", {})
        for target in material.get("exposure_chain", [code])[1:]:
            target_report = target_reports.get(target)
            if not target_report:
                actions.append({"action": "COLLECT_TARGET_REPORT", "target_fund_code": target})
                continue
            reviewed = reviewed_targets.get(target, {}).get("report", {})
            if reviewed.get("id") != target_report.get("id"):
                actions.append(
                    {
                        "action": "REVIEW_NEW_LOOKTHROUGH_REPORT",
                        "target_fund_code": target,
                        "report_id": target_report.get("id"),
                        "report_end": target_report["endDate"],
                    }
                )
            missing = [h["stockCode"] for h in target_report.get("holdings", []) if h["stockCode"] not in companies]
            incomplete = [
                h["stockCode"]
                for h in target_report.get("holdings", [])
                if h["stockCode"] in companies
                and (not companies[h["stockCode"]].get("financials") or not quote_ready(companies[h["stockCode"]]))
            ]
            if missing or incomplete:
                actions.append(
                    {
                        "action": "SUPPLEMENT_HOLDING_MATERIALS",
                        "target_fund_code": target,
                        "new_securities": sorted(set(missing)),
                        "incomplete_securities": sorted(set(incomplete)),
                        "required": [
                            "VERIFY_SECURITY_IDENTITY",
                            "BASE_DATE_QUOTE",
                            "APPLICABLE_FINANCIAL_AND_BUSINESS_ORIGINALS",
                        ],
                    }
                )
    return [{"fund_code": code, **a} for a in actions]


def verify_outputs(
    code, window, dates, records, actions, exceptions=(), *, materials_hash=None, materials_input_hash=None
):
    """逐周期核对真实本人记录；待核验无论写成何种状态都不会被当作成功。

    records必须由调用方从正式数据库读取并核对摘要及归档状态，不能来自调用方
    自填的success布尔值。外部阻断是单独终态，仍不会计入已生成数量。
    """
    rows = []
    for horizon in HORIZONS:
        expected = dates.get(horizon, {})
        # 资料身份、公告、新持仓等共用行动继续阻断全部周期；仅明确绑定其他
        # 周期的日历行动不会把本周期已正式保存的结果误报成未完成。
        period_actions = [a for a in actions if not a.get("horizon") or a["horizon"] == horizon]
        matches = [
            r
            for r in records
            if r.get("horizon") == horizon
            and r.get("fund_code") == code
            and r.get("base_date") == window["base_nav_date"]
            and r.get("target_date") == window["target_nav_date"]
            and r.get("owner_verified") is True
            and (
                (
                    materials_input_hash is not None
                    and r.get("snapshot_verified") is True
                    and r.get("materials_input_hash") == materials_input_hash
                )
                or (
                    materials_input_hash is None
                    and (
                        materials_hash is None
                        or (r.get("snapshot_verified") is True and r.get("materials_hash") == materials_hash)
                    )
                )
            )
        ]
        if matches and not period_actions and (horizon == "T1" or expected.get("endDateStatus") == "RESOLVED"):
            matches = [r for r in matches if r.get("end_date") == expected.get("endDate", window["target_nav_date"])]
            if matches:
                rows.append(
                    {"horizon": horizon, "status": "SAVED_AND_READ_BACK", "record_ids": [r["id"] for r in matches]}
                )
                continue
        if expected.get("endDateStatus") == "PENDING_OFFICIAL_CALENDAR":
            rows.append({"horizon": horizon, "status": "EXTERNAL_BLOCKED", "reason": "PENDING_OFFICIAL_CALENDAR"})
            continue
        backend = expected.get("backendFailure", {})
        details = backend.get("details", {})
        proof = expected.get("coverageProof") or {}
        insufficient_sessions = (
            proof.get("baseDate") == window["base_nav_date"]
            and proof.get("verifiedCoverageEnd") == details.get("coverageEnd")
            and type(proof.get("availableSessionsAfterBase")) is int
            and type(proof.get("requiredSessions")) is int
            and 0 <= proof["availableSessionsAfterBase"] < proof["requiredSessions"]
        )
        outside = (
            expected.get("endDateStatus") == "OUTSIDE_VERIFIED_CALENDAR"
            and backend.get("code") == "CALENDAR_RANGE_MISSING"
            and details.get("calendarId")
            and details.get("coverageEnd")
            and (details.get("requestedDate", "") > details["coverageEnd"] or insufficient_sessions)
        )
        unsupported = (
            expected.get("endDateStatus") == "UNSUPPORTED_PRODUCT_CALENDAR"
            and backend.get("code") == "CALENDAR_POLICY_MISSING"
            and details.get("fundCode") == code
            and backend.get("summary") == "该基金不在本次境内股票、混合和债券基金的已支持范围"
        )
        if (
            not period_actions
            and expected.get("fundCode") == code
            and len(expected.get("calendarSourceHash", "")) == 64
            and (outside or unsupported)
        ):
            rows.append({"horizon": horizon, "status": "EXTERNAL_BLOCKED", "reason": backend["code"]})
            continue
        exception = next((e for e in exceptions if e.get("fund_code") == code and e.get("horizon") == horizon), None)
        if exception and _verified_exception(exception, window, period_actions):
            rows.append({"horizon": horizon, "status": "EXTERNAL_BLOCKED", "reason": exception["reason"]})
        else:
            rows.append({"horizon": horizon, "status": "ACTION_REQUIRED"})
    pending = any(r["status"] == "ACTION_REQUIRED" for r in rows)
    return {
        "fund_code": code,
        "status": "ACTION_REQUIRED" if pending else "CLOSED",
        "actions": actions,
        "horizons": rows,
    }


def _verified_exception(exception, window, actions):
    """外部失败须绑定本期和真实尝试证据；一个“待核验”标签不能终结任务。"""
    if (
        exception.get("target_date") != window["target_nav_date"]
        or exception.get("reason") not in EXTERNAL_REASONS
        or len(exception.get("explanation", "").strip()) < 80
        or not exception.get("attempts")
    ):
        return False
    # 对可处理的原件身份、新报告审阅不能用“尚无方向优势”跳过资料准备。
    if actions and exception["reason"] == "NO_DIRECTION_ADVANTAGE":
        return False
    if exception["reason"] == "VERIFIED_NON_TRADING_TARGET" and any(
        a.get("action") not in {"REVIEW_COMPREHENSIVE_MATERIALS", "RECHECK_CURRENT_INPUT_AND_CALENDAR"} for a in actions
    ):
        # 公告身份、新证券、原件解析等内部待办不能被休市证据一并关闭。
        return False
    from app.services.fund_exposure_common import read

    allowed_outcomes = {
        "SOURCE_UNAVAILABLE": {"NETWORK_ERROR", "HTTP_ERROR", "EMPTY_PUBLIC_SOURCE"},
        "AUTH_REQUIRED": {"HTTP_UNAUTHORIZED", "HTTP_FORBIDDEN", "SOURCE_PERMISSION_DENIED"},
        "IDENTITY_CONFLICT": {"SOURCE_IDENTITY_CONFLICT"},
        "NO_DIRECTION_ADVANTAGE": {"REVIEW_COMPLETED_WITHOUT_DIRECTION"},
        "VERIFIED_NON_TRADING_TARGET": {"OFFICIAL_CALENDAR_EXCLUDES_TARGET"},
    }
    for attempt in exception["attempts"]:
        try:
            proof = read(Path(attempt["receipt_path"]))
            at = datetime.fromisoformat(attempt["attempted_at"])
            if (
                digest(proof) != attempt["receipt_hash"]
                or at.tzinfo is None
                or at.astimezone(ZONE).date() != datetime.now(ZONE).date()
                or at > datetime.now(ZONE)
                or proof.get("fund_code") != exception["fund_code"]
                or proof.get("target_date") != window["target_nav_date"]
                or proof.get("reason") != exception["reason"]
                or proof.get("attempted_at") != attempt["attempted_at"]
                or proof.get("outcome") not in allowed_outcomes[exception["reason"]]
                or not proof.get("source_evidence")
            ):
                return False
            for source in proof["source_evidence"]:
                if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
                    return False
            if exception["reason"] == "VERIFIED_NON_TRADING_TARGET":
                from app.services.fund_prediction_calendar_block import verified_non_trading_target

                if not verified_non_trading_target(proof, window):
                    return False
        except (OSError, ValueError, KeyError, TypeError):
            return False
    return True
