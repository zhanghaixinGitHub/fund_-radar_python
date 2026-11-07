"""从原件目录、逐项原句与研究适用判断编译重大事项范围。

本模块不把“搜索没找到”变为无事发生，也不把持仓发行人诉讼变为基金
自身失效。pending由已绑定的全部事项对象、状态、严重性共同导出。
"""

import importlib
import re
from datetime import datetime
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

_source = importlib.import_module((__package__ + "." if __package__ else "") + "fund_rating_thematic_source")
plain = _source.plain

FUND_PATTERN = r"诉讼|仲裁|处罚|调查|立案|稽查|违约|清算|终止|暂停|澄清|估值方法调整|风险警示|重大事件"


def need(ok, code):
    if not ok:
        raise ValueError("MATERIAL_REVIEW_" + code)


def compile_material_review(inputs, source):
    ref = inputs["material_event_review_ref"]
    need(source.sources[ref]["archive_kind"] == "DERIVED", "TYPED_RESEARCH_REQUIRED")
    r = source.json(ref)
    need(
        r["version"] == "018590_MATERIAL_APPLICABILITY_ORIGINALS_V1"
        and r["as_of_date"] == inputs["as_of_date"]
        and r["window_start"] == "2023-09-30",
        "TIME_CONTRACT",
    )
    catalogs = {}
    coverage = {}
    for code, refs in r["fund_catalogs"].items():
        need(code in inputs["share_binding"]["column_codes"], "SHARE_SCOPE")
        rows = []
        totals = set()
        for page, digest in enumerate(refs, 1):
            source.original(digest)
            params = (
                source.sources[digest].get("request_params_without_credentials")
                or source.sources[digest].get("params")
                or {}
            )
            need(
                params.get("type") in {0, "0"} and str(params.get("fundcode")) == code,
                "FUND_CATALOG_UNFILTERED_REQUEST",
            )
            d = source.json(digest)
            need(d.get("ErrCode") == 0 and d["PageIndex"] == page and d["PageSize"] == 100, "FUND_CATALOG_PAGE")
            totals.add(d["TotalCount"])
            part = d["Data"]
            need(part and all(x["FUNDCODE"] == code for x in part), "FUND_CATALOG_IDENTITY")
            rows += part
        need(
            len(totals) == 1
            and (min(x["PUBLISHDATEDesc"][:10] for x in rows) < r["window_start"] or len(rows) >= next(iter(totals))),
            "FUND_CATALOG_TRUNCATED",
        )
        need(len({x["ID"] for x in rows}) == len(rows), "FUND_CATALOG_DUPLICATED")
        filtered = [x for x in rows if r["window_start"] <= x["PUBLISHDATEDesc"][:10] <= r["as_of_date"]]
        for x in filtered:
            catalogs.setdefault(x["ID"], x)
        coverage[code] = {
            "pages": len(refs),
            "window_rows": len(filtered),
            "oldest_returned": min(x["PUBLISHDATEDesc"][:10] for x in rows),
            "source_refs": refs,
        }
    need(set(r["fund_catalogs"]) == set(inputs["share_binding"]["column_codes"]), "BOTH_SHARES_CLOSED")
    expected = {k for k, v in catalogs.items() if re.search(FUND_PATTERN, v["TITLE"])}
    need({x["id"] for x in r["fund_events"]} == expected, "FUND_EVENT_REVIEW_CLOSURE")
    all_events = []
    for e in r["fund_events"]:
        row = catalogs[e["id"]]
        need(e["title"] == row["TITLE"] and e["published_on"] == row["PUBLISHDATEDesc"][:10], "EVENT_CATALOG_BINDING")
        for claim in e["claims"]:
            source.claim(claim)
        text = plain("".join(source.body(h) for h in e["source_refs"]))
        if e["applicability"] == "OTHER_FUNDS_HOLIDAY":
            need(
                "体育文化" not in text and "香港联合交易所" in text and e["published_on"] in text,
                "HOLIDAY_PRODUCT_OR_DATE",
            )
        elif e["applicability"] == "THIRD_PARTY_IMPERSONATION":
            need("不法分子" in text, "SCAM_SUBJECT")
        elif e["status"] == "CHANNEL_TERMINATION":
            need("终止" in text and "销售" in text, "CHANNEL_SCOPE")
        elif e["status"] == "CHANNEL_SUSPENSION":
            need(
                "海银" in text and "赎回" in text and ("转托管" in text or "转出" in text),
                "DISTRIBUTOR_REDEMPTION_SCOPE",
            )
        elif e["applicability"] == "VALUATION_POLICY":
            need("估值" in text and ("托管" in text or "协商" in text), "VALUATION_DISCLOSURE")
        elif e["status"] == "DATED_MAINTENANCE_SCHEDULE":
            need(
                any(word in text for word in ["维护", "系统数据库迁移", "系统升级"])
                and "恢复" in text
                and "暂停" in text
                and re.search(r"20\d{2}年\d{1,2}月\d{1,2}日", text),
                "MAINTENANCE_WINDOW",
            )
        else:
            need(e["applicability"] == "FUND_CONFIRMED", "UNREVIEWED_APPLICABILITY")
        all_events.append(e)
    issuer_ids = {}
    issuer_coverage = {}
    for security, refs in r["issuer_catalogs"].items():
        rows = []
        last = None
        total = None
        for page, digest in enumerate(refs, 1):
            source.original(digest)
            params = (
                source.sources[digest].get("request_params_without_credentials")
                or source.sources[digest].get("params")
                or {}
            )
            need(
                params.get("seDate") == "2025-10-01~2026-09-30"
                and str(params.get("stock", "")).startswith(security + ",")
                and int(params.get("pageNum", 0)) == page,
                "ISSUER_FULL_YEAR_REQUEST",
            )
            d = source.json(digest)
            part = d.get("announcements")
            need(isinstance(part, list) and all(x["secCode"] == security for x in part), "ISSUER_CATALOG_IDENTITY")
            need(total in {None, d["totalAnnouncement"]}, "ISSUER_CATALOG_TOTAL")
            total = d["totalAnnouncement"]
            rows += part
            last = d
        need(
            last is not None and last.get("hasMore") is False and len({x["announcementId"] for x in rows}) == total,
            "ISSUER_CATALOG_TRUNCATED",
        )
        issuer_ids[security] = {x["announcementId"]: x for x in rows}
        issuer_coverage[security] = {
            "source_refs": refs,
            "count": total,
            "window_start": "2025-10-01",
            "window_end": "2026-09-30",
        }
    for e in r["issuer_events"]:
        row = issuer_ids[e["security"]].get(e["id"])
        need(row is not None, "ISSUER_EVENT_CATALOG_BINDING")
        day = datetime.fromtimestamp(row["announcementTime"] / 1000, ZoneInfo("Asia/Shanghai")).date().isoformat()
        need(day == e["published_on"] and day <= r["as_of_date"], "ISSUER_EVENT_DATE")
        need(e["applicability"] == "ISSUER_HOLDING" and e.get("assessment"), "ISSUER_IS_NOT_FUND")
        need(bool(e["claims"]), "ISSUER_NO_ORIGINAL_QUOTE")
        # 必须至少有该目录事件自己的附件；同公司相关说明可附加，不能
        # 用另一则公告的真实原文顶替本事件。巨潮id与原附件路径逐件相同。
        need(
            any(
                urlsplit(source.sources[h]["url"]).path.lower().endswith("/" + e["id"] + ".pdf")
                for h in e["source_refs"]
            ),
            "ISSUER_EVENT_ACTUAL_ATTACHMENT",
        )
        for claim in e["claims"]:
            need(claim["source_hash"] in e["source_refs"], "ISSUER_CLAIM_SOURCE_BINDING")
            source.claim(claim)
        need(set(e["source_refs"]) <= source.sources.keys(), "ISSUER_SOURCE_REF")
        all_events.append(e)
    # 已知的最终处罚及未决仲裁是强制复核点，不允许调用方删除后保留“闭合”。
    need(
        {"1224854546", "1225270915", "1225583096", "1225021111", "1224976935", "1224797176"}
        <= {e["id"] for e in r["issuer_events"]},
        "KNOWN_NEGATIVE_FACT_DROPPED",
    )
    need(
        next(e for e in r["issuer_events"] if e["id"] == "1225583096")["status"] == "UNRESOLVED",
        "KNOWN_ARBITRATION_UNRESOLVED",
    )
    for row in r["financial_reviews"]:
        need({d["period"] for d in row["documents"]} == {"FY2025", "H126"}, "FINANCIAL_PERIOD_COVERAGE")
        for d in row["documents"]:
            source.original(d["source_hash"])
            body = source.body(d["source_hash"])
            need(row["security"] in plain(body)[:20000], "FINANCIAL_ISSUER_IDENTITY")
            from importlib import import_module

            execution = import_module((__package__ + "." if __package__ else "") + "fund_rating_thematic_execution")
            need(
                execution.company_period(body) == {"FY2025": "2025-12-31", "H126": "2026-06-30"}[d["period"]],
                "FINANCIAL_ACTUAL_ECONOMIC_PERIOD",
            )
            need(d["claims"], "FINANCIAL_CLAIMS")
            for claim in d["claims"]:
                need(claim["source_hash"] == d["source_hash"], "FINANCIAL_CLAIM_SOURCE_BINDING")
                source.claim(claim)
    need(
        len(r["financial_reviews"]) == 22
        and set(r["issuer_catalogs"]) == {d["security"] for d in r["financial_reviews"]},
        "ALL_22_ISSUERS",
    )
    full = []
    need(r["fund_full_reports"] == inputs["execution"]["full_reports"], "TARGET_REPORT_BINDING")
    for period, digest in sorted(r["fund_full_reports"].items()):
        text = plain(source.body(digest))
        need(inputs["legal_name"] in text[:18000], "TARGET_IDENTITY")
        litigation = re.search(r"本报告期内[,，]?(?:无|没有)涉及(?:本)?基金管理人、基金财产、基金托管业务的诉讼", text)
        manager = re.search(r"本报告期内[,，]?基金管理人(?:及其相关高级管理人员)?无受(?:稽查|调查)或处罚等情况", text)
        need(litigation is not None and manager is not None, "TARGET_ACTUAL_NEGATIVE_DISCLOSURE:" + period)
        full.append(
            {"period": period, "source_hash": digest, "litigation_quote": litigation[0], "manager_quote": manager[0]}
        )
    e = r["known_manager_event"]
    need(
        e["event_on"] < r["window_start"] and e["status"] == "REMEDIATION_COMPLETION_UNPROVEN", "KNOWN_MANAGER_HISTORY"
    )
    for claim in e["claims"]:
        source.claim(claim)
    need(
        all(
            w in plain(e["claims"][0]["quote"])
            for w in ["银华基金管理股份有限公司", "2022年11月24日", "警示函", "内控管理", "销售业务管理"]
        ),
        "KNOWN_MANAGER_WARNING_ORIGINAL",
    )
    all_events.append(e)
    pending = any(
        e["applicability"] == "FUND_CONFIRMED"
        and e["status"] == "UNRESOLVED"
        and e["severity"] in {"MATERIAL", "SEVERE"}
        for e in all_events
    )
    return {
        "material_event_pending": pending,
        "events": all_events,
        "anchors": {"material_events": {inputs["fund_code"]: all_events}},
        "fund_catalog_coverage": coverage,
        "issuer_catalog_coverage": issuer_coverage,
        "fund_report_actual_disclosures": full,
        "source_bound_review_ref": ref,
        "known_context_outside_window": [e],
        "known_applicable_events": [x for x in all_events if x["applicability"] == "FUND_CONFIRMED"],
        "outside_window_policy": (
            "2022-11-24措施及2023-03-31原公开均在本窗口外；完整保留历史，不宣称整改，"
            "也不把未证明完成推成当前本基金重大未决。窗内新原件如证明持续影响需重新纳入。"
        ),
        "scope": (
            "基金三年公告目录及六期定期披露；22持仓发行人一年完整公告目录、2025年报和2026中报及已知历史事项；"
            "不声称监管全量历史核查。"
        ),
    }
