"""日常资料续办的通用实现；复用本基金原件，逐项重试，不包含日期或基金白名单。"""

import json
import re
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

from app.core.config import get_settings
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import now, read
from app.services.fund_materials import snapshot
from app.services.fund_materials_store import versioned_save
from app.services.fund_workflow_materials import receipt_path, workflow_root
from app.services.material_content import content_digest, source_bytes, source_exists


def normalized_materials(material):
    """只转换不可入库的 NUL；原PDF不动，保留字节摘要及转换前文本摘要。

    替代字符保持字符位置；未经真实原件哈希核验不能给文本增加转换声明。
    引用和事实同步转换，随后仍须原契约及聊天审阅，不能直接批准。
    """

    def convert(value):
        if isinstance(value, str):
            return value.replace("\x00", "\ufffd")
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    result = convert(deepcopy(material))
    for before, after in zip(material["documents"], result["documents"], strict=True):
        count = before["body"].count("\x00")
        if not count:
            continue
        raw = Path(before["raw_path"])
        if (
            content_digest(source_bytes(raw)) != before["raw_sha256"]
            or digest(before["body"]) != before["body_hash"]
        ):
            raise ValueError("TEXT_NORMALIZATION_ORIGINAL_HASH_MISMATCH")
        after.update(
            body_hash=digest(after["body"]),
            body_scope="VERIFIED_SOURCE_TRANSFORM",
            text_normalization={
                "operation": "NUL_TO_REPLACEMENT_CHARACTER",
                "count": count,
                "original_body_hash": before["body_hash"],
                "semantic_characters_not_removed": True,
            },
        )
        after.pop("source_hash", None)
    return result


def replay_originals(fund, output):
    """已有正文但解析未完成时只重放原件；与定期维护共用锁，不强制重查目录。"""
    from app.integrations.market_fund_reports import MarketReportClient, read_source
    from app.services.fund_exposure_runtime import execution_lock
    from app.services.fund_holdings_publication import publish_report
    from app.services.market_holdings_parse import parse_report

    cfg = get_settings()
    directory = Path(cfg.fund_material_directory)
    root = directory / "market-maintenance"
    state_path = root / "funds" / (fund["fund_code"] + ".json")
    if not state_path.exists():
        return []
    state, results = read(state_path), []
    before_reports = {r["id"]: r for r in (snapshot(fund["fund_code"]) or {}).get("reports", [])}
    before = set(before_reports)
    client = MarketReportClient(root)
    try:
        with execution_lock() as locked:
            if not locked:
                raise ValueError("SHARED_HOLDINGS_LOCK_BUSY")
            for article, item in state.get("entries", {}).items():
                prior_report = before_reports.get(item.get("report_id"), {})
                legacy_quarter = (
                    item.get("status") == "PUBLISHED"
                    and item.get("entry", {}).get("kind") == "QUARTER"
                    and prior_report.get("holdingsPublicationVersion") == 1
                    and (
                        prior_report.get("reportParserVersion", 0) < 2
                        or any(
                            re.fullmatch(r"\d{5,6}|[A-Z]{2,6}", h.get("stockCode", ""))
                            for h in prior_report.get("holdings", [])
                        )
                    )
                    and prior_report.get("endDate") == max(
                        (r.get("endDate", "") for r in before_reports.values()), default=""
                    )
                )
                if (
                    item.get("status") not in {"PARTIAL", "FAILED", "DOWNLOADED"}
                    and not legacy_quarter
                ) or not item.get("source"):
                    continue
                first = deepcopy(item)
                try:
                    read_source(root, item["source"])
                    report, issues = parse_report(client, root, fund, item["entry"], item["source"])
                    published = publish_report(fund["fund_code"], report, directory=directory)
                    current_report = next(r for r in published["reports"] if r["id"] == report["id"])
                    item.update(status="PARTIAL" if issues else "PUBLISHED", issues=issues, report_id=report["id"])
                    results.append(
                        {
                            "article_id": article,
                            "status": item["status"],
                            "issues": issues,
                            "changed": before_reports.get(report["id"]) != current_report,
                            "original": item["source"],
                            "first_attempt": first,
                        }
                    )
                except Exception as error:
                    results.append(
                        {
                            "article_id": article,
                            "status": "ACTION_REQUIRED",
                            "first_attempt": first,
                            "failure": type(error).__name__ + ":" + str(error)[:400],
                        }
                    )
            versioned_save(state_path, state)
    finally:
        client.close()
    after = {r["id"] for r in (snapshot(fund["fund_code"]) or {}).get("reports", [])}
    if not before <= after:
        raise ValueError("HISTORICAL_REPORT_LOST")
    versioned_save(output / "report-replay.json", results)
    return results


def public_financial_originals(code, output, window):
    """正常结构化财务缺失时补查巨潮公开原件；原件取得后继续交聊天解析和用途审阅。"""
    from app.integrations.cninfo_exposure import LIVE, PublicClient, acquire_announcements
    from app.services.fund_exposure_common import ROOT
    from app.services.fund_rating_public_market import public_document

    cutoff = str(now().date())
    start = str(date.fromisoformat(window["base_nav_date"]) - timedelta(days=190))
    scope = {
        "windows": {code: [[start, cutoff]]},
        "cutoff": cutoff,
        "check_id": cutoff,
        "maximum_pages_per_window": 8,
        "maximum_new_requests": 17,
        "training_eligible": False,
    }
    result = acquire_announcements(live_scope=scope)
    catalog = read(LIVE / "supplement/company-announcements" / (code + ".json"))
    if catalog.get("scope_hash") != digest(scope):
        raise ValueError("PUBLIC_FINANCIAL_CATALOG_SCOPE_CHANGED")
    selected = [
        entry
        for entry in catalog.get("rows", [])
        if re.search(r"(?:季度报告|半年度报告|年度报告|上市公告书|招股说明书)", entry.get("title_plain", ""))
        and not re.search("摘要|英文|说明会|提示|制度|取消|回复", entry.get("title_plain", ""))
    ]
    selected.sort(key=lambda e: (e["announced_at_source"], e["announcementId"]), reverse=True)
    client = PublicClient(cache_period=cutoff, request_limit=4)
    originals = []
    try:
        for entry in selected[:2]:
            try:
                url = "https://static.cninfo.com.cn/" + entry["adjunctUrl"]
                raw, receipt = client.fetch(url)
                if not raw.startswith(b"%PDF"):
                    raise ValueError("PUBLIC_FINANCIAL_NOT_PDF")
                body = "\n".join(public_document(raw, receipt, code[:6], entry["title_plain"])[0])
                if code[:6] not in body[:10000] or len(body) < 200:
                    raise ValueError("PUBLIC_FINANCIAL_SECURITY_IDENTITY")
                value = {
                    "security_code": code,
                    "status": "ORIGINAL_COLLECTED_PARSE_REQUIRED",
                    "body": body,
                    "entry": entry,
                    "url": url,
                    "raw_path": str((ROOT / receipt["file"]).resolve()),
                    "raw_sha256": receipt["sha256"],
                    "received_at": receipt["received_at"],
                }
            except Exception as error:
                value = {
                    "security_code": code,
                    "entry": entry,
                    "status": "ACTION_REQUIRED",
                    "failure": type(error).__name__ + ":" + str(error)[:300],
                }
            versioned_save(output / "financial-originals" / (entry["announcementId"] + ".json"), value)
            originals.append(value)
    finally:
        client.close()
    versioned_save(output / "financial-catalog.json", {"collection": result, "catalog": catalog})
    return originals


def recover(code, output, window, *, collector=None):
    """先重试正常来源，再整理候选包；读过的新原文才由独立审阅入口批准。"""
    from app.services.fund_market_news import refresh_material_news_link
    from app.services.fund_market_preparation import MarketFundPreparation, market_scope
    from app.services.market_company_materials import MarketCompanyCollector

    fund = next((f for f in market_scope() if f["fund_code"] == code), None)
    if not fund:
        raise ValueError("WATCHED_FUND_NOT_IN_ACTIVE_MARKET")
    replay = replay_originals(fund, output)
    root = workflow_root()
    news_path = root / "news/funds" / (code + ".json")
    news = read(news_path) if news_path.exists() else {}
    if news.get("status") != "SUCCEEDED" or news.get("coverage", {}).get("end", "") < str(now().date()):
        prep = MarketFundPreparation()
        try:
            news_receipt = prep.sync_news(code)
        finally:
            prep.close()
        versioned_save(output / "notice-recovery.json", news_receipt)
        if news_receipt["status"] == "SUCCEEDED":
            refresh_material_news_link(code)
    if any(row.get("changed") for row in replay):
        # 原件重放有实际变化才更新标准资料指纹；禁止旧下游快照冒充本次新持仓。
        prep = MarketFundPreparation()
        try:
            current_news = read(news_path) if news_path.exists() else {}
            canonical_receipt = prep.sync_material(fund, current_news)
            versioned_save(output / "canonical-refresh.json", canonical_receipt)
        finally:
            prep.close()
    handoff = receipt_path(code)
    prior = read(handoff) if handoff.exists() else {}
    path = Path(prior["materials_path"]) if prior.get("materials_path") else None
    if not path or not source_exists(path):
        # 缺通用包不等于缺原件；给出已收录报告和证券原文的具体整理入口。
        value = {
            "status": "ASSEMBLE_OWN_MATERIALS",
            "reports": (snapshot(code) or {}).get("reports", []),
            "originals_root": str(Path(get_settings().fund_material_directory)),
            "replay": replay,
        }
        versioned_save(output / "supplement.json", value)
        return value
    material = read(path)
    if material.get("fund_code") != code:
        raise ValueError("RECOVERY_CROSS_FUND_MATERIALS")
    from app.services.fund_prediction_lookthrough import recover_target_reports

    target_reports = recover_target_reports(material, output)
    candidate = normalized_materials(material)
    from app.services.fund_prediction_market_recovery import refresh as refresh_market

    market_receipts = refresh_market(candidate, window["base_nav_date"])
    from app.services.fund_prediction_bond_recovery import refresh as refresh_bonds

    bond_receipts = refresh_bonds(candidate, window["base_nav_date"], output.parents[1] / "public")
    from app.services.fund_prediction_news_recovery import refresh as refresh_news

    notice_receipt = refresh_news(candidate, root=root, checked_at=now())
    if candidate.get("calendar_evidence"):
        # 重新绑定本次窗口；正式校验仍逐项核对本基金条款、官方日历与实际行情。
        candidate["calendar_evidence"].update(
            base_date=window["base_nav_date"], target_date=window["target_nav_date"]
        )
    reports = [r for r in (snapshot(code) or {}).get("reports", []) if r["publishedDate"] <= str(now().date())]
    latest = max(reports, key=lambda r: (r["endDate"], r["publishedDate"])) if reports else {}
    from app.services.fund_prediction_report_recovery import bind_latest_partial

    bind_latest_partial(candidate, latest)
    companies = {c["code"]: c for c in candidate.get("companies", [])}
    required = {h["stockCode"] for h in latest.get("holdings", [])} | set(companies)
    required |= {h["stockCode"] for r in target_reports for h in r.get("report", {}).get("holdings", [])}
    receipts = []
    own_collector = collector is None
    collector = collector or MarketCompanyCollector(root)
    try:
        for security in sorted(required):
            existing = companies.get(security, {})
            if security.endswith(".JP"):
                # 当期报价齐备不代表财报仍是最新。先核真实原件目录，发现新半年报
                # 就保留全文并追加未批准候选，不能只复用上期季度包或自动放行方向。
                from app.services.fund_prediction_jp_financials import collect as collect_jp

                refs = {v.get("sourceRef") for v in existing.get("financials", []) if isinstance(v, dict)}
                documents = [d for d in candidate["documents"] if d["id"] in refs or d.get("issuer_code") == security]
                result = collect_jp(security, output / "companies" / security / "jp-financials", documents=documents)
                receipts.append({"code": security, "japanese_financials": result})
                if result["status"] == "ORIGINAL_PARSED_READING_REQUIRED":
                    ref = "jp-financial:" + security + ":" + result["raw_sha256"]
                    document = {
                        "id": ref, "kind": "COMPANY_FINANCIAL", "title": result["title"], "url": result["url"],
                        "body": result["body"], "body_hash": digest(result["body"]), "body_scope": "FULL_TEXT",
                        "codes": [code, security], "issuer_code": security, "published_date": result["published_date"],
                        "available_at": result["received_at"], "raw_path": result["raw_path"],
                        "raw_sha256": result["raw_sha256"],
                    }
                    if not any(d["id"] == ref for d in candidate["documents"]):
                        candidate["documents"].append(document)
                    if security in companies and not any(
                        v.get("sourceRef") == ref for v in existing.get("financials", []) if isinstance(v, dict)
                    ):
                        companies[security].setdefault("financials", []).append({
                            "sourceRef": ref, "endDate": result["report_end"],
                            "publishedDate": result["published_date"], "scope": "ORIGINAL_PARSED_READING_REQUIRED",
                        })
                continue
            if existing.get("financials") and (existing.get("quote") or {}).get("date") == window["base_nav_date"]:
                continue
            # 本基金原报告代码决定市场；不把港股代码猜成A股，也不借用别的基金权重。
            value = (
                collector.company(security)
                if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", security)
                else {"code": security, "gaps": ["需按本证券真实市场补充行情及经营原文"], "sources": []}
            )
            quote = value.get("quote") or {}
            if security in companies and quote.get("date") == window["base_nav_date"]:
                source = next((r for r in value["sources"] if r.get("api") == "daily"), None)
                if source and quote.get("pre_close") is not None:
                    ref = "daily-quote:" + security + ":" + window["base_nav_date"]
                    body = json.dumps(quote, ensure_ascii=False)
                    document = {
                        "id": ref,
                        "kind": "MARKET_DATA",
                        "title": "本证券实际基准日行情",
                        "url": get_settings().tushare_api_url,
                        "body": body,
                        "body_hash": digest(body),
                        "body_scope": "VERIFIED_SOURCE_TRANSFORM",
                        "published_date": window["base_nav_date"],
                        "available_at": source["received_at"],
                        "codes": [code],
                        "raw_path": str((root / source["file"]).resolve()),
                        "raw_sha256": source["sha256"],
                    }
                    candidate["documents"] = [d for d in candidate["documents"] if d["id"] != ref] + [document]
                    companies[security]["quote"] = {
                        "date": quote["date"],
                        "close": quote["close"],
                        "previous_close": quote["pre_close"],
                        "change_pct": quote["pct_chg"],
                        "source_ref": ref,
                    }
            if re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", security) and not existing.get("financials"):
                try:
                    value["public_originals"] = public_financial_originals(
                        security, output / "companies" / security, window
                    )
                except Exception as error:
                    value["public_fallback_failure"] = type(error).__name__ + ":" + str(error)[:300]
            receipts.append(value)
    finally:
        if own_collector:
            collector.close()
    candidate = normalized_materials(candidate)
    from app.services.fund_prediction_hk_recovery import refresh as refresh_hk

    hk_receipts = refresh_hk(candidate, window["base_nav_date"], output.parents[1] / "public")
    from app.services.fund_prediction_overseas_recovery import refresh as refresh_overseas

    overseas_receipts = refresh_overseas(candidate, window["base_nav_date"], output.parents[1] / "public")
    from app.services.fund_prediction_corporate_recovery import refresh as refresh_corporate

    corporate_receipts = refresh_corporate(candidate, window)
    # 股票更新完成后重建当前事实及覆盖状态；此时已取得的对照报价不会重复请求。
    refresh_market(candidate, window["base_nav_date"])
    if candidate != material:
        path = output / "materials-normalized.json"
        versioned_save(path, candidate)
    # 原包或新候选包均不自动套用审阅；新证券和原文差异交给本聊天继续处理。
    value = {
        "status": "SUPPLEMENTED_REVIEW_REQUIRED",
        "materials_path": str(path.resolve()),
        "latest_report": latest,
        "securities": sorted(required),
        "company_attempts": receipts,
        "market_attempts": market_receipts,
        "bond_attempts": bond_receipts,
        "hk_attempts": hk_receipts,
        "overseas_attempts": overseas_receipts,
        "corporate_action_checks": corporate_receipts,
        "notice_attempt": notice_receipt,
        "report_replay": replay,
        "target_reports": target_reports,
        "canonical_path": str(root / "materials" / (code + ".json")),
    }
    versioned_save(output / "supplement.json", value)
    return value
