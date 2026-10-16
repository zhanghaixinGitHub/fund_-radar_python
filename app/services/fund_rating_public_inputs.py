"""公开资料评级的真实输入适配与本机使用边界，不制造机构内部持有人/收付款明细。"""

import hashlib
import io
import json
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

from pypdf import PdfReader

from app.core.config import get_settings
from app.services.fund_rating_disclosures import appointed_teams, compact, equity_balance, key_information
from app.services.fund_rating_metrics import (
    equity_structure,
    history_metrics,
    reconcile_equity_assets,
    standard_cost,
    total_return,
)
from app.services.fund_rating_public_policy import (
    BASIS,
    ETF_BASIS,
    GLOBAL_BASIS,
    INDEX_BASES,
    INDEX_BASIS,
    MARKET_BASES,
    MARKET_BASIS,
    RECONCILED_BASIS,
)
from app.services.fund_rating_rules import decimal


def report_facts(pages: list[str]) -> dict:
    """只解析已核对的报告明确陈述；没找到否定语句不能解释为零或无风险。"""
    text = compact("\n".join(pages))
    statements = {
        "new_issue": "本基金本报告期无因认购新发/增发证券而于期末持有的流通受限证券。",
        "suspended": "本基金本报告期末未持有暂时停牌等流通受限股票。",
        "lending": "截至本报告期末，本基金无参与转融通证券出借业务的证券。",
        "holder": "本报告期内无单一投资者持有基金份额比例达到或超过20%的情况",
        "strategy": "本报告期本基金投资策略未发生改变。",
    }
    if not all(v in text for v in statements.values()):
        raise ValueError("RATING_PUBLIC_REPORT_PATTERN_UNSUPPORTED")
    start = text.rfind("10.6.1管理人受调查或处罚等情况")
    end = text.find("10.6.2", start)
    if start < 0 or end < 0:
        raise ValueError("RATING_PUBLIC_GOVERNANCE_MISSING")
    section = text[start:end]
    if "行政监管措施" in section and "截至报告期末，公司已完成整改，所有业务均正常开展" in section:
        governance = 1
    elif "未受到调查或处罚" in section:
        governance = 2
    elif "行政监管措施" in section:
        governance = 0
    else:
        raise ValueError("RATING_PUBLIC_GOVERNANCE_UNRECOGNIZED")
    return {
        "restricted_stock_amount": "0",
        "holder_risk_band": 0,
        "strategy_continuity": 1,
        "governance_state": governance,
        "statements": statements,
        "governance_excerpt": section,
    }


def validate_public_sources(bundle: dict, frozen: datetime, *, root: Path | None = None) -> None:
    """本机公开事实分析的独立来源资格，不写入或冒充商业授权登记。

    限官方德邦原文及本机运行；原文、哈希、取得时间与来源域名全部校验。
    原商业/授权来源的规范化包仍走原门禁，部署到非local环境时本口径失效。
    """
    if get_settings().app_environment != "local" or bundle.get("source_scope") != "PUBLIC_DISCLOSURE_LOCAL_ANALYSIS":
        raise ValueError("RATING_PUBLIC_LOCAL_SCOPE_REQUIRED")
    etf = bundle.get("basis") == ETF_BASIS
    market = bundle.get("basis") in {*MARKET_BASES, *INDEX_BASES}
    if not (
        (market and re_market_code(bundle.get("fund_code")))
        or (etf and re_market_code(bundle.get("fund_code")))
        or (bundle.get("basis") == BASIS and bundle.get("fund_code") == "002112")
    ):
        raise ValueError("RATING_PUBLIC_CASE_UNSUPPORTED")
    # 同一资料包共用已经解析的根目录，避免每个原件都重复遍历 Windows 路径。
    # 原件路径仍逐件 resolve，并检查目录范围、大小和真实内容哈希。
    root = (root or Path(get_settings().fund_insight_directory) / "ratings").resolve()
    sources_directory = root / "sources"
    refs = bundle["raw_archive"]
    if not 1 <= len(refs) <= 200 or len({r["sha256"] for r in refs}) != len(refs):
        raise ValueError("RATING_PUBLIC_SOURCE_COUNT")
    expanded = [v for r in refs for v in (r, *r.get("aliases", []))]
    if len(expanded) > 200 or any(
        a["sha256"] != r["sha256"] or a["file"] != r["file"] for r in refs for a in r.get("aliases", [])
    ):
        raise ValueError("RATING_PUBLIC_SOURCE_ALIAS")
    from app.services.fund_rating_public_market import HOSTS

    allowed_hosts = HOSTS if market or etf else {"www.dbfund.com.cn", "dbfund.com.cn"}

    for ref in expanded:
        url = urlsplit(ref["url"])
        if url.scheme != "https" or url.hostname not in allowed_hosts or url.username or url.password:
            raise ValueError("RATING_PUBLIC_SOURCE_HOST")
        received = datetime.fromisoformat(ref["acquired_at"])
        if received.tzinfo is None or received > frozen:
            raise ValueError("RATING_PUBLIC_SOURCE_FUTURE")
        path = (root / ref["file"]).resolve()
        if not path.is_relative_to(sources_directory) or not path.is_file() or path.stat().st_size > 20_000_000:
            raise ValueError("RATING_PUBLIC_SOURCE_PATH")
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref["sha256"]:
            raise ValueError("RATING_PUBLIC_SOURCE_HASH")
    if market or etf:
        # dfcfw 附件主机按东财公告目录逐件绑定身份；白名单不等于整域信任。
        from app.services.fund_rating_public_market import ATTACHMENT_HOST, attachment_identity

        by_label = {r["label"]: r for r in refs if "label" in r}
        for ref in refs:
            if urlsplit(ref["url"]).hostname == ATTACHMENT_HOST:
                code = bundle["fund_code"]
                if etf and ref["label"].startswith("target-"):
                    # 穿透目标 ETF 的披露附件身份绑定到目标基金代码。
                    code = bundle["classification"]["scope"]["target_fund"]
                attachment_identity(by_label, root, ref["label"], code)
    hashes = {r["sha256"] for r in refs}
    if not {bundle["report_hash"], bundle["summary_hash"], *bundle["classification"]["source_refs"]} <= hashes:
        raise ValueError("RATING_PUBLIC_SOURCE_REFERENCE")


def build_public_metrics(bundle: dict, as_of: date, frozen: datetime) -> tuple[dict, dict, datetime]:
    if bundle.get("basis") == GLOBAL_BASIS:
        from app.services.fund_rating_public_global import metrics

        return metrics(bundle, as_of, frozen)
    if bundle.get("basis") == RECONCILED_BASIS:
        from app.services.fund_rating_public_reconciled import metrics

        return metrics(bundle, as_of, frozen)
    if bundle.get("basis") == INDEX_BASIS:
        from app.services.fund_rating_public_index import metrics

        return metrics(bundle, as_of, frozen)
    """从完整事实重算全部八维指标；有界披露字段采用风险档，不强迫提供不存在的精确值。"""
    if bundle.get("basis") == MARKET_BASIS:
        from app.services.fund_rating_public_market import metrics

        return metrics(bundle, as_of, frozen)
    if bundle.get("basis") == ETF_BASIS:
        from app.services.fund_rating_public_etf import build_etf_metrics

        return build_etf_metrics(bundle, as_of, frozen)
    if (
        bundle["basis"] != BASIS
        or bundle["family"] != "ACTIVE_EQUITY"
        or bundle["currency"] != "CNY"
        or bundle["as_of_date"] != str(as_of)
    ):
        raise ValueError("RATING_PUBLIC_SCOPE")
    h, holding, info, manager, disclosure = (
        bundle[k] for k in ("history", "holdings", "key_information", "manager", "disclosure")
    )
    if (
        info["fund_code"] != bundle["fund_code"]
        or info["currency"] != bundle["currency"]
        or date.fromisoformat(info["published_on"]) > as_of
    ):
        raise ValueError("RATING_PUBLIC_DOCUMENT_SCOPE")
    metrics = public_history(h, as_of)
    report = date.fromisoformat(holding["report_date"])
    if not 0 <= (as_of - report).days <= 210:
        raise ValueError("RATING_PUBLIC_HOLDINGS_EXPIRED")
    reconcile_equity_assets(holding["positions"], holding["net_assets"], holding["balance_sheet"])
    metrics.update(equity_structure(holding["positions"]))
    if (
        disclosure["holder_risk_band"] not in (0, 1, 2)
        or disclosure["strategy_continuity"] not in (0, 1)
        or disclosure["governance_state"] not in (0, 1, 2)
    ):
        raise ValueError("RATING_PUBLIC_DISCLOSURE_STATE")
    metrics.update({k: decimal(disclosure[k]) for k in ("holder_risk_band", "strategy_continuity", "governance_state")})
    metrics["net_assets"] = decimal(holding["net_assets"])
    metrics["restricted_ratio"] = decimal(disclosure["restricted_stock_amount"]) / metrics["net_assets"]
    metrics["cash_ratio"] = decimal(holding["balance_sheet"]["other_assets"]["货币资金"]) / metrics["net_assets"]
    for key in ("restricted_ratio", "cash_ratio"):
        if not 0 <= metrics[key] <= 1:
            raise ValueError("RATING_PUBLIC_ASSET_RATIO")
    metrics.update({k: decimal(manager[k]) for k in ("team_months", "team_changes")})
    if metrics["team_months"] < 0 or metrics["team_changes"] < 0:
        raise ValueError("RATING_PUBLIC_MANAGER_RANGE")
    fees = info["fees"]
    metrics["cost_1y"] = standard_cost(
        *(fees[k] for k in ("management", "custody", "sales", "subscription", "redemption")),
        mode=fees["subscription_mode"],
        operating_cost=fees["operating_cost"],
    )
    expiry = public_valid_until(as_of, report, datetime.fromisoformat(bundle["documents_checked_at"]), frozen)
    facts = {"nav": str(as_of), "holdings": str(report), "manager": info["checked_on"], "fees": info["checked_on"]}
    return metrics, facts, expiry


def public_history(h: dict, as_of: date) -> dict:
    """公开权益与ETF共用的真实历史指标；不需要任何持仓或持有人占位值。"""
    if not h.get("events_complete") or h["missing_dates"] or h["event_coverage"]["dates_checked"] != len(h["dates"]):
        raise ValueError("RATING_PUBLIC_EVENTS_INCOMPLETE")
    from app.services.direction_1d_protocol import calendar

    sessions, calendar_hash = calendar()
    days = [date.fromisoformat(d) for d in h["dates"]]
    if not days or calendar_hash != h["calendar_hash"] or days != [d for d in sessions if days[0] <= d <= as_of]:
        raise ValueError("RATING_PUBLIC_CALENDAR_GAP")
    if max(d for d in sessions if (d.year, d.month) == (as_of.year, as_of.month)) != as_of:
        raise ValueError("RATING_PUBLIC_MONTH_END")
    index = total_return(h["unit_nav"], h["cash_dividend_per_share"], h["split_factor"])
    metrics = history_metrics(
        days, index, [], [date.fromisoformat(d) for d in h["month_ends"]], 252, require_sharpe=False
    )
    metrics.pop("sharpe")
    endpoints = [index[h["dates"].index(d)] for d in h["month_ends"]]
    metrics["rolling_positive"] = Decimal(sum(x > 0 for x in metrics["rolling"])) / len(metrics["rolling"])
    metrics["monthly_positive"] = Decimal(sum(b > a for a, b in zip(endpoints, endpoints[1:], strict=False))) / (
        len(endpoints) - 1
    )
    # 回撤为零时不伪造分母；只用政策已饱和的端点表达正、零、负收益。
    metrics["calmar"] = (
        metrics["return_3y"] / metrics["drawdown"]
        if metrics["drawdown"]
        else Decimal(2 if metrics["return_3y"] > 0 else 0 if metrics["return_3y"] == 0 else -1)
    )
    return metrics


def public_valid_until(as_of: date, report: date, checked: datetime, frozen: datetime) -> datetime:
    """复核、持仓和月度宽限取最早到期点；未来报告或已过期输入拒绝生成等级。"""
    from app.services.direction_1d_protocol import calendar

    sessions, _ = calendar()
    if not 0 <= (as_of - report).days <= 210:
        raise ValueError("RATING_PUBLIC_HOLDINGS_EXPIRED")
    if checked.tzinfo is None or checked > frozen or frozen - checked > timedelta(days=31):
        raise ValueError("RATING_PUBLIC_DOCUMENTS_EXPIRED")
    # 遵循既有月度更新宽限；宽限不超过原文复核/持仓有效期。
    next_month = (as_of.replace(day=28) + timedelta(days=4)).replace(day=1)
    update_month = (next_month.replace(day=28) + timedelta(days=4)).replace(day=1)
    update_days = [d for d in sessions if (d.year, d.month) == (update_month.year, update_month.month)]
    if len(update_days) < 6:
        raise ValueError("RATING_PUBLIC_EXPIRY_CALENDAR")
    expiry = min(
        datetime.combine(update_days[5], datetime.min.time(), ZoneInfo("Asia/Shanghai")),
        checked + timedelta(days=31),
        datetime.combine(report + timedelta(days=211), datetime.min.time(), UTC),
    )
    if expiry <= frozen:
        raise ValueError("RATING_PUBLIC_ALREADY_EXPIRED")
    return expiry


def prepare_public_bundle(session, as_of: date, frozen: datetime) -> dict:
    """复用已有002112原件并提取新增公开段落，先留存原文后返回待导入包；不直接激活评级。"""
    from app.services.fund_rating_inputs import canonical, catalog
    from app.services.fund_rating_materials import _pages, archive_materials, inspect_case

    material = inspect_case(session, as_of, frozen)
    if material is None:
        raise ValueError("RATING_PUBLIC_MATERIALS_MISSING")
    source_root = Path(material["source_root"])
    report_ref = material["sources"][0]
    disclosure = report_facts(_pages(source_root, report_ref, frozen))
    pdf_refs = [r for r in material["sources"] if r.get("url", "").endswith(".pdf")]
    summaries = [(r, _pages(source_root, r, frozen)) for r in pdf_refs if "概要" in r["url"]]
    if len(summaries) != 1 or "基金类型混合型交易币种人民币" not in compact("\n".join(summaries[0][1])):
        raise ValueError("RATING_PUBLIC_FUND_TYPE")
    summary_ref, summary_pages = summaries[0]
    info = key_information(summary_pages)
    info["checked_on"] = (
        datetime.fromisoformat(summary_ref["received_at"]).astimezone(ZoneInfo("Asia/Shanghai")).date().isoformat()
    )
    facts = material["facts"]
    # 本口径不使用国债/股票行情资料，不把这些来源的使用权限当作出级前提。
    retained = {"source_root": str(source_root), "sources": pdf_refs, "facts": {"history": facts["history"]}}
    raw_refs = pdf_refs + facts["history"]["event_coverage"]["receipts"]
    by_hash = {r["sha256"]: r for r in raw_refs}
    archives = archive_materials(retained, Path(get_settings().fund_insight_directory) / "ratings", frozen)
    for r in archives:
        receipt = by_hash[r["sha256"]]
        r.update(url=receipt["url"], acquired_at=receipt["received_at"])
    row = next(r for r in catalog(session) if r["fund_code"] == "002112")
    public_sources = []
    for dimension in material["dimensions"].values():
        for r in dimension["sources"]:
            if r["url"].endswith(".pdf") and r not in public_sources:
                public_sources.append(r)
    sources = [{"title": r["title"], "url": r["url"], "published_on": r["publishedOn"]} for r in public_sources]
    result = {
        "basis": BASIS,
        "fund_code": "002112",
        "family": "ACTIVE_EQUITY",
        "currency": "CNY",
        "as_of_date": str(as_of),
        "found_date": str(row["found_date"]),
        "source_scope": "PUBLIC_DISCLOSURE_LOCAL_ANALYSIS",
        "sources": sources,
        "raw_archive": archives,
        "report_hash": report_ref["sha256"],
        "summary_hash": summary_ref["sha256"],
        "documents_checked_at": min(r["received_at"] for r in pdf_refs[1:]),
        "history": facts["history"],
        "holdings": facts["holdings"],
        "manager": facts["manager"],
        "key_information": info,
        "disclosure": disclosure,
        "category_label": "境内主动混合基金（公开资料评价）",
        "classification": {
            "product_id": info["product_id"],
            "family": "ACTIVE_EQUITY",
            "relationship_verified": True,
            "effective_from": str(as_of),
            "source_refs": [summary_ref["sha256"]],
            "scope": {"basis": BASIS, "asset": "DOMESTIC_ACTIVE_MIXED", "currency": "CNY"},
        },
    }
    return json.loads(canonical(result))


def verify_public_originals(bundle: dict) -> None:
    if bundle.get("basis") == GLOBAL_BASIS:
        from app.services.fund_rating_public_global import verify_originals

        return verify_originals(bundle)
    if bundle.get("basis") == RECONCILED_BASIS:
        from app.services.fund_rating_public_reconciled import verify_originals

        return verify_originals(bundle)
    if bundle.get("basis") == INDEX_BASIS:
        from app.services.fund_rating_public_index import verify_originals

        return verify_originals(bundle)
    """导入时重读关键PDF，拒绝把手工改写的监管/流通受限事实冒充原文。"""
    if bundle.get("basis") == MARKET_BASIS:
        from app.services.fund_rating_public_market import verify_originals

        verify_originals(bundle)
        return
    if bundle.get("basis") == ETF_BASIS:
        from app.services.fund_rating_public_etf import verify_etf_originals

        verify_etf_originals(bundle)
        return
    root = Path(get_settings().fund_insight_directory) / "ratings"
    refs = {r["sha256"]: r for r in bundle["raw_archive"]}
    raw = (root / refs[bundle["report_hash"]]["file"]).read_bytes()
    pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages]
    if report_facts(pages) != bundle["disclosure"]:
        raise ValueError("RATING_PUBLIC_REPORT_FACT_CONFLICT")
    balance = equity_balance(pages, bundle["holdings"]["positions"])
    if any(balance[k] != bundle["holdings"][k] for k in ("net_assets", "balance_sheet")):
        raise ValueError("RATING_PUBLIC_BALANCE_FACT_CONFLICT")
    raw = (root / refs[bundle["summary_hash"]]["file"]).read_bytes()
    pages = [p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages]
    if "基金类型混合型交易币种人民币" not in compact("\n".join(pages)):
        raise ValueError("RATING_PUBLIC_FUND_TYPE")
    info = key_information(pages)
    actual = deepcopy(bundle["key_information"])
    actual.pop("checked_on")
    if actual != info:
        raise ValueError("RATING_PUBLIC_FEE_FACT_CONFLICT")
    prospectus = [r for r in refs.values() if "招募" in r["url"] and r["url"].endswith(".pdf")]
    if len(prospectus) != 1:
        raise ValueError("RATING_PUBLIC_MANAGER_SOURCE")
    raw = (root / prospectus[0]["file"]).read_bytes()
    teams = appointed_teams(
        [p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages],
        info,
        date.fromisoformat(bundle["history"]["dates"][0]),
        date.fromisoformat(bundle["as_of_date"]),
    )
    if teams != bundle["manager"]:
        raise ValueError("RATING_PUBLIC_MANAGER_FACT_CONFLICT")


def re_market_code(code) -> bool:
    """公开市场适配仍限六位公开基金代码，不接受调用方路径或任意标识。"""
    return isinstance(code, str) and len(code) == 6 and code.isascii() and code.isdigit()
