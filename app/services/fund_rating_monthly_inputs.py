"""境内被动基金滚动月评的真实输入准备；不批准用途，不发布评分。

净值使用已授权规范记录，官网曲线按真实新窗口重取。新协议只绑定实际请求
和实际公布的日期/重复行；公式、精度及缺项保护仍由原专用编译器重放。
"""

import json
import re
from copy import deepcopy
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_nav_preview_engine
from app.services.direction_1d_protocol import calendar, digest
from app.services.fund_exposure_common import now
from app.services.fund_materials_store import versioned_save
from app.services.fund_rating_direct_index import Originals
from app.services.fund_rating_pipeline import archive_json
from app.services.fund_rating_pipeline_prepare import canonical_source
from app.services.fund_rating_public_market import provider_history


def provider_snapshot(code, target, *, session):
    """读取本份额三年真实规范记录；缺日、未授权或重复日仍明确拒绝。"""
    sessions, calendar_hash = calendar()
    start_month = target.replace(year=target.year - 3, day=1)
    starts = [d for d in sessions if (d.year, d.month) == (start_month.year, start_month.month)]
    if not starts or target not in sessions:
        raise ValueError("MONTHLY_OFFICIAL_VALUATION_WINDOW_REQUIRED")
    start = max(starts)
    days = [d for d in sessions if start <= d <= target]
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
        raise ValueError("MONTHLY_PROVIDER_AUTHORIZATION_REQUIRED")
    rows = []
    for year in range(start.year, target.year + 1):
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
                    "end": min(target, date(year, 12, 31)),
                },
            )
            .mappings()
            .all()
        )
    by_day = {r["nav_date"]: r for r in rows}
    if len(by_day) != len(rows):
        raise ValueError("MONTHLY_PROVIDER_DUPLICATE_DAY")
    missing = sorted(set(days) - set(by_day))
    if missing:
        raise ValueError("MONTHLY_PROVIDER_DAYS_MISSING:" + ",".join(map(str, missing[:10])))
    managers = (
        session.execute(
            text(
                "SELECT manager_name,ann_date,begin_date,end_date,education FROM fund_manager_assignment "
                "WHERE fund_code=:code ORDER BY begin_date,manager_name"
            ),
            {"code": code},
        )
        .mappings()
        .all()
    )
    dividends = (
        session.execute(
            text(
                "SELECT ann_date,implementation_ann_date,base_date,process_status,record_date,ex_date,pay_date,"
                "earnings_pay_date,nav_ex_date,cash_dividend,base_unit,distributable_earnings,earnings_amount,"
                "reinvestment_arrival_date,base_year FROM fund_dividend WHERE fund_code=:code "
                "ORDER BY pay_date,source_event_key"
            ),
            {"code": code},
        )
        .mappings()
        .all()
    )
    month_ends = [max(d for d in days if (d.year, d.month) == ym) for ym in sorted({(d.year, d.month) for d in days})]
    if len(month_ends) != 37:
        raise ValueError("MONTHLY_PROVIDER_THREE_YEAR_WINDOW")

    def serial(values):
        return [{k: str(v) if v is not None else None for k, v in row.items()} for row in values]

    return {
        "fund_code": code,
        "source": {
            "source_code": "TUSHARE_PRO_FUND",
            "enabled": True,
            "authorization_verified_at": source["authorization_verified_at"].isoformat(),
        },
        "history": {
            "dates": list(map(str, days)),
            "month_ends": list(map(str, month_ends)),
            "annual_days": 252,
            "calendar_hash": calendar_hash,
            "supplementary_non_trading_dates": sorted(str(d) for d in set(by_day) - set(days)),
            "rows": serial(rows),
        },
        "manager_records": serial(managers),
        "dividend_records": serial(dividends),
        "normalized_records_observed_at": now().isoformat(),
    }


def _derived(root, value, *, label):
    return {
        **archive_json(root, value),
        "kind": "json",
        "archive_kind": "DERIVED",
        "label": label,
        "url": "LOCAL_READ_ONLY_NORMALIZED_SOURCE_RECORDS",
        "acquired_at": now().isoformat(),
    }


def _acquire(reference, params, *, root, output, client):
    """只重放已支持的官网只读接口，失败原文和真实取得时刻先留证。"""
    allowed = {
        "https://www.cmfchina.com/ws-business-server/fund/getIncomeList",
        "https://www.southernfund.com/nfwebApi/fund/earningsMovements",
        "https://www.southernfund.com/nfwebApi/fund/netValue",
        "https://www.southernfund.com/nfwebApi/fund/dividends",
        "https://e.gtfund.com/Etrade/Public/cochin/info.fund.growth_plot",
    }
    url = reference["url"]
    if url not in allowed or urlsplit(url).scheme != "https":
        raise ValueError("MONTHLY_ISSUER_ADAPTER_REQUIRED")
    method = "POST_FORM" if urlsplit(url).hostname == "www.southernfund.com" else "GET"
    if urlsplit(url).hostname == "e.gtfund.com":
        method = "POST_JSON"
    request_key = digest({"url": url, "method": method, "params": params})
    if url == "https://www.cmfchina.com/ws-business-server/fund/getIncomeList":
        from app.integrations.cmf_public import income_list

        response = income_list(client, params, audit_directory=Path(output) / "public-protocol")
    elif method == "POST_JSON":
        response = client.post(url, json={"api": "info.fund.growth_plot", "params": params}, follow_redirects=True)
    else:
        response = client.post(url, data=params) if method == "POST_FORM" else client.get(url, params=params)
    raw = response.content
    if len(raw) > 5_000_000:
        raise ValueError("MONTHLY_ISSUER_RESPONSE_SIZE")
    import hashlib

    sha = hashlib.sha256(raw).hexdigest()
    destination = Path(output) / "originals" / (sha + ".json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        destination.write_bytes(raw)
    receipt = {
        "url": url,
        "method": method,
        "params": params,
        "status": response.status_code,
        "received_at": now().isoformat(),
        "raw_path": str(destination.resolve()),
        "sha256": sha,
    }
    versioned_save(Path(output) / "attempts" / (request_key + ".json"), receipt)
    response.raise_for_status()
    payload = json.loads(raw)
    if url.startswith("https://www.cmfchina.com/") and payload.get("code") != 0:
        raise ValueError(
            "MONTHLY_ISSUER_BUSINESS_RESPONSE:" + str(payload.get("code")) + ":" + str(payload.get("msg", ""))[:100]
        )
    if url.startswith("https://www.southernfund.com/") and payload.get("code") != "ETS-5BP00000":
        raise ValueError("MONTHLY_ISSUER_BUSINESS_RESPONSE:" + str(payload.get("code")))
    if method == "POST_JSON":
        reference = {**reference, "request": {"api": "info.fund.growth_plot", "params": params}}
    return canonical_source(
        root,
        {
            **reference,
            "path": str(destination.resolve()),
            "file": str(destination.resolve()),
            "params": params,
            "method": method,
            "sha256": sha,
            "status": response.status_code,
            "bytes": len(raw),
            "acquired_at": receipt["received_at"],
            "final_url": str(response.url).split("?", 1)[0]
            if url == "https://www.cmfchina.com/ws-business-server/fund/getIncomeList"
            else str(response.url),
            "archive_kind": "ORIGINAL",
        },
    )


def _index_daily(code, start, end, *, root, index_client=None):
    from app.integrations.tushare_market_reference import TushareMarketReferenceClient

    settings = get_settings()
    owned = index_client is None
    client = index_client or TushareMarketReferenceClient(
        token=settings.tushare_token.get_secret_value(),
        api_url=settings.tushare_api_url,
        connect_timeout_seconds=5,
        read_timeout_seconds=20,
        max_retries=0,
        catalog_max_rows_per_query=settings.tushare_catalog_max_rows_per_query,
        max_rows_per_query=settings.tushare_market_reference_max_rows_per_query,
    )
    try:
        values = client.list_index_daily(code, start_date=date.fromisoformat(start), end_date=date.fromisoformat(end))
    finally:
        if owned:
            client.close()
    rows = [[v.index_code, str(v.trade_date), str(v.close_price)] for v in values]
    ref = _derived(
        root,
        {"code": 0, "data": {"fields": ["ts_code", "trade_date", "close"], "items": rows}},
        label="monthly-index-daily",
    )
    ref["url"] = settings.tushare_api_url
    return ref


def _bind(source, inputs):
    inputs["sources"][source["sha256"]] = source
    return source["sha256"]


def _refresh_protocols(inputs, history, *, root, output, client, index_client=None):
    source = Originals(inputs, root, lazy=True)
    refs = inputs["target_refs"]
    first, last = history["dates"][0], history["dates"][-1]
    if "cmf_benchmark_reconstruction" in inputs:
        params = {
            **inputs["sources"][refs["benchmark"]].get("params", {}),
            "productCode": inputs["fund_code"],
            "startDate": first,
            "endDate": last,
            "queryMonth": "",
        }
        current = _acquire(inputs["sources"][refs["benchmark"]], params, root=root, output=output, client=client)
        refs["benchmark"] = _bind(current, inputs)
        node = inputs["cmf_benchmark_reconstruction"]
        protocol = deepcopy(source.json(node["protocol_hash"]))
        protocol.update(as_of=last, frozen_at=now().isoformat())
        node["protocol_hash"] = _bind(_derived(root, protocol, label="monthly-cmf-protocol"), inputs)
        node["index_daily"] = _bind(
            _index_daily(inputs["index"]["code"], first, last, root=root, index_client=index_client), inputs
        )
    elif "gt_protocol" in inputs:
        from datetime import timedelta

        from app.services.fund_rating_gt_benchmark import _day

        def stamp(day):
            return str(date.fromisoformat(day) - timedelta(days=1)) + "T16:00:00.000Z"

        params = {"start": stamp(first), "end": stamp(last), "fundCode": inputs["fund_code"], "order": "asc"}
        refs["benchmark"] = _bind(
            _acquire(inputs["sources"][refs["benchmark"]], params, root=root, output=output, client=client), inputs
        )
        current = Originals(inputs, root, lazy=True).json(refs["benchmark"])
        protocol = deepcopy(source.json(inputs["gt_protocol"]))
        protocol.update(
            as_of=last,
            first=first,
            last=last,
            frozen_at=now().isoformat(),
            published_nonvaluation_dates_retained=sorted({_day(row[0]) for row in current} - set(history["dates"])),
        )
        inputs["gt_protocol"] = _bind(_derived(root, protocol, label="monthly-gt-protocol"), inputs)
    elif "southern_protocol" in inputs:
        protocol = deepcopy(source.json(inputs["southern_protocol"]))
        scope = {"fundCode": inputs["fund_code"], "startDate": first.replace("-", ""), "endDate": last.replace("-", "")}
        for key in ["benchmark", "issuer-nav"]:
            refs[key] = _bind(
                _acquire(inputs["sources"][refs[key]], scope, root=root, output=output, client=client), inputs
            )
        event_params = {
            "fundCode": inputs["fund_code"],
            "startDate": "20090101",
            "endDate": scope["endDate"],
            "curPage": 1,
            "pageSize": protocol.get("event_page_size", 10),
        }
        refs["issuer-events"] = _bind(
            _acquire(inputs["sources"][refs["issuer-events"]], event_params, root=root, output=output, client=client),
            inputs,
        )
        current = Originals(inputs, root, lazy=True).json(refs["benchmark"])["data"]
        own_dates = [date.fromisoformat(str(row["fdate"])).isoformat() for row in current["thisfundlist"]]
        extra = sorted(set(own_dates) - set(history["dates"]))
        duplicates = [day for n, day in enumerate(own_dates) if day in own_dates[:n]]
        protocol.update(
            as_of=last,
            first=first,
            last=last,
            events_cutoff=last,
            frozen_at=now().isoformat(),
            published_nonvaluation_dates_retained=extra,
            identical_own_terminal_duplicates_retained=duplicates,
        )
        inputs["southern_protocol"] = _bind(_derived(root, protocol, label="monthly-southern-protocol"), inputs)


def apply_management_review(bundle, review, *, root, history):
    """只消费实际提交的新期职责审阅；不从公告未命中或旧日期自动批准续任。"""
    from app.services.fund_rating_management_anchor import appointment_verified

    inputs = bundle["evidence_v1_calculation"]["inputs"]
    if (
        review.get("fund_code") != inputs["fund_code"]
        or review.get("as_of_date") != inputs["as_of_date"]
        or len(review.get("reason", "").strip()) < 80
        or not review.get("role_quote")
    ):
        raise ValueError("MONTHLY_ACTUAL_MANAGEMENT_REVIEW_REQUIRED")
    reference = review["source"]
    if not reference.get("published_on") or reference["published_on"] > inputs["as_of_date"]:
        raise ValueError("MONTHLY_MANAGEMENT_FUTURE_DISCLOSURE")
    raw = inputs["management_inputs"]
    _bind(reference, inputs)
    source = Originals(inputs, root, lazy=True)
    pages = source.pages(reference["sha256"])
    text_body = "\n".join(pages)
    from app.services.fund_rating_market_review import normalized_text

    if (
        review["role_quote"] not in text_body
        or inputs["fund_code"] not in text_body
        or normalized_text(inputs["legal_name"]) not in normalized_text(text_body)
    ):
        raise ValueError("MONTHLY_MANAGEMENT_OWN_REPORT_PROOF")
    active = [v for v in raw["continuity"]["intervals"] if v.get("end") is None]

    def verified(span):
        begin = date.fromisoformat(span["start"])
        if appointment_verified(pages, span["name"], begin):
            return True
        # 人员行的年月、下一行月日是表格排版，不能跨任意履历查找。
        pattern = (
            re.escape(span["name"])
            + r"\s+[^\n]{0,45}"
            + str(begin.year)
            + r"\s*年\s*"
            + str(begin.month)
            + r"\s+[^\n]{0,45}\n[^\n]{0,30}月\s*"
            + str(begin.day)
            + r"\s*日"
        )
        return any(re.search(pattern, page) for page in pages)

    if not active or any(not verified(span) for span in active):
        raise ValueError("MONTHLY_MANAGER_TIMELINE_REBUILD_REQUIRED")
    raw["sources"][reference["sha256"]] = reference
    claim = {"source_hash": reference["sha256"], "quote": review["role_quote"]}
    if claim not in raw["claims"]:
        raw["claims"].append(claim)
    if reference["sha256"] not in raw["experience"]["source_refs"]:
        raw["experience"]["source_refs"].append(reference["sha256"])
    raw["as_of_date"] = inputs["as_of_date"]
    raw["continuity"].update(period_start=history["dates"][0], period_end=inputs["as_of_date"])
    if reference["sha256"] not in raw["continuity"]["source_refs"]:
        raw["continuity"]["source_refs"].append(reference["sha256"])
    raw["continuity"]["review_scope"] = review["reason"]
    for span in active:
        if reference["sha256"] not in span["source_refs"]:
            span["source_refs"].append(reference["sha256"])
    bundle["monthly_execution_review"] = {"checked_at": now().isoformat(), **review}


def build_monthly_inputs(
    code,
    previous_bundle,
    target_date,
    root,
    output,
    *,
    management_review=None,
    client=None,
    index_client=None,
    provider=None,
):
    """返回未批准候选及预览；只登记来源，不自动发布或关闭缺项。"""
    from app.services.fund_rating_evidence_preview import preview

    target = date.fromisoformat(str(target_date))
    bundle = deepcopy(previous_bundle)
    if (
        bundle.get("fund_code") != code
        or bundle.get("family") != "INDEX"
        or re.search(r"QDII|海外|日本|越南|纳斯达克|原油|黄金", bundle.get("identity", {}).get("legal_name", ""), re.I)
    ):
        raise ValueError("MONTHLY_DOMESTIC_PASSIVE_SCOPE_REQUIRED")
    inputs = bundle["evidence_v1_calculation"]["inputs"]
    if "reviewed_management" in inputs and not inputs.get("management_inputs"):
        bundle["monthly_prior_management_observation"] = inputs.pop("reviewed_management")
    if provider is None:
        with Session(get_nav_preview_engine()) as session:
            provider = provider_snapshot(code, target, session=session)
    _, history = provider_history(provider, target)
    provider_ref = _derived(root, provider, label="monthly-provider-" + code)
    inputs["target_refs"]["provider"] = _bind(provider_ref, inputs)
    inputs["as_of_date"] = bundle["as_of_date"] = str(target)
    inputs["verified_at"] = bundle["verified_at"] = now().isoformat()
    bundle["data_dates"]["nav"] = str(target)
    owned = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(20, connect=5),
        trust_env=False,
        follow_redirects=False,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    errors = []
    try:
        try:
            _refresh_protocols(inputs, history, root=root, output=output, client=http, index_client=index_client)
        except Exception as error:
            errors.append({"step": "ISSUER_BENCHMARK", "reason": type(error).__name__ + ":" + str(error)[:180]})
        if management_review:
            apply_management_review(bundle, management_review, root=root, history=history)
        else:
            errors.append({"step": "MANAGEMENT", "reason": "ACTUAL_NEW_PERIOD_ROLE_REVIEW_REQUIRED"})
    finally:
        if owned:
            http.close()
    inputs["verified_at"] = bundle["verified_at"] = now().isoformat()
    result = preview(bundle, root, review_required=False)
    Path(output).mkdir(parents=True, exist_ok=True)
    candidate = Path(output) / "candidate-bundle.json"
    candidate.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    value = {
        "fund_code": code,
        "as_of_date": str(target),
        "checked_at": now().isoformat(),
        "reviewed": False,
        "candidate_path": str(candidate.resolve()),
        "attempt_errors": errors,
        "preview": result,
    }
    versioned_save(Path(output) / "preparation.json", value)
    return value
