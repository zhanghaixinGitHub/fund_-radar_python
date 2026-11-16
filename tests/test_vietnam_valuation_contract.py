"""越南估值隔离用例；所有正向汇率、原件和日历均为测试样例，不写生产。"""

import hashlib
import json
from contextlib import nullcontext
from datetime import date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from app.services.direction_1d_protocol import digest
from app.services.fund_information_contracts import time_context
from app.services.prediction_v2_baseline import baseline_ready
from app.services.qdii_valuation_contract import verified_qdii_window
from app.services.vietnam_valuation_contract import POLICY, assess_vietnam_valuation, future_sessions

AT = datetime.fromisoformat("2026-11-14T19:00:00+08:00")
WINDOW = {"base_nav_date": "2026-11-13", "target_nav_date": "2026-11-16"}


def doc(identifier, kind, body, url="https://www.thfund.com.cn/test", **extra):
    text = json.dumps(body, ensure_ascii=False) if isinstance(body, dict) else body
    return {
        "id": identifier,
        "kind": kind,
        "body": text,
        "body_hash": digest(text),
        "url": url,
        "available_at": "2026-11-14T08:00:00+08:00",
        "codes": ["008764"],
        **extra,
    }


def replace_body(document, body):
    document.update(body=json.dumps(body, ensure_ascii=False), body_hash=digest(json.dumps(body, ensure_ascii=False)))


def sample():
    clauses = {
        "day": "本基金的估值日为本基金相关的证券交易场所的交易日。",
        "close": "股票以估值日的收盘价估值。",
        "fx": "采用人民币中间价；彭博提供伦敦时间16:00与美元折算率。无法取得时用基金托管人或境外托管人提供的价格。",
    }
    calendar = {
        "year": 2026,
        "closed_dates": [f"2026-01-{i:02}" for i in range(1, 11)],
        "makeup_saturdays_not_trading": ["2026-01-10"],
    }
    stock_rows = [
        {"symbol": "HPG", "date": "2026-11-12", "close": "21.05"},
        {"symbol": "HPG", "date": "2026-11-13", "close": "21.05"},
    ]
    index_rows = [
        {"symbol": "VN30", "date": "2026-11-12", "close": "1839.43"},
        {"symbol": "VN30", "date": "2026-11-13", "close": "1844.91"},
    ]
    stock_proof = json.dumps(stock_rows[1])
    index_proof = json.dumps(index_rows[1])
    docs = {
        "own": doc("own", "PROSPECTUS", "\n".join(clauses.values())),
        "report": doc("report", "REPORT", "本基金为越南市场股票人民币C份额"),
        "calendar": doc("calendar", "EXCHANGE_CALENDAR", calendar, "https://www.hsx.vn/calendar"),
        "hours": doc(
            "hours", "EXCHANGE_TRADING_HOURS", "HOSE ATC 14h45; put-through ends 15h00", "https://www.ssi.com.vn/hsx"
        ),
        "stock": doc("stock", "MARKET_DATA", {"symbol": "HPG", "rows": stock_rows}, "https://api.dnse.com.vn/HPG"),
        "index": doc("index", "MARKET_DATA", {"symbol": "VN30", "rows": index_rows}, "https://api.dnse.com.vn/VN30"),
        "units": doc("units", "PROVIDER_UNIT_EXCERPT", "Giá x 1000 VND", "https://developers.dnse.com.vn/units"),
        "official": doc(
            "official",
            "FX_REFERENCE_DATA",
            {"rates": [{"currency": "USD", "date": "2026-11-13", "cny_per_unit": "6.7"}]},
            "https://www.safe.gov.cn/rates",
        ),
        "fixing": doc(
            "fixing",
            "FX_FIXING_DATA",
            {
                "date": "2026-11-13",
                "base_currency": "USD",
                "quote_currency": "VND",
                "rate": "25000",
                "rate_type": "BLOOMBERG_LONDON_1600",
                "fixing_at": "2026-11-13T16:00:00+00:00",
            },
            "https://www.bloomberg.com/fixing",
        ),
    }
    evidence = {
        "policy": POLICY,
        "base_date": WINDOW["base_nav_date"],
        "target_date": WINDOW["target_nav_date"],
        "prospectus_ref": "own",
        "report_ref": "report",
        "valuation_quotes": clauses,
        "calendar_refs": {"VN": "calendar"},
        "calendar_review": {
            "body_hash": docs["calendar"]["body_hash"],
            "closed_dates_hash": digest(calendar["closed_dates"]),
            "coverage_start": "2026-01-01",
            "coverage_end": "2026-12-31",
        },
        "market_hours_ref": "hours",
        "fx_ref": "official",
        "vnd_fx_ref": "fixing",
    }
    common = {
        "date": "2026-11-13",
        "currency": "VND",
        "valuation_market": "VN",
        "change_pct": 0,
        "session": "COMPLETED_DAILY_HISTORY",
        "closed_at": "2026-11-13T15:00:00+07:00",
        "close_timestamp_kind": "EXCHANGE_COMPLETION_BOUND",
    }
    stock = {
        **common,
        "close": "21050",
        "provider_close": "21.05",
        "previous_close": "21050",
        "provider_to_vnd_multiplier": 1000,
        "price_unit": "VND_PER_SHARE",
        "unit_evidence_ref": "units",
        "unit_quote": "Giá x 1000 VND",
        "source_ref": "stock",
        "price_quote": stock_proof,
    }
    index = {
        **common,
        "close": "1844.91",
        "previous_close": "1839.43",
        "change_pct": str((Decimal("1844.91") / Decimal("1839.43") - 1) * 100),
        "price_unit": "INDEX_POINTS",
        "source_ref": "index",
        "price_quote": index_proof,
    }
    material = {
        "fund_code": "008764",
        "fund_name": "天弘越南市场股票(QDII)-C",
        "companies": [{"code": "VN000000HPG4", "ticker": "HPG", "quote": stock}],
        "market": {"VN30": index},
        "calendar_evidence": evidence,
    }
    return {"fund_name": material["fund_name"]}, material, docs, dict(WINDOW)


def status(p, m, d, w):
    return assess_vietnam_valuation(p, m, d, w, as_of=AT)


def test_complete_market_and_exact_fixing_are_verified_without_inventing_fund_nav():
    p, m, d, w = sample()
    result = status(p, m, d, w)
    assert result["ready"] and result["supported"]
    assert Decimal(result["fx"]["vnd_cny"]) == Decimal("6.7") / 25000
    assert verified_qdii_window(p, m, d, w, as_of=AT)
    data = {
        **m,
        "as_of": AT.isoformat(),
        "window": w,
        "nav": [{"nav_date": "2026-11-12"}],
        "latest_nav_date": "2026-11-12",
        "valuation_status": result,
    }
    data["time_context"] = time_context(data, ["2026-11-12", "2026-11-13", "2026-11-16"])
    assert not baseline_ready(data)


@pytest.mark.parametrize(
    "problem", ["missing", "spot", "time", "date", "nan", "retail", "future", "proxy", "duplicate"]
)
def test_missing_or_wrong_fixing_never_becomes_ready(problem):
    p, m, d, w = sample()
    row = json.loads(d["fixing"]["body"])
    if problem == "missing":
        m["calendar_evidence"].pop("vnd_fx_ref")
    elif problem == "spot":
        row["rate_type"] = "SPOT"
    elif problem == "time":
        row["fixing_at"] = "2026-11-13T09:00:00+07:00"
    elif problem == "date":
        row["date"] = "2026-11-12"
    elif problem == "nan":
        row["rate"] = "NaN"
    elif problem == "retail":
        d["fixing"].update(kind="CUSTODIAN_FX_REFERENCE", url="https://www.hsbc.com.vn/retail")
    elif problem == "future":
        d["fixing"]["available_at"] = "2026-11-15T01:00:00+08:00"
    elif problem == "proxy":
        d["official"]["url"] = "https://www.ecb.europa.eu/rates"
    else:
        official = json.loads(d["official"]["body"])
        official["rates"] *= 2
        replace_body(d["official"], official)
    replace_body(d["fixing"], row)
    result = status(p, m, d, w)
    assert result["supported"] and not result["ready"] and not result["fx_ready"]
    assert not verified_qdii_window(p, m, d, w, as_of=AT)


def test_inverse_pair_is_explicit_and_custodian_fallback_requires_actual_application():
    p, m, d, w = sample()
    row = json.loads(d["fixing"]["body"])
    row.update(base_currency="VND", quote_currency="USD", rate="0.00004")
    replace_body(d["fixing"], row)
    assert Decimal(status(p, m, d, w)["fx"]["vnd_cny"]) == Decimal("6.7") * Decimal("0.00004")
    quote = "008764 2026-11-13 香港上海汇丰银行有限公司已提供本基金估值用替代汇率"
    row.update(
        rate_type="CUSTODIAN_APPLIED_FALLBACK",
        reason="CONTRACT_RATE_UNAVAILABLE",
        applied_for_fund_valuation=True,
        application_text=quote,
    )
    d["fixing"].update(kind="CUSTODIAN_FX_REFERENCE", url="https://www.hsbc.com.hk/valuation")
    m["calendar_evidence"]["custodian_application_quote"] = quote
    replace_body(d["fixing"], row)
    assert status(p, m, d, w)["ready"]
    row["applied_for_fund_valuation"] = False
    replace_body(d["fixing"], row)
    assert not status(p, m, d, w)["ready"]


def test_london_summer_fixing_is_not_fixed_utc_or_vietnam_time():
    p, m, d, w = sample()
    p, m, d, w = json.loads(
        json.dumps([p, m, d, w])
        .replace("2026-11-12", "2026-07-09")
        .replace("2026-11-13", "2026-07-10")
        .replace("2026-11-16", "2026-07-13")
    )
    for source in d.values():
        source["body_hash"] = digest(source["body"])
    assert not status(p, m, d, w)["ready"]
    row = json.loads(d["fixing"]["body"])
    row["fixing_at"] = "2026-07-10T16:00:00+01:00"
    replace_body(d["fixing"], row)
    assert status(p, m, d, w)["ready"]


def test_old_generic_policy_does_not_implicitly_admit_vietnam():
    from app.services.qdii_valuation_contract import POLICY as generic_policy

    p, m, d, w = sample()
    m["calendar_evidence"]["policy"] = generic_policy
    assert not verified_qdii_window(p, m, d, w, as_of=AT)


@pytest.mark.parametrize("problem", ["index_change", "stock_change", "borrowed_row"])
def test_daily_percent_and_security_must_match_the_same_actual_record(problem):
    p, m, d, w = sample()
    if problem == "index_change":
        m["market"]["VN30"]["change_pct"] = "999"
    elif problem == "stock_change":
        m["companies"][0]["quote"]["change_pct"] = "-99"
    else:
        payload = json.loads(d["stock"]["body"])
        payload["rows"][1]["symbol"] = "VIC"
        replace_body(d["stock"], payload)
        m["companies"][0]["quote"]["price_quote"] = json.dumps(payload["rows"][1])
    assert not status(p, m, d, w)["market_ready"]


def test_provider_vietnamese_dong_unit_is_preserved_without_faking_english_word():
    p, m, d, w = sample()
    proof = "Giá hiển thị x 1.000 = Giá thực tế. Ví dụ 25.30 tương ứng 25.300 đồng/cp."
    d["units"].update(body=proof, body_hash=digest(proof))
    m["companies"][0]["quote"]["unit_quote"] = proof
    assert status(p, m, d, w)["ready"]


@pytest.mark.parametrize("problem", [None, "raw_changed", "economic_source", "no_receipt", "after_cutoff"])
def test_undated_static_reference_uses_observed_time_and_original_hash(tmp_path, problem):
    from app.services.fund_comprehensive_materials import checked_documents

    raw = "Giá x 1000 đồng".encode()
    path = tmp_path / "unit.raw"
    path.write_bytes(raw)
    reference = doc(
        "unit",
        "PROVIDER_UNIT_EXCERPT",
        raw.decode(),
        "https://banggia.dnse.com.vn/units",
        published_date=None,
        publication_date_status="NOT_STATED",
        http_status=200,
        raw_path=str(path),
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )
    if problem == "raw_changed":
        path.write_bytes(b"changed")
    elif problem == "economic_source":
        reference["kind"] = "FX_FIXING_DATA"
    elif problem == "no_receipt":
        reference["http_status"] = 403
    elif problem == "after_cutoff":
        reference["available_at"] = "2026-11-15T01:00:00+08:00"
    material = {"fund_code": "008764", "documents": [reference]}
    if problem is not None:
        with pytest.raises(ValueError, match="COMPREHENSIVE_SOURCE_INVALID"):
            checked_documents(AT, "008764", material)
    else:
        assert checked_documents(AT, "008764", material)[0]["published_date"] is None


def test_dnse_normalization_binds_raw_bytes_request_and_local_daily_dates():
    from app.services.vietnam_valuation_contract import normalize_dnse_history

    raw = json.dumps({"t": [1794448800, 1794535200], "c": [21.05, 21.05]}).encode()
    original = doc(
        "raw",
        "MARKET_DATA",
        raw.decode(),
        "https://api.dnse.com.vn/chart-api/v2/ohlcs/stock?symbol=HPG&resolution=1D",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )
    normalized = normalize_dnse_history(original, raw, "HPG", at=AT)
    rows = json.loads(normalized["body"])["rows"]
    assert [r["date"] for r in rows] == ["2026-11-12", "2026-11-13"]
    assert all(r["close"] == "21.05" for r in rows), "日线表示不猜千盾单位或擅自缩放"
    with pytest.raises(ValueError, match="RAW_IDENTITY"):
        normalize_dnse_history(original, raw + b" ", "HPG", at=AT)
    with pytest.raises(ValueError, match="RAW_IDENTITY"):
        normalize_dnse_history(original, raw, "VIC", at=AT)


@pytest.mark.parametrize("second_transform", [False, True])
def test_later_normalization_does_not_turn_intraday_raw_into_final_close(second_transform):
    from app.services.vietnam_valuation_contract import normalize_dnse_history

    p, m, d, w = sample()
    raw = json.dumps({"t": [1794448800, 1794535200], "c": [21.05, 21.05]}).encode()
    original = doc(
        "raw",
        "MARKET_DATA",
        raw.decode(),
        "https://api.dnse.com.vn/chart-api/v2/ohlcs/stock?symbol=HPG&resolution=1D",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        available_at="2026-11-13T14:00:00+07:00",
    )
    normalized = normalize_dnse_history(original, raw, "HPG", at=AT)
    if second_transform:
        normalized = normalize_dnse_history(normalized, raw, "HPG", at=AT)
    assert normalized["original_available_at"] == "2026-11-13T14:00:00+07:00"
    d["stock"] = normalized
    q = m["companies"][0]["quote"]
    q["source_ref"] = "stock"
    q["price_quote"] = json.dumps(json.loads(normalized["body"])["rows"][1], sort_keys=True)
    result = status(p, m, d, w)
    assert not result["ready"]
    assert any(g["code"] == "VN_INTRADAY_SOURCE_REJECTED" for g in result["gaps"])


@pytest.mark.parametrize(
    "problem", ["stale", "intraday", "bar_time", "unit", "wrong_provider", "wrong_price", "wrong_security", "nan"]
)
def test_stock_dates_units_and_completed_session_are_real_gates(problem):
    p, m, d, w = sample()
    q = m["companies"][0]["quote"]
    if problem == "stale":
        q["date"] = "2026-10-28"
    elif problem == "intraday":
        d["stock"]["available_at"] = "2026-11-13T14:00:00+07:00"
    elif problem == "bar_time":
        q["closed_at"] = "2026-11-13T09:00:00+07:00"
    elif problem == "unit":
        q.pop("unit_evidence_ref")
    elif problem == "wrong_provider":
        d["units"]["url"] = "https://banggia.vndirect.com.vn/units"
    elif problem == "wrong_price":
        q["close"] = "21.05"
    elif problem == "wrong_security":
        m["companies"][0]["ticker"] = "ACB"
    else:
        q["change_pct"] = "NaN"
    result = status(p, m, d, w)
    assert result["supported"] and result["fx_ready"] and not result["ready"]
    assert any(g.get("security") == "VN000000HPG4" for g in result["gaps"])


@pytest.mark.parametrize("problem", ["own", "fx_clause", "calendar_host", "calendar_digest", "year", "weekend"])
def test_actual_contract_and_reviewed_calendar_cannot_be_waived(problem):
    p, m, d, w = sample()
    if problem == "own":
        d["own"]["codes"] = ["OTHER"]
    elif problem == "fx_clause":
        m["calendar_evidence"]["valuation_quotes"]["fx"] = "采用人民币中间价。"
    elif problem == "calendar_host":
        d["calendar"]["url"] = "https://example.com/hsx"
    elif problem == "calendar_digest":
        parsed = json.loads(d["calendar"]["body"])
        parsed["closed_dates"].append("2026-11-16")
        replace_body(d["calendar"], parsed)
    elif problem == "year":
        w["target_nav_date"] = m["calendar_evidence"]["target_date"] = "2027-01-04"
    else:
        w["target_nav_date"] = m["calendar_evidence"]["target_date"] = "2026-11-14"
    assert not status(p, m, d, w)["supported"]


def test_future_calendar_works_while_fx_is_missing_and_stops_at_actual_mismatch():
    p, m, d, w = sample()
    m["calendar_evidence"].pop("vnd_fx_ref")
    parsed = json.loads(d["calendar"]["body"])
    parsed["closed_dates"].append("2026-11-20")
    replace_body(d["calendar"], parsed)
    m["calendar_evidence"]["calendar_review"].update(
        body_hash=d["calendar"]["body_hash"], closed_dates_hash=digest(parsed["closed_dates"])
    )
    days = tuple(
        date(2026, 11, 12) + timedelta(days=i)
        for i in range(50)
        if (date(2026, 11, 12) + timedelta(days=i)).weekday() < 5
    )
    data = {**m, "as_of": AT.isoformat(), "window": w}
    result, end, _ = future_sessions(data, d, days, date(2026, 12, 31), "cn-hash", at=AT)
    assert end == date(2026, 11, 19) and max(result) == end
    assert status(p, m, d, w)["supported"] and not status(p, m, d, w)["ready"]


def test_general_calendar_route_is_independent_from_missing_nav_and_fx(monkeypatch):
    from app.services import comprehensive_multi_prediction as service
    from app.services import trading_calendar
    from app.services.prediction_contract import target_dates

    p, m, d, w = sample()
    m["calendar_evidence"].pop("vnd_fx_ref")
    days = tuple(
        date(2026, 1, 1) + timedelta(days=i) for i in range(365) if (date(2026, 1, 1) + timedelta(days=i)).weekday() < 5
    )
    calendar = SimpleNamespace(
        sessions=days,
        content_hash="cn-hash",
        definition=SimpleNamespace(coverage_start=date(2026, 1, 1), coverage_end=date(2026, 12, 31)),
    )
    monkeypatch.setattr(trading_calendar, "load_live_calendars", lambda: [calendar])
    monkeypatch.setattr(service, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(None)))
    monkeypatch.setattr(service.repo, "profiles", lambda *args: [p])
    w["calendar_version"] = digest(["cn-hash"])
    data = {
        **m,
        "as_of": AT.isoformat(),
        "window": w,
        "documents": list(d.values()),
        "time_context": {"baseline_complete": False},
    }
    result = service.review_calendar(data, "008764", AT)
    assert target_dates(result, AT, "T20_V1")["endDate"] == "2026-12-11"
    assert target_dates(result, AT, "M6_V1")["endDateStatus"] == "PENDING_OFFICIAL_CALENDAR"
    data.update(nav=[{"nav_date": "2026-11-13"}], latest_nav_date="2026-11-13", valuation_status=status(p, m, d, w))
    data["time_context"] = time_context(data, ["2026-11-12", "2026-11-13", "2026-11-16"])
    assert data["time_context"]["nav_status"] == "READY"
    assert not data["time_context"]["baseline_complete"] and not baseline_ready(data)
    data["time_context"]["baseline_complete"] = True
    assert not baseline_ready(data), "单个baseline布尔值不能跳过指定汇率核验"


@pytest.mark.parametrize("market_ready,nav_ready", [(False, True), (True, False)])
def test_old_and_new_direction_publish_paths_cannot_skip_partial_valuation(market_ready, nav_ready, monkeypatch):
    from app.services import comprehensive_prediction_policy, fund_information_analysis
    from app.services.prediction_review_v2 import validate_review_v2
    from tests.test_comprehensive_prediction import sample as old_sample
    from tests.test_prediction_review_v2 import sample as v2_sample

    packet, data, review = v2_sample()
    data.update(valuation_status={"ready": market_ready}, time_context={"baseline_complete": nav_ready})
    assert packet["source_ready"] is True
    with pytest.raises(ValueError, match="V2_TARGET_OR_SOURCE_UNVERIFIED"):
        validate_review_v2(packet, data, review)
    data, review, facts = old_sample()
    data.update(
        valuation_status={"ready": market_ready}, time_context={"baseline_complete": nav_ready}, fund_code="008764"
    )
    with pytest.raises(ValueError, match="COMPREHENSIVE_VN_VALUATION_INPUTS_REQUIRED"):
        comprehensive_prediction_policy.validate_review(data, review, facts)
    monkeypatch.setattr(fund_information_analysis, "parse_documents", lambda *a, **k: pytest.fail("不应调用付费解析"))
    with pytest.raises(ValueError, match="VN_VALUATION_INPUTS_REQUIRED"):
        fund_information_analysis.analyze(data, None)
