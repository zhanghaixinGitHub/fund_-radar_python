"""第二版资料冻结的隔离验收；不访问真实本人接口或业务库。"""

import hashlib
import json
from copy import deepcopy
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from app.commands import fund_prediction_v2_sources as command
from app.services import prediction_trial_v2 as trial
from app.services.direction_1d_protocol import digest
from app.services.fund_comprehensive_materials import checked_documents
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save
from app.services.fund_prediction_gold_recovery import overnight_observation
from app.services.prediction_contract import PredictionFailure

AT = datetime.fromisoformat("2026-11-12T17:00:00+08:00")
WINDOW = {"base_nav_date": "2026-11-12", "target_nav_date": "2026-11-13", "status": "OPEN"}


def test_same_day_gold_overnight_quote_has_verified_date_time_and_raw_body(tmp_path):
    code = "021740"
    raw_path = tmp_path / "sge-night.html"
    raw = (
        '<h1>上海黄金交易所2026年11月25日延时行情</h1><table>'
        '<tr><th>合约</th><th>最新价</th><th>最高价</th><th>最低价</th><th>今开盘</th></tr>'
        '<tr><td>Au99.99</td><td>765.8</td><td>784.0</td><td>765.8</td><td>780.0</td></tr></table>'
    ).encode()
    raw_path.write_bytes(raw)
    day, received = "2026-11-25", "2026-11-25T03:23:40+08:00"
    body = overnight_observation(raw, day)
    document = {
        "id": "gold-night", "kind": "MARKET_DATA", "codes": [code],
        "url": "https://www.sge.com.cn/sjzx/yshqbg", "published_date": day, "available_at": received,
        "body": body, "body_hash": digest(body), "raw_path": str(raw_path),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "gold_overnight_observation": {"server_response_at": "2026-11-24T19:23:35+00:00"},
    }
    at = datetime.fromisoformat("2026-11-25T03:24:00+08:00")
    assert checked_documents(at, code, {"fund_code": code, "documents": [document]})[0]["body"] == body

    for invalid in (
        {"published_date": "2026-11-26"},
        {"url": "https://www.sge.com.cn.example.org/sjzx/yshqbg"},
        {"gold_overnight_observation": {"server_response_at": "2026-11-24T18:20:00+00:00"}},
        {"gold_overnight_observation": {"server_response_at": "2026-11-24T19:23:42+00:00"}},
        {"body": body.replace("765.8", "765.0"), "body_hash": digest(body.replace("765.8", "765.0"))},
        {"raw_sha256": "0" * 64},
    ):
        with pytest.raises(ValueError, match="COMPREHENSIVE_SOURCE_INVALID"):
            checked_documents(at, code, {"fund_code": code, "documents": [{**document, **invalid}]})


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    material = {"fund_code": "002112", "documents": [], "facts": {}}
    source = tmp_path / "materials.json"
    versioned_save(source, material)
    inventory = {
        "owner_hash": digest("owner"),
        "scope": ["002112"],
        "profiles": [{"fund_code": "002112", "fund_type": "EQUITY"}],
    }
    data = {
        "fund_code": "002112",
        "window": WINDOW,
        "as_of": AT.isoformat(),
        "materials_path": str(source),
        "materials_hash": digest(material),
        "documents": [],
        "facts": {},
        "time_context": {"baseline_complete": True},
    }
    handoff = {
        "materials_path": str(source),
        "materials_hash": digest(material),
        "reviewed": False,
        "gaps": ["旧长期字段待核，本命令不代替第二版审阅"],
    }
    monkeypatch.setattr(command, "local_path", lambda p: Path(p))
    monkeypatch.setattr(trial, "local_path", lambda p: Path(p))
    monkeypatch.setattr(command, "window", lambda at: deepcopy(WINDOW))
    monkeypatch.setattr(command, "registered_materials", lambda *a, **k: deepcopy(handoff))
    monkeypatch.setattr(command, "prepare", lambda at, code, path: deepcopy(data))
    monkeypatch.setattr(command, "review_calendar", lambda *args: object())

    def dates(calendar, at, horizon):
        return {
            "horizonId": horizon,
            "baseNavDate": "2026-11-12",
            "startDate": "2026-11-13",
            "endDate": None if horizon == "M6_V1" else "2026-11-19",
            "nominalEndDate": "2027-05-12" if horizon == "M6_V1" else None,
            "endDateStatus": "PENDING_OFFICIAL_CALENDAR" if horizon == "M6_V1" else "RESOLVED",
            "calendarId": "verified-test-calendar",
            "calendarHash": "verified-test-hash",
        }

    monkeypatch.setattr(command, "target_dates", dates)
    return inventory, data, handoff


def test_freeze_selection_can_directly_prepare_without_v1_review(tmp_path, isolated):
    inventory, _, _ = isolated
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    selection = json.loads(Path(result["selection_path"]).read_text(encoding="utf-8"))
    assert result["directions_generated"] == 0
    guidance = read(Path(result["review_guidance_path"]))
    assert result["review_guidance_hash"] == digest(guidance)
    assert guidance["changes_strict_validation"] is False
    assert result["source_ready_count"] == 1
    assert selection[0]["source_kind"] == "FROZEN"
    source = read(Path(selection[0]["source_path"]))
    assert source["format"] == "CODEX_FUND_INPUT_V1"
    assert source["input_hash"] == digest(source["input"])
    trial.prepare_trial(tmp_path / "trial", Path(result["selection_path"]), holdings=inventory, clock=AT)
    packet = read(tmp_path / "trial/inputs/002112.json")
    assert packet["periods"]["T5_V1"]["status"] == "RESOLVED"
    assert packet["periods"]["M6_V1"]["status"] == "RESOLVED"
    assert packet["periods"]["M6_V1"]["end_date"] == "2027-05-12"
    assert packet["periods"]["M6_V1"]["end_date_basis"] == "CALENDAR_MONTH"
    assert packet["periods"]["M6_V1"]["nominal_end_date"] == "2027-05-12"
    assert "decision" not in source


def test_baseline_gap_is_not_promoted_to_source_ready(tmp_path, isolated):
    inventory, data, _ = isolated
    data["time_context"] = {"baseline_complete": False, "nav": {"status": "MISSING"}}
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    assert result["source_ready_count"] == 0
    assert result["funds"][0]["status"] == "FROZEN_WITH_BASELINE_GAPS"
    trial.prepare_trial(tmp_path / "trial", Path(result["selection_path"]), holdings=inventory, clock=AT)
    packet = read(tmp_path / "trial/inputs/002112.json")
    assert packet["source_ready"] is False
    assert packet["periods"]["T1"]["status"] == "UNVERIFIED"
    assert packet["periods"]["M6_V1"]["status"] == "RESOLVED"
    assert packet["periods"]["M6_V1"]["end_date"] == "2027-05-12"
    assert packet["periods"]["M6_V1"]["end_date_basis"] == "CALENDAR_MONTH"


def test_midnight_freeze_keeps_actual_fund_dates_without_using_older_nav(tmp_path, isolated, monkeypatch):
    from datetime import date, timedelta

    from app.services import prediction_window_v2
    from app.services.prediction_contract import ValuationCalendar, target_dates

    inventory, data, _ = isolated
    at = datetime.fromisoformat("2026-11-12T00:30:00+08:00")
    start, end = date(2026, 11, 9), date(2026, 12, 31)
    domestic = tuple(day for n in range((end - start).days + 1)
                     if (day := start + timedelta(days=n)).weekday() < 5)
    fund_days = tuple(day for day in domestic if day != date(2026, 11, 13))
    calendar = ValuationCalendar("verified-product", fund_days, start, end, "verified")
    monkeypatch.setattr(prediction_window_v2, "calendar", lambda: (domestic, "domestic"))
    monkeypatch.setattr(command, "window", prediction_window_v2.window)
    monkeypatch.setattr(command, "review_calendar", lambda *args: calendar)
    monkeypatch.setattr(command, "target_dates", target_dates)
    data.update(fund_name="测试QDII", nav=[{"nav_date": "2026-11-11"}],
                time_context={"baseline_complete": False, "nav_status": "STALE"})
    monkeypatch.setattr(command, "prepare", lambda at, code, path, *, planning, **kwargs:
                        {**deepcopy(data), "window": planning, "as_of": at.isoformat()})

    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=at)
    trial.prepare_trial(tmp_path / "trial", Path(result["selection_path"]), holdings=inventory, clock=at)
    packet = read(tmp_path / "trial/inputs/002112.json")
    frozen = read(Path(packet["source_path"]))["input"]
    assert packet["source_ready"] is True
    assert "reference_nav_date" not in frozen
    assert frozen["planning_window"]["target_nav_date"] == "2026-11-13"
    assert {p["base_date"] for p in packet["periods"].values()} == {"2026-11-12"}
    assert {p["start_date"] for p in packet["periods"].values()} == {"2026-11-16"}
    assert {h: p["end_date"] for h, p in packet["periods"].items()} == {
        "T1": "2026-11-16", "T5_V1": "2026-11-20", "T20_V1": "2026-12-11", "M6_V1": "2027-05-12",
    }

    for hour in (10, 15, 22):
        assert prediction_window_v2.window(at.replace(hour=hour)) == result["window"]
    assert prediction_window_v2.window(at - timedelta(minutes=31))["target_nav_date"] == "2026-11-12"
    saturday = at.replace(day=14)
    assert prediction_window_v2.window(saturday)["target_nav_date"] == "2026-11-16"
    assert prediction_window_v2.window(saturday)["base_nav_date"] == "2026-11-13"


def test_qdii_prior_nav_keeps_forecast_window_and_anchors_half_year_target(tmp_path, isolated):
    inventory, data, _ = isolated
    data.update(
        fund_name="测试基金(QDII)",
        nav=[{"nav_date": "2026-11-11", "ann_date": "2026-11-12", "source_id": "test", "unit_nav": "1.04"}],
        latest_nav_date="2026-11-11",
        companies=[],
        stock_market_applicable=False,
        market={code: {"date": "2026-11-12", "change_pct": 1.0} for code in ("000300.SH", "000905.SH")},
    )
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    assert result["funds"][0]["reference_nav_date"] == "2026-11-11"
    assert result["funds"][0]["source_ready"] is True
    trial.prepare_trial(tmp_path / "trial", Path(result["selection_path"]), holdings=inventory, clock=AT)
    packet = read(tmp_path / "trial/inputs/002112.json")
    assert packet["periods"]["T1"]["start_date"] == "2026-11-13"
    assert packet["periods"]["M6_V1"]["base_date"] == "2026-11-12"
    assert packet["periods"]["M6_V1"]["end_date"] == "2027-05-11"


@pytest.mark.parametrize(('reference', 'end_date'), [(None, '2027-05-12'), ('2026-11-11', '2027-05-11')])
def test_partial_materials_keep_real_gap_while_resolving_natural_month_target(tmp_path, isolated, reference, end_date):
    inventory, data, _ = isolated
    if reference:
        data["reference_nav_date"] = reference
    source = tmp_path / "actual-partial-materials.json"
    versioned_save(source, data)
    selection = tmp_path / "partial-selection.json"
    selection.write_text(json.dumps([{
        "fund_code": "002112", "source_kind": "PARTIAL_MATERIALS", "source_path": str(source),
        "sample_label": "本期真实缺件资料", "reason": "当前基准及行情尚未全部齐备，仅供缺件审阅",
    }]), encoding="utf-8")
    trial.prepare_trial(tmp_path / "trial", selection, holdings=inventory, clock=AT)
    packet = read(tmp_path / "trial/inputs/002112.json")
    assert packet["source_ready"] is False
    assert packet["periods"]["T1"]["status"] == "UNVERIFIED"
    assert packet["periods"]["M6_V1"]["end_date"] == end_date
    assert packet["periods"]["M6_V1"]["end_date_basis"] == "CALENDAR_MONTH"


def test_v2_freeze_and_trial_share_partial_stock_baseline_boundary(tmp_path, isolated):
    inventory, data, _ = isolated
    data["nav"] = [{"nav_date": "2026-11-12"}]
    data["market"] = {"ETF": {"date": "2026-11-12", "change_pct": -1.2}}
    data["time_context"] = {
        "baseline_complete": False,
        "baseline_date": "2026-11-12",
        "nav_status": "READY",
        "latest_nav_date": "2026-11-12",
        "stocks": {"SUSPENDED": {"date": "2026-10-30", "status": "STALE"}},
        "markets": {"ETF": {"date": "2026-11-12", "status": "READY"}},
    }
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    assert result["source_ready_count"] == 1
    trial.prepare_trial(tmp_path / "trial", Path(result["selection_path"]), holdings=inventory, clock=AT)
    packet = read(tmp_path / "trial/inputs/002112.json")
    assert packet["source_ready"] is True
    original = read(Path(packet["source_path"]))["input"]
    assert original["time_context"]["baseline_complete"] is False
    assert original["time_context"]["stocks"]["SUSPENDED"]["status"] == "STALE"
    assert packet["periods"]["T1"]["status"] == "RESOLVED"


def test_calendar_failure_is_preserved_not_filled_with_weekdays(tmp_path, isolated, monkeypatch):
    inventory, _, _ = isolated

    def missing(*args):
        raise PredictionFailure("CALENDAR_POLICY_MISSING", "CALENDAR", "产品估值条款仍待核验")

    monkeypatch.setattr(command, "review_calendar", missing)
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    for horizon in command.HORIZONS:
        packet = read(tmp_path / "sources/funds/002112" / horizon / "input.json")
        dates = packet["input"]["dates"]
        assert dates["endDateStatus"] == "UNVERIFIED"
        assert dates["endDate"] is None
        assert dates["missingEvidence"]["code"] == "CALENDAR_POLICY_MISSING"
    assert result["funds"][0]["periods"]["M6_V1"]["nominal_end_date"] == "2027-05-12"


def test_prepare_failure_returns_concrete_metadata_and_no_fake_selection(tmp_path, isolated, monkeypatch):
    inventory, _, _ = isolated
    monkeypatch.setattr(
        command, "prepare", lambda *a: (_ for _ in ()).throw(ValueError("COMPREHENSIVE_CALENDAR_POLICY_REQUIRED"))
    )
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    assert result["frozen_count"] == 0
    assert result["status"] == "WAITING_MATERIALS"
    assert result["funds"][0]["failure"]["code"] == "COMPREHENSIVE_CALENDAR_POLICY_REQUIRED"
    assert json.loads(Path(result["selection_path"]).read_text(encoding="utf-8")) == []
    assert not (tmp_path / "sources/funds/002112/T1/input.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"owner_hash": "other"},
        {"policy_hash": "old"},
        {"window": {"base_nav_date": "2026-11-11", "target_nav_date": "2026-11-12"}},
        {"materials_hash": "old-material"},
    ],
)
def test_request_wrong_owner_window_policy_or_material_is_rejected(tmp_path, isolated, change):
    inventory, _, handoff = isolated
    request = {
        "format": "FUND_PREDICTION_V2_DAILY_ANALYSIS_REQUEST",
        "fund_code": "002112",
        "owner_hash": inventory["owner_hash"],
        "window": WINDOW,
        "policy_hash": digest(command.policy_v2()),
        "materials_hash": handoff["materials_hash"],
        "materials_input_hash": handoff["materials_hash"],
        **change,
    }
    path = tmp_path / "request.json"
    versioned_save(path, request)
    if "materials_hash" not in change:
        with pytest.raises(ValueError, match="V2_SOURCE_REQUEST_OWNER_POLICY_OR_WINDOW_MISMATCH"):
            command.freeze_sources(
                tmp_path / "sources", analysis_requests=[path], holdings_provider=lambda: inventory, clock=AT
            )
    else:
        result = command.freeze_sources(
            tmp_path / "sources", analysis_requests=[path], holdings_provider=lambda: inventory, clock=AT
        )
        assert result["funds"][0]["failure"]["code"] == "V2_SOURCE_REQUEST_MATERIALS_CHANGED"


def test_removed_holding_never_produces_selection(tmp_path, isolated):
    inventory, _, _ = isolated
    states = iter([inventory, {**inventory, "scope": []}])
    with pytest.raises(ValueError, match="V2_SOURCE_HOLDINGS_CHANGED_DURING_FREEZE"):
        command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: next(states), clock=AT)
    assert not (tmp_path / "sources/selection.json").exists()


def _qdii_calendar_case(tmp_path, isolated, monkeypatch, error="COMPREHENSIVE_CALENDAR_POLICY_REQUIRED"):
    inventory, _, handoff = isolated
    inventory["profiles"][0].update(fund_name="本基金股票(QDII)", fund_type="STOCK")
    raw = tmp_path / "own-contract.txt"
    raw.write_bytes(b"002112 actual own contract original")
    documents = []
    for ref, kind in (("own-contract", "PROSPECTUS"), ("own-report", "REPORT")):
        body = "本基金002112原件正文，估值日按收盘价及人民币中间价计算。"
        documents.append({
            "id": ref, "kind": kind, "codes": ["002112"], "url": "https://example.com/" + ref,
            "body": body, "body_hash": digest(body), "published_date": "2026-11-11",
            "available_at": "2026-11-11T18:00:00+08:00", "raw_path": str(raw),
            "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        })
    material = {"fund_code": "002112", "documents": documents, "facts": {},
                "calendar_evidence": {"prospectus_ref": "own-contract", "report_ref": "own-report"}}
    versioned_save(Path(handoff["materials_path"]), material)
    handoff["materials_hash"] = digest(material)
    monkeypatch.setattr(command, "prepare", lambda *args: (_ for _ in ()).throw(ValueError(error)))
    return inventory, handoff, material, raw


@pytest.mark.parametrize('calendar_failure', [
    'COMPREHENSIVE_CALENDAR_POLICY_REQUIRED', 'COMPREHENSIVE_VN_CONTRACT_CALENDAR_REQUIRED',
])
@pytest.mark.parametrize(('nav_day', 'identity_matches', 'expected_reference'), [
    (12, True, None), (11, True, '2026-11-11'), (11, False, None),
])
def test_qdii_exact_calendar_failure_prepares_false_partial_with_real_nav_identity(
    tmp_path, isolated, monkeypatch, nav_day, identity_matches, expected_reference, calendar_failure
):
    inventory, handoff, material, _ = _qdii_calendar_case(tmp_path, isolated, monkeypatch, error=calendar_failure)
    statements = []
    from app.services.fund_prediction_nav_identity import nav_identity

    nav_date = date(2026, 11, nav_day)
    nav = {"nav_date": nav_date, "ann_date": nav_date, "source_id": "actual-source", "unit_nav": "2.04980000"}

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement, parameters):
            statements.append(str(statement))
            return self

        def mappings(self):
            return [{**nav, "content_hash": "upstream",
                     "updated_at": AT - timedelta(hours=1), "version_id": None}]

    monkeypatch.setattr("app.db.session.get_nav_preview_engine", lambda: type(
        "ReadOnlyEngine", (), {"connect": lambda self: Connection()})())
    monkeypatch.setattr("app.services.fund_prediction_nav_identity.current_prediction_nav_identity",
                        lambda code, base, *, at: nav_identity([nav] if identity_matches else [], nav_date))
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    assert result["frozen_count"] == 1 and result["source_ready_count"] == 0
    assert result["funds"][0]["status"] == "FROZEN_WITH_CALENDAR_GAPS"
    assert result["funds"][0]["failure"]["code"] == calendar_failure
    selection = json.loads(Path(result["selection_path"]).read_text())
    assert selection[0]["source_kind"] == "PARTIAL_MATERIALS"
    data = read(Path(selection[0]["source_path"]))
    assert data["valuation_status"] == {"ready": False, "reason": calendar_failure}
    assert data["materials_path"] == handoff["materials_path"]
    assert data["materials_hash"] == digest(material)
    assert data["nav"][0]["source_id"] == "actual-source"
    assert data["nav"][0]["nav_date"] == str(nav_date)
    assert data.get("reference_nav_date") == expected_reference
    assert read(Path(handoff["materials_path"])) == material
    assert len(statements) == 1 and statements[0].lstrip().startswith("SELECT")
    trial.prepare_trial(tmp_path / "trial", Path(result["selection_path"]), holdings=inventory, clock=AT)
    packet = read(tmp_path / "trial/inputs/002112.json")
    assert packet["source_ready"] is False
    assert packet["periods"]["T1"]["status"] == "UNVERIFIED"
    assert packet["periods"]["M6_V1"]["end_date"] == ('2027-05-11' if expected_reference else '2027-05-12')
    with pytest.raises(ValueError, match="V2_SOURCE_DIRECTORY_MUST_BE_NEW"):
        command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)


def test_qdii_other_technical_failure_does_not_fall_back(tmp_path, isolated, monkeypatch):
    inventory, _, _, _ = _qdii_calendar_case(tmp_path, isolated, monkeypatch, error="COMPREHENSIVE_SOURCE_INVALID")
    monkeypatch.setattr(command, "_partial_calendar_materials", lambda *args: pytest.fail("unexpected fallback"))
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    assert result["frozen_count"] == 0
    assert result["funds"][0]["failure"]["code"] == "COMPREHENSIVE_SOURCE_INVALID"


def test_qdii_missing_original_is_not_promoted_to_partial(tmp_path, isolated, monkeypatch):
    inventory, _, _, raw = _qdii_calendar_case(tmp_path, isolated, monkeypatch)
    raw.unlink()
    monkeypatch.setattr("app.db.session.get_nav_preview_engine", lambda: pytest.fail("NAV read before original proof"))
    result = command.freeze_sources(tmp_path / "sources", holdings_provider=lambda: inventory, clock=AT)
    assert result["frozen_count"] == 0
    assert result["funds"][0]["failure"]["code"] == "COMPREHENSIVE_CALENDAR_POLICY_REQUIRED"
    assert result["funds"][0]["partial_failure"]["code"] == "SOURCE_FILE_MISSING"
