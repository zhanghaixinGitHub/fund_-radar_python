"""Codex独立持仓维护：节奏、动态市场范围、断点复用、历史与失败隔离。"""

import hashlib
import json
from contextlib import contextmanager
from datetime import date

import pytest
from app.integrations.market_fund_reports import MarketReportClient
from app.services import market_holdings_maintenance as maintenance
from app.services.fund_exposure_common import read, save


@contextmanager
def unlocked():
    yield True


@pytest.mark.parametrize(
    "day,last,expected",
    [
        ("2026-10-16", "2026-10-15", True),
        ("2026-10-16", "2026-10-16", False),
        ("2026-03-15", "2026-03-14", True),
        ("2026-08-31", "2026-08-30", True),
        ("2026-05-11", "2026-05-04", True),
        ("2026-05-12", "2026-05-11", False),
        ("2026-05-13", "2026-05-04", True),
        ("2026-05-13", None, True),
    ],
)
def test_disclosure_daily_normal_weekly_and_missed_week_catchup(day, last, expected):
    assert maintenance.due(date.fromisoformat(day), last) is expected


class Client:
    def __init__(self, entries):
        self.entries, self.body_downloads, self.count = entries, 0, 0

    def catalog(self, code):
        self.count += 1
        if code == "000000":
            raise ValueError("SOURCE_FAILED")
        return self.entries

    def body(self, code, entry, previous=None):
        if previous and previous.get("source"):
            return previous["source"]
        self.body_downloads += 1
        return {"sha256": hashlib.sha256(entry["title"].encode()).hexdigest(), "file": "source.json"}


def report(source, entry):
    return {
        "id": source["sha256"],
        "title": entry["title"],
        "endDate": entry["end"],
        "publishedDate": entry["published"],
        "fullDisclosure": False,
        "holdings": [{"stockCode": "000001.SZ", "stockName": "公司", "weightPct": 1, "marketValue": 100}],
        "investmentGroups": [],
        "holdingsPublicationVersion": 1,
    }


@pytest.fixture
def setup(tmp_path, monkeypatch):
    directory, root = tmp_path / "materials", tmp_path / "maintenance"
    monkeypatch.setattr(maintenance, "execution_lock", unlocked)
    monkeypatch.setattr(
        maintenance, "parse_report", lambda client, root, fund, entry, source: (report(source, entry), [])
    )
    entry = {
        "id": "AN202607211827185219",
        "title": "测试基金2026年第2季度报告",
        "end": "2026-06-30",
        "kind": "QUARTER",
        "published": "2026-07-21",
    }
    return maintenance.MarketHoldingsMaintenance(root, directory), Client([entry]), directory, root


def test_second_run_checks_directory_but_never_redownloads_and_adds_new_market_funds(setup, monkeypatch):
    service, client, directory, root = setup
    funds = [{"fund_code": "008888", "fund_name": "测试基金"}]
    monkeypatch.setattr(maintenance, "market_scope", lambda: list(funds))
    assert service.check(force=True, client=client)["published"] == 1
    assert client.body_downloads == 1
    assert service.check(force=True, client=client)["published"] == 0
    assert client.body_downloads == 1
    funds.append({"fund_code": "001021", "fund_name": "另一个市场基金"})
    result = service.check(force=True, client=client)
    assert result["scope_count"] == 2 and result["published"] == 1
    assert read(directory / "008888.json")["reports"]
    assert read(directory / "001021.json")["reports"]


def test_parse_failure_reuses_downloaded_body_and_keeps_previous_reports(setup, monkeypatch):
    service, client, directory, root = setup
    old = {
        "id": "old",
        "title": "测试基金2026年第1季度报告",
        "endDate": "2026-03-31",
        "publishedDate": "2026-04-20",
        "fullDisclosure": False,
        "holdings": [],
        "investmentGroups": [],
    }
    save(directory / "008888.json", {"fundCode": "008888", "reports": [old], "companies": [], "documents": []})
    original = (directory / "008888.json").read_bytes()
    monkeypatch.setattr(maintenance, "parse_report", lambda *args: (_ for _ in ()).throw(ValueError("PARSE_FAILED")))
    funds = [{"fund_code": "008888", "fund_name": "测试基金"}]
    assert service.check(force=True, funds=funds, client=client)["status"] == "PARTIAL_SUCCESS"
    assert (directory / "008888.json").read_bytes() == original
    monkeypatch.setattr(
        maintenance, "parse_report", lambda client, root, fund, entry, source: (report(source, entry), [])
    )
    assert service.check(force=True, funds=funds, client=client)["status"] == "SUCCEEDED"
    assert client.body_downloads == 1
    assert old in read(directory / "008888.json")["reports"]


def test_source_revision_keeps_original_and_fund_failure_does_not_stop_others(setup):
    service, client, directory, root = setup
    funds = [{"fund_code": "000000", "fund_name": "失败基金"}, {"fund_code": "008888", "fund_name": "测试基金"}]
    result = service.check(force=True, funds=funds, client=client)
    assert [f["status"] for f in result["funds"]] == ["FAILED", "SUCCEEDED"]
    original = read(directory / "008888.json")["reports"][0]
    client.entries[0] = {**client.entries[0], "title": client.entries[0]["title"] + "（更正版）"}
    service.check(force=True, funds=funds[1:], client=client)
    assert len(read(directory / "008888.json")["reports"]) == 2
    assert original in read(directory / "008888.json")["reports"]


def test_existing_reports_are_seeded_without_download_and_not_due_never_opens_database(setup, monkeypatch):
    service, client, directory, root = setup
    entry = client.entries[0]
    old = report({"sha256": "known"}, entry)
    save(directory / "008888.json", {"fundCode": "008888", "reports": [old], "companies": [], "documents": []})
    service.check(force=True, funds=[{"fund_code": "008888", "fund_name": "测试基金"}], client=client)
    assert client.body_downloads == 0
    monkeypatch.setattr(maintenance, "market_scope", lambda: (_ for _ in ()).throw(AssertionError("数据库不应连接")))
    assert service.check(client=client)["status"] == "NOT_DUE"


def test_catalog_rejects_cross_fund_rows(tmp_path, monkeypatch):
    client = MarketReportClient(tmp_path)
    value = {
        "ErrCode": 0,
        "PageIndex": 1,
        "TotalCount": 1,
        "Data": [{"FUNDCODE": "001021", "ID": "AN202607211827185219", "TITLE": "测试基金2026年第2季度报告"}],
    }
    monkeypatch.setattr(client, "_get", lambda *args: json.dumps(value).encode())
    try:
        with pytest.raises(ValueError, match="IDENTITY"):
            client.catalog("008888")
    finally:
        client.close()


def test_relative_configuration_paths_are_resolved_before_verifying_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    service = maintenance.MarketHoldingsMaintenance(directory=tmp_path / "materials", root=tmp_path / "cache")
    assert service.directory.is_absolute() and service.root.is_absolute()
    relative = maintenance.MarketHoldingsMaintenance(directory=maintenance.Path("materials"))
    assert relative.directory == tmp_path / "materials"
    assert relative.root == tmp_path / "materials" / "market-maintenance"


def test_newly_published_revision_of_older_period_is_not_filtered_out(setup):
    service, client, directory, root = setup
    old = report({"sha256": "known"}, {**client.entries[0], "published": "2026-08-31"})
    save(directory / "008888.json", {"fundCode": "008888", "reports": [old], "companies": [], "documents": []})
    client.entries[0] = {
        **client.entries[0],
        "end": "2025-12-31",
        "published": "2026-10-15",
        "title": "旧期报告（更正版）",
    }
    result = service.check(force=True, funds=[{"fund_code": "008888", "fund_name": "测试基金"}], client=client)
    assert result["published"] == 1 and client.body_downloads == 1
    assert old in read(directory / "008888.json")["reports"]


def test_retrying_identical_partial_parse_never_republishes_or_rewrites_snapshot(setup, monkeypatch):
    service, client, directory, root = setup
    monkeypatch.setattr(
        maintenance,
        "parse_report",
        lambda client, root, fund, entry, source: (report(source, entry), [{"part": "funds", "reason": "PENDING"}]),
    )
    funds = [{"fund_code": "008888", "fund_name": "测试基金"}]
    assert service.check(force=True, funds=funds, client=client)["published"] == 1
    previous = (directory / "008888.json").read_bytes()
    assert service.check(force=True, funds=funds, client=client)["published"] == 0
    assert client.body_downloads == 1
    assert (directory / "008888.json").read_bytes() == previous


def test_new_article_after_baseline_downloads_even_if_period_and_publication_day_match(setup):
    service, client, directory, root = setup
    funds = [{"fund_code": "008888", "fund_name": "测试基金"}]
    service.check(force=True, funds=funds, client=client)
    original = read(directory / "008888.json")["reports"][0]
    client.entries = [{**client.entries[0], "id": "AN202607211827185220", "title": "同日发布的另一份完整报告"}]
    assert service.check(force=True, funds=funds, client=client)["published"] == 1
    assert client.body_downloads == 2
    assert original in read(directory / "008888.json")["reports"]
