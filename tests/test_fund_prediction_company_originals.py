from copy import deepcopy
from datetime import datetime
from hashlib import sha256
import json
from zoneinfo import ZoneInfo

import pytest

from app.services import fund_prediction_company_originals as service


@pytest.fixture
def source(tmp_path, monkeypatch):
    raw = b"%PDF-test-issuer-300308"
    (tmp_path / "source.pdf").write_bytes(raw)
    stamp = datetime(2026, 11, 17, 16, tzinfo=ZoneInfo("Asia/Shanghai"))
    monkeypatch.setattr("app.integrations.dbfund_supplement.article_text",
                        lambda raw, pdf: (["证券代码300308。实际新增事项全文。"], "TEXT_EXTRACTED"))
    monkeypatch.setattr(service, "window", lambda at: dict(base_nav_date="2026-11-17", target_nav_date="2026-11-18"))
    record = dict(stock_code="300308", announcement_id="123", title="新增实际事项",
        announced_at_source="2026-10-30T00:00:00+08:00",
        receipt=dict(url="https://static.cninfo.com.cn/finalpage/source.PDF", file="source.pdf",
                     sha256=sha256(raw).hexdigest(), received_at=stamp.isoformat()))
    material = dict(fund_code="002112", companies=[dict(code="300308.SZ", financials=[{"existing": True}],
        quote={"date": "2026-11-17"})], documents=[])
    return tmp_path, stamp, record, material


def test_existing_financials_and_current_quote_do_not_hide_new_original(source):
    root, at, record, material = source
    registered = service.register(material, "300308.SZ", record, root=root, raw_root=root, at=at)
    assert registered["status"] == "REGISTERED_UNREVIEWED"
    result = service.refresh(material, root=root, checked_at=at)
    assert result["status"] == "NEW_ORIGINALS_REVIEW_REQUIRED"
    assert len(material["documents"]) == 1
    assert material["documents"][0]["codes"] == ["002112"]
    assert service.refresh(material, root=root, checked_at=at)["status"] == "UNCHANGED"
    assert len(material["documents"]) == 1
    assert service.register(material, "300308.SZ", record, root=root, raw_root=root, at=at)["status"] == "UNCHANGED"


def test_unrelated_company_cannot_be_registered_or_appended(source):
    root, at, record, material = source
    other = deepcopy(material)
    other["companies"] = [dict(code="600000.SH")]
    with pytest.raises(ValueError, match="UNRELATED_SECURITY"):
        service.register(other, "300308.SZ", record, root=root, raw_root=root, at=at)
    service.register(material, "300308.SZ", record, root=root, raw_root=root, at=at)
    assert service.refresh(other, root=root, checked_at=at)["documents_added"] == []
    assert not other["documents"]


def test_changed_bytes_and_future_availability_rejected(source):
    root, at, record, material = source
    bad = deepcopy(record)
    bad["receipt"]["received_at"] = "2026-11-18T00:00:00+08:00"
    with pytest.raises(ValueError, match="ACTUAL_AVAILABILITY"):
        service.register(material, "300308.SZ", bad, root=root, raw_root=root, at=at)


def test_issuer_reply_without_ticker_requires_actual_catalog_binding(source, monkeypatch):
    root, at, record, material = source
    monkeypatch.setattr("app.integrations.dbfund_supplement.article_text",
                        lambda raw, pdf: (["中际旭创股份有限公司，关于实际事项的完整回复。"], "TEXT_EXTRACTED"))
    with pytest.raises(ValueError, match="BODY_ISSUER"):
        service.register(material, "300308.SZ", record, root=root, raw_root=root, at=at)
    original = dict(announcements=[dict(announcementId="123", secCode="300308", secName="中际旭创",
                                      adjunctUrl="finalpage/source.PDF")])
    raw = json.dumps(original, ensure_ascii=False).encode()
    path = root / "catalog.json"
    path.write_bytes(raw)
    record["catalog_proof"] = dict(path=str(path), sha256=sha256(raw).hexdigest())
    assert service.register(material, "300308.SZ", record, root=root, raw_root=root, at=at)["status"] == "REGISTERED_UNREVIEWED"
    bad = deepcopy(record)
    bad["receipt"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SOURCE_HASH"):
        service.register(material, "300308.SZ", bad, root=root, raw_root=root, at=at)
