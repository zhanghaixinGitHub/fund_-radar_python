"""份额身份、来源完整性和失败隔离；使用隔离公开资料夹具，不读取实际业务库。"""

import hashlib
from contextlib import contextmanager
from datetime import timedelta

import pytest
from app.integrations.fund_share_class_source import parse_identity, parse_profile
from app.services import fund_share_classes as service
from app.services.fund_exposure_common import now, read
from app.services.fund_materials_store import versioned_save

BODY = """§2 基金产品概况
2.1 基金基本情况
基金主代码 000001
基金管理人 测试基金管理有限公司
下属分级基金的基金简称 测试基金A 测试基金C
下属分级基金的交易代码 000001 000002
报告期末下属分级基金的份额总额 100.00份 200.00份
2.1.1 目标基金基本情况
基金主代码 159999
§3 主要财务指标和基金净值表现
"""


def test_identity_comes_only_from_share_code_row_and_excludes_target_and_holdings():
    assert parse_identity(BODY + "股票代码 600000", "000002") == ("000001", ("000001", "000002"))
    with pytest.raises(ValueError):
        parse_identity(BODY, "159999")
    with pytest.raises(ValueError):
        parse_identity(BODY.replace("000002\n", "000001\n"), "000001")


def test_wrapped_column_labels_do_not_include_front_back_charging_codes():
    wrapped = BODY.replace("下属分级基金的交易代码 000001 000002", """下属分级基金 000001 000002
的交易代码
下属分级基金
的前端交易代 000001
码
下属分级基金
的后端交易代 000003
码""")
    assert parse_identity(wrapped, "000002")[1] == ("000001", "000002")


def test_single_share_is_confirmed_only_by_complete_own_identity():
    single = "基金产品概况\n基金主代码 000001\n基金管理人 测试公司\n主要财务指标"
    assert parse_identity(single, "000001") == ("000001", ("000001",))
    for body, code in ((single, "000002"), (single.split("主要财务")[0], "000001"),
                       (single.replace("基金管理人", "下属分级基金的交易代码"), "000001")):
        with pytest.raises(ValueError):
            parse_identity(body, code)


@pytest.mark.parametrize("name,share,currency", [
    ("测试基金C-CNY", "C", "CNY"), ("测试基金美元现汇A", "A", "美元现汇"),
    ("测试基金人民币C", "C", "人民币"), ("测试ETF", None, None), ("测试REIT", None, None),
])
def test_share_and_currency_labels_do_not_confuse_currency_or_product_acronyms(name, share, currency):
    html = f"<title>{name}(000001)基金基本概况</title>".encode()
    assert parse_profile(html, "000001") == {
        "fundCode": "000001", "fundName": name, "shareClass": share, "currency": currency,
    }
    with pytest.raises(ValueError):
        parse_profile(html, "000002")


class Source:
    def __init__(self, root):
        self.root, self.calls, self.fail = root, [], set()

    def archive(self, value):
        raw = value.encode()
        sha = hashlib.sha256(raw).hexdigest()
        path = self.root / "sources" / (sha + ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return {"file": path.relative_to(self.root).as_posix(), "sha256": sha}

    def catalog(self, code):
        self.calls.append(code)
        if code in self.fail:
            raise ValueError("TEST_SOURCE_UNAVAILABLE")
        return [{"end": "2026-09-30", "published": "2026-10-26", "id": "test"}]

    def report(self, code, entry):
        return BODY, self.archive(BODY)

    def profile(self, code):
        return {"fundCode": code, "fundName": "测试份额" + code,
                "shareClass": "A" if code == "000001" else "C", "currency": None}, self.archive(code)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    @contextmanager
    def lock():
        yield
    monkeypatch.setattr(service, "_sync_lock", lock)
    return tmp_path, Source(tmp_path)


def test_dynamic_scope_reciprocal_lookup_reuse_and_new_fund(isolated, monkeypatch):
    root, source = isolated
    codes = ["000001"]
    monkeypatch.setattr(service, "scope_codes", lambda: tuple(codes))
    assert service.synchronize(root=root, source=source)["verified"] == 1
    assert service.read_relations("000002", root=root)["status"] == "VERIFIED"
    codes.append("000002")
    result = service.synchronize(root=root, source=source)
    assert [item["status"] for item in result["items"]] == ["REUSED", "VERIFIED"]
    assert source.calls == ["000001", "000002"]
    assert len(list((root / "versions").glob("*.json"))) == 2


def test_failure_preserves_old_facts_as_stale_and_new_failure_remains_unknown(isolated):
    root, source = isolated
    service.synchronize(("000001",), root=root, source=source)
    source.fail = {"000001", "000003"}
    result = service.synchronize(("000001", "000003"), root=root, source=source, due_only=False)
    assert result["unverified"] == 2
    assert service.read_relations("000002", root=root)["status"] == "STALE"
    assert service.read_relations("000003", root=root)["shares"] == []


def test_corrupt_original_is_not_served_and_is_not_skipped_as_recent_success(isolated):
    root, source = isolated
    service.synchronize(("000001",), root=root, source=source)
    next((root / "sources").glob("*.txt")).write_bytes(b"corrupt")
    assert service.read_relations("000001", root=root)["status"] == "UNVERIFIED"
    service.synchronize(("000001",), root=root, source=source)
    assert source.calls == ["000001", "000001"]
    assert service.read_relations("000001", root=root)["status"] == "VERIFIED"


def test_preview_does_not_publish_and_conflicting_membership_closes_relation(isolated):
    root, source = isolated
    service.synchronize(("000001",), root=root, source=source, publish=False)
    assert not (root / "current.json").exists()
    service.synchronize(("000001",), root=root, source=source)
    packet = read(root / "current.json")
    packet["checks"]["000002"] = {"error": "ValueError:SHARE_PRODUCT_CONFLICT"}
    versioned_save(root / "current.json", packet)
    assert service.read_relations("000001", root=root)["shares"] == []


def test_future_or_old_check_time_never_looks_current(isolated):
    root, source = isolated
    service.synchronize(("000001",), root=root, source=source)
    packet = read(root / "current.json")
    for age, expected in ((timedelta(days=-1), "UNVERIFIED"), (timedelta(days=36), "STALE")):
        packet["products"]["000001"]["checkedAt"] = (now() - age).isoformat()
        versioned_save(root / "current.json", packet)
        assert service.read_relations("000001", root=root)["status"] == expected


def test_internal_api_requires_service_token_and_never_collects(monkeypatch):
    from app.api.dependencies import require_service_token
    from app.main import app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(service, "read_relations", lambda code: {
        "fundCode": code, "status": "UNVERIFIED", "shares": [],
    })
    client = TestClient(app)
    assert client.get("/internal/v1/funds/000001/share-classes").status_code in (401, 403)
    app.dependency_overrides[require_service_token] = lambda: None
    try:
        response = client.get("/internal/v1/funds/000001/share-classes")
        assert response.status_code == 200 and response.json()["status"] == "UNVERIFIED"
        assert client.get("/internal/v1/funds/abcdef/share-classes").status_code == 422
    finally:
        app.dependency_overrides.pop(require_service_token, None)
