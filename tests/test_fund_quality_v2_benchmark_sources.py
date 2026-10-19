"""基准原响应复用的拒收边界；合成资料只作工程测试，不能进入产品校准。"""

import hashlib
import json
from datetime import date, datetime

import pytest
from app.services.fund_quality_v2_benchmark_sources import reuse_benchmark_originals
from app.services.fund_quality_v2_source_reuse import reuse_originals


def fixture(tmp_path, issuer="fg"):
    source, target = tmp_path / "originals", tmp_path / "ratings"
    (source / "raw").mkdir(parents=True)
    if issuer == "fg":
        url = "https://www.fullgoal.com.cn/ws-business-server/fund/getIncomeList?productCode=123456"
        params = {"productCode": "123456"}
        body = {"code": 0, "data": [{"productCode": "123456", "navDate": "2026-09-30",
                                     "benchmarkRate": "0", "syNavRate": "0"}]}
        extra = {}
    else:
        url = "https://www.wjasset.com/common-web/ets40/yieldListQueryAPI?fundCode=123456&incomeType=8"
        params = {"fundCode": "123456", "incomeType": "8"}
        body = {"code": "ETS-5BP0000", "results": [{"tradeDate": "20260930",
                                                   "benchmarkRatio": "0", "returnRatio": "0"}]}
        extra = {"format": "ISSUER_DAILY_BENCHMARK_WJ"}
    item = {"fund_code": "123456", "source_kind": "ISSUER_PRODUCT_DAILY_BENCHMARK", "title": "测试原响应非校准",
            "receipt": {"method": "GET", "status": 200, "url": url, "params": params,
                        "file": "raw/response.json", "received_at": "2026-10-17T10:00:00+08:00"}, **extra}
    return source, target, item, body


def persist(source, item, body):
    raw = json.dumps(body).encode()
    (source / "raw/response.json").write_bytes(raw)
    item["receipt"]["sha256"] = hashlib.sha256(raw).hexdigest()
    (source / "quality-benchmark-originals.json").write_text(json.dumps({
        "schema": "QUALITY_BENCHMARK_SOURCE_RECEIPTS_V1", "items": [item]}), encoding="utf-8")
    return raw


@pytest.mark.parametrize("issuer", ["fg", "wj"])
def test_actual_bytes_dates_and_zero_values_are_retained_but_source_is_not_complete_curve_proof(tmp_path, issuer):
    source, target, item, body = fixture(tmp_path, issuer)
    raw = persist(source, item, body)
    refs = reuse_originals("123456", "测试-C", source, target, date(2026, 9, 30),
                           datetime.fromisoformat("2026-10-17T11:00:00+08:00"))
    assert len(refs) == 1 and (target / refs[0]["file"]).read_bytes() == raw
    assert refs[0]["acquired_at"] == item["receipt"]["received_at"]
    assert refs[0]["params"] == item["receipt"]["params"] and refs[0]["label"] == "benchmark"
    assert "published_on" not in refs[0] and "verified" not in refs[0]


@pytest.mark.parametrize("failure", [
    "hash", "foreign_query", "foreign_params", "foreign_row", "business_failure", "duplicate_day", "outside_path",
    "future", "naive_time", "http", "foreign_host", "port", "fragment", "credentials", "post", "status",
    "missing_values", "nan", "bad_number", "nonpositive", "bad_date",
])
def test_success_flag_and_metadata_cannot_replace_original_and_request_identity(tmp_path, failure):
    source, target, item, body = fixture(tmp_path)
    receipt = item["receipt"]
    if failure == "foreign_query":
        receipt["url"] = receipt["url"].replace("123456", "654321")
    elif failure == "foreign_params":
        receipt["params"]["productCode"] = "654321"
    elif failure == "foreign_row":
        body["data"][0]["productCode"] = "654321"
    elif failure == "business_failure":
        body["code"] = 1
    elif failure == "duplicate_day":
        body["data"] *= 2
    elif failure == "outside_path":
        receipt["file"] = "../response.json"
    elif failure == "future":
        receipt["received_at"] = "2026-10-18T00:00:00+08:00"
    elif failure == "naive_time":
        receipt["received_at"] = "2026-10-17T10:00:00"
    elif failure == "http":
        receipt["url"] = receipt["url"].replace("https:", "http:")
    elif failure == "foreign_host":
        receipt["url"] = receipt["url"].replace("www.fullgoal.com.cn", "www.fullgoal.com.cn.example.org")
    elif failure == "port":
        receipt["url"] = receipt["url"].replace(".cn/", ".cn:8443/")
    elif failure == "fragment":
        receipt["url"] += "#another-source"
    elif failure == "credentials":
        receipt["url"] = receipt["url"].replace("https://", "https://test-only@")
    elif failure == "post":
        receipt["method"] = "POST"
    elif failure == "status":
        receipt["status"] = 302
    elif failure == "missing_values":
        body["data"][0].update(benchmarkRate=None, syNavRate=None)
    elif failure in {"nan", "bad_number", "nonpositive"}:
        body["data"][0]["benchmarkRate"] = {"nan": "NaN", "bad_number": "N/A", "nonpositive": "-100"}[failure]
    elif failure == "bad_date":
        body["data"][0]["navDate"] = None
    persist(source, item, body)
    if failure == "hash":
        (source / "raw/response.json").write_bytes(b"changed-test-only")
    with pytest.raises(ValueError):
        reuse_benchmark_originals("123456", source, target, datetime.fromisoformat("2026-10-17T11:00:00+08:00"))
    assert not (target / "sources").exists()


def test_unrelated_product_manifest_is_not_shared_between_a_and_c_shares(tmp_path):
    source, target, item, body = fixture(tmp_path)
    persist(source, item, body)
    assert reuse_benchmark_originals("654321", source, target,
                                    datetime.fromisoformat("2026-10-17T11:00:00+08:00")) == []
