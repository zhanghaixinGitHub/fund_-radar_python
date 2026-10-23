"""管理人净值原件的身份、数值冲突和来源隔离；不访问真实服务或改写净值库。"""

import hashlib
import json
from copy import deepcopy

import pytest
from app.services.direction_1d_protocol import digest
from app.services.fund_material_nav import DISTRIBUTOR_POLICY, POLICY, supplement_nav


def sample(tmp_path):
    body = 'mk_010737_1y="0_1_2_3_4;20261019_0.0778_1.0778_1.0778_-0.19;20261020_0.0715_1.0715_1.0715_-0.58;";'
    raw = tmp_path / "original.js"
    raw.write_text(body, encoding="utf-8")
    doc = {
        "id": "nav-original",
        "kind": "FUND_NAV_DATA",
        "codes": ["010737"],
        "url": "https://cdn.efunds.com.cn/market/2.0/010737_1y.js",
        "body_scope": "FULL_TEXT",
        "body": body,
        "body_hash": digest(body),
        "raw_path": str(raw),
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "receipt_http": 200,
        "available_at": "2026-10-20T20:00:00+08:00",
    }
    profile = {"fund_code": "010737", "manager_name": "易方达基金管理有限公司"}
    material = {"fund_code": "010737", "nav_supplement": {"policy": POLICY, "document_ref": doc["id"]}}
    window = {"base_nav_date": "2026-10-20", "deadline_at": "2026-10-21T15:00:00+08:00"}
    observed = [{"nav_date": "2026-10-19", "unit_nav": "1.0778", "source_id": "provider-id"}]
    return profile, material, {doc["id"]: doc}, observed, window


def test_official_baseline_keeps_distinct_provenance_without_changing_provider_rows(tmp_path):
    args = sample(tmp_path)
    before = deepcopy(args)
    rows, proof = supplement_nav(*args)
    assert args == before
    assert rows[0] == args[3][0]
    assert rows[-1]["unit_nav"] == "1.0715"
    assert rows[-1]["source_id"] is None
    assert rows[-1]["source_code"] == "EFUNDS_OFFICIAL_PUBLIC_DOCUMENT"
    assert rows[-1]["observed_at"] == "2026-10-20T20:00:00+08:00"
    assert rows[-1]["ann_date"] is None
    assert proof["nav_daily_unchanged"] is True


@pytest.mark.parametrize("value", ["1.0715", "1.071500"])
def test_agreeing_same_day_provider_value_is_retained_without_duplicate(tmp_path, value):
    args = sample(tmp_path)
    args[3].append({"nav_date": "2026-10-20", "unit_nav": value, "source_id": "provider-id"})
    rows, proof = supplement_nav(*args)
    assert rows is args[3]
    assert len(rows) == 2 and rows[-1]["source_id"] == "provider-id"
    assert proof["provider_same_day_value_present"] is True


def test_conflicting_provider_value_cannot_be_overwritten(tmp_path):
    args = sample(tmp_path)
    args[3].append({"nav_date": "2026-10-20", "unit_nav": "1.0716"})
    with pytest.raises(ValueError, match="NAV_SOURCE_CONFLICT"):
        supplement_nav(*args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "https://cdn.efunds.com.cn.evil.test/market/2.0/010737_1y.js"),
        ("url", "https://cdn.efunds.com.cn/market/2.0/010736_1y.js"),
        ("url", "https://cdn.efunds.com.cn/market/2.0/010737_1y.js?changed=1"),
        ("codes", ["010736"]),
        ("kind", "MARKET_DATA"),
        ("body_scope", "VERIFIED_EXCERPT"),
        ("receipt_http", 404),
        ("raw_sha256", "0" * 64),
        ("body_hash", "0" * 64),
    ],
)
def test_wrong_identity_or_changed_original_is_rejected(tmp_path, field, value):
    args = sample(tmp_path)
    args[2]["nav-original"][field] = value
    with pytest.raises(ValueError):
        supplement_nav(*args)


@pytest.mark.parametrize("field,value", [("manager_name", "其他管理人"), ("fund_code", "010736")])
def test_other_manager_or_share_class_cannot_use_the_document(tmp_path, field, value):
    args = sample(tmp_path)
    args[0][field] = value
    with pytest.raises(ValueError, match="NAV_PUBLISHER_MISMATCH"):
        supplement_nav(*args)


@pytest.mark.parametrize(
    "old,new,error",
    [
        ("20261020", "20261021", "NAV_BASE_REQUIRED"),
        ("1.0715_1.0715", "NaN_1.0715", "NAV_VALUE_INVALID"),
        ("1.0715_1.0715", "0_1.0715", "NAV_VALUE_INVALID"),
        ("0_1_2_3_4", "0_2_1_3_4", "NAV_FORMAT_INVALID"),
        ("mk_010737", "mk_010736", "NAV_FORMAT_INVALID"),
    ],
)
def test_old_date_unknown_field_order_and_invalid_value_cannot_clear_baseline(tmp_path, old, new, error):
    args = sample(tmp_path)
    doc = args[2]["nav-original"]
    doc["body"] = doc["body"].replace(old, new)
    from pathlib import Path

    raw = Path(doc["raw_path"])
    raw.write_text(doc["body"], encoding="utf-8")
    doc.update(raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(), body_hash=digest(doc["body"]))
    with pytest.raises(ValueError, match=error):
        supplement_nav(*args)


def test_absent_supplement_keeps_existing_readiness_unchanged(tmp_path):
    args = sample(tmp_path)
    args[1].pop("nav_supplement")
    assert supplement_nav(*args) == (args[3], None)


def distributor_sample(tmp_path):
    profile = {"fund_code": "007832", "manager_name": "博道基金管理有限公司", "fund_type": "STOCK"}
    title = "博道伍佰智航股票型证券投资基金"
    clause = "通过指定网站、基金销售机构网站或者营业网点披露开放日的各类基金份额净值和基金份额累计净值。"
    material = {
        "fund_code": "007832",
        "nav_supplement": {
            "policy": DISTRIBUTOR_POLICY,
            "document_ref": "nav",
            "identity_ref": "identity",
            "prospectus_ref": "prospectus",
            "report_ref": "report",
            "disclosure_quote": clause,
            "fund_title": title,
        },
    }
    response = {
        "ErrCode": 0,
        "PageIndex": 1,
        "Data": {
            "LSJZList": [
                {"FSRQ": "2026-10-20", "DWJZ": "1.7426", "LJJZ": "2.0346", "NAVTYPE": "1"},
                {"FSRQ": "2026-10-19", "DWJZ": "1.7293", "LJJZ": "2.0213", "NAVTYPE": "1"},
            ]
        },
    }
    body = json.dumps(response)
    path = tmp_path / "nav.json"
    path.write_text(body, encoding="utf-8")
    docs = {
        "nav": {
            "id": "nav",
            "kind": "FUND_NAV_DATA",
            "codes": ["007832"],
            "body_scope": "FULL_TEXT",
            "receipt_http": 200,
            "body": body,
            "body_hash": digest(body),
            "raw_path": str(path),
            "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "url": "https://api.fund.eastmoney.com/f10/lsjz?fundCode=007832&pageIndex=1&pageSize=5",
            "available_at": "2026-10-20T20:06:00+08:00",
        },
        "identity": {
            "id": "identity",
            "kind": "FUND_DISTRIBUTOR_IDENTITY",
            "codes": ["007832"],
            "url": "https://fund.eastmoney.com/007832.html",
            "body_hash": "identity-hash",
            "body": "007832 上海天天基金销售有限公司 证监会批准 单位净值(2026-10-20)1.7426 累计净值2.0346",
        },
        "prospectus": {
            "id": "prospectus",
            "kind": "PROSPECTUS",
            "codes": ["007832"],
            "url": "https://np-cnotice-fund.eastmoney.com/api/content/ann?art_code=own",
            "body": title + "博道基金管理有限公司" + clause,
            "body_hash": "prospectus-hash",
        },
        "report": {
            "id": "report",
            "kind": "REPORT",
            "codes": ["007832"],
            "body": title + "007832",
            "body_hash": "report-hash",
        },
    }
    observed = [{"nav_date": "2026-10-19", "unit_nav": "1.7293", "source_id": "provider", "content_hash": "previous"}]
    return (
        profile,
        material,
        docs,
        observed,
        {"base_nav_date": "2026-10-20", "deadline_at": "2026-10-21T15:00:00+08:00"},
    )


def test_distributor_requires_own_disclosure_clause_identity_page_and_provider_overlap(tmp_path):
    args = distributor_sample(tmp_path)
    before = deepcopy(args)
    rows, proof = supplement_nav(*args)
    assert args == before and len(rows) == 2
    assert rows[-1]["unit_nav"] == "1.7426" and rows[-1]["source_id"] is None
    assert rows[-1]["source_code"] == "TTFUND_PUBLIC_DISCLOSED_NAV_DOCUMENT"
    assert proof["provider_overlap"] == args[3]
    assert proof["prospectus_ref"] == "prospectus" and proof["identity_ref"] == "identity"


def test_distributor_page_does_not_join_nav_and_following_change_percentage(tmp_path):
    args = distributor_sample(tmp_path)
    args[2]["identity"]["body"] = (
        "007832 上海天天基金销售有限公司 证监会批准\n单位净值\n(\n2026-10-20)\n1.7426\n0.77%\n累计净值\n2.0346"
    )
    rows, _ = supplement_nav(*args)
    assert rows[-1]["unit_nav"] == "1.7426"


def test_fund_name_line_breaks_do_not_change_the_verified_fund_identity(tmp_path):
    args = distributor_sample(tmp_path)
    args[2]["prospectus"]["body"] = args[2]["prospectus"]["body"].replace("伍佰智航", "伍佰\n智航")
    assert supplement_nav(*args)[0][-1]["unit_nav"] == "1.7426"


@pytest.mark.parametrize(
    "problem",
    [
        "clause",
        "other_fund",
        "host",
        "duplicate_query",
        "page_price",
        "old_page",
        "missing_overlap",
        "conflicting_overlap",
        "raw_changed",
        "fund_type",
    ],
)
def test_distributor_cannot_clear_gaps_with_mismatched_proof(tmp_path, problem):
    args = distributor_sample(tmp_path)
    if problem == "clause":
        args[1]["nav_supplement"]["disclosure_quote"] = "无法从原文证明的条款"
    elif problem == "other_fund":
        args[2]["report"]["body"] = "另一基金007831"
    elif problem == "host":
        args[2]["nav"]["url"] = args[2]["nav"]["url"].replace(
            "api.fund.eastmoney.com", "api.fund.eastmoney.com.evil.test"
        )
    elif problem == "duplicate_query":
        args[2]["nav"]["url"] += "&fundCode=007831"
    elif problem == "page_price":
        args[2]["identity"]["body"] = args[2]["identity"]["body"].replace("1.7426", "1.7427")
    elif problem == "old_page":
        args[2]["identity"]["body"] = args[2]["identity"]["body"].replace("2026-10-20", "2026-10-19")
    elif problem == "missing_overlap":
        args[3].clear()
    elif problem == "conflicting_overlap":
        args[3][0]["unit_nav"] = "1.7294"
    elif problem == "raw_changed":
        args[2]["nav"]["body"] += " "
        args[2]["nav"]["body_hash"] = "changed"
    else:
        args[0]["fund_type"] = "MONEY_MARKET"
    with pytest.raises(ValueError):
        supplement_nav(*args)


@pytest.mark.parametrize("problem", ["estimate", "corporate_action", "duplicate_date", "nan", "future_row"])
def test_distributor_nav_keeps_special_events_and_bad_values_blocked(tmp_path, problem):
    from pathlib import Path

    args = distributor_sample(tmp_path)
    doc = args[2]["nav"]
    raw = json.loads(doc["body"])
    rows = raw["Data"]["LSJZList"]
    if problem == "estimate":
        rows[0]["NAVTYPE"] = "2"
    elif problem == "corporate_action":
        rows[0]["FHSP"] = "每份现金分红"
    elif problem == "duplicate_date":
        rows.append(rows[0].copy())
    elif problem == "nan":
        rows[0]["DWJZ"] = "NaN"
    else:
        rows.append({**rows[0], "FSRQ": "2026-10-21"})
    doc["body"] = json.dumps(raw)
    path = Path(doc["raw_path"])
    path.write_text(doc["body"], encoding="utf-8")
    doc.update(raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), body_hash=digest(doc["body"]))
    with pytest.raises(ValueError):
        supplement_nav(*args)
