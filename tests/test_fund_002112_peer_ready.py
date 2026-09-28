"""公开日期协议准入测试；只有合成数据和本地临时原文，不导入模型或访问数据库。"""

from copy import deepcopy

import pytest
from app.services.fund_002112_peer_ready import (
    EvidenceFiles,
    assert_fit_budget,
    correction_conflicts,
    nav_evidence,
    public_nav_rows,
    report_date_check,
    training_time_ok,
)
from app.services.fund_002112_zero_fit_review import file_hash, save_once


def nav_pair():
    frozen = {"date": "2023-03-30", "ann_date": "2023-03-31", "nav": "1.00000000", "source_hash": "a" * 64}
    database = {"nav_date": "2023-03-30", "ann_date": "2023-03-31", "unit_nav": "1.0000", "content_hash": "a" * 64}
    return frozen, database


def test_original_dated_provider_plus_crosscheck_can_pass_without_local_old_download():
    frozen, database = nav_pair()
    result = nav_evidence(frozen, database, "1.0")
    assert result["known_version_public_date"] == "2023-03-31"
    assert not result["independent_historical_capture_verified"]


@pytest.mark.parametrize(
    "field,value", [("ann_date", None), ("ann_date", "2023-03-29"), ("date", "2025-01-02"), ("source_hash", "changed")]
)
def test_unknown_date_scope_or_source_conflict_blocks(field, value):
    frozen, database = nav_pair()
    frozen[field] = value
    with pytest.raises(ValueError):
        nav_evidence(frozen, database, "1")


@pytest.mark.parametrize("value", ["1.000000001", "0", "NaN", "Infinity"])
def test_cross_source_disagreement_or_invalid_value_blocks(value):
    frozen, database = nav_pair()
    with pytest.raises(ValueError, match="NAV_VALUE_CONFLICT"):
        nav_evidence(frozen, database, value)


def test_known_revision_is_never_backdated():
    frozen, database = nav_pair()
    frozen["revised_at"] = "2023-04-20T15:00:00+08:00"
    assert nav_evidence(frozen, database, "1")["known_version_public_date"] == "2023-04-20"
    frozen["revised_at"] = None
    with pytest.raises(ValueError, match="PUBLIC_DATE_UNKNOWN"):
        nav_evidence(frozen, database, "1")


def test_known_revision_without_date_is_not_assumed_original():
    frozen, database = nav_pair()
    frozen["revision_known"] = True
    with pytest.raises(ValueError, match="KNOWN_REVISION_DATE_UNKNOWN"):
        nav_evidence(frozen, database, "1")


def test_native_report_record_and_original_date_agree():
    assert report_date_check("2023-07-21", "2023-07-21", "2023-07-20", "2023-07-21") == "2023-07-21"
    # 有已知修订时采用更晚日期，不把原来的公开日改写成新版的时间。
    assert report_date_check("2023-07-21", "2023-07-21", "2023-07-25", "2023-07-25") == "2023-07-25"


def test_late_pdf_without_explained_native_revision_stops():
    with pytest.raises(ValueError, match="PDF_LATE_MODIFICATION"):
        report_date_check("2023-07-21", "2023-07-21", "2023-07-25")
    with pytest.raises(ValueError, match="REPORT_PUBLICATION_CONFLICT"):
        report_date_check("2023-07-21", "2023-07-20", "2023-07-20")


def test_correction_catalog_does_not_ignore_unknown_or_nav_report_corrections():
    titles = ["关于修订基金合同的公告", "2023年中期报告更正公告", "单位净值更正公告", "勘误公告"]
    assert correction_conflicts([{"title": s} for s in titles]) == titles[1:]
    assert correction_conflicts([{"title": "基金合同及季度报告更正公告"}])


def test_both_label_endpoints_and_known_revision_must_mature():
    row = {"target": "2023-03-29", "label_publication": "2023-03-30", "mature_at": "2023-03-31T08:00:00+08:00"}
    assert training_time_ok(row, "2023-04-01", "2023-03-29")
    assert not training_time_ok(row, "2023-04-01", "2023-03-31")
    assert not training_time_ok(row, "2023-04-01", "2023-04-20")


@pytest.mark.parametrize("budget", [0, -1, None, True, 0.5])
def test_material_ready_cannot_borrow_old_fit_budget(budget):
    with pytest.raises(ValueError, match="NEW_FIT_BUDGET_NOT_AUTHORIZED"):
        assert_fit_budget(budget)


def synthetic_pages(tmp_path):
    receipt = tmp_path / "receipt.json"
    raw = tmp_path / "raw.json"
    save_once(
        raw,
        {
            "PageIndex": 1,
            "TotalCount": 1,
            "ErrCode": 0,
            "Data": {"LSJZList": [{"FSRQ": "2023-03-30", "DWJZ": "1.0000"}]},
        },
    )
    save_once(
        receipt,
        {
            "params": {"fundCode": "160323", "startDate": "2023-03-30", "endDate": "2023-03-30"},
            "path": str(raw),
            "sha256": file_hash(raw),
        },
    )
    return {"fund_code": "160323", "receipts": [str(receipt)], "rows": [{"date": "2023-03-30", "unit_nav": "1.0000"}]}


def test_raw_page_chain_reconstructs_exact_crosscheck(tmp_path):
    cross = synthetic_pages(tmp_path)
    assert public_nav_rows(cross, EvidenceFiles()) == {"2023-03-30": "1.0000"}


def test_summary_value_cannot_replace_raw_response(tmp_path):
    cross = synthetic_pages(tmp_path)
    cross["rows"][0]["unit_nav"] = "2.0000"
    with pytest.raises(ValueError, match="PAGINATION_INCOMPLETE"):
        public_nav_rows(cross, EvidenceFiles())


def test_duplicate_page_is_not_full_history(tmp_path):
    cross = synthetic_pages(tmp_path)
    cross["receipts"] *= 2
    with pytest.raises(ValueError, match="PAGE_INVALID"):
        public_nav_rows(cross, EvidenceFiles())


def test_other_fund_page_is_not_allowed(tmp_path):
    cross = deepcopy(synthetic_pages(tmp_path))
    cross["fund_code"] = "017493"
    with pytest.raises(ValueError, match="REQUEST_SCOPE"):
        public_nav_rows(cross, EvidenceFiles())


def final_package(tmp_path, monkeypatch, finalize=True):
    """最小合成包用于验证最后一道冻结检查；不是训练数据。"""
    from scripts import fund_002112_peer_ready as cli

    monkeypatch.setattr(cli, "PACKAGE", tmp_path)
    sources, decision = tmp_path / "sources.json", tmp_path / "decision.json"
    save_once(sources, {"files": {}})
    save_once(
        decision,
        {
            "material_ready": True,
            "folds": {"Q2": {"gate": {"passed": True}}},
            "train_pool_rows": 1,
            "status": "MATERIAL_READY_AWAIT_NEW_FIT_BUDGET",
        },
    )
    save_once(tmp_path / "ready.json", {"artifact_hashes": {str(p): file_hash(p) for p in (sources, decision)}})
    if finalize:
        audit, code = tmp_path / "independent-audit.json", tmp_path / "code.txt"
        save_once(audit, {"passed": True})
        code.write_text("synthetic verified code", encoding="utf-8")
        save_once(
            tmp_path / "execution-protocol.json",
            {"files": {str(p): file_hash(p) for p in (audit, code)}, "current_execute_budget": 0},
        )
    return cli


def test_independent_audit_and_freeze_required_for_final_readiness(tmp_path, monkeypatch):
    cli = final_package(tmp_path, monkeypatch, finalize=False)
    result = cli.preflight()
    assert result["material_checks_passed"] and not result["material_ready"]


def test_ready_package_still_has_zero_fit_authorization(tmp_path, monkeypatch):
    cli = final_package(tmp_path, monkeypatch)
    result = cli.preflight()
    assert result["material_ready"] and not result["fit_execution_allowed"]
    assert result["current_fit_budget"] == result["actual_new_fits"] == 0


def test_code_changed_after_freeze_blocks_final_preflight(tmp_path, monkeypatch):
    cli = final_package(tmp_path, monkeypatch)
    (tmp_path / "code.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="EXECUTION_PROTOCOL_SOURCE_CHANGED"):
        cli.preflight()


def test_public_fetch_cannot_expand_to_sealed_nav_year():
    from scripts.fund_002112_peer_ready_sources import fetch

    with pytest.raises(ValueError, match="NAV_DATE_SCOPE"):
        fetch(
            "synthetic-forbidden-date",
            "https://api.fund.eastmoney.com/f10/lsjz",
            params={
                "fundCode": "160323",
                "startDate": "2025-01-01",
                "endDate": "2025-01-02",
                "pageIndex": 1,
                "pageSize": 20,
            },
        )


def test_public_fetch_cannot_call_paid_provider_endpoint():
    from scripts.fund_002112_peer_ready_sources import fetch

    with pytest.raises(ValueError, match="PUBLIC_ENDPOINT_NOT_ALLOWED"):
        fetch("synthetic-forbidden-provider", "https://api.tushare.pro", form={})
