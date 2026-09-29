"""验证可复用批次的路径边界、主体归属和恢复期间不换计划。"""

import pytest
from app.services.fund_earnings_batch_v3 import EarningsBatch, day_set, issuer_identity, run_path
from app.services.fund_information_history_v1 import save


@pytest.mark.parametrize("name", ["../20260928-earnings-v3", "C:/temp/test", "20260928-v1", "20260928-earnings-v0"])
def test_batch_names_cannot_escape_or_reuse_training_directory(name):
    with pytest.raises(ValueError, match="INVALID_DATA_BATCH_NAME"):
        run_path(name)


def test_distinct_batches_do_not_share_output():
    assert run_path("20260928-earnings-v3") != run_path("20260928-earnings-v4")


def test_explicit_subsidiary_code_blocks_parent_even_if_parent_code_is_mentioned():
    row = {"secCode": "601318", "title_plain": "2022年第一季度报告"}
    pages = ["证券代码：000001 证券简称：平安银行 2022年第一季度报告 母公司601318为重要股东"]
    proof = issuer_identity(row, pages)
    assert not proof["passed"] and proof["reason"] == "EXPLICIT_BODY_ISSUER_DIFFERS_FROM_CATALOG"


def test_own_code_and_title_pass_without_semantic_admission():
    row = {"secCode": "603613", "title_plain": "2022年第一季度业绩预增公告"}
    result = issuer_identity(row, ["证券代码：603613证券简称：国联股份2022年第一季度业绩预增公告"])
    assert result["passed"] and not result["semantic_verified"]


def test_calendar_window_includes_weekends_and_both_bounds():
    assert day_set("2022-01-01", "2022-01-03") == {"2022-01-01", "2022-01-02", "2022-01-03"}


def test_resume_never_replans_or_changes_predecessor(tmp_path, monkeypatch):
    batch = EarningsBatch("20260928-earnings-v3")
    batch.out = tmp_path
    save(tmp_path / "plan.json", {})
    monkeypatch.setattr(batch, "check_plan", lambda: {"previous_run": "C:/data/20260928-earnings-v2", "windows": [1]})
    assert batch.prepare("20260928-earnings-v2")["reused_frozen_plan"]
    with pytest.raises(ValueError, match="PREDECESSOR_CHANGED"):
        batch.prepare("20260928-earnings-v1")
