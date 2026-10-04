"""搜索前冻结、后段隔离和共同日期的边界测试；不调用提供商、不进行真实训练。"""

import pytest

from scripts import fund_002112_all_information_search_v1 as search


def test_no_audit_training_before_development_selection(tmp_path):
    search.io.save(tmp_path / "base-freeze.json", {"files": {}})
    with pytest.raises(ValueError, match="SELECTION_MUST_PRECEDE"):
        search.fit(tmp_path, "N", "LR_C1", "T")
    assert not (tmp_path / "fit-ledger.jsonl").exists()


def test_unfrozen_input_cannot_train(tmp_path):
    with pytest.raises(ValueError, match="NOT_FROZEN"):
        search.check_freeze(tmp_path)


def test_changed_frozen_file_rejected(tmp_path):
    path = tmp_path / "source.json"
    path.write_text("original", encoding="utf-8")
    search.io.save(tmp_path / "base-freeze.json", {"files": {str(path): search.io.sha(path)}})
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="FREEZE_CHANGED"):
        search.check_freeze(tmp_path)


def test_ensemble_rejects_different_date_coverage():
    with pytest.raises(ValueError, match="ENSEMBLE_DATES_DIFFER"):
        search.ensemble([[{"target": "2025-01-01"}], [{"target": "2025-01-02"}]])


def test_scoring_rejects_swapped_or_removed_dates():
    with pytest.raises(ValueError, match="COVERAGE_CHANGED"):
        search.score_rows([{"target": "2025-01-01"}], {}, [{"target": "2025-01-02"}])


def test_base_model_does_not_consume_unavailable_semantics():
    rows = [{"groups": {"N": [1, 2], "COUNTS": None, "SEMANTICS": None}}]
    assert search.matrix(rows, "N").tolist() == [[1.0, 2.0]]
