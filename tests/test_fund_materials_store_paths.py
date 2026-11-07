"""冻结版本中的审阅归档不能因Windows长路径失败或混淆基金。"""

from app.services import fund_materials_store as store
from app.services.fund_exposure_common import read


def test_nested_prediction_version_keeps_both_old_and_new_review(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PROJECT", tmp_path / "project")
    path = tmp_path / "funds" / "123456" / "T1" / "versions" / ("a" * 64) / "review.json"
    first, second = {"input_hash": "a" * 64, "reading": "原版本"}, {"input_hash": "a" * 64, "reading": "更正版本"}
    store.versioned_save(path, first)
    store.versioned_save(path, second)
    assert read(path) == second
    archives = list((tmp_path / "project/.local-runs/material-version-archives").glob("*.json"))
    assert {read(p)["value"]["reading"] for p in archives} == {"原版本", "更正版本"}
    assert all(read(p)["source_path"] == str(path.resolve()) for p in archives)


def test_long_path_archive_does_not_merge_same_named_fund_reviews(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "PROJECT", tmp_path / "project")
    for code in ("123456", "654321"):
        path = tmp_path / "funds" / code / "T1" / "versions" / ("a" * 64) / "review.json"
        store.versioned_save(path, {"same_value": True})
    archives = list((tmp_path / "project/.local-runs/material-version-archives").glob("*.json"))
    assert len(archives) == 2
    assert {read(p)["source_path"] for p in archives} == {
        str((tmp_path / "funds" / c / "T1" / "versions" / ("a" * 64) / "review.json").resolve())
        for c in ("123456", "654321")
    }
