"""训练就绪以真实日期、同基金身份和完整输入为门槛，不能靠重复份额或 ready 标记绕过。"""

from copy import deepcopy
from datetime import date, timedelta

import pytest
from app.services import fund_training_package as package
from app.services.direction_1d_protocol import digest
from app.services.fund_exposure_common import read, save


def samples():
    rows = []
    for i in range(300):
        day = str(date(2022, 1, 1) + timedelta(days=i))
        rows.append(
            {"fund_code": "002112", "family": "FUND_001412", "target": day, "actual_direction": package.CLASSES[i % 3]}
        )
    return rows


def test_gate_requires_dates_and_each_direction_without_flat_zero_fill():
    rows = samples()
    assert package.eligibility(rows, [{"target": "2024-01-02", "actual_direction": "UP"}])["training_ready"]
    bad = deepcopy(rows)
    for r in bad:
        if r["actual_direction"] == "FLAT":
            r["actual_direction"] = "DOWN"
    assert not package.eligibility(bad, [rows[0]])["training_ready"]
    assert not package.eligibility(rows[:100], [rows[0]])["training_ready"]


def test_duplicate_share_family_is_rejected_instead_of_counted_twice():
    rows = samples()
    with pytest.raises(ValueError, match="DUPLICATE_FAMILY_DATE"):
        package.eligibility(rows + [{**rows[0], "fund_code": "001412"}], [rows[0]])


def test_many_funds_on_one_flat_date_cannot_replace_thirty_market_dates():
    rows = [r for r in samples() if r["actual_direction"] != "FLAT"]
    rows.extend(
        {"fund_code": f"{i:06}", "family": str(i), "target": "2023-01-03", "actual_direction": "FLAT"}
        for i in range(100)
    )
    assert package.summarize(rows)["classes"]["FLAT"] == 100
    assert package.summarize(rows)["class_distinct_dates"]["FLAT"] == 1
    assert not package.eligibility(rows, [rows[0]])["training_ready"]


def test_peer_data_cannot_replace_all_target_fund_history():
    rows = [{**r, "fund_code": "006038", "family": "FUND_090019"} for r in samples()]
    assert not package.eligibility(rows, [rows[0]])["training_ready"]


def test_latest_listing_uses_report_period_and_actual_public_date():
    rows = [
        {"report_end": "2022-12-31", "available_at": "2023-03-31T08:00:00+08:00"},
        {"report_end": "2023-03-31", "available_at": "2023-04-22T08:00:00+08:00"},
        {"report_end": "2022-12-31", "available_at": "2023-05-01T08:00:00+08:00"},
    ]
    assert package.latest_listing(rows, "2023-04-01T08:00:00+08:00") == rows[0]
    assert package.latest_listing(rows, "2023-05-02T08:00:00+08:00") == rows[1]


def test_cohort_requires_strategy_and_own_share_evidence():
    report = {
        "fund_master_code": "090019",
        "title": "大成景恒混合2023年年度报告",
        "product_description_excerpt": "006038 投资目标长期增长 投资策略多因子 业绩比较基准沪深300 风险收益特征",
        "raw": {"sha256": "abc", "url": "https://example.test"},
        "report_end": "2023-12-31",
    }
    assert package.cohort_evidence("006038", [report])["family"] == "FUND_090019"
    for changed in ["其他基金 投资目标 投资策略 业绩比较基准", "006038 没有策略依据"]:
        with pytest.raises(ValueError, match="COHORT"):
            package.cohort_evidence("006038", [{**report, "product_description_excerpt": changed}])


def test_cached_ready_is_not_trusted_without_matching_dataset_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "STORE", tmp_path)
    save(tmp_path / "plan.json", package.PLAN)
    save(tmp_path / "ready.json", {"file": "datasets/abc.json", "sha256": "abc", "training_ready": True})
    save(tmp_path / "datasets/abc.json", {"train": []})
    with pytest.raises(ValueError, match="DATASET_HASH_CHANGED"):
        package.training_inputs()
    assert read(tmp_path / "ready.json")["training_ready"] is True


def test_training_loader_keeps_three_classes_and_common_rows_for_all_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "STORE", tmp_path)
    data = {
        "train": [{"x": list(range(20)), "actual_direction": k} for k in package.CLASSES],
        "development": [{"x": list(range(20)), "actual_direction": "FLAT"}],
        "weights": [1, 1, 1],
    }
    key = digest(data)
    save(tmp_path / f"datasets/{key}.json", data)
    calls = []

    def checked(*args, **kwargs):
        calls.append(kwargs)
        return {"training_ready": True, "file": f"datasets/{key}.json", "sha256": key}

    monkeypatch.setattr(package, "verify", checked)
    for name, size in package.PLAN["variants"].items():
        result = package.training_inputs(name)
        assert len(result["X_train"]) == 3 and len(result["X_train"][0]) == size
        assert result["y_train"] == list(package.CLASSES) and result["training_runs"] == 0
    assert calls == [{"pointer": "ready.json", "publish": False}] * 3
    with pytest.raises(ValueError, match="VARIANT_INVALID"):
        package.training_inputs("UNKNOWN")
