"""综合模型接入边界：五类输入实际影响、作用可还原、版本变化与时间/范围隔离。"""

import copy
import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from app.services import direction_1d_information as info
from app.services.direction_1d_explanation import explain_original
from app.services.direction_1d_protocol import ZONE, digest
from app.services.fund_exposure_common import save


def fixture():
    names = info.recipe.NAMES["C_EVENTS"]
    n = len(names)
    dimension = 2 * n - 7 + 2
    model = {
        "fund_code": "002112",
        "protocol": "DIRECTION_1D_V2",
        "feature_version": info.FEATURE_VERSION,
        "features": names,
        "classes": ["DOWN", "UP"],
        "unsupported_classes": ["FLAT"],
        "adoption_basis": "EXPLICIT_USER_SELECTION",
        "group_id": info.GROUPS["MORNING_0830"],
        "recipe_hashes": {},
        "medians": [0.0] * n,
        "mean": [0.0] * (2 * n - 7),
        "scale": [1.0] * (2 * n - 7),
        "coef": [0.0] * dimension,
        "intercept": -1.0,
        "tfidf": {"vocabulary": {"政策": 0, "新闻": 1}, "idf": [1.0, 1.0]},
    }
    for name in (
        "return_5d",
        "holding_return_1",
        "ANNOUNCEMENT_5_observed_count",
        "NEWS_5_observed_count",
        "POLICY_5_observed_count",
    ):
        # 行情名称沿用冻结版本，避免测试另外造一套业务字段。
        index = 7 if name == "holding_return_1" else names.index(name)
        model["coef"][index] = 0.5
    model["coef"][-2:] = [0.3, -0.4]
    full = {
        "version": info.FEATURE_VERSION,
        "numeric": [0.0] * n,
        "text": "政策 新闻",
        "market_date": "2026-10-08",
        "holding_report_date": "2026-06-30",
        "holding_coverage": 0.63,
        "counts": {"ANNOUNCEMENT": 1, "NEWS": 1, "POLICY": 1},
        "public_material_through": "2026-10-08",
        "limitations": ["资料覆盖不完整。"],
    }
    return model, full


def test_all_five_input_categories_really_affect_score():
    model, full = fixture()
    info.validate_model(model)
    initial = info.predict(model, full)["class_scores"]["UP"]
    for index in (0, 7, *[model["features"].index(f"{kind}_5_observed_count") for kind in info.recipe.KINDS]):
        changed = copy.deepcopy(full)
        changed["numeric"][index] = 1
        assert info.predict(model, changed)["class_scores"]["UP"] > initial
        assert digest(full) != digest(changed)
    changed = {**full, "text": "新闻"}
    assert info.predict(model, changed)["class_scores"]["UP"] != initial
    assert full["numeric"] == [0.0] * 87


def test_full_explanation_restores_original_and_rejects_corruption():
    model, full = fixture()
    prediction = info.predict(model, full)
    source = {
        "fund_code": "002112",
        "feature_version": info.FEATURE_VERSION,
        "features": full["numeric"][:7],
        "information": full,
    }
    body = {
        "fund_code": "002112",
        "input": source,
        "input_hash": digest(source),
        "base_nav_date": "2026-10-08",
        "target_nav_date": "2026-10-09",
        "branches": [
            {
                "branch_id": b,
                "model_id": "saved",
                "model_hash": "hash",
                "status": "AVAILABLE",
                "score": prediction["score"],
                "predicted_direction": prediction["direction"],
                "class_scores": prediction["class_scores"],
            }
            for b in ("FIXED", "WEEKLY")
        ],
    }
    restored = explain_original(body, lambda *_: model)
    branch = restored["branches"][0]
    margin = branch["intercept"] + sum(f["contribution"] for f in branch["factors"])
    assert abs(1 / (1 + np.exp(-margin)) - prediction["score"]) < 1e-12
    assert set(f["feature"] for f in branch["factors"]) == set(info.FACTOR_NAMES)
    body["branches"][0]["score"] = 0.99
    with pytest.raises(ValueError, match="PREDICTION_RESTORE_MISMATCH"):
        explain_original(body, lambda *_: model)


def test_missing_news_never_becomes_a_claim_of_policy_support():
    model, full = fixture()
    full["counts"].update(NEWS=0, POLICY=0)
    branch = {**info.explain(model, full), "modelHash": "one"}
    prose = info.narrative({"branches": [branch]})
    assert "政策" not in prose["summary"] and "新闻" not in prose["summary"]
    assert "未找到可用的相关政策、新闻" in prose["context"]
    assert "不代表没有" in prose["context"]


def test_selection_only_changes_002112_and_keeps_target_across_midnight(tmp_path, monkeypatch):
    active = tmp_path / "active.json"
    monkeypatch.setattr(info, "ACTIVE_FILE", active)
    assert info.selection("002112", datetime(2026, 10, 8, 23, tzinfo=ZONE)) is None
    save(
        active,
        {
            "fund_code": "002112",
            "feature_version": info.FEATURE_VERSION,
            "enabled": True,
            "models": {phase: {"model_id": phase} for phase in info.GROUPS},
        },
    )
    evening = info.selection("002112", datetime(2026, 10, 8, 23, tzinfo=ZONE))
    morning = info.selection("002112", datetime(2026, 10, 9, 0, tzinfo=ZONE))
    assert evening["phase"] == "EVENING_2300" and morning["phase"] == "MORNING_0830"
    assert info.selection("001412", datetime(2026, 10, 9, 9, tzinfo=ZONE)) is None
    assert info.selection("002112", datetime(2026, 10, 9, 15, tzinfo=ZONE))["phase"] == "EVENING_2300"


def test_receipt_rejects_future_expired_changed_and_outside_files(tmp_path, monkeypatch):
    monkeypatch.setattr(info, "ROOT", tmp_path / "sources")
    info.ROOT.mkdir()
    raw = info.ROOT / "raw.json"
    raw.write_text("original", encoding="utf-8")
    now = datetime(2026, 10, 9, 10, tzinfo=ZONE)
    receipt = {
        "file": "raw.json",
        "sha256": info.recipe.sha(raw),
        "received_at": (now - timedelta(days=1)).isoformat(),
        "expires_at": (now + timedelta(days=1)).isoformat(),
    }
    assert info.receipt_available(receipt, now)
    for key, value in (("received_at", now + timedelta(seconds=1)), ("expires_at", now)):
        with pytest.raises(ValueError, match="NOT_AVAILABLE"):
            info.receipt_available({**receipt, key: value.isoformat()}, now)
    raw.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        info.receipt_available(receipt, now)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1"])
def test_invalid_extended_numbers_rejected(bad):
    model, full = fixture()
    full["numeric"][12] = bad
    with pytest.raises(ValueError, match="INPUT_INVALID"):
        info.predict(model, full)


def test_export_runtime_uses_json_only():
    model, full = fixture()
    restored = json.loads(json.dumps(model))
    assert info.predict(restored, full) == info.predict(model, full)
    restored["fund_code"] = "001412"
    with pytest.raises(ValueError, match="MODEL_INVALID"):
        info.validate_model(restored)


def test_same_announcement_is_not_counted_again_as_downloaded_pdf(tmp_path, monkeypatch):
    """历史事实哈希和PDF哈希不同，也不能使同一来源编号重复入模。"""
    catalog = tmp_path / "catalog.json"
    save(catalog, {"rows": [{"announcementId": "123", "announced_at_source": "2026-10-08"}]})
    monkeypatch.setattr(info, "source_path", lambda *args: catalog)
    source = {"documents": [{"document_id": "company-123", "source_hash": "fact-hash"}]}
    info.add_current_documents(
        source, {"holdings": [{"stock_code": "000001"}]}, datetime(2026, 10, 9, 10, tzinfo=ZONE), {}
    )
    assert len(source["documents"]) == 1
