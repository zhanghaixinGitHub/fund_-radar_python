"""审阅启用的八维口径：资料篡改与缺失拒绝、关键短板及真实隔离库发布。

本文件全部数据都是测试替身，不用于方法准入或真实产品独立验证。
"""

import hashlib
import importlib.util
import json
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.models.fund_rating import RatingBatch, RatingClassification, RatingCurrent, RatingEvidence, RatingMethodology
from app.repositories.fund_rating import RatingConflict, rollback_switch
from app.services import fund_rating_reviewed_quality as quality
from app.services.fund_rating import _category, read
from app.services.fund_rating_inputs import digest
from sqlalchemy.orm import Session

from test_fund_rating_database import db  # noqa: F401


def metrics(basis=quality.ACTIVE_BASIS):
    """用可辨别的原始观测测试分支，不直接填写维度分。"""
    config = quality.policy(basis)
    result = {"target_product_aliases": ["product:target"], "material_event_pending": False, "observations": {}}
    for name in {n for terms in config["terms"].values() for n, _ in terms}:
        if name in {"monthly_success", "annual_success"}:
            n = 36 if name == "monthly_success" else 25
            node = {
                "kind": "windows",
                "expected_windows": n,
                "rows": [{"end": str(i), "target": "2", "reference": "1"} for i in range(n)],
            }
        elif name == "asset_quality":
            node = {
                "kind": "asset_quality",
                "stock_total": "100",
                "positions": [
                    {
                        "security": "fixture",
                        "amount": "100",
                        "state": "NORMAL_REVIEWED",
                        "reason": "测试替身中的已核事实",
                    }
                ],
            }
        elif name == "restricted":
            node = {"kind": "restricted", "restricted_amount": "0", "net_assets": "100", "scope_closed": True}
        elif name == "known_payment":
            node = {"kind": "payment", "cash_principal": "20", "all_book_liabilities": "10", "reason": "测试"}
        elif name in {"allocation", "mandate", "experience", "continuity", "governance", "holders"}:
            node = {"kind": "state", "state": "NORMAL_REVIEWED", "reason": "测试替身中的已核事实"}
        else:
            higher = name.startswith("return_")
            node = {
                "kind": "rank",
                "target": "1",
                "direction": "higher" if higher else "lower",
                "expected_products": ["peer:1"],
                "peers": [{"product_id": "peer:1", "value": "0" if higher else "2"}],
            }
        node["source_refs"] = ["fixture_hash"]
        result["observations"][name] = node
    return result


def test_missing_dimension_never_renormalizes():
    value = metrics()
    del value["observations"]["cost_365"]
    with pytest.raises(ValueError, match="REQUIRED_OBSERVATIONS"):
        quality.calculate(value, config=quality.policy(quality.ACTIVE_BASIS))


def test_cannot_replace_raw_return_with_a_filled_score():
    value = metrics()
    value["observations"]["return_1y"] = {"kind": "state", "state": "CROSS_PERIOD_STRONG", "reason": "伪分数"}
    with pytest.raises(ValueError, match="OBSERVATION_TYPE"):
        quality.calculate(value, config=quality.policy(quality.ACTIVE_BASIS))


def test_missing_peer_and_target_product_in_peer_set_are_rejected():
    value = metrics()
    node = value["observations"]["market_capacity"]
    node["peers"] = []
    with pytest.raises(ValueError, match="PEER_CLOSURE"):
        quality.calculate(value, config=quality.policy(quality.ACTIVE_BASIS))
    node.update(peers=[{"product_id": "product:target", "value": "1"}], expected_products=["product:target"])
    with pytest.raises(ValueError, match="PEER_CLOSURE"):
        quality.calculate(value, config=quality.policy(quality.ACTIVE_BASIS))


def test_unknown_cash_is_not_zero_and_known_gap_not_automatically_normal():
    value = metrics()
    value["observations"]["known_payment"]["cash_principal"] = None
    with pytest.raises(ValueError, match="VALUE_MISSING"):
        quality.calculate(value, config=quality.policy(quality.ACTIVE_BASIS))
    value["observations"]["known_payment"]["cash_principal"] = "1"
    with pytest.raises(ValueError, match="PAYMENT_GAP_UNRESOLVED"):
        quality.calculate(value, config=quality.policy(quality.ACTIVE_BASIS))


def test_short_monthly_data_cannot_replace_36_months():
    value = metrics()
    value["observations"]["monthly_success"]["rows"].pop()
    with pytest.raises(ValueError, match="WINDOW_CLOSURE"):
        quality.calculate(value, config=quality.policy(quality.ACTIVE_BASIS))


def test_cap_changes_grade_but_never_deducts_score_again():
    value = metrics()
    before = quality.calculate(value, config=quality.policy(quality.ACTIVE_BASIS))
    for name in ["extra_drawdown", "extra_downside"]:
        value["observations"][name]["peers"][0]["value"] = "0"
    after = quality.calculate(value, config=quality.policy(quality.ACTIVE_BASIS))
    assert after["raw_grade"] == "GOOD" and after["grade"] == "AVERAGE"
    assert float(before["score"]) - float(after["score"]) == 20
    assert after["grade_cap_reasons"] == ["KEY_DIMENSION_WEAK:risk"]


def test_passive_keeps_tracking_and_replication_instead_of_active_concentration():
    value = metrics(quality.PASSIVE_BASIS)
    config = quality.policy(quality.PASSIVE_BASIS)
    assert config["weights"]["efficiency"] == 25 and config["weights"]["cost"] == 15
    assert "issuer_concentration" not in value["observations"]
    assert "issuer_replication" in value["observations"]
    assert quality.calculate(value, config=config)["grade"]


def packet(tmp_path, monkeypatch):
    """临时文件和隔离库专用原件替身；永不进入真实资料目录。"""
    root = tmp_path / "ratings"
    (root / "sources").mkdir(parents=True)
    raw = b'{"fund_code":"002112","legal_product":"001412","fixture":true}'
    sha = hashlib.sha256(raw).hexdigest()
    path = root / "sources" / (sha + ".json")
    path.write_bytes(raw)
    monkeypatch.setattr(
        quality,
        "get_settings",
        lambda: SimpleNamespace(
            app_environment="local", fund_insight_directory=str(tmp_path), tushare_api_url="https://api.tushare.pro"
        ),
    )
    m = metrics()
    for node in m["observations"].values():
        node["source_refs"] = [sha]
    for name in ["extra_drawdown", "extra_downside"]:
        m["observations"][name]["peers"][0]["value"] = "0"
    now = datetime.now(UTC).isoformat()
    view = {
        "summary": "隔离测试替身",
        "comparison": {
            "category": "测试",
            "currency": "CNY",
            "productCount": 0,
            "scope": "仅隔离测试",
            "period": "2023-09-28 至 2026-09-30",
        },
        "dimensions": [
            {
                "key": k,
                "name": k,
                "grade": None,
                "gradeLabel": None,
                "explanation": "测试",
                "metrics": [],
                "sources": [],
            }
            for k in quality.policy(quality.ACTIVE_BASIS)["weights"]
        ],
        "strengths": [],
        "weaknesses": [],
        "dataDates": {"nav": "2026-09-30", "holdings": "2026-06-30", "manager": "2026-06-30", "fees": "2026-06-30"},
        "limitations": ["测试替身"],
    }
    return {
        "basis": quality.ACTIVE_BASIS,
        "source_scope": "PUBLIC_DISCLOSURE_LOCAL_ANALYSIS",
        "fund_code": "002112",
        "family": "ACTIVE_EQUITY",
        "currency": "CNY",
        "found_date": "2015-06-19",
        "as_of_date": "2026-09-30",
        "verified_at": now,
        "sources": [],
        "quality_metrics": m,
        "quality_public": view,
        "classification": {
            "product_id": "product:target",
            "relationship_verified": True,
            "source_refs": [sha],
            "effective_from": "2015-06-19",
            "identity_bindings": [{"source_ref": sha, "required_text": ["002112", "001412"]}],
        },
        "raw_archive": [
            {
                "file": "sources/" + path.name,
                "sha256": sha,
                "kind": "ORIGINAL",
                "url": "https://www.dbfund.com.cn/fixture.json",
                "published_on": "2026-08-31",
                "acquired_at": now,
            }
        ],
    }


def test_source_tampering_and_wrong_product_identity_fail(tmp_path, monkeypatch):
    value = packet(tmp_path, monkeypatch)
    quality.validate_sources(value, datetime.now(UTC))
    bad = deepcopy(value)
    bad["classification"]["identity_bindings"][0]["required_text"].append("999999")
    with pytest.raises(ValueError, match="IDENTITY_ORIGINAL_MISMATCH"):
        quality.validate_sources(bad, datetime.now(UTC))
    source = tmp_path / "ratings" / value["raw_archive"][0]["file"]
    source.write_bytes(b"modified")
    with pytest.raises(ValueError, match="SOURCE_HASH"):
        quality.validate_sources(value, datetime.now(UTC))


def test_postgres_publish_capped_grade_readback_idempotency(db, tmp_path, monkeypatch):  # noqa: F811
    """真实PG仅使用随机schema；验证等级上限约束、发布、读回与重复调用。"""
    value = packet(tmp_path, monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "quality_caps_migration", Path("alembic/versions/20261020_37_rating_quality_grade_caps.py")
    )
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    with db.begin() as connection, Operations.context(MigrationContext.configure(connection)):
        migration.upgrade()
    config = quality.policy(quality.ACTIVE_BASIS)
    category = digest({"basis": quality.ACTIVE_BASIS, "currency": "CNY"})
    with Session(db) as session, session.begin():
        key = quality.ingest(session, value)
        session.add(
            RatingMethodology(
                methodology_id=digest(config),
                family="ACTIVE_EQUITY",
                config=config,
                active=True,
                created_at=datetime.now(UTC),
                validation={
                    "fixture": "isolated only",
                    "report": {"categories": [{"category": category, "eligible": True}]},
                },
            )
        )
    m, facts, expiry = quality.build_metrics(value, date(2026, 9, 30), datetime.now(UTC))
    members = [
        {
            "fund_code": "002112",
            "product_id": "product:target",
            "family": "ACTIVE_EQUITY",
            "representative": True,
            "admitted": True,
            "reasons": [],
            "metrics": m,
            "facts": facts,
            "valid_until": expiry,
            "evidence_hash": key,
            "bundle": value,
        }
    ]
    first = _category(db, tmp_path / "ratings", category, members, date(2026, 9, 30), datetime.now(UTC))
    again = _category(db, tmp_path / "ratings", category, members, date(2026, 9, 30), datetime.now(UTC))
    assert first["batch"] == again["batch"]
    response = read(["002112"], detail=True, rating_ref=first["batch"], engine=db)
    assert response["grade"] == "AVERAGE" and len(response["dimensions"]) == 8
    from sqlalchemy import text as sql_text

    with Session(db) as session:
        stored = session.execute(sql_text(
            "SELECT score FROM fund_rating_result WHERE batch_id=:batch AND fund_code='002112'"
        ), {"batch": first["batch"]}).scalar_one()
    assert response["score"] == f"{stored:.2f}"
    assert all(d["score"] is not None for d in response["dimensions"])
    assert json.dumps(response, ensure_ascii=False)
    from app.services.fund_rating_admin import ingest

    with Session(db) as session, session.begin(), pytest.raises(ValueError, match="RATING_REVIEWED_INPUT_REQUIRED"):
        ingest(session, {"fund_code": "002112", "basis": "PUBLIC_DISCLOSURE_EQUITY_V1"})
    legacy_member = deepcopy(members[0])
    legacy_member.update(
        admitted=False,
        representative=False,
        metrics={},
        evidence_hash=None,
        bundle={},
        public=value["quality_public"],
        reasons=["NO_NEW_REVIEWED_DOSSIER"],
    )
    with pytest.raises(ValueError, match="RATING_REVIEWED_INPUT_REQUIRED"):
        _category(db, tmp_path / "ratings", "legacy-unverified", [legacy_member], date(2026, 9, 30), datetime.now(UTC))
    with Session(db) as session:
        assert session.get(RatingCurrent, category).batch_id == first["batch"]
        assert session.get(RatingEvidence, key).revoked is False
    # 模拟相同资料形成的旧类别，实际切换到新类别后，再原子恢复旧指针。
    old_category = "old-reviewed-" + uuid4().hex
    with Session(db) as session, session.begin():
        method = session.get(RatingMethodology, digest(config))
        validation = deepcopy(method.validation)
        validation["report"]["categories"].append({"category": old_category, "eligible": True})
        method.validation = validation
    old = _category(db, tmp_path / "ratings", old_category, members, date(2026, 9, 30), datetime.now(UTC))
    with Session(db) as session, session.begin():
        rollback_switch(session, first["batch"], old["batch"], "隔离测试跨口径恢复")
    with Session(db) as session:
        assert session.get(RatingCurrent, category) is None
        assert session.get(RatingCurrent, old_category).batch_id == old["batch"]
        assert session.get(RatingBatch, first["batch"]) is not None
    with Session(db) as session, session.begin(), pytest.raises(RatingConflict, match="RATING_SWITCH_CURRENT_CHANGED"):
        rollback_switch(session, first["batch"], old["batch"], "拒绝重复撤销过期指针")


def test_policy_survives_json_snapshot_roundtrip():
    config = quality.policy(quality.ACTIVE_BASIS)
    assert quality.calculate(metrics(), config=json.loads(json.dumps(config))) == quality.calculate(
        metrics(), config=config
    )


def test_ingest_reuses_existing_legal_identity_without_revocation(db, tmp_path, monkeypatch):  # noqa: F811
    value = packet(tmp_path, monkeypatch)
    key = "a" * 64
    with Session(db) as session, session.begin():
        session.add(
            RatingClassification(
                classification_id=key,
                fund_code="002112",
                product_id="product:target",
                category_code="old-category",
                family="ACTIVE_EQUITY",
                currency="CNY",
                eligibility="VERIFIED",
                effective_from=date(2015, 6, 19),
                source_refs=[],
                verified_at=datetime.now(UTC),
                revoked=False,
            )
        )
    with Session(db) as session, session.begin():
        evidence = quality.ingest(session, value)
    with Session(db) as session:
        assert session.get(RatingEvidence, evidence).classification_id == key
        assert session.get(RatingClassification, key).revoked is False
        assert len(session.query(RatingClassification).all()) == 1
