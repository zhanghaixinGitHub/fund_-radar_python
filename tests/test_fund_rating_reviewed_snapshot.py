"""部分评分的真实性、缺项、过期、篡改及真实PostgreSQL事务边界。"""

import hashlib
import json
import os
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, getcontext
from types import SimpleNamespace
from uuid import uuid4

import pytest
from app.core.config import get_settings
from app.db.base import Base
from app.models.fund import FundMaster, FundShareClass
from app.models.fund_rating import RatingBatch, RatingClassification, RatingCurrent, RatingMember, RatingResult
from app.services import fund_rating_reviewed_snapshot as snapshots
from app.services.fund_rating import read
from app.services.fund_rating_inputs import digest
from app.services.fund_rating_money_public import concentration
from app.services.fund_rating_reviewed_snapshot import (
    BASIS,
    KEYS,
    archive_root,
    calculate,
    public_evidence,
    publish_reviewed,
    validate_sources,
)
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session


@pytest.fixture
def dossier(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "fund_insight_directory", str(tmp_path))
    root = archive_root()
    (root / "sources").mkdir(parents=True)

    def archive(name, data, kind="json"):
        content = json.dumps(data, ensure_ascii=False).encode() if kind == "json" else data
        path = root / "sources" / name
        path.write_bytes(content)
        return {"file": "sources/" + name, "sha256": hashlib.sha256(content).hexdigest(), "kind": kind}

    raw = archive("report.pdf", b"%PDF-isolated-fixture", "pdf")
    identity = archive(
        "identity.json",
        {
            "code": "000009",
            "identity_verified": True,
            "published_on": "2026-08-31",
            "title": "隔离测试基金2026年中期报告",
            "pages": ["基金管理人：隔离测试管理人 000009"],
            "source": {"sha256": raw["sha256"]},
        },
    )
    doc = {
        "code": "000009",
        "as_of": "2026-09-30",
        "outer_weights_percent": dict(
            zip(
                ["return", "risk", "execution", "stability", "management", "holdings", "liquidity", "fee"],
                [15, 20, 10, 10, 15, 10, 10, 10],
                strict=True,
            )
        ),
        "dimensions": {
            k: "75" for k in ["return", "risk", "execution", "stability", "management", "holdings", "liquidity", "fee"]
        },
    }
    doc["dimensions"]["execution"] = None
    research = archive("research.json", doc)
    entry = {
        "fund_code": "000009",
        "research_archive": research,
        "research_format": "single",
        "identity_archive": identity,
        "original_report_archive": raw,
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "verified_at": datetime.now(UTC).isoformat(),
        "family": "MONEY",
        "share_class": "A",
        "public_name": "隔离测试基金A",
        "identity": {"legal_name": "隔离测试基金", "manager_name": "隔离测试管理人"},
        "report_title": "隔离原件",
        "report_url": "https://example.org/report.pdf",
        "report_published_on": "2026-08-31",
        "extra_archives": [],
    }
    return {"basis": BASIS, "entries": [entry]}, root, doc


def bundle_from(dossier):
    data, _, doc = dossier
    return {**deepcopy(data["entries"][0]), "basis": BASIS, "reviewed_document": deepcopy(doc)}


def test_missing_dimension_preserved_no_full_score(dossier):
    computed = calculate(bundle_from(dossier))
    assert computed["status"] == "PARTIAL" and computed["point_weight"] == "90"
    assert Decimal(computed["partial_score"]) == 75 and computed["score"] is None and computed["grade"] is None
    assert computed["missing_weight"] == "10" and not computed["has_full_informative_interval"]


def test_list_verification_reuses_published_result_without_replaying_originals(dossier, monkeypatch):
    """发布默认仍完整复算；列表只读冻结结果，但输入摘要和八维结构仍须完整。"""
    bundle = bundle_from(dossier)
    computed = calculate(bundle)
    code = bundle["fund_code"]
    batch = SimpleNamespace(input_snapshot={
        "method": {"approved_artifacts": {code: bundle["research_archive"]["sha256"]}},
        "members": [{"fund_code": code, "bundle": bundle, "evidence_hash": digest(bundle)}],
    })
    result = SimpleNamespace(
        fund_code=code,
        status=computed["status"],
        score=Decimal(computed["score"]) if computed["score"] is not None else None,
        grade=None,
        dimension_values={KEYS[key]: value["candidate"] for key, value in computed["dimensions"].items()},
        public_evidence=public_evidence(bundle, computed),
    )

    def unexpected_replay(_bundle):
        raise RuntimeError("unexpected original replay")

    monkeypatch.setattr(snapshots, "calculate", unexpected_replay)
    with pytest.raises(RuntimeError, match="unexpected original replay"):
        snapshots.verify_result(result, batch)
    snapshots.verify_result(result, batch, replay=False)
    bundle["public_name"] = "篡改后的名称"
    with pytest.raises(ValueError, match="RATING_SNAPSHOT_BUNDLE_HASH"):
        snapshots.verify_result(result, batch, replay=False)


def test_list_input_validation_does_not_replay_public_sources(dossier, monkeypatch):
    """证据哈希和撤回状态逐次核验，公开原件只在发布路径重新读取。"""
    from app.repositories.fund_rating import RatingValidationRows, inputs_valid
    from app.services import fund_rating_public_inputs as public_inputs

    bundle = bundle_from(dossier)
    evidence_hash = digest(bundle)
    evidence = SimpleNamespace(payload=bundle, evidence_hash=evidence_hash, revoked=False)
    classification = SimpleNamespace(revoked=False)
    member = SimpleNamespace(evidence_hash=evidence_hash)
    batch = SimpleNamespace(batch_id="batch")
    context = RatingValidationRows(
        members={"batch": [member]},
        results={},
        evidence={evidence_hash: (evidence, classification)},
        enabled_sources=set(),
    )

    def unexpected_replay(*_args, **_kwargs):
        raise RuntimeError("unexpected source replay")

    monkeypatch.setattr(public_inputs, "validate_public_sources", unexpected_replay)
    with pytest.raises(RuntimeError, match="unexpected source replay"):
        inputs_valid(None, batch, context=context)
    assert inputs_valid(None, batch, context=context, replay_sources=False)


@pytest.mark.parametrize("field", ["active_execution_calculation", "stability_calculation", "passive_performance_calculation",
                                  "gold_components_calculation", "active_broad_components_calculation"])
@pytest.mark.parametrize("damage", [None, "inline_changed", "archive_changed"])
def test_new_component_original_input_archive_must_match_embedded_inputs(dossier, field, damage):
    """批次内嵌原始输入与不可变归档必须一致，不能只检查研究分数表。"""
    from app.services.fund_rating_pipeline import archive_json

    _, root, _ = dossier
    value = bundle_from(dossier)
    raw = {"fund_code": value["fund_code"], "isolated_observation": "original"}
    reference = archive_json(root, raw)
    value[field] = {"inputs": raw, "input_archive": reference}
    if damage == "inline_changed":
        raw["isolated_observation"] = "changed_without_new_archive"
    elif damage == "archive_changed":
        (root / reference["file"]).write_bytes(b'{"tampered":true}')
    if damage:
        with pytest.raises(ValueError, match="COMPONENT_INPUT_ARCHIVE_MISMATCH|ARCHIVE_HASH"):
            validate_sources(value, datetime.now(UTC))
    else:
        validate_sources(value, datetime.now(UTC))


@pytest.mark.parametrize("field,kind,changed_ref,limit", [
    ("active_broad_components_calculation", "DERIVED_MANDATE_REVIEW", False, 50_000_000),
    ("active_broad_components_calculation", "DERIVED_MANDATE_REVIEW", True, 25_000_000),
    ("active_broad_components_calculation", "pdf", False, 25_000_000),
    ("mixed_performance_calculation", "pdf", False, "MIXED_REGISTERED_ORIGINAL_VERIFIER"),
])
def test_typed_large_source_limit_keeps_exact_binding_and_ordinary_25mb(dossier, monkeypatch, field, kind, changed_ref, limit):
    """仅验证大小边界分派；读回、SHA和路径本身由其余真实隔离文件反例覆盖。"""
    from app.services import fund_rating_reviewed_snapshot as implementation
    from app.services.fund_rating_pipeline import archive_json

    _, root, _ = dossier
    value = bundle_from(dossier)
    reference = {"sha256": "typed-large-fixture", "file": "sources/large-fixture", "kind": kind,
                 "archive_kind": "RESEARCH" if kind.startswith("DERIVED") else "ORIGINAL"}
    raw = {"sources": {reference["sha256"]: reference}}
    value[field] = {"inputs": raw, "input_archive": archive_json(root, raw)}
    value["extra_archives"] = [{**reference, **({"role": "different-metadata"} if changed_ref else {})}]
    original_read = implementation._read
    seen = []
    if field == "mixed_performance_calculation":
        # mixed维持专属身份/50MB原件验证器，不降级成通用_read豁免。
        from app.services import fund_rating_mixed_performance as mixed

        monkeypatch.setattr(mixed, "verify_registered_original", lambda *args, **kwargs: seen.append(limit))

    def read(ref, archive_root, *, max_bytes=25_000_000):
        if ref["sha256"] == reference["sha256"]:
            seen.append(max_bytes)
            return None
        return original_read(ref, archive_root, max_bytes=max_bytes)

    monkeypatch.setattr(implementation, "_read", read)
    implementation.validate_sources(value, datetime.now(UTC))
    assert seen == [limit]


def test_execution_originals_replayed_and_all_points_still_require_full_r5_package(dossier, monkeypatch):
    from app.services import fund_rating_passive_execution

    bundle = bundle_from(dossier)
    bundle['family'] = 'INDEX'
    bundle['reviewed_document']['dimensions']['execution'] = '75'
    raw = {'fund_code': bundle['fund_code'], 'as_of_date': bundle['as_of_date'],
           'holding_date': bundle['holding_date'], 'legal_name': bundle['identity']['legal_name'],
           'primary_ref': bundle['original_report_archive']['sha256']}
    bundle['passive_execution_calculation'] = {'inputs': raw}
    monkeypatch.setattr(fund_rating_passive_execution, 'calculate', lambda *a, **k: {'score': '75'})
    result = calculate(bundle)
    assert result['point_dimensions'] == 8 and result['status'] == 'PARTIAL' and result['score'] is None
    raw['fund_code'] = '999999'
    with pytest.raises(ValueError, match='EXECUTION_INPUT_SCOPE'):
        calculate(bundle)
    raw['fund_code'] = bundle['fund_code']
    monkeypatch.setattr(fund_rating_passive_execution, 'calculate', lambda *a, **k: {'score': '76'})
    with pytest.raises(ValueError, match='EXECUTION_DIMENSION_RECALCULATION'):
        calculate(bundle)


def test_fee_supplement_rejects_wrong_share_and_changed_point(dossier, monkeypatch):
    from app.services import fund_rating_fee_inputs

    bundle = bundle_from(dossier)
    raw = {'fund_code': bundle['fund_code'], 'as_of_date': bundle['as_of_date'],
           'share_class': bundle['share_class'], 'legal_name': bundle['identity']['legal_name'],
           'primary_ref': bundle['original_report_archive']['sha256']}
    bundle['fee_calculation'] = {'inputs': raw}
    monkeypatch.setattr(fund_rating_fee_inputs, 'calculate', lambda *a, **k: {'score': '75'})
    result = calculate(bundle)
    assert result['status'] == 'PARTIAL' and result['score'] is None
    raw['share_class'] = 'C'
    with pytest.raises(ValueError, match='FEE_INPUT_SCOPE'):
        calculate(bundle)
    raw['share_class'] = bundle['share_class']
    monkeypatch.setattr(fund_rating_fee_inputs, 'calculate', lambda *a, **k: {'score': '76'})
    with pytest.raises(ValueError, match='FEE_DIMENSION_RECALCULATION'):
        calculate(bundle)


def test_passive_holdings_supplement_replays_originals_and_cannot_create_full_score(dossier, monkeypatch):
    from app.services import fund_rating_passive_holdings

    bundle = bundle_from(dossier)
    bundle["family"] = "INDEX"
    raw = {"fund_code": bundle["fund_code"], "as_of_date": bundle["as_of_date"],
           "holding_date": bundle["holding_date"], "legal_name": bundle["identity"]["legal_name"],
           "primary_ref": bundle["original_report_archive"]["sha256"]}
    bundle["passive_holdings_calculation"] = {"inputs": raw}
    monkeypatch.setattr(fund_rating_passive_holdings, "calculate", lambda *a, **k: {"score": "75"})
    value = calculate(bundle)
    assert value["status"] == "PARTIAL" and value["score"] is None
    monkeypatch.setattr(fund_rating_passive_holdings, "calculate", lambda *a, **k: {"score": "76"})
    with pytest.raises(ValueError, match="HOLDINGS_DIMENSION_RECALCULATION"):
        calculate(bundle)
    raw["fund_code"] = "999999"
    with pytest.raises(ValueError, match="HOLDINGS_INPUT_SCOPE"):
        calculate(bundle)


def test_all_unknown_cannot_become_partial_rating(dossier):
    bundle = bundle_from(dossier)
    bundle["reviewed_document"]["dimensions"] = {k: None for k in bundle["reviewed_document"]["dimensions"]}
    computed = calculate(bundle)
    assert (
        computed["status"] == "NOT_RATED" and computed["partial_score"] is None and computed["missing_weight"] == "100"
    )


def test_market_partial_detail_keeps_business_limits_and_avoids_fake_peer_count(dossier):
    bundle = bundle_from(dossier)
    bundle.update(public_limitations=["未评分项仍为空，部分分不代表完整评价。"], comparison_product_count=0)
    detail = public_evidence(bundle, calculate(bundle))
    assert detail["comparison"]["productCount"] == 0
    assert detail["limitations"] == bundle["public_limitations"]


def test_artifact_tampering_and_wrong_fund_rejected(dossier):
    bundle = bundle_from(dossier)
    validate_sources(bundle, datetime.now(UTC), root=dossier[1])
    bundle["reviewed_document"]["dimensions"]["return"] = "99"
    with pytest.raises(ValueError, match="DOCUMENT_MISMATCH"):
        validate_sources(bundle, datetime.now(UTC), root=dossier[1])
    bundle = bundle_from(dossier)
    bundle["fund_code"] = "000003"
    with pytest.raises(ValueError, match="CASE_IDENTITY"):
        validate_sources(bundle, datetime.now(UTC), root=dossier[1])


def test_expired_verification_and_future_original_rejected(dossier):
    bundle = bundle_from(dossier)
    bundle["verified_at"] = (datetime.now(UTC) - timedelta(days=32)).isoformat()
    with pytest.raises(ValueError, match="EXPIRED"):
        validate_sources(bundle, datetime.now(UTC), root=dossier[1])
    bundle = bundle_from(dossier)
    bundle["as_of_date"] = "2026-07-31"
    with pytest.raises(ValueError, match="ORIGINAL_FUTURE"):
        validate_sources(bundle, datetime.now(UTC), root=dossier[1])


@pytest.mark.parametrize("damage", ["hash", "missing", "outside"])
def test_parallel_archives_rechecked_and_late_damage_rejected(dossier, damage):
    """并行队列后部的原件也必须校验；上次核验成功不能掩盖本次篡改或丢失。"""
    bundle = bundle_from(dossier)
    root = dossier[1]
    refs = []
    for index in range(32):
        content = json.dumps({"source": index}).encode()
        relative = f"sources/extra-{index}.json"
        (root / relative).write_bytes(content)
        refs.append({"file": relative, "sha256": hashlib.sha256(content).hexdigest(), "kind": "json"})
    bundle["extra_archives"] = refs
    validate_sources(bundle, datetime.now(UTC), root=root)
    last = root / refs[-1]["file"]
    if damage == "hash":
        last.write_text('{"source":"changed"}', encoding="utf-8")
        expected = "ARCHIVE_HASH"
    elif damage == "missing":
        last.unlink()
        expected = "ARCHIVE_PATH"
    else:
        outside = root / "outside.json"
        outside.write_bytes(last.read_bytes())
        refs[-1]["file"] = "outside.json"
        expected = "ARCHIVE_PATH"
    with pytest.raises(ValueError, match=expected):
        validate_sources(bundle, datetime.now(UTC), root=root)


def test_fixed_eight_weights_and_interval_cannot_become_point(dossier):
    bundle = bundle_from(dossier)
    bundle["reviewed_document"]["dimensions"]["execution"] = {"lower": "20", "upper": "80", "candidate": None}
    computed = calculate(bundle)
    assert computed["has_full_informative_interval"] and computed["interval_weight"] == "10"
    assert computed["dimensions"]["execution"]["candidate"] is None
    bundle["reviewed_document"]["outer_weights_percent"]["risk"] = 0
    with pytest.raises(ValueError, match="EIGHT_WEIGHTS"):
        calculate(bundle)


def test_money_precision_does_not_mutate_other_business_context():
    before = getcontext().prec
    assert concentration(".1", ".2") > Decimal("66")
    assert getcontext().prec == before


@pytest.fixture
def db():
    if os.getenv("RATING_POSTGRES_TEST") != "1":
        pytest.skip("显式允许才创建本机随机隔离schema")
    from sqlalchemy.engine import make_url

    url = get_settings().ai_database_url
    assert make_url(url).host in {"localhost", "127.0.0.1", "::1"}
    admin = create_engine(url)
    schema = "rating_partial_" + uuid4().hex
    with admin.begin() as con:
        con.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(url, connect_args={"options": f"-c search_path={schema}"})
    try:
        names = [
            "fund_master",
            "fund_share_class",
            "fund_rating_methodology",
            "fund_rating_classification",
            "fund_rating_evidence",
            "fund_rating_batch",
            "fund_rating_member",
            "fund_rating_result",
            "fund_rating_current",
            "fund_rating_audit",
        ]
        Base.metadata.create_all(engine, tables=[Base.metadata.tables[n] for n in names], checkfirst=False)
        with engine.connect() as con:
            assert con.scalar(text("SELECT current_schema()")) == schema
            assert con.scalar(text("SELECT count(*) FROM fund_share_class")) == 0
        yield engine
    finally:
        engine.dispose()
        assert schema.startswith("rating_partial_") and len(schema) == 47
        with admin.begin() as con:
            con.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def test_real_postgres_publish_idempotency_read_and_tamper(db, dossier, monkeypatch):
    from app.services import fund_rating_public_inputs as public_inputs

    data, root, _ = dossier
    with Session(db) as session, session.begin():
        first = publish_reviewed(session, data, "隔离测试用户明确审阅", root=root)
    with Session(db) as session, session.begin():
        second = publish_reviewed(session, data, "隔离测试幂等复核", root=root)
    assert first["results"] == second["results"]
    batch_id = first["results"][0]["batch"]

    def unexpected_replay(*_args, **_kwargs):
        pytest.fail("页面读取已发布版本时不应重放原件或复算")

    with monkeypatch.context() as patch:
        patch.setattr(snapshots, "calculate", unexpected_replay)
        patch.setattr(public_inputs, "validate_public_sources", unexpected_replay)
        current = read(["000009"], detail=True, engine=db)
        assert current["ratingRef"] == batch_id
        assert read(["000009"], detail=True, rating_ref=batch_id, engine=db)["dimensions"] == current["dimensions"]
    assert current["status"] == "PARTIAL" and current["score"] is None and current["grade"] is None
    assert current["partialScore"] == "75.00" and current["scoreCoverage"]["pointWeight"] == "90"
    assert len(current["dimensions"]) == 8 and current["dimensions"][2]["score"] is None
    with Session(db) as session, session.begin():
        row = session.get(RatingResult, (batch_id, "000009"))
        row.public_evidence = {**row.public_evidence, "partialScore": "99.99"}
    with pytest.raises(ValueError, match="RESULT_MISMATCH"):
        read(["000009"], detail=True, replay=True, engine=db)


def test_real_postgres_failure_rolls_back_catalogue_and_pointer(db, dossier):
    data, root, _ = dossier
    second = deepcopy(data["entries"][0])
    second["fund_code"] = "000003"
    data["entries"].append(second)
    with pytest.raises(ValueError), Session(db) as session, session.begin():
        publish_reviewed(session, data, "隔离测试失败回滚", root=root)
    with db.connect() as con:
        assert con.scalar(text("SELECT count(*) FROM fund_rating_current")) == 0
        assert con.scalar(text("SELECT count(*) FROM fund_share_class")) == 0


def test_pending_event_hides_partial_score_in_list_and_detail_but_keeps_explicit_history(db, dossier, monkeypatch):
    """待复核状态不能在列表泄漏旧部分分；显式历史引用仍能回看原结果。"""
    from app.services import fund_market_review

    data, root, _ = dossier
    with Session(db) as session, session.begin():
        published = publish_reviewed(session, data, "隔离测试部分分的待复核保护", root=root)
    monkeypatch.setattr(fund_market_review, "rating_review_required", lambda code: True)
    current_summary = read(["000009"], engine=db)["items"][0]
    current_detail = read(["000009"], detail=True, engine=db)
    for current in (current_summary, current_detail):
        assert current["status"] == "STALE" and current["grade"] is None
        assert current["partialScore"] is None and current["scoreCoverage"] is None
        assert current["scoreRange"] is None
    historical = read(["000009"], rating_ref=published["results"][0]["batch"], detail=True, engine=db)
    assert historical["status"] == "PARTIAL"
    assert historical["partialScore"] == "75.00" and historical["scoreCoverage"]["pointWeight"] == "90"


def test_real_postgres_explicit_catalog_binding_preserves_existing_identity(db, dossier):
    data, root, _ = dossier
    with Session(db) as session, session.begin():
        publish_reviewed(session, data, "隔离测试已有产品", root=root)
    with db.connect() as con:
        original = dict(con.execute(text(
            "SELECT s.fund_master_id,m.fund_name FROM fund_share_class s "
            "JOIN fund_master m USING(fund_master_id) WHERE s.fund_code='000009'"
        )).mappings().one())
    entry = data["entries"][0]
    entry["existing_catalog_binding"] = {
        "fund_master_id": str(original["fund_master_id"]), "catalog_legal_name": original["fund_name"]}
    with Session(db) as session, session.begin():
        publish_reviewed(session, data, "隔离测试保留已有目录身份", root=root)
    with db.connect() as con:
        assert con.scalar(text("SELECT count(*) FROM fund_master")) == 1
    entry["existing_catalog_binding"]["fund_master_id"] = str(uuid4())
    with pytest.raises(ValueError, match="CATALOG_BINDING"), Session(db) as session, session.begin():
        publish_reviewed(session, data, "隔离测试拒绝错误产品绑定", root=root)


def test_real_postgres_method_switch_reuses_live_classification_without_revoking_history(db, dossier):
    data, root, _ = dossier
    entry = data["entries"][0]
    master_id = uuid4()
    with Session(db) as session, session.begin():
        session.add(FundMaster(fund_master_id=master_id, fund_name="旧目录简称",
                               manager_name="隔离测试管理人", fund_type="MONEY", status="ACTIVE"))
        session.flush()
        session.add(FundShareClass(fund_code="000009", fund_master_id=master_id, share_class="A",
                                  fund_name="旧目录简称A", fund_type="MONEY", status="ACTIVE",
                                  source_code="PUBLIC_DISCLOSURE", source_fund_code="000009"))
        session.add(RatingClassification(classification_id="legacy-classification", fund_code="000009",
                                        product_id=str(master_id), category_code="legacy-category", family="MONEY",
                                        currency="CNY", eligibility="PUBLIC_LOCAL_ANALYSIS",
                                        effective_from=date(2020, 1, 1), effective_to=None,
                                        source_refs=[entry["original_report_archive"]["sha256"]],
                                        verified_at=datetime.now(UTC), revoked=False))
    entry["existing_catalog_binding"] = {"fund_master_id": str(master_id), "catalog_legal_name": "旧目录简称"}
    with Session(db) as session, session.begin():
        publish_reviewed(session, data, "隔离测试旧分类切换新评分方法", root=root)
    with db.connect() as con:
        rows = con.execute(text("SELECT classification_id,revoked FROM fund_rating_classification")).all()
        assert rows == [("legacy-classification", False)]
        assert con.scalar(text("SELECT count(*) FROM fund_master")) == 1
    assert read(["000009"], detail=True, engine=db)["status"] == "PARTIAL"


def test_current_read_ignores_superseded_archive_but_never_falls_back_after_withdrawal(db, dossier):
    """历史归档损坏不拖累新结果；指定历史仍报错，新结果撤回后不偷用旧分。"""
    from app.repositories.fund_rating import withdraw

    data, root, _ = dossier
    with Session(db) as session, session.begin():
        receipt = publish_reviewed(session, data, "隔离测试当前有效结果", root=root)
    old_time = datetime.now(UTC) - timedelta(days=1)
    with Session(db) as session, session.begin():
        session.add(RatingBatch(batch_id="superseded-archive", category_code="old-category",
                               methodology_id=None, as_of_date=date(2026, 9, 30), frozen_at=old_time,
                               input_hash="broken-archive-hash", sample_hash="old", input_path="inputs/old.json",
                               input_snapshot={"method": None}, member_count=1, product_count=1,
                               status="PUBLISHED", completed_at=old_time, withdrawn=False))
        session.flush()
        session.add(RatingCurrent(category_code="old-category", batch_id="superseded-archive", updated_at=old_time))
        session.add(RatingMember(batch_id="superseded-archive", fund_code="000009", product_id=None,
                                 representative=True, admitted=False, reasons=[], evidence_hash=None))
        session.add(RatingResult(batch_id="superseded-archive", fund_code="000009", status="NOT_RATED",
                                 score=None, grade=None, dimension_values=None, public_evidence={}, valid_until=None))
    assert read(["000009"], detail=True, engine=db)["status"] == "PARTIAL"
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        read(["000009"], rating_ref="superseded-archive", detail=True, engine=db)
    with Session(db) as session, session.begin():
        withdraw(session, receipt["results"][0]["batch"], "隔离测试撤回当前结果")
    assert read(["000009"], detail=True, engine=db)["status"] == "WITHDRAWN"
