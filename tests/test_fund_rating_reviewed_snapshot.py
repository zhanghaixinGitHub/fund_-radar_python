"""部分评分的真实性、缺项、过期、篡改及真实PostgreSQL事务边界。"""

import hashlib
import json
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal, getcontext
from uuid import uuid4

import pytest
from app.core.config import get_settings
from app.db.base import Base
from app.models.fund_rating import RatingResult
from app.services.fund_rating import read
from app.services.fund_rating_money_public import concentration
from app.services.fund_rating_reviewed_snapshot import (
    BASIS,
    archive_root,
    calculate,
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


def test_all_unknown_cannot_become_partial_rating(dossier):
    bundle = bundle_from(dossier)
    bundle["reviewed_document"]["dimensions"] = {k: None for k in bundle["reviewed_document"]["dimensions"]}
    computed = calculate(bundle)
    assert (
        computed["status"] == "NOT_RATED" and computed["partial_score"] is None and computed["missing_weight"] == "100"
    )


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


def test_real_postgres_publish_idempotency_read_and_tamper(db, dossier):
    data, root, _ = dossier
    with Session(db) as session, session.begin():
        first = publish_reviewed(session, data, "隔离测试用户明确审阅", root=root)
    with Session(db) as session, session.begin():
        second = publish_reviewed(session, data, "隔离测试幂等复核", root=root)
    assert first["results"] == second["results"]
    current = read(["000009"], detail=True, engine=db)
    assert current["status"] == "PARTIAL" and current["score"] is None and current["grade"] is None
    assert current["partialScore"] == "75.00" and current["scoreCoverage"]["pointWeight"] == "90"
    assert len(current["dimensions"]) == 8 and current["dimensions"][2]["score"] is None
    with Session(db) as session, session.begin():
        row = session.get(RatingResult, (first["results"][0]["batch"], "000009"))
        row.public_evidence = {**row.public_evidence, "partialScore": "99.99"}
    with pytest.raises(ValueError, match="RESULT_MISMATCH"):
        read(["000009"], detail=True, engine=db)


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
