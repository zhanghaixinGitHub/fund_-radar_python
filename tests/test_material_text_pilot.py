"""试点的实质边界：完整读库、不可变历史、逐图覆盖、摘要不能代替图表数据。"""

import hashlib
import json

import pytest
from app.models.material_content import MaterialContent, MaterialDocument, MaterialLocation, MaterialPageRepresentation
from app.services import material_content as store
from app.services import material_text_pilot as pilot
from app.services.material_document import get_review, read_document
from sqlalchemy import create_engine, update
from tests.test_material_content import pdf_bytes


@pytest.fixture
def setup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'pilot.db'}")
    for table in (MaterialContent, MaterialLocation, MaterialDocument, MaterialPageRepresentation):
        table.__table__.create(engine)
    yield tmp_path, store.MaterialRepository(engine)
    engine.dispose()


def candidate(setup, *, regions=None, issues=None):
    root, repo = setup
    path = root / "data/sources/008888.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_bytes("Fee -1.25%; effective 2026-11-03; excluded if closed"))
    base = repo.ingest(path)
    text = base.extracted["pdfium"][0]
    for engine in ("PYPDF_TEXT_V1", "PYPDF_LAYOUT_V1"):
        repo.page(base.source_hash, engine, 0, text)
    record = {
        "page": 1,
        "native_page_index": 0,
        "native_text": text,
        "independent_text": text,
        "third_text": text,
        "layout_engine": "PYPDF_LAYOUT_V1",
        "layout_hash": store.checksum(text.encode()),
        "independent_text_hash": store.checksum(text.encode()),
        "tables": [],
        "visual_regions": regions or [],
        "issues": issues or [],
        "preview_sha256": "a" * 64,
    }
    ref = pilot._part(repo, base.source_hash, pilot.PAGE_ENGINE, 0, record)
    pilot._save(
        repo,
        {
            "schema": pilot.CANDIDATE,
            "source_hash": base.source_hash,
            "base_content_hash": base.content_hash,
            "format": "PDF",
            "ready": False,
            "content": {"page_count": 1, "pages": [ref]},
            "issues": [],
        },
    )
    return path, base, repo


def test_published_text_is_selected_and_exact_values_survive_without_original(setup):
    path, base, repo = candidate(setup)
    before = get_review(base.source_hash, repo=repo, policy=pilot.CANDIDATE)["document_hash"]
    result = pilot.publish(base.source_hash, repo=repo)
    assert result["ready"]
    path.unlink()
    with store.database_only():
        doc = read_document(base.source_hash, repo=repo)
    text = doc["content"]["pages"][0]["page_content"]["text"]
    assert "-1.25%" in text and "2026-11-03" in text and "excluded if closed" in text
    assert get_review(base.source_hash, repo=repo, policy=pilot.CANDIDATE)["document_hash"] == before
    # 迁移输入副本清理后，发布入口仍识别已完成版本，而不是误报资料待处理。
    assert pilot.publish(base.source_hash, repo=repo)["document_hash"] == result["document_hash"]


def test_unreviewed_image_is_not_published(setup):
    _, base, repo = candidate(setup, regions=[{"region_hash": "raster-one"}])
    result = pilot.publish(base.source_hash, repo=repo)
    assert not result["ready"] and result["pending"][0]["page"] == 1
    assert pilot.selected_policy(base.source_hash, repo) is None


def test_chart_summary_without_data_is_rejected(setup):
    _, base, repo = candidate(setup, regions=[{"region_hash": "chart-one"}])
    review = {
        "source_hash": base.source_hash,
        "page": 1,
        "preview_sha256": "a" * 64,
        "remaining": [],
        "regions": [
            {
                "region_hash": "chart-one",
                "kind": "line_chart",
                "status": "VERIFIED",
                "remaining": [],
                "complete_transcription": "The fund increased.",
            }
        ],
    }
    with pytest.raises(ValueError, match="CHART_SUMMARY_IS_INSUFFICIENT"):
        pilot.publish(base.source_hash, repo=repo, reviews={"1": review})


def test_wrong_source_or_preview_cannot_certify_a_page(setup):
    _, base, repo = candidate(setup, regions=[{"region_hash": "one"}])
    wrong = {"source_hash": base.source_hash, "page": 1, "preview_sha256": "b" * 64, "remaining": []}
    assert not pilot.publish(base.source_hash, repo=repo, reviews={"1": wrong})["ready"]


def test_nonempty_trend_field_does_not_replace_complete_chart_table(setup):
    _, base, repo = candidate(setup, regions=[{"region_hash": "chart-one"}])
    review = {
        "source_hash": base.source_hash,
        "page": 1,
        "preview_sha256": "a" * 64,
        "remaining": [],
        "regions": [
            {
                "region_hash": "chart-one",
                "kind": "bar_chart",
                "status": "VERIFIED",
                "remaining": [],
                "complete_transcription": "增长",
                "axes": {"unit": "%"},
                "series": ["基金"],
                "data": {"trend": "增长"},
            }
        ],
    }
    with pytest.raises(ValueError, match="FULL_TABLE_REQUIRED"):
        pilot.publish(base.source_hash, repo=repo, reviews={"1": review})


def test_later_cross_source_values_are_not_inserted_into_original_source_text(setup):
    _, base, repo = candidate(setup, regions=[{"region_hash": "chart-one"}])
    review = {
        "source_hash": base.source_hash,
        "page": 1,
        "preview_sha256": "a" * 64,
        "remaining": [],
        "regions": [
            {
                "region_hash": "chart-one",
                "kind": "pie_chart",
                "status": "VERIFIED",
                "remaining": [],
                "complete_transcription": "原图占比10%",
                "axes": {"unit": "%"},
                "series": ["资产"],
                "data": {
                    "columns": ["资产", "占比"],
                    "rows": [["股票", "10%"]],
                    "exact_values_from_other_verified_text": {"later_document_value": "LATER_VALUE"},
                },
            }
        ],
    }
    pilot.publish(base.source_hash, repo=repo, reviews={"1": review})
    page = read_document(base.source_hash, repo=repo)["content"]["pages"][0]
    assert "10%" in page["page_content"]["text"] and "LATER_VALUE" not in page["page_content"]["text"]
    assert page["review"]["regions"][0]["cross_source_verification"][
        "point_in_time_use_requires_own_publication_cutoff"
    ]


def test_source_part_tampering_blocks_readback(setup):
    _, base, repo = candidate(setup)
    pilot.publish(base.source_hash, repo=repo)
    with repo.engine.begin() as c:
        c.execute(
            update(MaterialPageRepresentation)
            .where(
                MaterialPageRepresentation.source_hash == base.source_hash,
                MaterialPageRepresentation.engine == pilot.PAGE_ENGINE,
            )
            .values(content_hash="0" * 64)
        )
    with pytest.raises(ValueError, match="MATERIAL_PAGE_CONTENT_MISMATCH"):
        read_document(base.source_hash, repo=repo)


@pytest.mark.parametrize("value", [{"data_base64": "AA=="}, {"svg": "text"}, {"text": "data:image/png;base64,AAA"}])
def test_encoded_images_cannot_be_disguised_as_text(value):
    with pytest.raises(ValueError, match="VISUAL_PAYLOAD"):
        pilot._text_only(value)


def test_real_pdf_extraction_preserves_all_pages_and_never_stores_images(setup):
    pytest.importorskip("pymupdf")
    root, repo = setup
    path = root / "data/sources/actual.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pdf_bytes("Exact fee: -1.25%; exception retained"))
    base = repo.ingest(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = pilot.extract_pdf(path, repo=repo, temp_root=root / "temporary")
    assert result["content"]["page_count"] == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    pilot._text_only(read_document(base.source_hash, repo=repo, policy=pilot.CANDIDATE))
    with repo.engine.connect() as c:
        assert (
            c.scalar(
                MaterialDocument.__table__.select()
                .with_only_columns(MaterialDocument.ready)
                .where(MaterialDocument.policy == pilot.CANDIDATE)
            )
            is False
        )


def test_failed_integrity_check_prevents_cleanup_plan(tmp_path):
    from app.commands.material_text_pilot import cleanup_plan

    original = tmp_path / "original.pdf"
    original.write_bytes(b"sole-source")
    scope = {"sources": [{"source_hash": "h", "existing_paths": [str(original)]}]}
    verify = {"complete": False, "sources": [{"source_hash": "h", "complete_text_ready": False}]}
    (tmp_path / "scope.json").write_text(json.dumps(scope), encoding="utf-8")
    (tmp_path / "verification.json").write_text(json.dumps(verify), encoding="utf-8")
    result = cleanup_plan(tmp_path)
    assert result["can_cleanup"] is False and result["originals_deleted"] == 0
    assert original.read_bytes() == b"sole-source"
