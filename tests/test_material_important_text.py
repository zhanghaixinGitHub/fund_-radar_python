"""重要信息发布与真实读取契约；未知区域不能因布尔完成标记而被遗漏。"""

import copy

import pytest
from app.models.material_content import MaterialContent, MaterialDocument, MaterialLocation, MaterialPageRepresentation
from app.services import material_content as store
from app.services import material_important_text as important
from app.services.material_document import read_document
from sqlalchemy import create_engine, delete
from tests.test_material_content import pdf_bytes


@pytest.fixture
def setup(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'materials.db'}")
    for table in (MaterialContent, MaterialLocation, MaterialDocument, MaterialPageRepresentation):
        table.__table__.create(engine)
    repo = store.MaterialRepository(engine)
    with store.using_repository(repo):
        yield tmp_path, repo
    engine.dispose()


def prepared(setup):
    root, repo = setup
    path = root / "sources/report.pdf"
    value = store.extract(pdf_bytes(), ".pdf")
    repo.put(path, value)
    repo.page(value.source_hash, "PYPDF_LAYOUT_V1", 0, "OLD_LAYOUT")
    repo.page(value.source_hash, "PYPDF_TEXT_V1", 0, "OLD_NATIVE")
    doc = {
        "schema": important.POLICY,
        "source_hash": value.source_hash,
        "base_content_hash": value.content_hash,
        "format": "PDF",
        "coverage": "IMPORTANT_INFORMATION",
        "expected_region_ids": ["p1r0"],
        "source_review": "独立对照原图及图注，份额/日期/单位/列对应已核对",
        "remaining": [],
        "regions": [
            {
                "region_id": "p1r0",
                "page": 1,
                "decision": "TEXTUALIZED_IMPORTANT",
                "kind": "asset_allocation",
                "reason": "资产比例影响风险判断",
                "review": "对照原图标签",
                "text": "截至2025-12-31，权益投资99.58%，现金0.30%，其他0.12%。",
                "data": {"columns": ["类别", "比例"], "unit": "%", "rows": [["权益", "99.58"]]},
            }
        ],
    }
    return path, value, doc


def test_published_business_text_reaches_both_pdf_readers_and_canonical_document(setup):
    _, repo = setup
    path, value, doc = prepared(setup)
    important.publish(doc, repo=repo)
    original_hash = value.content_hash
    with store.database_only():
        raw = store.source_bytes(path)
        with store.document_reader(raw) as reader:
            assert "99.58%" in reader.pages[0].extract_text()
            assert reader.pages[0].extract_text(extraction_mode="layout") == "OLD_LAYOUT"
        with store.document_pdfium(raw) as reader:
            assert "99.58%" in reader[0].get_textpage().get_text_range()
        result = read_document(value.source_hash, repo=repo)
    assert result["coverage"] == "IMPORTANT_INFORMATION"
    assert "99.58%" in result["content"]["pages"][0]["text"]
    assert repo.get(source_hash=value.source_hash).content_hash == original_hash


def test_common_policy_publishes_and_reads_without_using_pilot_identity(setup):
    _, repo = setup
    path, value, doc = prepared(setup)
    doc["schema"] = important.COMMON_POLICY
    important.publish(doc, repo=repo)
    assert important.release(value.source_hash, repo)["policy"] == important.COMMON_POLICY
    assert important.selected_policy(value.source_hash, repo) == important.COMMON_POLICY
    with store.database_only():
        assert "99.58%" in store.document_reader(store.source_bytes(path)).pages[0].extract_text()
        restored = read_document(value.source_hash, repo=repo)
    assert restored["schema"] == important.COMMON_POLICY
    assert "99.58%" in restored["content"]["pages"][0]["text"]


def test_missing_region_prevents_publication(setup):
    _, _, doc = prepared(setup)
    doc["expected_region_ids"].append("p2r0")
    with pytest.raises(ValueError, match="REGION_COVERAGE"):
        important.publish(doc)


def test_uncertain_region_cannot_be_excluded(setup):
    _, _, doc = prepared(setup)
    region = doc["regions"][0]
    region.update(decision="EXCLUDED_LOW_IMPORTANCE", uncertain=True, contains_unique_business_fact=False)
    with pytest.raises(ValueError, match="UNCERTAIN_REGION"):
        important.publish(doc)


def test_summary_cannot_replace_explicit_financial_table(setup):
    _, _, doc = prepared(setup)
    del doc["regions"][0]["data"]
    with pytest.raises(ValueError, match="EXACT_TABLE"):
        important.publish(doc)


def test_unpublished_curve_cannot_claim_exact_values(setup):
    _, _, doc = prepared(setup)
    doc["regions"][0].update(
        kind="performance_curve", data={"series": ["基金", "基准"], "period": ["2020", "2025"], "features": ["上涨"]}
    )
    with pytest.raises(ValueError, match="CURVE_PRECISION"):
        important.publish(doc)


def test_missing_published_part_fails_without_opening_original(setup, monkeypatch):
    _, repo = setup
    path, value, doc = prepared(setup)
    important.publish(doc)
    with repo.engine.begin() as c:
        c.execute(
            delete(MaterialPageRepresentation).where(
                MaterialPageRepresentation.source_hash == value.source_hash,
                MaterialPageRepresentation.engine == important.PAGE_ENGINE,
            )
        )
    with pytest.raises(ValueError, match="PAGE_MISSING"):
        with store.document_reader(store.source_bytes(path)) as reader:
            reader.pages[0].extract_text()


def test_republishing_conflicting_review_is_rejected(setup):
    _, _, doc = prepared(setup)
    important.publish(doc)
    changed = copy.deepcopy(doc)
    changed["regions"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="MATERIAL_PAGE_CONTENT_MISMATCH"):
        important.publish(changed)


def test_existing_public_document_cache_does_not_hide_new_important_text(setup):
    from app.services.fund_rating_public_market import public_document

    path, value, doc = prepared(setup)
    previous = store.source_bytes(path)
    old_hash = hash(previous)
    assert "99.58%" not in public_document(previous, {}, "reference", "")[0][0]
    important.publish(doc)
    current = store.source_bytes(path)
    assert hash(previous) == old_hash and current != previous
    assert current.source_hash == previous.source_hash and current.content_hash == previous.content_hash
    assert "99.58%" in public_document(current, {}, "reference", "")[0][0]


def test_html_business_reader_includes_catalog_but_exact_source_is_preserved(setup):
    root, repo = setup
    path = root / "sources/page.html"
    raw = b"<html><body>Original body</body></html>"
    value = repo.put(path, store.extract(raw, ".html"))
    doc = {
        "schema": important.POLICY,
        "source_hash": value.source_hash,
        "base_content_hash": value.content_hash,
        "format": "HTML",
        "coverage": "IMPORTANT_INFORMATION",
        "expected_region_ids": ["iframe"],
        "source_review": "原始列表与独立分页对应",
        "remaining": [],
        "regions": [
            {
                "region_id": "iframe",
                "page": 1,
                "kind": "announcement_index",
                "decision": "TEXTUALIZED_IMPORTANT",
                "reason": "公告来源重要",
                "review": "核对分页日期和标题",
                "text": "2025-12-31公告索引，来源时点单独保留。",
            }
        ],
    }
    important.publish(doc)
    assert "公告索引" in store.source_text(path)
    assert store.exact_source_text(store.source_bytes(path)) == raw.decode()
    assert "公告索引" in read_document(value.source_hash)["content"]["text"]


def test_shared_copy_exit_requires_exact_scope_and_preserves_business_read(setup, monkeypatch):
    from types import SimpleNamespace

    from app.services import material_source_archive as archive

    root, repo = setup
    path, value, doc = prepared(setup)
    path.parent.mkdir(parents=True)
    path.write_bytes(pdf_bytes())
    disk = root.parent / (root.name + "-archive")
    target = disk / "originals" / value.source_hash[:2] / (value.source_hash + ".pdf")
    target.parent.mkdir(parents=True)
    target.write_bytes(pdf_bytes())
    repo.link(target, value.source_hash)
    monkeypatch.setattr(store, "PROJECT", root)
    monkeypatch.setattr(
        archive,
        "get_settings",
        lambda: SimpleNamespace(fund_source_archive_root=str(disk), fund_source_archive_funds="008888"),
    )
    important.publish(doc)
    with pytest.raises(ValueError, match="NOT_IN_EXPLICIT_SCOPE"):
        archive.remove_reviewed_copy(path, source_hash=value.source_hash, approved_paths=[])
    assert path.is_file()
    archive.remove_reviewed_copy(path, source_hash=value.source_hash, approved_paths=[str(path)])
    assert not path.exists() and target.is_file()
    assert "99.58%" in store.document_reader(store.source_bytes(path)).pages[0].extract_text()
    # 同源的新共享别名即使没有008888目录，也不再重新生成C盘原件。
    shared = path.with_name("another-shared-source.pdf")
    store.write_source(shared, pdf_bytes())
    assert not shared.exists()
    assert store.source_bytes(shared).source_hash == value.source_hash


def test_archive_status_recognizes_important_completion_without_claiming_full_visual_equivalence(setup):
    from app.commands.material_source_archive import capture_text

    _, repo = setup
    path, value, doc = prepared(setup)
    path.parent.mkdir(parents=True)
    path.write_bytes(pdf_bytes())
    important.publish(doc)
    status = capture_text(path, repo)
    assert status["status"] == "IMPORTANT_INFORMATION_VERIFIED"
    assert status["coverage"] == "IMPORTANT_INFORMATION"
