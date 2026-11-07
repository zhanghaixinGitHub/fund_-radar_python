"""XML工作簿数值词元、稀疏/合并位置、原件不变及业务只读库。"""
import json

import pytest
from app.models.material_content import MaterialDocument
from app.services import material_content as store
from app.services.material_archive_text import capture
from app.services.material_spreadsheet_text import capture_spreadsheetml, read_spreadsheetml
from tests.test_material_source_archive import setup  # noqa: F401


def workbook(extra=""):
    return ('<?xml version="1.0"?><ss:Workbook xmlns:ss="urn:schemas-microsoft-com:office:spreadsheet">'
            '<ss:Styles><ss:Style ss:ID="currency"><ss:NumberFormat ss:Format="0.00000 CNY"/>'
            '</ss:Style></ss:Styles><ss:Worksheet ss:Name="History"><ss:Table><ss:Row ss:Index="7">'
            '<ss:Cell ss:Index="3" ss:MergeAcross="1" ss:StyleID="currency"><ss:Data ss:Type="Number">'
            '-1234567890.000001</ss:Data></ss:Cell><ss:Cell ss:HRef="https://example.com?a=1&b=2">'
            '<ss:Data ss:Type="String">条件：ETF除外</ss:Data></ss:Cell></ss:Row></ss:Table>'
            + extra + '</ss:Worksheet></ss:Workbook>')


def test_workbook_preserves_full_cells_and_business_read_without_original(setup, monkeypatch):  # noqa: F811
    path, disk, repo = setup
    MaterialDocument.__table__.create(repo.engine)
    raw = workbook().encode()
    store.write_source(path.with_suffix(".xls"), raw)
    target = next((disk / "originals").rglob("*.html"))
    outcome = capture(target, repo=repo)
    assert outcome["status"] == "IMPORTANT_TEXT_VERIFIED"
    assert target.read_bytes() == raw
    monkeypatch.setattr(type(target), "read_bytes", lambda self: pytest.fail("reopened workbook source"))
    with store.using_repository(repo), store.database_only():
        actual = read_spreadsheetml(store.checksum(raw), repo=repo)
        assert actual["all_cell_data_independently_equal"]
        assert actual["bare_ampersands_escaped_in_memory"] == 1
        row = actual["sheets"][0]["tables"][0]["rows"][0]
        assert row["row"] == 7
        assert [cell["column"] for cell in row["cells"]] == [3, 5]
        assert row["cells"][0]["value"] == "-1234567890.000001"
        assert row["cells"][1]["value"] == "条件：ETF除外"


def test_workbook_visual_region_stays_pending_and_no_pixel_payload_is_stored(setup):  # noqa: F811
    _, _, repo = setup
    raw = workbook('<Image>do-not-store-image-pixels</Image>')
    value = store.extract(raw.encode(), ".xls")
    assert value.kind == "UNSUPPORTED" and not value.complete
    repo.put(store.PROJECT / "data/raw/source.xls", value)
    assert 'do-not-store-image-pixels' not in json.dumps(repo.get(source_hash=value.source_hash).extracted)
    result = capture_spreadsheetml(raw, value.source_hash, repo)
    assert result["pending"]
    restored = read_spreadsheetml(value.source_hash, repo=repo)
    assert 'do-not-store-image-pixels' not in json.dumps(restored)
