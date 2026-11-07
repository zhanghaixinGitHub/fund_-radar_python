"""文本优先不得丢失扫描图、独有图形或精确正文。"""

import base64
import zlib

import pytest
from app.services import material_raster, material_text_first

SVG = ('<svg xmlns="http://www.w3.org/2000/svg"><defs/>'
       '<use data-text="金额"/><use data-text="1.25%"/></svg>')


def page():
    return {"native_text": "金额 1.25%", "independent_text": "金额 1.25%",
            "size": [100, 100], "layout": {"blocks": [{"lines": [{"spans": [
                {"text": "金额 1.25%", "bbox": [1, 1, 90, 10]}]}]}]},
            "annotations": [], "widgets": [], "links": [], "visual": {"old": True}}


def test_text_page_preserves_exact_values_and_layout():
    old = page()
    proof = material_text_first.text_only_proof(old, SVG)
    assert proof["text_only"]
    new = material_text_first.plain_page(old, proof)
    assert new["native_text"] == old["native_text"]
    assert new["layout"] == old["layout"]
    assert "visual" not in new and old["visual"] == {"old": True}


@pytest.mark.parametrize("change", ["scan", "changed_number", "annotation", "chart"])
def test_unverified_content_keeps_visual(change):
    original = page()
    visual = SVG
    if change == "scan":
        original["native_text"] = original["independent_text"] = ""
    elif change == "changed_number":
        original["independent_text"] = "金额 1.25"
    elif change == "annotation":
        original["annotations"] = [{"text": "限制条件"}]
    else:
        visual = SVG.replace("</svg>", '<path d="M1 2 L3 4"/></svg>')
    assert not material_text_first.text_only_proof(original, visual)["text_only"]


def test_existing_jpeg_survives_svg_extraction_and_asset_integrity_check():
    jpeg = b"\xff\xd8\xffsource-already-compressed\xff\xd9"
    svg = ('<svg xmlns="http://www.w3.org/2000/svg"><image href="data:image/jpeg;base64,'
           + " \n".join([base64.b64encode(jpeg).decode()[:8], base64.b64encode(jpeg).decode()[8:]]) + '"/></svg>')
    assets = {}

    def save(raw, mime):
        reference, body = material_raster.payload(raw, mime)
        assets[reference["source_hash"]] = body
        return reference

    packed = material_text_first.compact_necessary_svg(svg, save, lambda node: None)
    ref = packed["raster_bindings"][0]["asset"]
    body = assets[ref["source_hash"]]
    restored = material_raster.validate_asset(ref["source_hash"], ref["content_hash"], zlib.compress(body))
    assert base64.b64decode(restored["data_base64"]) == jpeg
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        material_raster.validate_asset(ref["source_hash"], ref["content_hash"], zlib.compress(body + b"x"))
