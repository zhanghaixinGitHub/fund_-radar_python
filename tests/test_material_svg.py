"""共享字形必须保持全部SVG内容，缺失或改变一个字形也不能通过还原校验。"""

from copy import deepcopy
from xml.etree import ElementTree as ET

import pytest
from app.services.material_svg import glyph_payload, pack_svg, restore_svg

SOURCE = '''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<defs>
<path id="font_1_1" d="M0 0L2 2Z"/>
<clipPath id="clip"><path d="M1 1L4 4Z"/></clipPath>
<path id="font_1_2" d="M0 0L2 2Z"/>
</defs>
<use data-text="条" xlink:href="#font_1_1" x="12" y="30"/>
<use data-text="件" xlink:href="#font_1_2" x="24" y="30"/>
<path id="chart" fill="red" d="M10 10L30 40Z"/>
</svg>'''


def packed_source():
    assets = {}

    def store(node):
        reference, _ = glyph_payload(node)
        assets[reference["source_hash"]] = {"glyph_node": node}
        return reference

    return pack_svg(SOURCE, store), assets


def test_shared_glyphs_deduplicate_without_changing_ids_positions_or_graphics():
    packed, assets = packed_source()
    assert len(assets) == 1 and len(packed["glyphs"]) == 2
    assert "font_1_1" in packed["svg"] and "chart" in packed["svg"]
    restored = restore_svg(packed["svg"], packed["glyphs"], packed["tree_sha256"],
                           lambda reference: assets[reference["source_hash"]])
    assert restored == ET.tostring(ET.fromstring(SOURCE), encoding="unicode")
    assert "条" in restored and "件" in restored and 'fill="red"' in restored


def test_missing_or_changed_shared_glyph_cannot_be_accepted_as_complete():
    packed, assets = packed_source()
    with pytest.raises(KeyError):
        restore_svg(packed["svg"], packed["glyphs"], packed["tree_sha256"], lambda reference: {})
    wrong = deepcopy(assets)
    next(iter(wrong.values()))["glyph_node"]["attributes"][0][1] = "M0 0L9 9Z"
    with pytest.raises(ValueError, match="RECONSTRUCTION_MISMATCH"):
        restore_svg(packed["svg"], packed["glyphs"], packed["tree_sha256"],
                    lambda reference: wrong[reference["source_hash"]])
