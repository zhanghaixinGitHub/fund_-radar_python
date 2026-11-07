"""PDF逐页SVG的共享字形编码；仅去除重复定义，不改变文字、路径、坐标或图像。

每个字形保存完整路径属性及文本/尾随空白；页面保留原ID、属性位置和定义位置。
还原后以规范化ElementTree串的完整SHA256核对，因此共享存储不能漏字或改变图形。
"""

import hashlib
from xml.etree import ElementTree as ET

from app.services.material_content import encoded

ASSET_KIND = "SVG_GLYPH_V1"
ASSET_FORMAT = "SVG_GLYPH"
ASSET_PREFIX = b"HISTORY_SHARED_SVG_GLYPH_V1\x00"
SVG = "{http://www.w3.org/2000/svg}"


def glyph_payload(node):
    """返回不可变字形的来源键和内容摘要；字形ID保留在各页面中，不参与共享键。"""
    data = encoded({"glyph_node": node})
    return {
        "asset_kind": ASSET_KIND,
        "source_hash": hashlib.sha256(ASSET_PREFIX + data).hexdigest(),
        "content_hash": hashlib.sha256(data).hexdigest(),
    }, data


def pack_svg(svg, store_glyph):
    """提取根defs中的无子元素字形路径，其他图形及资源全部原样保留。"""
    tree = ET.fromstring(svg)
    canonical = ET.tostring(tree, encoding="unicode")
    bindings = []
    for parent_index, parent in enumerate(list(tree)):
        if parent.tag != SVG + "defs":
            continue
        for child_index, child in enumerate(list(parent)):
            identity = child.attrib.get("id", "")
            if child.tag != SVG + "path" or not identity.startswith("font_") or len(child):
                continue
            attributes = list(child.attrib.items())
            id_position = next(index for index, (key, _) in enumerate(attributes) if key == "id")
            node = {
                "node_tag": child.tag,
                "attributes": [[key, value] for key, value in attributes if key != "id"],
                "node_text": child.text,
                "tail": child.tail,
            }
            reference = store_glyph(node)
            bindings.append({
                "id": identity, "id_attribute_position": id_position,
                "defs_index": parent_index, "child_index": child_index, "asset": reference,
            })
            parent.remove(child)
    return {
        "format": "SVG_SHARED_GLYPHS_V1",
        "svg": ET.tostring(tree, encoding="unicode"),
        "glyphs": bindings,
        "tree_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def restore_svg(svg, bindings, expected_tree_hash, load_glyph):
    """按原定义位置还原完整SVG；缺失字形、ID冲突、属性或内容变化均拒绝返回。"""
    tree = ET.fromstring(svg)
    known = {node.attrib["id"] for node in tree.iter() if "id" in node.attrib}
    for binding in sorted(bindings, key=lambda node: (node["defs_index"], node["child_index"])):
        if binding["id"] in known:
            raise ValueError("SVG_SHARED_GLYPH_ID_CONFLICT")
        node = load_glyph(binding["asset"])["glyph_node"]
        parent = tree[binding["defs_index"]]
        if parent.tag != SVG + "defs" or node["node_tag"] != SVG + "path":
            raise ValueError("SVG_SHARED_GLYPH_DEFINITION_INVALID")
        attributes = [tuple(value) for value in node["attributes"]]
        attributes.insert(binding["id_attribute_position"], ("id", binding["id"]))
        element = ET.Element(node["node_tag"], dict(attributes))
        element.text, element.tail = node["node_text"], node["tail"]
        parent.insert(binding["child_index"], element)
        known.add(binding["id"])
    restored = ET.tostring(tree, encoding="unicode")
    if hashlib.sha256(restored.encode("utf-8")).hexdigest() != expected_tree_hash:
        raise ValueError("SVG_SHARED_GLYPH_RECONSTRUCTION_MISMATCH")
    return restored
