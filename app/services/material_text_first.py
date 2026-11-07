"""文本优先转换的可验证规则；正文/表格布局保持原值，不能只按体积舍弃图像。

纯文字页面须有两份原生文字、布局文字与SVG字形文字一致，并且没有未表达的
可见图形、批注或表单。含扫描图、图表或不确定内容的页面仍保留必要图形；
其栅格图片独立共享，字体轮廓复用，避免反复保存整套资源。
"""

import base64
import copy
import hashlib
import io
import re
import unicodedata
from xml.etree import ElementTree as ET

from app.services import material_raster
from app.services.material_svg import pack_svg

POLICY = "TEXT_FIRST_CONTENT_V1"


def normalized(value):
    return "".join(unicodedata.normalize("NFKC", value).split())


def svg_inventory(svg):
    """统计实际绘制内容；defs/裁剪路径本身不当作页面图像，未知use仍保留。"""
    tree = ET.fromstring(svg)
    text, graphics = [], []

    def walk(element, definitions=False):
        tag = element.tag.rsplit("}", 1)[-1]
        definitions = definitions or tag in {"defs", "clipPath"}
        if not definitions:
            if tag == "use" and "data-text" in element.attrib:
                text.append(element.attrib["data-text"])
            elif tag not in {"svg", "g"}:
                graphics.append(tag)
        for child in element:
            walk(child, definitions)

    walk(tree)
    return tree, "".join(text), graphics


def layout_text(page):
    spans = []
    for block in page.get("layout", {}).get("blocks", []):
        for line in block.get("lines", []):
            spans.extend(span for span in line.get("spans", []) if "text" in span)
    return "".join(span["text"] for span in spans), spans


def text_only_proof(page, svg, source_png=None):
    """只在文字覆盖可独立复查时移除页面画面；可疑字符或剩余墨迹不放行。"""
    _, glyphs, graphics = svg_inventory(svg)
    native = page.get("native_text", "")
    independent = page.get("independent_text", "")
    laid_out, spans = layout_text(page)
    expected = normalized(native)
    if (not expected or graphics or page.get("annotations") or page.get("widgets")
            or "\ufffd" in native or any(unicodedata.category(char) == "Co" for char in native)
            or expected != normalized(independent) or expected != normalized(glyphs)
            or expected != normalized(laid_out)):
        return {"text_only": False, "reason": "NON_TEXT_OR_UNCONFIRMED_CONTENT"}
    if source_png is not None:
        from PIL import Image, ImageChops, ImageDraw

        with Image.open(io.BytesIO(source_png)) as image:
            bitmap = image.convert("RGB")
        width, height = page["size"]
        sx, sy = bitmap.width / width, bitmap.height / height
        draw = ImageDraw.Draw(bitmap)
        for span in spans:
            x0, y0, x1, y1 = span["bbox"]
            draw.rectangle((max(0, int(x0 * sx) - 2), max(0, int(y0 * sy) - 2),
                            min(bitmap.width - 1, int(x1 * sx) + 2),
                            min(bitmap.height - 1, int(y1 * sy) + 2)), fill="white")
        residual = ImageChops.difference(bitmap, Image.new("RGB", bitmap.size, "white"))
        if residual.getbbox() is not None:
            return {"text_only": False, "reason": "SOURCE_PIXELS_OUTSIDE_CONFIRMED_TEXT"}
    return {"text_only": True, "rule": "NATIVE_TEXT_LAYOUT_GLYPHS_AGREE_V1",
            "normalized_text_sha256": hashlib.sha256(expected.encode()).hexdigest(),
            "source_backup_ink_checked": source_png is not None}


def compact_necessary_svg(svg, put_raster, put_glyph):
    """必要图形保留全部坐标/图形/字形，内嵌栅格改为共享资产，不做有损转码。"""
    tree = ET.fromstring(svg)
    bindings = []
    for element in tree.iter():
        if element.tag.rsplit("}", 1)[-1] != "image":
            continue
        for key, value in list(element.attrib.items()):
            if key.rsplit("}", 1)[-1] != "href":
                continue
            match = re.fullmatch(r"data:([^;,]+);base64,([\s\S]+)", value)
            if match is None:
                raise ValueError("TEXT_FIRST_NON_SELF_CONTAINED_RASTER")
            # PDF导出的SVG可能在Base64中插入空白；仅移除编码允许的空白，
            # 再严格校验其余字符，不能因换行导致扫描件被误判为不可解析。
            raw = base64.b64decode(re.sub(r"\s+", "", match[2]), validate=True)
            encoded, mime, audit = material_raster.optimise(raw, match[1])
            asset = put_raster(encoded, mime)
            placeholder = "material-raster:" + str(len(bindings))
            element.attrib[key] = placeholder
            bindings.append({"placeholder": placeholder, "asset": asset, "source_encoding": audit})
    normalised = ET.tostring(tree, encoding="unicode")
    packed = pack_svg(normalised, put_glyph)
    packed.update(format="SVG_RASTER_ASSETS_V1", raster_bindings=bindings)
    return packed


def plain_page(page, proof):
    """只去除已证明冗余的画面；正文、坐标、批注、链接、表单等原值均保留。"""
    result = copy.deepcopy({key: value for key, value in page.items()
                            if key not in {"visual", "visual_verification"}})
    result["text_first_proof"] = proof
    return result
