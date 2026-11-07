"""原网页局部编码异常的无损底稿；可读正文与异常原字节分开，不猜测缺字。"""

import re


def escape_invalid_utf8_html(raw):
    """仅处理明确声明UTF-8的HTML；返回安全正文及可精确还原的字节位置表。

    异常字节显示为ASCII的\\xNN，位置表区分原文已有的相同字样；不将其冒充中文。
    其他格式、无编码声明或大量异常不自动猜测，继续交给既有严格解析流程处理。
    """
    head = raw[:8192]
    if not re.match(br"\s*(?:<!doctype\s+html|<html\b)", head, re.I):
        raise ValueError("SOURCE_NOT_DECLARED_UTF8_HTML")
    if not re.search(br"charset\s*=\s*[\"']?utf-?8\b", head, re.I):
        raise ValueError("SOURCE_NOT_DECLARED_UTF8_HTML")
    decoded = raw.decode("utf-8", "surrogateescape")
    matches = list(re.finditer("[\udc80-\udcff]", decoded))
    if not matches or len(matches) > min(1024, max(4, len(raw) // 100)):
        raise ValueError("SOURCE_UTF8_ANOMALY_REQUIRES_REVIEW")
    pieces, escapes = [], []
    cursor = byte_offset = display_offset = 0
    for match in matches:
        regular = decoded[cursor:match.start()]
        pieces.append(regular)
        byte_offset += len(regular.encode("utf-8"))
        display_offset += len(regular)
        value = ord(match.group()) - 0xDC00
        marker = f"\\x{value:02X}"
        pieces.append(marker)
        escapes.append({"character_start": display_offset, "character_end": display_offset + len(marker),
                        "byte_offset": byte_offset, "hex": f"{value:02x}"})
        byte_offset += 1
        display_offset += len(marker)
        cursor = match.end()
    pieces.append(decoded[cursor:])
    body = "".join(pieces)
    if restore_escaped_utf8(body, escapes) != raw:
        raise ValueError("SOURCE_UTF8_BYTE_ESCAPE_ROUNDTRIP_FAILED")
    return body, escapes


def restore_escaped_utf8(body, escapes):
    """从库内正文及位置表还原源文字字节；位置、标记或字节值不一致即拒绝。"""
    output = bytearray()
    cursor = 0
    for entry in escapes:
        start, end = entry["character_start"], entry["character_end"]
        value = bytes.fromhex(entry["hex"])
        if not cursor <= start < end <= len(body) or len(value) != 1 or value[0] < 128:
            raise ValueError("SOURCE_BYTE_ESCAPE_LOCATION_INVALID")
        if body[start:end] != f"\\x{value[0]:02X}":
            raise ValueError("SOURCE_BYTE_ESCAPE_MARKER_CHANGED")
        output.extend(body[cursor:start].encode("utf-8"))
        if len(output) != entry["byte_offset"]:
            raise ValueError("SOURCE_BYTE_ESCAPE_OFFSET_CHANGED")
        output.extend(value)
        cursor = end
    output.extend(body[cursor:].encode("utf-8"))
    return bytes(output)
