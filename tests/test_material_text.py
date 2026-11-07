"""异常源网页按原始字节保全，禁止把猜测的中文或替换字符写成原文。"""

import copy

import pytest
from app.services.material_text import escape_invalid_utf8_html, restore_escaped_utf8


def test_invalid_html_bytes_and_existing_escape_literals_roundtrip_exactly():
    raw = b'<!doctype html><html><meta charset="UTF-8"><p>\\xC8 ' + '原文金额123.45'.encode() + b'\xc8x\xcf</p></html>'
    body, escapes = escape_invalid_utf8_html(raw)
    assert len(escapes) == 2
    assert "原文金额123.45" in body
    assert restore_escaped_utf8(body, escapes) == raw
    wrong = copy.deepcopy(escapes)
    wrong[0]["byte_offset"] += 1
    with pytest.raises(ValueError, match="OFFSET_CHANGED"):
        restore_escaped_utf8(body, wrong)


def test_unknown_encodings_or_non_html_are_not_silently_guessed():
    with pytest.raises(ValueError, match="NOT_DECLARED"):
        escape_invalid_utf8_html(b'<html><meta charset="GBK">\xc8x</html>')
    with pytest.raises(ValueError, match="NOT_DECLARED"):
        escape_invalid_utf8_html(b'plain text \xc8x')
