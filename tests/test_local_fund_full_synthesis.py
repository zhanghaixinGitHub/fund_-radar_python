"""全文封装必须可逆，正文中的类似分隔符和否定条件不能丢失。"""

from copy import deepcopy

import pytest
from scripts.local_fund_full_synthesis import decode_packet, encode_packet, text_request
from scripts.local_fund_token_budget import LocalTokenBudget


def packet():
    return {
        "facts": {
            "F1": {
                "category": "公告",
                "direction_eligible": False,
                "body_truncated": True,
                "text_parts": [{"source": "D001", "quote": "P0001"}, {"source": "D001", "quote": "P0002"}],
            }
        },
        "sources": {
            "D001": {
                "id": "doc1",
                "title": "含条件的公告",
                "date": "2026-10-08",
                "body_scope": "VERIFIED_EXCERPT",
                "body_truncated": True,
            }
        },
        "quote_pool": {
            "P0001": "尚需股东会批准。\r\n年均基数；不是三年累计。",
            "P0002": "QUOTE P0001 999\nHEADER 123\n\n不计利息；空列：A | | B。",
        },
    }


def test_roundtrip_preserves_every_character_flag_and_source_identity():
    value = packet()
    original = deepcopy(value)
    encoded = encode_packet(value)
    assert decode_packet(encoded) == original
    assert value == original
    assert value["quote_pool"]["P0001"] in encoded
    assert "\\r\\n" not in encoded


def test_partial_or_duplicated_frames_do_not_decode_as_complete_evidence():
    encoded = encode_packet(packet())
    with pytest.raises(ValueError, match="LENGTH_INVALID"):
        decode_packet(encoded[:-2])
    with pytest.raises(ValueError, match="FRAME_INVALID"):
        decode_packet(encoded + "QUOTE P0001 1\n字\n")


def test_raw_text_budget_never_discards_a_large_quote_to_fit():
    budget = LocalTokenBudget.__new__(LocalTokenBudget)
    budget.calibrated = True
    budget.count = len
    value = packet()
    value["quote_pool"]["P0001"] = "所有原文保留。" * 10000
    data = {
        "fund_code": "002112",
        "window": {},
        "as_of": "2026-10-09T17:21:00+08:00",
        "latest_nav_date": "2026-10-08",
        "inventory": [],
        "overflow": False,
    }
    with pytest.raises(ValueError, match="PACKET_TOO_LARGE"):
        text_request(data, value["facts"], {}, value, budget)
