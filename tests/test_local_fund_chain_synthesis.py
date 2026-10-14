"""验证预算压缩严格保留全部事实及引句，而不是筛选后重新摘要。"""

from scripts.local_fund_chain_synthesis import pack_books, unpack_books


def test_pack_keeps_all_rows_quotes_nulls_and_types():
    books = {
        "doc-01": {
            "source_scope": "VERIFIED_EXCERPT",
            "source_truncated": True,
            "facts": [
                {
                    "id": "F0001",
                    "kind": "cell",
                    "data": {"amount": None, "unit": "万元", "condition": True},
                    "quote": "合计空列",
                },
                {
                    "id": "F0002",
                    "kind": "cell",
                    "data": {"amount": "85.73", "unit": "万元", "condition": 1},
                    "quote": "收益85.73",
                },
                {
                    "id": "F0003",
                    "kind": "cell",
                    "data": {"amount": "8324.58", "unit": "万元", "condition": True},
                    "quote": "未收回本金8324.58",
                },
            ],
        }
    }
    packed = pack_books(books)
    assert packed["documents"]["doc-01"]["groups"]["cell"]["constants"] == {"unit": "万元"}
    rows = unpack_books(packed)
    assert len(rows) == 3
    assert rows["doc-01:F0001"][1]["amount"] is None
    assert rows["doc-01:F0002"][1]["condition"] == 1
    assert rows["doc-01:F0003"][2:] == ["未收回本金8324.58", "VERIFIED_EXCERPT", True, []]


def test_quote_reuse_does_not_drop_fact():
    book = {
        "source_scope": "FULL_TEXT",
        "source_truncated": False,
        "facts": [
            {"id": f"F{i:04}", "kind": "quantity", "data": {"value": str(i), "unit": "股"}, "quote": "原文同一行"}
            for i in range(1, 4)
        ],
    }
    packed = pack_books({"doc-02": book})
    assert packed["documents"]["doc-02"]["quotes"] == ["原文同一行"]
    assert len(unpack_books(packed)) == 3


def test_scope_follows_shareholder_items_without_merging_next_clause():
    from scripts.local_fund_chain_synthesis import scope_context

    body = (
        "（二）公司出现下列情形之一的，由股东会决定是否继续执行：\n"
        "1、控制权变更；\n2、合并分立。\n（三）虚假披露应返还权益。"
    )
    quote = "（二）公司出现下列情形之一的，由股东会决定是否继续执行：\n1、控制权变更；"
    contexts = scope_context({"kind": "literal_condition", "quote": quote, "source_refs": []}, {"body": body}, None)
    assert "2、合并分立。" in contexts[0]
    assert "虚假披露" not in contexts[0]


def test_shared_strings_keep_every_character_and_json_type():
    from scripts.local_fund_chain_synthesis import share_strings, unshare_strings

    original = {
        "one": "原始条件必须保留完整，不能删掉任何一个限定词",
        "two": ["原始条件必须保留完整，不能删掉任何一个限定词", None, True, 1, "1"],
    }
    packed = share_strings(original)
    assert len(packed["shared_strings"]) == 1
    assert unshare_strings(packed) == original


def test_shared_string_tag_collision_blocks_instead_of_changing_data():
    import pytest
    from scripts.local_fund_chain_synthesis import share_strings

    with pytest.raises(ValueError, match="TAG_COLLISION"):
        share_strings({"unexpected": {"s": 3}})
