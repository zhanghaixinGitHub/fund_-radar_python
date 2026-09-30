"""旧结论合并仅验证绑定范围，不能把其他通过标志扩大为身份或全文语义通过。"""

from scripts.fund_002112_closure_prior_evidence_v3 import anchors_passed, identity_passed


def test_generic_pass_does_not_mean_identity():
    assert not identity_passed({"passed": True}, "/asof_snapshots/0")
    assert identity_passed({"passed": True}, "/prior_english_identities/0")
    assert identity_passed({"source_identity_verified": True}, "/claims/0")
    assert not identity_passed({"title": {"passed": True}}, "/identity_supplements/0")


def test_exact_page_and_text_are_required():
    assert anchors_passed({"anchor": {"page": 1, "text": "净利润 10 万元"}}, ["净利润10万元"])
    assert not anchors_passed({"anchor": {"page": 2, "text": "净利润10万元"}}, ["净利润10万元"])
    assert not anchors_passed({"anchor": {"page": 1, "text": "净利润10亿元"}}, ["净利润10万元"])
    assert not anchors_passed({"anchor": {"page": 1, "text": "净利润11万元"}}, ["净利润10万元"])
