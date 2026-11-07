"""本份额费用原件、日期和篡改边界；测试公告为隔离合成文本。"""

import hashlib
import json
from copy import deepcopy
from decimal import Decimal

import pytest
from app.services.fund_rating_fee_inputs import VERSION, calculate
from app.services.fund_rating_reviewed_quality import PASSIVE_BASIS


@pytest.fixture
def case(tmp_path):
    text = (
        "隔离基金(A类份额) 基金代码999999 送出日期2026年7月29日"
        "三、投资本基金涉及的费用 (一)基金销售相关费用 申购费(前收费) M<100万元1.2% "
        "赎回费 N<7天1.5% 7天≤N<180天0.25% 180天≤N0% "
        "(二)基金运作相关费用 (三)基金运作综合费用测算 基金运作综合费率(年化)0.20%"
    )
    raw = json.dumps({"data": {"notice_content": text}}, ensure_ascii=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (tmp_path / "sources").mkdir()
    path = tmp_path / "sources" / f"{digest}.json"
    path.write_bytes(raw)
    inputs = {
        "version": VERSION,
        "basis": PASSIVE_BASIS,
        "fund_code": "999999",
        "share_class": "A",
        "legal_name": "隔离基金",
        "as_of_date": "2026-09-30",
        "summary_ref": digest,
        "sources": {
            digest: {
                "sha256": digest,
                "file": f"sources/{digest}.json",
                "published_on": "2026-07-29",
                "archive_kind": "ORIGINAL",
            }
        },
    }
    return inputs, tmp_path, path


def test_a_share_public_fees_use_existing_r5_cash_cost_and_external_subscription(case):
    inputs, root, _ = case
    result = calculate(inputs, root=root)
    assert Decimal(result["score"]).quantize(Decimal(".01")) == Decimal("65.42")
    assert result["fees"] == {
        "subscription": "0.012",
        "redemption": "0",
        "operating_cost": "0.002",
        "subscription_mode": "EXTERNAL",
    }


def test_supporting_prospectus_must_be_the_same_legal_product(case):
    inputs, root, _ = case
    raw = json.dumps({'data': {'notice_content': '另一只基金更新招募说明书 C类基金份额不收取申购费用'}}).encode()
    digest = hashlib.sha256(raw).hexdigest()
    (root/'sources'/f'{digest}.json').write_bytes(raw)
    inputs['prospectus_ref'] = digest
    inputs['sources'][digest] = {'file': f'sources/{digest}.json', 'sha256': digest,
                                'published_on': '2026-07-29', 'archive_kind': 'ORIGINAL'}
    with pytest.raises(ValueError, match='SUPPORTING_PRODUCT_IDENTITY'):
        calculate(inputs, root=root)


@pytest.mark.parametrize(
    "change,error",
    [
        ("derived", "ORIGINAL_REQUIRED"),
        ("future", "FUTURE_DOCUMENT"),
        ("wrong_share", "身份尚未核准"),
        ("tamper", "SOURCE_HASH"),
    ],
)
def test_unverified_fee_source_cannot_get_a_point(case, change, error):
    inputs, root, path = deepcopy(case)
    reference = inputs["sources"][inputs["summary_ref"]]
    if change == "derived":
        reference["archive_kind"] = "DERIVED"
    elif change == "future":
        reference["published_on"] = "2026-10-01"
    elif change == "wrong_share":
        inputs["fund_code"] = "999998"
    elif change == "tamper":
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match=error):
        calculate(inputs, root=root)
