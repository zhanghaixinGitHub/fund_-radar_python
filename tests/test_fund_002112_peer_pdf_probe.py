"""公开报告补查的身份和缓存边界，所有测试均禁止外呼。"""

import hashlib
import json

import pytest
from scripts import fund_002112_peer_pdf_probe as probe


def test_annual_share_identity_on_page_five_is_valid():
    pages = ["东方红新动力年报", "目录", "目录", "目录", "基金主代码000480 下属分级基金代码 000480 017493"]
    probe.validate_pdf_identity(pages, "017493")


def test_correct_master_with_wrong_share_class_is_rejected():
    with pytest.raises(ValueError, match="SHARE_CLASS_NOT_IN_PRODUCT_SECTION"):
        probe.validate_pdf_identity(["东方红新动力 基金主代码000480 下属代码000480"], "017493")


def test_unplanned_fund_is_rejected():
    with pytest.raises(ValueError, match="SCOPE_INVALID"):
        probe.validate_pdf_identity(["华夏磐泰 基金主代码160323 999999"], "999999")


@pytest.mark.parametrize("tampered", [False, True])
def test_completed_probe_only_checks_cached_receipts_without_network(tmp_path, monkeypatch, tampered):
    monkeypatch.setattr(probe, "OUTPUT", tmp_path)
    folder = tmp_path / "public-pdf-probe"
    folder.mkdir()
    path = folder / "source.pdf"
    path.write_bytes(b"original")
    result = {"receipts": [{"path": str(path), "sha256": hashlib.sha256(b"original").hexdigest()}]}
    (folder / "results.json").write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(probe.httpx, "Client", lambda **_: pytest.fail("cached probe must not access network"))
    if tampered:
        path.write_bytes(b"changed")
        with pytest.raises(ValueError, match="RECEIPT_CHANGED"):
            probe.run()
    else:
        assert probe.run() == result
