import json
from datetime import datetime

from app.integrations.public_fund_reports import BODY_URL
from app.services import fund_notice_observed_proof as proof
from app.services.direction_1d_protocol import digest
from app.services.fund_materials_store import versioned_save


def test_real_public_response_proves_upper_bound_without_claiming_exact_publication(tmp_path, monkeypatch):
    code, article = "123456", "AN202611111831605448"
    received = "2026-11-11T10:00:00+08:00"
    receipt = {"url": BODY_URL, "params": {"art_code": article}, "received_at": received, "sha256": "a" * 64}
    raw = json.dumps(
        {"success": 1, "data": {"art_code": article, "notice_content": "实际公开原文", "security": [{"stock": code}]}}
    )
    versioned_save(tmp_path / "news/body-receipts" / code / (article + ".json"), receipt)
    versioned_save(
        tmp_path / "news/funds" / (code + ".json"),
        {"items": {article: {"source": receipt, "entry": {"published": "2026-11-11"}}}},
    )
    monkeypatch.setattr(proof, "read_source", lambda *args: raw)
    document = {
        "kind": "ANNOUNCEMENT",
        "codes": [code],
        "published_date": "2026-11-11",
        "body": "实际公开原文",
        "available_at": received,
        "raw_sha256": "a" * 64,
        "publication_observed_proof": {
            "kind": "PUBLIC_RESPONSE_FIRST_OBSERVED",
            "article_id": article,
            "receipt_hash": digest(receipt),
            "observed_at": received,
        },
    }
    assert proof.validate(document, code, datetime.fromisoformat("2026-11-11T11:00:00+08:00"), root=tmp_path)
    assert "published_at" not in document
    assert not proof.validate(document, code, datetime.fromisoformat("2026-11-11T09:00:00+08:00"), root=tmp_path)
    assert not proof.validate(
        {**document, "body": "改写后的公告"}, code, datetime.fromisoformat("2026-11-11T11:00:00+08:00"), root=tmp_path
    )
