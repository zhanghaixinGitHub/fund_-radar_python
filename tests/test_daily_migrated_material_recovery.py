"""历史路径已迁到D盘时，资料续办不能用旧路径物理不存在误判为无原件。"""

from datetime import datetime
from pathlib import Path

from app.services import fund_prediction_recovery as recovery
from app.services.fund_exposure_common import read
from app.services.fund_materials_store import versioned_save


def test_existing_migrated_package_is_read_through_source_router(tmp_path, monkeypatch):
    from app.services import fund_market_preparation, fund_prediction_lookthrough

    at = datetime.fromisoformat("2026-11-11T10:30:00+08:00")
    code = "123456"
    old_identity = Path("C:/historical-project/comprehensive/package.json")
    assert not old_identity.exists()
    versioned_save(tmp_path / "handoff.json", {"materials_path": str(old_identity)})
    versioned_save(
        tmp_path / "news/funds" / (code + ".json"),
        {
            "status": "SUCCEEDED",
            "catalog_complete": True,
            "coverage": {"start": "2026-11-01", "end": "2026-11-11"},
            "items": {},
        },
    )
    package = {"fund_code": code, "documents": [], "companies": [], "source_manifest": {}}
    original_read = recovery.read
    legacy_reads = []

    def read_routed(path):
        if Path(path) == old_identity:
            legacy_reads.append(Path(path))
            return package
        return original_read(path)

    monkeypatch.setattr(recovery, "read", read_routed)
    monkeypatch.setattr(recovery, "source_exists", lambda path: Path(path) == old_identity)
    monkeypatch.setattr(recovery, "now", lambda: at)
    monkeypatch.setattr(recovery, "receipt_path", lambda _: tmp_path / "handoff.json")
    monkeypatch.setattr(recovery, "workflow_root", lambda: tmp_path)
    monkeypatch.setattr(recovery, "replay_originals", lambda *args: [])
    monkeypatch.setattr(recovery, "snapshot", lambda _: {"reports": []})
    monkeypatch.setattr(fund_market_preparation, "market_scope", lambda: [{"fund_code": code}])
    monkeypatch.setattr(fund_prediction_lookthrough, "recover_target_reports", lambda *args: [])
    result = recovery.recover(code, tmp_path / "run", {"base_nav_date": "2026-11-10"}, collector=object())
    assert result["status"] == "SUPPLEMENTED_REVIEW_REQUIRED"
    assert legacy_reads == [old_identity]
    candidate = Path(read(tmp_path / "run/supplement.json")["materials_path"])
    assert candidate == (tmp_path / "run/materials-normalized.json").resolve()
    assert read(candidate)["source_manifest"]["current_notice_check"]["coverage"]["end"] == "2026-11-11"
    assert package["source_manifest"] == {}
