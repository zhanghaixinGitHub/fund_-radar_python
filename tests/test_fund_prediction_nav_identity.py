from decimal import Decimal

from app.services.fund_prediction_nav_identity import nav_identity


def test_late_arrival_and_real_revision_change_identity_but_archive_metadata_does_not():
    day = "2026-11-16"
    row = {"nav_date": day, "ann_date": "2026-11-17", "source_id": "provider", "unit_nav": "1.23000000"}
    captured = nav_identity([row], day)
    assert captured != nav_identity([], day)
    assert captured == nav_identity([{**row, "unit_nav": Decimal("1.23"), "content_hash": "new-envelope"}], day)
    assert captured != nav_identity([{**row, "unit_nav": "1.231"}], day)
