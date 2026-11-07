"""流动性独立接入边界：不携入未核Q，且不可绕过原字节与输入归档。"""

from copy import deepcopy
from hashlib import sha256

import pytest
from app.services.fund_rating_active_portfolio_components import (
    PRIMARY,
    VERSION,
    Sources,
    calculate,
    check_binding,
    validate_input_archive,
)


def test_unreviewed_holdings_quality_cannot_enter_with_liquidity(tmp_path):
    inputs = {
        "version": VERSION,
        "fund_code": "004605",
        "share_class": "C",
        "as_of_date": "2026-09-30",
        "holding_date": "2026-06-30",
        "primary_ref": PRIMARY,
        "selected_dimensions": ["holdings", "liquidity"],
    }
    with pytest.raises(ValueError, match="SELECTED_DIMENSIONS"):
        calculate(inputs, root=tmp_path)


def test_changed_original_bytes_are_rejected(tmp_path):
    original = b'{"actual":"original"}'
    digest = sha256(original).hexdigest()
    path = tmp_path / "sources" / (digest + ".json")
    path.parent.mkdir()
    path.write_bytes(b'{"actual":"changed"}')
    with pytest.raises(ValueError, match="SOURCE_HASH"):
        Sources(
            {
                "as_of_date": "2026-09-30",
                "sources": {digest: {"sha256": digest, "file": str(path.relative_to(tmp_path))}},
            },
            tmp_path,
        )


def test_archive_equality_precedes_raw_component_replay(tmp_path):
    bundle = {
        "active_portfolio_components_calculation": {
            "inputs": {"selected_dimensions": ["liquidity"]},
            "input_archive": {"sha256": "bound"},
        }
    }
    with pytest.raises(ValueError, match="INPUT_ARCHIVE_MISMATCH"):
        validate_input_archive(bundle, root=tmp_path, archive_reader=lambda ref, root: {})


def test_target_share_mismatch_rejected_before_external_sources():
    bundle = {
        "basis": "PUBLIC_REVIEWED_QUALITY_SNAPSHOT_V3",
        "family": "ACTIVE_EQUITY",
        "fund_code": "004605",
        "share_class": "A",
    }
    inputs = {"version": VERSION, "fund_code": "004605", "share_class": "C"}
    with pytest.raises(ValueError, match="BUNDLE_IDENTITY"):
        check_binding(deepcopy(bundle), inputs)
