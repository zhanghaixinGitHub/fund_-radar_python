"""检验截止时点、稀少样本权重、冻结来源及逐日身份，不运行真实模型。"""

import pytest
from app.services import fund_002112_peer_fold_impact as audit


@pytest.mark.parametrize(
    "target,publication,expected",
    [
        ("2023-03-29", "2023-03-30", "TIME_ELIGIBLE"),
        ("2023-03-30", "2023-03-31", "LABEL_NOT_MATURE_BEFORE_CUTOFF"),
        ("2023-03-31", "2023-04-21", "LABEL_NOT_PUBLIC_BEFORE_CUTOFF"),
        ("2023-03-30", "2023-04-01", "LABEL_NOT_PUBLIC_BEFORE_CUTOFF"),
        ("2023-03-29", None, "PUBLICATION_UNKNOWN"),
        ("2023-04-03", "2023-04-04", "TARGET_NOT_BEFORE_CUTOFF"),
    ],
)
def test_original_cutoff_does_not_mean_merely_target_before_start(target, publication, expected):
    assert audit.time_status(target, publication, "2023-04-01")[0] == expected


def test_calendar_day_maturity_keeps_weekend_and_conservative_publication():
    assert audit.maturity("2023-03-31", "2023-04-21") == "2023-04-22T08:00:00+08:00"
    assert audit.maturity("2023-03-31", "2023-03-30") == "2023-04-01T08:00:00+08:00"


@pytest.mark.parametrize(
    "target,publication,start",
    [
        ("2025-01-02", "2025-01-03", "2023-04-01"),
        ("2023-03-30", "2023-03-31", "2023-03-31"),
        ("2023-03-30", "2026-09-27", "2023-04-01"),
    ],
)
def test_forbidden_dates_and_changed_cutoff_are_rejected(target, publication, start):
    with pytest.raises(ValueError):
        audit.time_status(target, publication, start)


def test_more_rows_dilute_per_row_but_preserve_family_mass():
    counts = dict.fromkeys(audit.COHORT, 100)
    counts["017493"] = 5
    old = audit.weight_table(counts)
    new = audit.weight_table({**counts, "017493": 50})
    a, b = [x["by_fund"]["017493"] for x in (old, new)]
    assert a["per_row_relative_to_002112"] == 20
    assert b["per_row_relative_to_002112"] == 2
    assert b["normalized_per_row_weight"] == pytest.approx(a["normalized_per_row_weight"] / 10)
    assert a["family_weight_share"] == b["family_weight_share"] == 1 / 11
    for table in (old, new):
        assert sum(r["rows"] * r["raw_per_row_weight"] for r in table["by_fund"].values()) == pytest.approx(
            table["rows"]
        )
        assert sum(r["rows"] * r["normalized_per_row_weight"] for r in table["by_fund"].values()) == pytest.approx(1)


@pytest.mark.parametrize("change", ["absent", "zero", "negative", "float"])
def test_missing_family_never_becomes_zero_weight(change):
    counts = dict.fromkeys(audit.COHORT, 10)
    if change == "absent":
        counts.pop("017493")
    else:
        counts["017493"] = {"zero": 0, "negative": -1, "float": 1.5}[change]
    with pytest.raises(ValueError):
        audit.weight_table(counts)


@pytest.mark.parametrize("change", ["duplicate", "missing", "different_date"])
def test_daily_identity_cannot_shrink_or_expand(change):
    days = ["2023-03-30"]
    rows = [{"fund_code": c, "target": days[0]} for c in audit.PEERS]
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "missing":
        rows.pop()
    else:
        rows[0]["target"] = "2023-03-31"
    with pytest.raises(ValueError):
        audit.index_daily(rows, days)


def test_source_hash_and_result_are_immutable(tmp_path):
    path = tmp_path / "source.json"
    audit.save_once(path, {"count": 1})
    spec = {"sources": {"s": {"path": str(path), "sha256": audit.file_hash(path)}}}
    assert audit.load_source(spec, "s") == {"count": 1}
    with pytest.raises(ValueError, match="ALREADY_EXISTS"):
        audit.save_once(path, {"count": 2})
    path.write_text('{"count":2}', encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_CHANGED"):
        audit.load_source(spec, "s")
