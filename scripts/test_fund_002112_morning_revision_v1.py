"""08点覆盖反例：日期精度、已知时刻、修订与旧aux不能替代真正时间边界。"""

from scripts.fund_002112_morning_coverage_revision_v1 import availability, first_morning

SESSIONS = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]


def effective(**fields):
    base = {"published_date": "2025-01-02", "published_at": None, "revision_at": None, "force_next": False}
    base.update(fields)
    value, _, error = availability(base)
    assert error is None
    return first_morning(value, SESSIONS)


def test_date_only_does_not_enter_same_day_morning():
    assert effective() == "2025-01-03"


def test_verified_clock_before_at_and_after_morning_boundary():
    for clock in ("07:59:59", "08:00:00"):
        assert effective(published_at=f"2025-01-02T{clock}+08:00") == "2025-01-02"
    assert effective(published_at="2025-01-02T08:00:01+08:00") == "2025-01-03"


def test_later_revision_overrides_publication_and_old_aux():
    assert effective(published_at="2025-01-02T07:00:00+08:00", revision_at="2025-01-03 09:00:00") == "2025-01-06"
    assert effective(revision_at="2025-01-03", effective_session_aux="2025-01-03") == "2025-01-06"


def test_date_only_weekend_and_post_close_constraint():
    assert effective(published_date="2025-01-04") == "2025-01-06"
    assert effective(published_at="2025-01-02T07:00:00+08:00", force_next=True) == "2025-01-03"


def test_all_revision_sources_take_latest_and_unknown_time_stays_excluded():
    p, _, error = availability({"published_date": "2025-01-02"}, [("another_revision", "2025-01-06T09:00:00+08:00")])
    assert error is None and first_morning(p, SESSIONS) == "2025-01-07"
    assert availability({})[0] is None
    assert availability({"published_date": "not-a-date"})[0] is None
