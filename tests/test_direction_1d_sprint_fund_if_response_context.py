"""用刻意带未来标签/缺失标签的样本检查逐基金IF经验统计的成熟边界。"""

from datetime import date, timedelta

import pytest
from app.services import direction_1d_sprint_fund_if_response_context as s


def rows(n=130):
    output = []
    for i in range(n):
        day = date(2024, 1, 2) + timedelta(days=i)
        x = 1 if i % 2 else -1
        output.append(
            {
                "code": "sample",
                "t": str(day - timedelta(days=1)),
                "u": str(day),
                "mature": str(day + timedelta(days=1)),
                "y": int(x > 0),
                "market": {
                    "futures_intraday": {
                        "date": str(day - timedelta(days=1)),
                        "available": True,
                        "intraday_return_pct": x,
                        "close_location": x / 2,
                    }
                },
            }
        )
    return output


def test_current_and_unmatured_labels_are_never_read():
    values = rows()
    expected = s.prior(values, "2024-06-01")
    values += [
        {"code": "sample", "u": "2024-06-01", "mature": "2024-06-02"},
        {"code": "sample", "u": "2024-05-31", "mature": "2024-06-01"},
    ]
    assert s.prior(values, "2024-06-01") == expected
    assert expected["count"] == 126
    # x=±1、y=±1时协方差和方差均1，固定岭.25和收缩1/2给出0.4；第二列给0.5。
    assert expected["beta"] == pytest.approx([0.4, 0.5])


def test_short_history_explicitly_unavailable_and_no_nonzero_fill():
    c = s.prior(rows(62), "2024-06-01")
    assert not c["available"] and c["beta"] == [0, 0]
    v = s.feature({"date": "2024-05-31", "available": False}, c, "2024-06-01")
    assert not v["available"] and v["response_return"] == v["response_location"] == 0


@pytest.mark.parametrize("bad", ["duplicate", "fund", "nan", "label"])
def test_malformed_eligible_samples_rejected(bad):
    values = rows()
    if bad == "duplicate":
        values.append(values[-1])
    elif bad == "fund":
        values[-1]["code"] = "other"
    elif bad == "nan":
        values[-1]["market"]["futures_intraday"]["close_location"] = float("nan")
    else:
        values[-1]["y"] = True
    with pytest.raises(ValueError):
        s.prior(values, "2024-06-01")


def test_context_cannot_be_newer_than_prediction_or_contain_unmatured_data():
    c = s.prior(rows(), "2024-06-01")
    f = {"date": "2024-05-31", "available": True, "intraday_return_pct": 2, "close_location": 0.5}
    v = s.feature(f, c, "2024-06-01")
    assert v["response_return"] == pytest.approx(0.8) and v["response_location"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="CONTEXT_FROM_FUTURE"):
        s.feature(f, c, "2024-05-31")
    c["max_mature"] = c["cutoff"]
    with pytest.raises(ValueError, match="CONTEXT_NOT_MATURE"):
        s.feature(f, c, "2024-06-01")
