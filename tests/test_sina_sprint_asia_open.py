"""检验未来信息、收盘价替代开盘、错误代码和脚本注入等研究输入边界。"""

import pytest
from app.integrations import sina_sprint_asia_open as s


def fields(symbol="NKY"):
    return [
        "日经225指数" if symbol == "NKY" else "韩国KOSPI指数",
        "103",
        "3",
        "3",
        "0",
        "0",
        "2026-09-16",
        "08:01:00",
        "101",
        "100",
        "104",
        "99",
    ]


def raw():
    return "\n".join(
        "var hq_str_znb_" + symbol + '="' + ",".join(fields(symbol)) + '";' for symbol in s.SYMBOLS
    ).encode("gb18030")


def test_opening_uses_open_not_current_and_explicit_receipt():
    values = s.decode(raw())
    for symbol, value in values.items():
        out = s.opening(symbol, value, "2026-09-16", "2026-09-16T08:16:00+08:00")
        assert out["opening_gap_pct"] == pytest.approx(1.0)
        assert out["open"] == 101 and out["pre_close"] == 100


@pytest.mark.parametrize(
    "changed",
    [
        raw() + b"alert(1);",
        raw() + raw(),
        b'var hq_str_znb_NKY="";',
        raw().replace(b"znb_NKY", b"znb_SPX"),
        raw().replace(b"103", b"103\\n"),
    ],
)
def test_extra_code_unknown_or_duplicate_variable_rejected(changed):
    with pytest.raises(ValueError):
        s.decode(changed)


@pytest.mark.parametrize(
    "index,value",
    [
        (0, "日经期货"),
        (6, "2026-09-15"),
        (7, "09:01:00"),
        (7, "07:59:59"),
        (8, "nan"),
        (8, "0"),
        (8, "105"),
        (9, "inf"),
        (11, "102"),
    ],
)
def test_stale_future_or_bad_price_rejected(index, value):
    data = fields()
    data[index] = value
    with pytest.raises(ValueError):
        s.opening("NKY", data, "2026-09-16", "2026-09-16T08:16:00+08:00")


@pytest.mark.parametrize("received", ["2026-09-16T08:30:00+08:00", "2026-09-16T08:16:00", "2026-09-15T08:16:00+08:00"])
def test_late_naive_or_wrong_date_receipt_rejected(received):
    with pytest.raises(ValueError):
        s.opening("NKY", fields(), "2026-09-16", received)
