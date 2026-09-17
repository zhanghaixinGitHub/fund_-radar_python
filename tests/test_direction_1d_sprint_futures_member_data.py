"""排名特征只允许已知T数据，缺失与单位校验不通过不能变成有效输入。"""

import pytest
from app.services import direction_1d_sprint_futures_member_data as data

CAL = ["2026-09-14", "2026-09-15", "2026-09-16"]
POINT = {
    "main_contract": "IF2609.CFX",
    "level_imbalance": -0.1,
    "change_imbalance": 0.02,
    "long_members": 20,
    "short_members": 20,
}


def test_only_exact_t_is_used():
    actual = data.aligned(CAL[1], CAL[2], {CAL[1]: POINT, CAL[2]: POINT | {"level_imbalance": 0.9}}, CAL)
    assert actual["date"] == CAL[1] and actual["level_imbalance"] == -0.1
    missing = data.aligned(CAL[1], CAL[2], {CAL[0]: POINT, CAL[2]: POINT}, CAL)
    assert missing == {"date": CAL[1], "available": False}


@pytest.mark.parametrize("t,u", [(CAL[0], CAL[2]), (CAL[1], CAL[1]), (CAL[2], CAL[1]), ("2026-09-13", CAL[0])])
def test_nonadjacent_pair_rejected(t, u):
    with pytest.raises(ValueError, match="ADJACENCY"):
        data.aligned(t, u, {t: POINT}, CAL)


@pytest.mark.parametrize(
    "field,value",
    [
        ("level_imbalance", float("nan")),
        ("change_imbalance", 2),
        ("level_imbalance", True),
        ("long_members", 19),
        ("short_members", 21),
    ],
)
def test_invalid_point_rejected(field, value):
    with pytest.raises(ValueError):
        data.aligned(CAL[1], CAL[2], {CAL[1]: POINT | {field: value}}, CAL)


def test_original_row_and_inputs_preserved(monkeypatch):
    from datetime import date

    monkeypatch.setattr(data.base, "calendar", lambda: ([date.fromisoformat(d) for d in CAL], "test"))
    market = {"available": True, "other": {"x": 1}}
    output = data.extend(market, CAL[1], CAL[2], {CAL[1]: POINT})
    assert market == {"available": True, "other": {"x": 1}}
    assert data.original_row({"market": output, "y": 1}) == {"market": market, "y": 1}
