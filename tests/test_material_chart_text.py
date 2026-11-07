"""图面数值必须保留精度与缺失，不能由像素估读冒充正式日净值。"""

import numpy as np
import pytest
from app.services.material_chart_text import digitise


def chart():
    image = np.full((200, 400, 3), 255, dtype=np.uint8)
    image[10:161, 60] = image[10:161, 380] = [100, 100, 100]
    image[10, 60:381] = image[160, 60:381] = [100, 100, 100]
    labels = ["100.00%", "80.00%", "60.00%", "40.00%", "20.00%", "0.00%"]
    lines = []
    for i, value in enumerate(labels):
        y = 10 + 30 * i
        image[y, 61:380:2] = [210, 210, 210]
        lines.append({"text": value, "box": [[10, y - 2], [45, y - 2], [45, y + 2], [10, y + 2]]})
    dates = ["2026-01-01", "2026-01-17", "2026-02-02"]
    for x, value in zip([60, 220, 380], dates, strict=True):
        lines.append({"text": value, "box": [[x - 10, 180], [x + 10, 180], [x + 10, 190], [x - 10, 190]]})
    image[95, 61:380] = [0, 0, 180]
    image[110, 61:380] = [220, 0, 220]
    return image, {"lines": lines}, dates, labels


def test_intervals_preserve_observed_values_and_do_not_invent_missing_points():
    image, ocr, dates, labels = chart()
    image[95, 200] = [255, 255, 255]
    value = digitise(image, ocr, dates=dates, y_labels=labels, series={"fund": "基金C", "benchmark": "基准"})
    samples = value["data"][0]["samples"]
    assert len(samples) == 321
    assert value["semantic_review_required"]
    assert samples[140]["visibility"] == "NOT_VISIBLE_IN_SOURCE"
    assert samples[140]["visible_value_ranges_pct"] == []
    low, high = samples[50]["visible_value_ranges_pct"][0]
    assert low <= 43.3333333333 <= high
    assert all("2026-01-01" <= s["time_from"] <= s["time_to"] <= "2026-02-02T23:59:59" for s in samples)


def test_missing_axis_labels_block_digitisation():
    image, ocr, dates, labels = chart()
    ocr["lines"].pop()
    with pytest.raises(ValueError, match="CHART_AXIS_ANCHORS_INCOMPLETE"):
        digitise(image, ocr, dates=dates, y_labels=labels, series={"fund": "基金C", "benchmark": "基准"})


def test_missing_series_is_not_silently_filled():
    image, ocr, dates, labels = chart()
    image[110, 61:380] = [255, 255, 255]
    with pytest.raises(ValueError, match="CHART_SERIES_NOT_FOUND"):
        digitise(image, ocr, dates=dates, y_labels=labels, series={"fund": "基金C", "benchmark": "基准"})
