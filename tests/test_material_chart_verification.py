"""源图与文本的独立核验必须能检出遗漏、错系列及伪造完整标记。"""

import copy

import pytest
from app.services.material_chart_verification import extract_visible_curves, verify_curve
from tests.test_material_chart_text import chart


def prepared():
    image, ocr, dates, labels = chart()
    # 真实图框连续，背景虚线不会覆盖外框；补齐合成样本的连续边界。
    image[10, 60:381] = image[160, 60:381] = [100, 100, 100]
    ocr.update(bbox=[0, 0, image.shape[1] / 3, image.shape[0] / 3], text="基金C\n基准")
    data = extract_visible_curves(image, ocr, dates=dates, labels=labels, series=["基金C", "基准"])
    return image, ocr, data


def test_every_source_curve_point_is_covered_and_precision_is_explicit():
    image, ocr, data = prepared()
    assert verify_curve(image, ocr, data)["passed"]
    assert data["exact_daily_values"] == "NOT_PROVIDED_BY_SOURCE_FIGURE"
    assert data["semantic_review_required"]
    assert all(r["uncovered_points"] == 0 for r in verify_curve(image, ocr, data)["series"])


def test_missing_curve_segment_does_not_pass():
    image, ocr, data = prepared()
    data["data"][0]["samples"][140]["visible_value_ranges_pct"] = []
    assert not verify_curve(image, ocr, data)["passed"]


def test_swapped_curve_values_do_not_pass():
    image, ocr, data = prepared()
    data["data"][0]["samples"], data["data"][1]["samples"] = data["data"][1]["samples"], data["data"][0]["samples"]
    assert not verify_curve(image, ocr, data)["passed"]


def test_missing_series_and_wrong_fund_label_are_rejected():
    image, ocr, data = prepared()
    incomplete = copy.deepcopy(data)
    incomplete["data"].pop()
    with pytest.raises(ValueError, match="SERIES_COUNT_MISMATCH"):
        verify_curve(image, ocr, incomplete)
    data["series"][0] = "基金A"
    with pytest.raises(ValueError, match="SERIES_LABEL_MISMATCH"):
        verify_curve(image, ocr, data)
