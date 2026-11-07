"""图表文本的独立原图复核：核对业务坐标与可见数据，不能用摘要或哈希替代。

直接从PDF内原始图像取得像素，用不同于候选生成器的坐标拟合和色彩条件
对照时间/百分比区间。这里只验证原图实际提供的业务信息；未显示的精确
每日点值仍为源未提供，不能插值补造、冒充官方净值或清理未核实的内容。
"""

from __future__ import annotations

import hashlib
from datetime import datetime


def original_image(document, page_index, region):
    """按页面位置定位内嵌原图，避免拿候选生成时的重采样图自证。"""
    import io

    import numpy as np
    from PIL import Image

    matches = []
    for image in document[page_index].get_image_info(hashes=True, xrefs=True):
        if max(abs(a - b) for a, b in zip(image["bbox"], region["bbox"], strict=True)) < 0.01:
            matches.append(image)
    if len(matches) != 1 or not matches[0]["xref"]:
        raise ValueError("CHART_ORIGINAL_IMAGE_IDENTITY_AMBIGUOUS")
    image = matches[0]
    if image["digest"].hex() != region["source_pixel_digest"]:
        raise ValueError("CHART_ORIGINAL_IMAGE_DIGEST_CHANGED")
    raw = document.extract_image(image["xref"])["image"]
    with Image.open(io.BytesIO(raw)) as source:
        pixels = np.array(source.convert("RGB"))
    return pixels, {
        "encoded_image_sha256": hashlib.sha256(raw).hexdigest(),
        "source_pixel_digest": image["digest"].hex(),
        "original_dimensions": [image["width"], image["height"]],
    }


def _ordinal(value):
    when = datetime.fromisoformat(value)
    return when.toordinal() + (when.hour * 3600 + when.minute * 60 + when.second) / 86400


def _axes(pixels, ocr, labels, *, snap_grid=True):
    """从原图及OCR框重新定位；以成对斜率中位数拟合，独立于候选的最小二乘。"""
    import numpy as np

    h, w = pixels.shape[:2]
    rgb = pixels.astype(np.int16)
    dark = (rgb.max(2) - rgb.min(2) < 20) & (rgb.mean(2) < 185)
    horizontal = np.flatnonzero(dark.mean(1) > 0.55)
    vertical = np.flatnonzero(dark.mean(0) > 0.55)
    if len(horizontal) < 2 or len(vertical) < 2:
        raise ValueError("CHART_ORIGINAL_PLOT_BORDER_MISSING")
    left, right = int(vertical.min()), int(vertical.max())
    top, bottom = int(horizontal.min()), int(horizontal.max())
    if right - left < w * 0.5 or bottom - top < h * 0.5:
        raise ValueError("CHART_ORIGINAL_PLOT_BORDER_INVALID")
    # OCR是在3倍页面裁切上运行；源图可能是790×465，按真实尺寸重新换算。
    box_w = 3 * (ocr["bbox"][2] - ocr["bbox"][0])
    box_h = 3 * (ocr["bbox"][3] - ocr["bbox"][1])
    grey = (rgb.max(2) - rgb.min(2) < 22) & (rgb.mean(2) < 245)
    score = grey[:, left + 3 : right - 3].mean(1)
    anchors = []
    expected = {str(v).replace(" ", "") for v in labels}
    for line in ocr["lines"]:
        value = "".join(line["text"].split())
        xs = sum(p[0] for p in line["box"]) / 4 * w / box_w
        y = sum(p[1] for p in line["box"]) / 4 * h / box_h
        if value not in expected or xs >= left:
            continue
        if snap_grid:
            choices = np.flatnonzero(score[max(0, round(y - 4)) : min(h, round(y + 5))] > 0.12)
            if not len(choices):
                raise ValueError("CHART_ORIGINAL_TICK_NOT_FOUND")
            choices = choices + max(0, round(y - 4))
            location = float(choices[np.argmin(abs(choices - y))])
        else:
            location = float(y)
        anchors.append((location, float(value.removesuffix("%"))))
    if len(anchors) < len(labels) - 2 or len(anchors) < 5:
        raise ValueError("CHART_ORIGINAL_AXIS_COVERAGE_INCOMPLETE")
    slopes = [(v2 - v1) / (p2 - p1) for p1, v1 in anchors for p2, v2 in anchors if p2 > p1 + 5]
    slope = float(np.median(slopes))
    intercept = float(np.median([v - slope * y for y, v in anchors]))
    residual = float(max(abs(slope * y + intercept - v) for y, v in anchors))
    if slope >= 0 or residual > abs(slope) * 1.6:
        raise ValueError("CHART_ORIGINAL_CALIBRATION_INVALID")
    return (left, top, right, bottom), slope, intercept, residual


def extract_visible_curves(pixels, ocr, *, dates, labels, series):
    """从内嵌原图直接提取全部可见曲线，消除渲染缩放和文字框中心的横轴偏移。

    每条记录是有日期/收益率单位的业务观测区间，不保存像素、颜色数组或
    矢量轮廓。被覆盖的曲线在原图本来不可见，明确保持未知，不补造点值。
    """
    import numpy as np

    bounds, slope, intercept, residual = _axes(pixels, ocr, labels)
    left, top, right, bottom = bounds
    rgb = pixels.astype(np.int16)
    masks = [
        (rgb[:, :, 2] - rgb[:, :, 0] > 15) & (rgb[:, :, 2] - rgb[:, :, 1] > 15),
        (rgb[:, :, 0] - rgb[:, :, 1] > 25)
        & (rgb[:, :, 2] - rgb[:, :, 1] > 25)
        & (abs(rgb[:, :, 0] - rgb[:, :, 2]) < 90),
    ]
    first, last = _ordinal(dates[0]), _ordinal(dates[-1])

    def iso(ordinal):
        from datetime import timedelta

        return (datetime.fromordinal(int(ordinal)) + timedelta(days=ordinal % 1)).isoformat(timespec="seconds")

    result = []
    for name, mask in zip(series, masks, strict=True):
        samples = []
        for x in range(left, right + 1):
            ys = np.flatnonzero(mask[top : bottom + 1, x]) + top
            groups = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1) if len(ys) else []
            values = [
                [
                    round(slope * (int(g[-1]) + 0.6) + intercept - residual, 6),
                    round(slope * (int(g[0]) - 0.6) + intercept + residual, 6),
                ]
                for g in groups
                if len(g)
            ]
            lower = first + (last - first) * max(0, x - left - 0.6) / (right - left)
            upper = first + (last - first) * min(right - left, x - left + 0.6) / (right - left)
            samples.append(
                {
                    "time_from": iso(lower),
                    "time_to": iso(upper),
                    "visible_value_ranges_pct": values,
                    "visibility": "VISIBLE" if values else "NOT_VISIBLE_IN_SOURCE",
                }
            )
        result.append(
            {
                "name": name,
                "unit": "%",
                "samples": samples,
                "source_absence_rule": "原图未显示或被另一条线遮挡，不补造原始数据",
            }
        )
    return {
        "kind": "SOURCE_VISIBLE_CURVE_TABLE",
        "data": result,
        "axes": {
            "x": {"labels": dates, "unit": "日期", "scale": "LINEAR_CALENDAR"},
            "y": {"labels": labels, "unit": "%", "scale": "LINEAR"},
        },
        "series": series,
        "source_precision": "源图显示精度的时间及累计收益率区间；不是精确每日净值",
        "exact_daily_values": "NOT_PROVIDED_BY_SOURCE_FIGURE",
        "date_resolution_days": (last - first) / (right - left),
        "value_resolution_pct": abs(slope),
        "axis_fit_uncertainty_pct": residual,
        "semantic_review_required": True,
    }


def verify_curve(pixels, ocr, content):
    """全部源图可见线芯必须被库内区间覆盖；未提供的遮挡数据不得补齐。"""
    import numpy as np

    if len(content.get("data", [])) != 2 or len(content.get("series", [])) != 2:
        raise ValueError("CHART_SOURCE_SERIES_COUNT_MISMATCH")
    labels = "".join(ocr.get("text", "").split())
    if any("".join(name.split()) not in labels for name in content["series"]):
        raise ValueError("CHART_SOURCE_SERIES_LABEL_MISMATCH")
    bounds, slope, intercept, residual = _axes(pixels, ocr, content["axes"]["y"]["labels"])
    left, top, right, bottom = bounds
    rgb = pixels.astype(np.int16)
    # 采用原图的饱和线芯而非候选的宽色阈值，排除重采样与抗锯齿造成的假点。
    masks = [
        (rgb[:, :, 2] - rgb[:, :, 0] > 65) & (rgb[:, :, 2] - rgb[:, :, 1] > 65),
        (rgb[:, :, 0] - rgb[:, :, 1] > 80) & (rgb[:, :, 2] - rgb[:, :, 1] > 80),
    ]
    dates = content["axes"]["x"]["labels"]
    first, last = _ordinal(dates[0]), _ordinal(dates[-1])
    results = []
    precision = abs(slope) + residual
    for index, mask in enumerate(masks):
        series = content["data"][index]
        samples = series["samples"]
        starts = np.array([_ordinal(s["time_from"]) for s in samples])
        ends = np.array([_ordinal(s["time_to"]) for s in samples])
        observed = missed = 0
        misses = []
        for x in range(left, right + 1):
            ys = np.flatnonzero(mask[top : bottom + 1, x]) + top
            time = first + (last - first) * (x - left) / (right - left)
            selected = np.flatnonzero((starts <= time) & (ends >= time))
            ranges = [pair for n in selected for pair in samples[n]["visible_value_ranges_pct"]]
            for y in ys:
                observed += 1
                value = slope * float(y) + intercept
                if not any(low - precision <= value <= high + precision for low, high in ranges):
                    missed += 1
                    if len(misses) < 5:
                        misses.append({"date_ordinal": time, "value_pct": value})
        results.append(
            {
                "series": series["name"],
                "source_visible_points": observed,
                "uncovered_points": missed,
                "examples": misses,
            }
        )
    if any(r["source_visible_points"] < 100 or r["uncovered_points"] for r in results):
        return {"passed": False, "series": results, "independent_source_precision_pct": precision}
    return {
        "passed": True,
        "series": results,
        "independent_source_precision_pct": precision,
        "scope": "源图全部可见曲线，非未公开精确日频序列",
        "method": "ORIGINAL_IMAGE_MEDIAN_AXES_V1",
    }


def extract_bars(pixels, ocr, *, years, labels, series):
    """按原图柱形逐年提取区间；原图未标出的精确值不伪装为原值。"""
    import numpy as np

    bounds, slope, intercept, residual = _axes(pixels, ocr, labels, snap_grid=False)
    left, top, right, bottom = bounds
    rgb = pixels.astype(np.int16)
    masks = [
        (rgb[:, :, 2] - rgb[:, :, 0] > 65) & (rgb[:, :, 2] - rgb[:, :, 1] > 65),
        (rgb[:, :, 0] - rgb[:, :, 1] > 80) & (rgb[:, :, 2] - rgb[:, :, 1] > 80),
    ]
    intercept = _bar_zero(masks, bounds, slope, intercept)
    zero = -intercept / slope
    rows = []
    for j, year in enumerate(years):
        x0 = round(left + (right - left) * j / len(years))
        x1 = round(left + (right - left) * (j + 1) / len(years))
        values = []
        for mask in masks:
            ys, xs = np.nonzero(mask[top : bottom + 1, x0:x1])
            if len(ys) < 20:
                raise ValueError("CHART_BAR_MISSING")
            absolute = ys + top
            end = int(absolute.min()) if np.median(absolute) < zero else int(absolute.max())
            center = slope * end + intercept
            margin = abs(slope) * 1.5 + residual
            values.append([round(center - margin, 6), round(center + margin, 6)])
        rows.append([year, *values])
    return {
        "columns": ["年度", series[0] + "图面百分比区间", series[1] + "图面百分比区间"],
        "rows": rows,
        "precision": "图面区间；图中未给出精确柱值，不作精确原值使用",
        "axes": {"x": years, "y": labels, "unit": "%"},
        "series": series,
        "independent_axis_residual_pct": residual,
    }


def _bar_zero(masks, bounds, slope, intercept):
    """零值取正负柱的共同基线，而不是易受负号字形影响的文字框中心。"""
    import numpy as np

    left, top, right, bottom = bounds
    positives, negatives = [], []
    rough = -intercept / slope
    for mask in masks:
        columns = np.flatnonzero(mask[top : bottom + 1, left : right + 1].any(0)) + left
        for group in np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1):
            if not len(group):
                continue
            yy, _ = np.nonzero(mask[top : bottom + 1, int(group[0]) : int(group[-1]) + 1])
            yy = yy + top
            if np.median(yy) < rough:
                positives.append(int(yy.max()))
            else:
                negatives.append(int(yy.min()))
    if not positives or not negatives or max(positives) > min(negatives) + 2:
        raise ValueError("CHART_BAR_ZERO_BASELINE_NOT_CONFIRMED")
    baseline = (float(np.median(positives)) + float(np.median(negatives))) / 2
    if abs(baseline - rough) > 3:
        raise ValueError("CHART_BAR_ZERO_BASELINE_DISAGREEMENT")
    return -slope * baseline


def verify_bars(pixels, ocr, content):
    """不按年份窗口提取：独立枚举全部彩色柱组件，再逐列对照文字表格。"""
    import numpy as np

    labels = "".join(ocr.get("text", "").split())
    if any("".join(name.split()) not in labels for name in content["series"]):
        raise ValueError("CHART_SOURCE_SERIES_LABEL_MISMATCH")
    bounds, slope, intercept, residual = _axes(pixels, ocr, content["axes"]["y"], snap_grid=False)
    left, top, right, bottom = bounds
    rgb = pixels.astype(np.int16)
    masks = [
        (rgb[:, :, 2] - rgb[:, :, 0] > 65) & (rgb[:, :, 2] - rgb[:, :, 1] > 65),
        (rgb[:, :, 0] - rgb[:, :, 1] > 80) & (rgb[:, :, 2] - rgb[:, :, 1] > 80),
    ]
    intercept = _bar_zero(masks, bounds, slope, intercept)
    checks = []
    for number, mask in enumerate(masks):
        columns = np.flatnonzero(mask[top : bottom + 1, left : right + 1].any(0)) + left
        groups = np.split(columns, np.flatnonzero(np.diff(columns) > 1) + 1)
        if len(groups) != len(content["rows"]):
            raise ValueError("CHART_BAR_COMPONENT_COUNT_MISMATCH")
        for row, group in zip(content["rows"], groups, strict=True):
            ys, _ = np.nonzero(mask[top : bottom + 1, int(group[0]) : int(group[-1]) + 1])
            ys = ys + top
            zero = -intercept / slope
            endpoint = ys.min() if np.median(ys) < zero else ys.max()
            value = slope * float(endpoint) + intercept
            low, high = row[number + 1]
            if not low <= value <= high:
                raise ValueError("CHART_BAR_VALUE_NOT_COVERED")
            checks.append(
                {
                    "year": row[0],
                    "series": content["series"][number],
                    "source_visible_points": len(ys),
                    "endpoint_covered": True,
                }
            )
    return {"passed": True, "bars": checks, "method": "ORIGINAL_BAR_COMPONENTS_V1"}
